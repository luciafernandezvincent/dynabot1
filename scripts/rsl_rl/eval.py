# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate a trained RSL-RL checkpoint headlessly for a fixed number of steps."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from delay_args import add_delay_args, maybe_wrap_with_delay  # isort: skip
from experiment_config import apply_experiment_config  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
add_delay_args(parser)
parser.add_argument("--num_steps", type=int, default=1000, help="Number of simulation steps to run the evaluation for.")
parser.add_argument(
    "--torque_legs", type=str, default="front_left,back_left",
    help=(
        "Patas cuyos torques se guardan en torques.csv, separadas por coma "
        "(front_left, front_right, back_left, back_right). Vacio ('') desactiva el CSV."
    ),
)
parser.add_argument(
    "--torque_env", type=int, default=0,
    help="Indice del entorno cuyos torques se guardan en torques.csv (el eval corre muchos en paralelo).",
)
parser.add_argument(
    "--experiment_config", type=str, default=None,
    help=(
        "Path to the SAME YAML used to train the checkpoint being evaluated. Necesario cuando el "
        "experimento cambia la ARQUITECTURA del modelo (agent.actor/critic.hidden_dims): sin esto "
        "eval.py reconstruye la red con la forma default de la tarea y falla al cargar el checkpoint."
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# evaluation always runs headless
args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import csv
import json
import os

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from scipy import signal

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab.utils.math as math_utils

import dynabot1.tasks  # noqa: F401
from dynabot1.wrappers import ActionDelayWrapper


# Punta del pie en el frame de hand_link, medida sobre la malla de colision de dyna1_gazebo3.usd
# (punto mas bajo del pie en la pose cero del USD; identico en las 4 patas). El origen de hand_link
# es la rodilla, asi que sin este offset el "despeje de pie" mediria la rodilla.
FOOT_TIP_OFFSET_IN_HAND_FRAME = (0.1863, -0.0413, 0.0)


def evaluate_step(env, obs, actions, rewards, dones, extras, step: int, state: dict):
    """Called once per simulation step. Fill this in with per-step evaluation logic."""
    term_manager = env.unwrapped.termination_manager
    base_contact = term_manager.get_term("base_contact")
    shoulder_contact = term_manager.get_term("shoulder_contact")
    state["base_falls"] += int(base_contact.sum().item())
    state["shoulder_falls"] += int(shoulder_contact.sum().item())
    state["episodes"] += int(dones.sum().item())

    # los mismos conteos SIN sumar sobre los envs. Con el delay estratificado cada env tiene un lag
    # conocido y fijo toda la corrida, asi que esto es lo que permite responder "se cae a lag 7?"
    # en vez de solo "se cae". Cuesta tres tensores de (num_envs,).
    if state["base_falls_per_env"] is None:
        zeros = torch.zeros_like(base_contact, dtype=torch.float)
        state["base_falls_per_env"] = zeros.clone()
        state["shoulder_falls_per_env"] = zeros.clone()
        state["episodes_per_env"] = zeros.clone()
    state["base_falls_per_env"] += base_contact.float()
    state["shoulder_falls_per_env"] += shoulder_contact.float()
    state["episodes_per_env"] += dones.float()

    # foot contact analysis: impact force, stride frequency, and duty factor per foot
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    if state["foot_ids"] is None:
        state["foot_ids"], state["foot_names"] = contact_sensor.find_bodies(".*hand_link")
        num_feet = len(state["foot_ids"])
        state["foot_touchdowns"] = torch.zeros(num_feet, device=env.unwrapped.device)
        state["foot_contact_steps"] = torch.zeros(num_feet, device=env.unwrapped.device)

    first_contact = contact_sensor.compute_first_contact(env.unwrapped.step_dt)[:, state["foot_ids"]]
    in_contact = contact_sensor.data.current_contact_time[:, state["foot_ids"]] > 0.0
    state["foot_touchdowns"] += first_contact.sum(dim=0).float()
    state["foot_contact_steps"] += in_contact.sum(dim=0).float()
    # idem, sin colapsar la dimension de env: (num_envs, num_feet)
    if state["foot_touchdowns_per_env"] is None:
        state["foot_touchdowns_per_env"] = torch.zeros_like(first_contact, dtype=torch.float)
        state["foot_contact_steps_per_env"] = torch.zeros_like(first_contact, dtype=torch.float)
    state["foot_touchdowns_per_env"] += first_contact.float()
    state["foot_contact_steps_per_env"] += in_contact.float()

    # altura de cada pie y si esta apoyado, para medir cuanto despega en el swing.
    #
    # OJO (fix 21/09/2026): los ids de `contact_sensor.find_bodies` estan en el orden del SENSOR
    # (DFS: base, BL shoulder/arm/hand, BR ...) y `body_pos_w` esta en el orden de la ARTICULACION
    # (BFS: base, 4 shoulders, 4 arms, 4 hands). Indexar uno con los ids del otro leia otros
    # cuerpos: el "pie" era [hombro FL, cadera BR, rodilla BL, rodilla FR]. Los ids de posicion se
    # resuelven contra la articulacion, en el mismo orden de nombres que el sensor.
    #
    # Ademas el origen de hand_link es la articulacion de la RODILLA (arm_to_hand), no el pie: la
    # punta del pie esta ~20 cm mas abajo sobre el mismo cuerpo. Se mide ese punto.
    articulation_for_feet = env.unwrapped.scene.articulations["robot"]
    if state["foot_body_ids"] is None:
        state["foot_body_ids"], _ = articulation_for_feet.find_bodies(state["foot_names"], preserve_order=True)
        state["foot_tip_offset"] = torch.tensor(
            FOOT_TIP_OFFSET_IN_HAND_FRAME, device=env.unwrapped.device
        ).repeat(env.unwrapped.num_envs, len(state["foot_body_ids"]), 1)
    hand_pos = articulation_for_feet.data.body_pos_w[:, state["foot_body_ids"]]
    hand_quat = articulation_for_feet.data.body_quat_w[:, state["foot_body_ids"]]
    foot_tip = hand_pos + math_utils.quat_apply(hand_quat, state["foot_tip_offset"])
    foot_z = foot_tip[..., 2].detach().cpu().numpy()
    if state["foot_z"] is None:
        state["foot_z"] = []
        state["foot_in_contact"] = []
    state["foot_z"].append(foot_z)
    state["foot_in_contact"].append(in_contact.detach().cpu().numpy())

    # contacto de codo (arm_link) contra el piso: no hay terminacion dura para esto (a diferencia
    # de base_link/shoulder_link), solo la penalizacion blanda de undesired_contacts. Se mide
    # aparte para saber si el robot "camina con los codos".
    if state["arm_ids"] is None:
        state["arm_ids"], state["arm_names"] = contact_sensor.find_bodies(".*arm_link")
    arm_in_contact = contact_sensor.data.current_contact_time[:, state["arm_ids"]] > 0.0

    # altura del segmento medio de la pata (arm_link, la "rodilla") sobre el piso. A diferencia
    # de elbow_contact_rate, que solo cuenta cuando el sensor YA esta en contacto, esto mide la
    # distancia real en todo momento: yendo hacia atras el robot puede pasar muy cerca del piso
    # (casi "gateando") sin llegar a disparar el umbral de contacto de undesired_contacts.
    #
    # fix 21/09/2026: el origen de arm_link es la CADERA (pivote shoulder_to_arm), siempre a la
    # altura del torso. La rodilla real es el origen de hand_link (pivote arm_to_hand, ~153 mm del
    # piso en reposo). Y los ids salen de la articulacion, no del sensor (ver el fix del pie arriba):
    # antes esto leia [hombro BR, cadera BL, cadera FR, rodilla FL].
    if state["knee_body_ids"] is None:
        state["knee_body_ids"], _ = articulation_for_feet.find_bodies(state["foot_names"], preserve_order=True)
        state["knee_names"] = [name.replace("_hand_link", "_knee") for name in state["foot_names"]]
    knee_z = articulation_for_feet.data.body_pos_w[:, state["knee_body_ids"], 2].detach().cpu().numpy()
    if state["knee_z"] is None:
        state["knee_z"] = []
        state["knee_backward_mask"] = []
        state["knee_forward_mask"] = []
    state["knee_z"].append(knee_z)
    state["arm_contact_steps"] += arm_in_contact.sum().item()
    arm_forces = torch.norm(contact_sensor.data.net_forces_w[:, state["arm_ids"], :], dim=-1)
    arm_force_in_contact = arm_forces[arm_in_contact]
    if arm_force_in_contact.numel() > 0:
        state["arm_contact_force_sum"] += arm_force_in_contact.sum().item()
        state["arm_contact_force_max"] = max(state["arm_contact_force_max"], arm_force_in_contact.max().item())

    foot_forces = torch.norm(contact_sensor.data.net_forces_w[:, state["foot_ids"], :], dim=-1)
    impact_forces = foot_forces[first_contact]
    if impact_forces.numel() > 0:
        state["impact_force_sum"] += impact_forces.sum().item()
        state["impact_force_sq_sum"] += (impact_forces**2).sum().item()
        state["impact_force_count"] += impact_forces.numel()
        state["impact_force_max"] = max(state["impact_force_max"], impact_forces.max().item())
    # impacto por env y paso (suma y cantidad de pisadas), para desglosarlo por direccion de comando
    state["impact_sum_frames"].append((foot_forces * first_contact).sum(dim=1).detach().cpu().numpy())
    state["impact_count_frames"].append(first_contact.sum(dim=1).detach().cpu().numpy())

    # Record joint positions for smoothness analysis
    articulation = env.unwrapped.scene.articulations["robot"]
    # desviacion respecto de la pose default: cuanto "flexionado" camina. En el robot real los
    # actuadores hacen mejor fuerza cerca de la default, asi que menos es mejor.
    default_joint_pos = articulation.data.default_joint_pos
    deviation = torch.abs(articulation.data.joint_pos - default_joint_pos)
    if state["joint_names"] is None:
        state["joint_names"] = articulation.data.joint_names
    state["joint_dev_sum"] += deviation.mean().item()
    state["joint_dev_per_joint_sum"] += deviation.mean(dim=0).detach().cpu().numpy()
    state["joint_dev_steps"] += 1

    joint_pos = articulation.data.joint_pos.detach().cpu().numpy()  # Shape: (num_envs, num_joints)
    if state["joint_positions"] is None:
        state["joint_positions"] = []
    state["joint_positions"].append(joint_pos)

    # Saturacion del motor sobre TODAS las articulaciones (no solo las dos patas del CSV).
    # En DCMotorCfg el techo de par no es fijo: es la curva par-velocidad, asi que el actuador
    # esta saturado exactamente cuando recorta el pedido del PD, o sea applied != computed.
    computed_torque = articulation.data.computed_torque
    applied_torque = articulation.data.applied_torque
    saturated = (computed_torque - applied_torque).abs() > 1e-4
    state["torque_sat_steps"] += saturated.sum(dim=0).float().detach().cpu().numpy()
    state["torque_abs_sum"] += applied_torque.abs().mean(dim=0).detach().cpu().numpy()
    applied_abs_max = applied_torque.abs().max(dim=0).values.detach().cpu().numpy()
    state["torque_abs_max"] = (
        applied_abs_max if state["torque_abs_max"] is None else np.maximum(state["torque_abs_max"], applied_abs_max)
    )
    state["torque_steps"] += 1

    # torque por articulacion de las patas pedidas, para exportar a CSV. Se guardan las dos series:
    # applied_torque (lo que el actuador realmente entrega) y computed_torque (el pedido crudo del
    # PD): donde se separan, el motor esta saturado, y el valor de applied es el techo en ese instante.
    if state["torque_joint_ids"] is None and args_cli.torque_legs.strip():
        legs = [leg.strip() for leg in args_cli.torque_legs.split(",") if leg.strip()]
        try:
            # el nombre de la pata va EN EL MEDIO en la articulacion de abduccion
            # (base_to_front_left_shoulder), asi que anclar el patron al inicio con f"{leg}_.*" la
            # perdia en silencio: matcheaba las otras dos de la pata y por eso no daba error.
            ids, names = articulation.find_joints([f".*{leg}.*" for leg in legs], preserve_order=True)
        except ValueError as exc:
            # una pata mal escrita no debe tirar abajo el eval entero: se avisa y se sigue sin CSV
            print(f"[WARN] --torque_legs='{args_cli.torque_legs}' no matchea articulaciones ({exc}). Sin torques.csv.")
            ids, names = [], []
        # se listan los nombres resueltos para que un match incompleto se vea, en vez de aparecer
        # como columnas faltantes recien al abrir el CSV
        print(f"[INFO] Torques a CSV para {len(names)} articulaciones: {names}")
        if args_cli.torque_env >= env.unwrapped.num_envs:
            print(
                f"[WARN] --torque_env={args_cli.torque_env} fuera de rango ({env.unwrapped.num_envs} envs)."
                " Sin torques.csv."
            )
            ids, names = [], []
        state["torque_joint_ids"] = ids
        state["torque_joint_names"] = names
        state["torque_rows"] = []
    if state["torque_joint_ids"]:
        ids = state["torque_joint_ids"]
        row_applied = applied_torque[args_cli.torque_env, ids].detach().cpu().numpy().tolist()
        row_computed = computed_torque[args_cli.torque_env, ids].detach().cpu().numpy().tolist()
        row_sat = saturated[args_cli.torque_env, ids].detach().cpu().numpy().astype(int).tolist()
        state["torque_rows"].append(
            [step, (step + 1) * env.unwrapped.step_dt] + row_applied + row_computed + row_sat
        )

    # Record base position and orientation for path tracking and stability
    root_state = articulation.data.root_state_w  # (num_envs, 13)
    base_pos = root_state[:, :3].detach().cpu().numpy()  # Position
    base_quat = root_state[:, 3:7].detach().cpu().numpy()  # Quaternion

    if state["base_positions"] is None:
        state["base_positions"] = []
        state["base_quats"] = []
        state["done_history"] = []

    state["base_positions"].append(base_pos)
    state["base_quats"].append(base_quat)
    # tracks, per step/env, whether this pose is a post-reset teleport rather than a continuous rotation
    state["done_history"].append(dones.detach().cpu().numpy().astype(bool))

    # Record commanded vs. actual body-frame velocity for velocity tracking accuracy
    if state["has_velocity_command"] is None:
        state["has_velocity_command"] = "base_velocity" in env.unwrapped.command_manager.active_terms
    if state["has_velocity_command"]:
        cmd_vel = env.unwrapped.command_manager.get_command("base_velocity").detach().cpu().numpy()  # (num_envs, 3): vx, vy, wz
        actual_lin_vel_b = articulation.data.root_lin_vel_b[:, :2].detach().cpu().numpy()
        actual_ang_vel_b = articulation.data.root_ang_vel_b[:, 2:3].detach().cpu().numpy()
        actual_vel = np.concatenate([actual_lin_vel_b, actual_ang_vel_b], axis=-1)  # (num_envs, 3): vx, vy, wz

        if state["cmd_vels"] is None:
            state["cmd_vels"] = []
            state["actual_vels"] = []
        state["cmd_vels"].append(cmd_vel)
        state["actual_vels"].append(actual_vel)
        # marca los steps donde se pidio caminar hacia atras/adelante, para poder aislar el
        # "gateo" hacia atras del caso hacia adelante (donde no se observo el problema)
        state["knee_backward_mask"].append(cmd_vel[:, 0] < -0.1)
        state["knee_forward_mask"].append(cmd_vel[:, 0] > 0.1)
        # rapidez horizontal real, para derivar el largo de paso (velocidad / frecuencia)
        state["speed_sum"] += float(np.linalg.norm(actual_lin_vel_b, axis=-1).mean())
        state["speed_steps"] += 1


def compute_movement_smoothness(joint_positions):
    """Calculate smoothness as mean of second derivatives (acceleration magnitude)."""
    if len(joint_positions) < 3:
        return 0.0

    positions = np.array(joint_positions)  # Shape: (num_steps, num_envs, num_joints)
    num_steps, num_envs, num_joints = positions.shape

    # Calculate velocity (first derivative)
    velocity = np.diff(positions, axis=0)  # Shape: (num_steps-1, num_envs, num_joints)

    # Calculate acceleration (second derivative)
    acceleration = np.diff(velocity, axis=0)  # Shape: (num_steps-2, num_envs, num_joints)

    # Mean magnitude of acceleration across all joints and envs
    acc_magnitude = np.mean(np.abs(acceleration))

    # Smoothness: inverse of acceleration (lower acceleration = smoother)
    smoothness = 1.0 / (1.0 + acc_magnitude)

    return float(smoothness)


def compute_gait_metrics(foot_touchdowns, foot_contact_steps, foot_names, num_envs: int, num_steps: int, step_dt: float):
    """Compute per-foot stride frequency (Hz) and duty factor from accumulated contact events."""
    total_time = num_steps * step_dt
    total_samples = num_envs * num_steps
    touchdowns = foot_touchdowns.cpu().numpy()
    contact_steps = foot_contact_steps.cpu().numpy()

    stride_frequency_per_foot = {}
    duty_factor_per_foot = {}
    for i, name in enumerate(foot_names):
        stride_frequency_per_foot[name] = float(touchdowns[i] / num_envs / total_time) if total_time > 0 else 0.0
        duty_factor_per_foot[name] = float(contact_steps[i] / total_samples) if total_samples > 0 else 0.0
    return stride_frequency_per_foot, duty_factor_per_foot


def compute_foot_clearance(foot_z, foot_in_contact, foot_names):
    """Cuanto despega cada pie del suelo durante el swing, en metros.

    El origen del body del pie no coincide con la planta, asi que la altura absoluta no sirve
    como despeje. Se mide relativo a la altura mediana del pie MIENTRAS ESTA APOYADO, que es el
    cero efectivo de ese pie. Complementa a stride_frequency/duty_factor, que solo miden tiempo
    de contacto: un pie puede levantarse milimetros y aun asi dar un duty factor perfecto.
    """
    if not foot_z:
        return {}, {}, 0.0, 0.0

    z = np.array(foot_z)  # (num_steps, num_envs, num_feet)
    contact = np.array(foot_in_contact)  # (num_steps, num_envs, num_feet)

    mean_per_foot = {}
    peak_per_foot = {}
    for i, name in enumerate(foot_names):
        z_foot = z[:, :, i]
        contact_foot = contact[:, :, i]
        if not contact_foot.any() or not (~contact_foot).any():
            mean_per_foot[name] = 0.0
            peak_per_foot[name] = 0.0
            continue
        stance_z = float(np.median(z_foot[contact_foot]))
        swing_clearance = z_foot[~contact_foot] - stance_z
        mean_per_foot[name] = float(np.mean(swing_clearance))
        peak_per_foot[name] = float(np.percentile(swing_clearance, 95))

    mean_all = float(np.mean(list(mean_per_foot.values()))) if mean_per_foot else 0.0
    peak_all = float(np.mean(list(peak_per_foot.values()))) if peak_per_foot else 0.0
    return mean_per_foot, peak_per_foot, mean_all, peak_all


def compute_knee_clearance(knee_z, knee_backward_mask, knee_forward_mask, arm_names):
    """Altura minima/media del segmento arm_link (aprox. la rodilla) respecto al piso.

    Complementa a elbow_contact_rate: esa metrica solo ve el instante en que el sensor de
    contacto YA se activo, esto mide la distancia real en todo momento. Se reporta en general y
    separado por direccion de comando (adelante / atras), porque el "gateo" se observo
    especificamente yendo hacia atras. Los desgloses direccionales quedan en None (no 0.0) si el
    eval no incluyo comando en esa direccion, para no confundir "no hay datos" con "toco el piso".

    Tambien se desglosa el minimo POR PATA y direccion (min_per_knee_backward/forward): el minimo
    agregado de las 4 patas puede esconder que una sola pata concentra todo el problema (visto en
    exp_040: el promedio general por pata no distinguia direccion y no alcanzaba para confirmar
    cual pata rozaba el piso yendo para atras).
    """
    if not knee_z:
        return {}, 0.0, 0.0, None, None, None, None, None, None

    z = np.array(knee_z)  # (num_steps, num_envs, num_knees)
    mean_per_knee = {name: float(np.mean(z[:, :, i])) for i, name in enumerate(arm_names)}
    mean_all = float(np.mean(z))
    min_all = float(np.min(z))

    def _masked_stats(mask_list):
        if not mask_list or len(mask_list) != z.shape[0]:
            return None, None
        mask = np.array(mask_list)  # (num_steps, num_envs)
        if not mask.any():
            return None, None
        z_masked = z[mask]  # (num_samples, num_knees)
        return float(np.mean(z_masked)), float(np.min(z_masked))

    def _masked_min_per_knee(mask_list):
        if not mask_list or len(mask_list) != z.shape[0]:
            return None
        mask = np.array(mask_list)  # (num_steps, num_envs)
        if not mask.any():
            return None
        z_masked = z[mask]  # (num_samples, num_knees)
        return {name: float(np.min(z_masked[:, i])) for i, name in enumerate(arm_names)}

    mean_backward, min_backward = _masked_stats(knee_backward_mask)
    mean_forward, min_forward = _masked_stats(knee_forward_mask)
    min_per_knee_backward = _masked_min_per_knee(knee_backward_mask)
    min_per_knee_forward = _masked_min_per_knee(knee_forward_mask)

    return (
        mean_per_knee, mean_all, min_all, mean_backward, min_backward, mean_forward, min_forward,
        min_per_knee_backward, min_per_knee_forward,
    )


def quat_to_euler(quat):
    """Convert quaternion (w,x,y,z) to euler angles (roll, pitch, yaw) in radians."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = np.arcsin(2 * (w * y - z * x))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    return np.stack([roll, pitch, yaw], axis=-1)


def compute_orientation_stability(base_quats, dt=0.02):
    """Measure orientation stability - how constant is roll/pitch (should be ~0)."""
    if len(base_quats) < 2:
        return 0.0

    quats = np.array(base_quats)  # Shape: (num_steps, num_envs, 4)

    # Convert to euler angles
    euler = quat_to_euler(quats)  # Shape: (num_steps, num_envs, 3)

    # Extract roll and pitch (yaw doesn't matter for stability)
    roll_pitch = euler[:, :, :2]  # Shape: (num_steps, num_envs, 2)

    # Variance over time per env/axis, then averaged across all envs and axes
    roll_pitch_variance = np.var(roll_pitch, axis=0).mean()

    # Return value from 0 to 1 (1 = perfectly stable)
    stability = 1.0 / (1.0 + roll_pitch_variance * 10)

    return float(stability)


def compute_orientation_smoothness(base_quats, done_history, dt=0.02):
    """Measure how smoothly orientation changes - no abrupt rotations."""
    if len(base_quats) < 3:
        return 0.0

    quats = np.array(base_quats)  # Shape: (num_steps, num_envs, 4)
    euler = quat_to_euler(quats)  # Shape: (num_steps, num_envs, 3)
    euler = np.unwrap(euler, axis=0)  # remove artificial +-pi wraparound jumps before differentiating

    # Calculate angular velocity (derivative of euler angles)
    angular_vel = np.diff(euler, axis=0) / dt  # Shape: (num_steps-1, num_envs, 3)

    # A step where an env resets teleports its pose to the spawn state, which is not a real rotation.
    # Mark velocity/acceleration samples that span such a reset as invalid so they don't skew the score.
    dones = np.array(done_history)  # Shape: (num_steps, num_envs)
    valid_vel = ~dones[1:]  # Shape: (num_steps-1, num_envs), aligned with angular_vel

    # Calculate angular acceleration (second derivative)
    angular_acc = np.diff(angular_vel, axis=0) / dt  # Shape: (num_steps-2, num_envs, 3)
    valid_acc = valid_vel[:-1] & valid_vel[1:]  # Shape: (num_steps-2, num_envs)

    if not valid_acc.any():
        return 0.0

    # Mean magnitude of angular acceleration, excluding samples that span a reset
    acc_magnitude = np.mean(np.abs(angular_acc[valid_acc]))

    # Smoothness: inverse of angular acceleration
    smoothness = 1.0 / (1.0 + acc_magnitude)

    return float(smoothness)


def compute_velocity_tracking_accuracy(cmd_vels, actual_vels, std: float = 0.5):
    """Measure how well body-frame velocity tracks the [vx, vy, wz] command, step-by-step and per env.

    Mirrors the exponential-kernel error used by the training rewards (track_lin_vel_xy_exp /
    track_ang_vel_z_exp), so the score is directly comparable to what the policy was optimized for.
    """
    if len(cmd_vels) < 1:
        return 0.0

    cmd = np.array(cmd_vels)  # Shape: (num_steps, num_envs, 3): vx, vy, wz
    actual = np.array(actual_vels)  # Shape: (num_steps, num_envs, 3): vx, vy, wz

    lin_vel_error = np.sum((cmd[..., :2] - actual[..., :2]) ** 2, axis=-1)  # Shape: (num_steps, num_envs)
    ang_vel_error = (cmd[..., 2] - actual[..., 2]) ** 2  # Shape: (num_steps, num_envs)

    lin_tracking = np.mean(np.exp(-lin_vel_error / std**2))
    ang_tracking = np.mean(np.exp(-ang_vel_error / std**2))

    return float((lin_tracking + ang_tracking) / 2.0)


def compute_directional_smoothness(state: dict, dt: float) -> dict:
    """Suavidad separada por direccion de comando (adelante: vx > 0.1, atras: vx < -0.1).

    `movement_smoothness` esta saturada (~0.988 en todos los experimentos) y mezcla direcciones,
    asi que no distingue si un cambio suaviza la marcha HACIA ADELANTE. Informativo: no entra al
    score. Menos es mejor en todas:
      joint_acc_mean      aceleracion articular media, rad/paso^2 (misma unidad que movement_smoothness)
      base_acc_z_mean     aceleracion vertical media del torso, m/s^2 (rebote)
      base_ang_acc_mean   aceleracion angular media de roll/pitch del torso, rad/s^2 (cabeceo brusco)
      impact_force_mean   fuerza media en el instante de la pisada, N
    """
    if not state["joint_positions"] or not state["knee_forward_mask"]:
        return {}
    q = np.array(state["joint_positions"])  # (T, E, J)
    if q.shape[0] < 3 or len(state["knee_forward_mask"]) != q.shape[0]:
        return {}
    dones = np.array(state["done_history"])  # (T, E)
    valid_vel = ~dones[1:]
    valid_acc = valid_vel[:-1] & valid_vel[1:]  # (T-2, E), sin muestras que crucen un reset

    joint_acc = np.abs(np.diff(q, n=2, axis=0)).mean(axis=-1)  # (T-2, E)
    z = np.array(state["base_positions"])[:, :, 2]
    base_acc_z = np.abs(np.diff(z, n=2, axis=0)) / dt**2
    euler = np.unwrap(quat_to_euler(np.array(state["base_quats"]))[:, :, :2], axis=0)
    base_ang_acc = np.abs(np.diff(euler, n=2, axis=0)).mean(axis=-1) / dt**2
    impact_sum = np.array(state["impact_sum_frames"])
    impact_cnt = np.array(state["impact_count_frames"])

    out = {}
    for direction, mask_list in (("forward", state["knee_forward_mask"]), ("backward", state["knee_backward_mask"])):
        mask = np.array(mask_list)  # (T, E)
        m = mask[1:-1] & valid_acc
        if not m.any():
            out[direction] = None
            continue
        entry = {
            "joint_acc_mean": float(joint_acc[m].mean()),
            "base_acc_z_mean": float(base_acc_z[m].mean()),
            "base_ang_acc_mean": float(base_ang_acc[m].mean()),
        }
        if impact_sum.shape == mask.shape:
            n = impact_cnt[mask].sum()
            entry["impact_force_mean"] = float(impact_sum[mask].sum() / n) if n > 0 else 0.0
        out[direction] = entry
    return out


def compute_results(env, state: dict) -> dict:
    """Called once after the simulation loop ends. Fill this in to aggregate the final metrics."""
    total_falls = state["base_falls"] + state["shoulder_falls"]
    episodes = state["episodes"]

    count = state["impact_force_count"]
    impact_mean = state["impact_force_sum"] / count if count > 0 else 0.0
    impact_var = state["impact_force_sq_sum"] / count - impact_mean**2 if count > 0 else 0.0

    # Calculate movement metrics
    smoothness = compute_movement_smoothness(state["joint_positions"]) if state["joint_positions"] else 0.0

    # Calculate gait metrics (stride frequency and duty factor) from foot contact events
    stride_frequency_per_foot = {}
    duty_factor_per_foot = {}
    if state["foot_touchdowns"] is not None:
        stride_frequency_per_foot, duty_factor_per_foot = compute_gait_metrics(
            state["foot_touchdowns"],
            state["foot_contact_steps"],
            state["foot_names"],
            env.unwrapped.num_envs,
            args_cli.num_steps,
            env.unwrapped.step_dt,
        )
    stride_frequency_mean = float(np.mean(list(stride_frequency_per_foot.values()))) if stride_frequency_per_foot else 0.0
    duty_factor_mean = float(np.mean(list(duty_factor_per_foot.values()))) if duty_factor_per_foot else 0.0

    # Despeje de pie: cuanto levanta las patas realmente (no solo cuanto tiempo estan sin contacto)
    clearance_mean_per_foot, clearance_peak_per_foot, clearance_mean, clearance_peak = compute_foot_clearance(
        state["foot_z"], state["foot_in_contact"], state["foot_names"] or []
    )

    # Altura de la rodilla (arm_link) sobre el piso: en general, y aislada por direccion de comando
    (
        knee_mean_per_link,
        knee_height_mean,
        knee_height_min,
        knee_height_mean_backward,
        knee_height_min_backward,
        knee_height_mean_forward,
        knee_height_min_forward,
        knee_height_min_per_link_backward,
        knee_height_min_per_link_forward,
    ) = compute_knee_clearance(
        state["knee_z"], state["knee_backward_mask"], state["knee_forward_mask"], state["knee_names"] or []
    )

    # Calculate orientation metrics
    orientation_stability = compute_orientation_stability(state["base_quats"]) if state["base_quats"] else 0.0
    orientation_smoothness = (
        compute_orientation_smoothness(state["base_quats"], state["done_history"], dt=env.unwrapped.step_dt)
        if state["base_quats"]
        else 0.0
    )

    # Saturacion de los actuadores: fraccion de muestras (env x step x joint) en las que el modelo
    # de motor tuvo que recortar el par pedido por el PD. Alta saturacion = la politica pide mas par
    # del que el motor real puede dar, asi que la marcha no se transfiere al hardware.
    torque_sat_rate = 0.0
    torque_sat_per_joint = {}
    torque_mean_per_joint = {}
    torque_max_per_joint = {}
    if state["torque_steps"] > 0 and state["joint_names"]:
        samples_per_joint = state["torque_steps"] * env.unwrapped.num_envs
        sat_per_joint = state["torque_sat_steps"] / samples_per_joint
        mean_per_joint = state["torque_abs_sum"] / state["torque_steps"]
        torque_sat_rate = float(np.mean(sat_per_joint))
        torque_sat_per_joint = {n: float(v) for n, v in zip(state["joint_names"], sat_per_joint)}
        torque_mean_per_joint = {n: float(v) for n, v in zip(state["joint_names"], mean_per_joint)}
        torque_max_per_joint = {n: float(v) for n, v in zip(state["joint_names"], state["torque_abs_max"])}

    # Calculate velocity tracking
    velocity_tracking = (
        compute_velocity_tracking_accuracy(state["cmd_vels"], state["actual_vels"]) if state["cmd_vels"] else 0.0
    )

    return {
        "num_envs": env.unwrapped.num_envs,
        "num_steps": args_cli.num_steps,
        "episodes_completed": episodes,
        "base_falls": state["base_falls"],
        "shoulder_falls": state["shoulder_falls"],
        "total_falls": total_falls,
        "fall_rate_per_episode": total_falls / episodes if episodes > 0 else 0.0,
        "num_footsteps": count,
        "impact_force_mean": impact_mean,
        "impact_force_std": impact_var**0.5 if impact_var > 0 else 0.0,
        "impact_force_max": state["impact_force_max"],
        "movement_smoothness": smoothness,
        "stride_frequency_hz_mean": stride_frequency_mean,
        "stride_frequency_hz_per_foot": stride_frequency_per_foot,
        "duty_factor_mean": duty_factor_mean,
        "duty_factor_per_foot": duty_factor_per_foot,
        "mean_speed_mps": (state["speed_sum"] / state["speed_steps"] if state["speed_steps"] > 0 else 0.0),
        # largo de paso = cuanto avanza el cuerpo por cada zancada de una pata. Es lo que se
        # percibe como paso "amplio" o "cortito", y no lo capturan ni el despeje (altura) ni el
        # swing (duracion) por separado.
        "step_length_m": (
            (state["speed_sum"] / state["speed_steps"]) / stride_frequency_mean
            if state["speed_steps"] > 0 and stride_frequency_mean > 0 else 0.0
        ),
        "joint_deviation_mean_rad": (
            state["joint_dev_sum"] / state["joint_dev_steps"] if state["joint_dev_steps"] > 0 else 0.0
        ),
        "joint_deviation_per_joint_rad": (
            {n: float(v) for n, v in zip(state["joint_names"], state["joint_dev_per_joint_sum"] / state["joint_dev_steps"])}
            if state["joint_dev_steps"] > 0 and state["joint_names"] else {}
        ),
        "torque_saturation_rate": torque_sat_rate,
        "torque_saturation_rate_per_joint": torque_sat_per_joint,
        "torque_applied_abs_mean_per_joint_Nm": torque_mean_per_joint,
        "torque_applied_abs_max_per_joint_Nm": torque_max_per_joint,
        "elbow_contact_rate": (
            state["arm_contact_steps"] / (env.unwrapped.num_envs * args_cli.num_steps * max(1, len(state["arm_names"] or [])))
            if state["arm_names"] else 0.0
        ),
        "elbow_contact_force_mean": (
            state["arm_contact_force_sum"] / state["arm_contact_steps"] if state["arm_contact_steps"] > 0 else 0.0
        ),
        "elbow_contact_force_max": state["arm_contact_force_max"],
        "foot_clearance_mean_m": clearance_mean,
        "foot_clearance_peak_m": clearance_peak,
        "foot_clearance_mean_per_foot_m": clearance_mean_per_foot,
        "foot_clearance_peak_per_foot_m": clearance_peak_per_foot,
        "knee_height_mean_m": knee_height_mean,
        "knee_height_min_m": knee_height_min,
        "knee_height_mean_per_link_m": knee_mean_per_link,
        "knee_height_mean_backward_m": knee_height_mean_backward,
        "knee_height_min_backward_m": knee_height_min_backward,
        "knee_height_mean_forward_m": knee_height_mean_forward,
        "knee_height_min_forward_m": knee_height_min_forward,
        "knee_height_min_per_link_backward_m": knee_height_min_per_link_backward,
        "knee_height_min_per_link_forward_m": knee_height_min_per_link_forward,
        "orientation_stability_0to1": orientation_stability,
        "orientation_smoothness_0to1": orientation_smoothness,
        "velocity_tracking_accuracy_0to1": velocity_tracking,
        "smoothness_by_direction": compute_directional_smoothness(state, env.unwrapped.step_dt),
        # version de las metricas de rodilla/pie. v2 = fix del 21/09/2026 (ids de la articulacion,
        # rodilla = origen de hand_link, pie = punta de hand_link). Los results.json sin esta clave
        # son v1 y sus knee_height_* / foot_clearance_* NO miden rodilla ni pie.
        "body_metrics_version": 2,
    }


def describe_delay_regime(delay_wrapper) -> dict:
    """Deja en el results.json bajo que retardo se midio esto.

    Sin esto, un eval a delay fijo 5 y uno a 5..8 producen JSON indistinguibles, y la tabla de
    resultados no tiene como saber que dos numeros no son comparables.
    """
    lags = delay_wrapper.get_delay_steps()
    regime = {
        "mode": (
            "fixed" if delay_wrapper.delay_range is None
            else ("stratified" if delay_wrapper.stratified else "randomized")
        ),
        "delay_range": list(delay_wrapper.delay_range) if delay_wrapper.delay_range else None,
        "action_history_length": delay_wrapper.action_history_length,
        "envs_per_lag": {str(lag): count for lag, count in sorted(delay_wrapper.lag_counts().items())},
    }
    if lags is not None:
        regime["lag_min"] = int(lags.min())
        regime["lag_max"] = int(lags.max())
    return regime


def _subset_frames(frames, env_idx):
    """Toma la lista de (num_envs, ...) por paso y se queda con los envs pedidos."""
    if not frames:
        return frames
    return [frame[env_idx] for frame in frames]


def compute_per_lag_results(env, state: dict, delay_wrapper) -> dict:
    """Recalcula las metricas por separado para cada lag de accion.

    Por que hace falta: casi todas las metricas agregan sobre los 1000 envs, y varias son un
    extremo, no un promedio. `knee_height_min_backward_m`, la del objetivo principal, es un
    `np.min` sobre todos los pasos x todos los envs x las 4 rodillas, asi que con lags mezclados
    el numero sale casi siempre de los envs del lag mas alto (el caso mas dificil). Una politica
    que da 115 mm a lag 4 y 72 mm a lag 7 reporta 72 mm, indistinguible de una que anda mal en
    todo el rango. Este desglose separa las dos situaciones.

    Solo se calcula con reparto estratificado: ahi el mapa env -> lag es fijo y conocido, asi que
    agrupar es exacto. Con el sorteo aleatorio el lag de cada env cambia en cada reset de
    episodio, y `lags` al final de la corrida NO dice a que lag corrio ese env la mayor parte del
    tiempo: agrupar por ese valor seria sencillamente incorrecto.
    """
    lags = delay_wrapper.get_delay_steps()
    if lags is None:
        return {}
    if not delay_wrapper.stratified:
        print("[WARN] Sin --action-delay-stratified el lag de cada env cambia en cada reset; "
              "se omite el desglose por lag porque agrupar por el lag final seria incorrecto.")
        return {}

    lags_np = lags.detach().cpu().numpy()
    distinct = sorted(set(int(lag) for lag in lags_np))
    if len(distinct) < 2:
        return {}  # un solo lag: el desglose seria una copia del agregado

    step_dt = env.unwrapped.step_dt
    num_steps = args_cli.num_steps
    foot_names = state["foot_names"] or []
    knee_names = state["knee_names"] or []
    touchdowns_cpu = (
        state["foot_touchdowns_per_env"].cpu() if state["foot_touchdowns_per_env"] is not None else None
    )
    contact_steps_cpu = (
        state["foot_contact_steps_per_env"].cpu() if state["foot_contact_steps_per_env"] is not None else None
    )

    per_lag = {}
    for lag in distinct:
        env_idx = np.flatnonzero(lags_np == lag)
        num_envs = int(env_idx.size)
        entry = {"delay": lag + 1, "num_envs": num_envs}

        # caidas: la pregunta central del retardo variable, "se rompe en el lag mas alto?"
        if state["episodes_per_env"] is not None:
            idx_t = torch.as_tensor(env_idx, device=state["episodes_per_env"].device)
            falls = float(
                state["base_falls_per_env"][idx_t].sum().item()
                + state["shoulder_falls_per_env"][idx_t].sum().item()
            )
            episodes = float(state["episodes_per_env"][idx_t].sum().item())
            entry["episodes_completed"] = int(episodes)
            entry["total_falls"] = int(falls)
            entry["fall_rate_per_episode"] = falls / episodes if episodes > 0 else 0.0

        # rodilla: el objetivo principal, y la metrica que mas se distorsiona al mezclar lags
        (
            _knee_mean_per_link,
            knee_height_mean,
            _knee_height_min,
            knee_height_mean_backward,
            knee_height_min_backward,
            _knee_height_mean_forward,
            knee_height_min_forward,
            knee_min_per_link_backward,
            _knee_min_per_link_forward,
        ) = compute_knee_clearance(
            _subset_frames(state["knee_z"], env_idx),
            _subset_frames(state["knee_backward_mask"], env_idx),
            _subset_frames(state["knee_forward_mask"], env_idx),
            knee_names,
        )
        entry["knee_height_mean_m"] = knee_height_mean
        entry["knee_height_mean_backward_m"] = knee_height_mean_backward
        entry["knee_height_min_backward_m"] = knee_height_min_backward
        entry["knee_height_min_forward_m"] = knee_height_min_forward
        entry["knee_height_min_per_link_backward_m"] = knee_min_per_link_backward

        # despeje de pie
        _cl_mean_per_foot, _cl_peak_per_foot, clearance_mean, clearance_peak = compute_foot_clearance(
            _subset_frames(state["foot_z"], env_idx),
            _subset_frames(state["foot_in_contact"], env_idx),
            foot_names,
        )
        entry["foot_clearance_mean_m"] = clearance_mean
        entry["foot_clearance_peak_m"] = clearance_peak

        # ritmo de la marcha: el otro objetivo del usuario (caminata lenta)
        if touchdowns_cpu is not None and foot_names:
            idx_cpu = torch.as_tensor(env_idx)
            stride_per_foot, duty_per_foot = compute_gait_metrics(
                touchdowns_cpu[idx_cpu].sum(dim=0),
                contact_steps_cpu[idx_cpu].sum(dim=0),
                foot_names,
                num_envs,
                num_steps,
                step_dt,
            )
            stride_mean = float(np.mean(list(stride_per_foot.values()))) if stride_per_foot else 0.0
            entry["stride_frequency_hz_mean"] = stride_mean
            entry["duty_factor_mean"] = float(np.mean(list(duty_per_foot.values()))) if duty_per_foot else 0.0
        else:
            stride_mean = 0.0

        # velocidad real y largo de paso, con la misma definicion que el agregado
        actual_sub = _subset_frames(state["actual_vels"], env_idx)
        if actual_sub:
            speed = float(np.mean(np.linalg.norm(np.array(actual_sub)[:, :, :2], axis=-1)))
            entry["mean_speed_mps"] = speed
            entry["step_length_m"] = speed / stride_mean if stride_mean > 0 else 0.0

        # suavidad y seguimiento
        joint_sub = _subset_frames(state["joint_positions"], env_idx)
        entry["movement_smoothness"] = compute_movement_smoothness(joint_sub) if joint_sub else 0.0
        quats_sub = _subset_frames(state["base_quats"], env_idx)
        if quats_sub:
            entry["orientation_stability_0to1"] = compute_orientation_stability(quats_sub)
            entry["orientation_smoothness_0to1"] = compute_orientation_smoothness(
                quats_sub, _subset_frames(state["done_history"], env_idx), dt=step_dt
            )
        cmd_sub = _subset_frames(state["cmd_vels"], env_idx)
        if cmd_sub:
            entry["velocity_tracking_accuracy_0to1"] = compute_velocity_tracking_accuracy(cmd_sub, actual_sub)

        per_lag[str(lag)] = entry

    return per_lag


def save_torques_csv(state: dict, out_dir: str):
    if not state["torque_rows"]:
        return
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "torques.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        names = state["torque_joint_names"]
        writer.writerow(
            ["step", "time_s"]
            # applied = par entregado; computed = par pedido por el PD; saturated = 1 cuando el
            # actuador recorto (applied != computed), es decir el motor esta contra su techo.
            + [f"{name}_applied_Nm" for name in names]
            + [f"{name}_computed_Nm" for name in names]
            + [f"{name}_saturated" for name in names]
        )
        writer.writerows(state["torque_rows"])
    print(f"[INFO] Saved joint torques (env {args_cli.torque_env}) to: {out_path}")


def print_per_lag_table(per_lag: dict) -> None:
    """Tabla compacta del desglose por lag, para leerlo en el log sin abrir el JSON."""
    columns = [
        ("rodilla atras mm", "knee_height_min_backward_m", 1000.0, "{:.1f}"),
        ("rodilla adel mm", "knee_height_min_forward_m", 1000.0, "{:.1f}"),
        ("despeje mm", "foot_clearance_peak_m", 1000.0, "{:.1f}"),
        ("zancada Hz", "stride_frequency_hz_mean", 1.0, "{:.2f}"),
        ("paso mm", "step_length_m", 1000.0, "{:.0f}"),
        ("vel_track", "velocity_tracking_accuracy_0to1", 1.0, "{:.3f}"),
        ("ori_suav", "orientation_smoothness_0to1", 1.0, "{:.3f}"),
        ("mov_suav", "movement_smoothness", 1.0, "{:.3f}"),
        ("caidas/ep", "fall_rate_per_episode", 1.0, "{:.3f}"),
    ]
    header = f"{'lag':>4} {'delay':>6} {'envs':>5} " + " ".join(f"{name:>16}" for name, *_ in columns)
    print("\n" + "=" * len(header))
    print("METRICAS POR LAG DE ACCION (reparto estratificado)")
    print("=" * len(header))
    print(header)
    for lag in sorted(per_lag, key=int):
        entry = per_lag[lag]
        cells = []
        for _name, key, scale, fmt in columns:
            value = entry.get(key)
            cells.append(f"{fmt.format(value * scale):>16}" if isinstance(value, (int, float)) else f"{'-':>16}")
        print(f"{int(lag):>4} {entry.get('delay', '-'):>6} {entry.get('num_envs', '-'):>5} " + " ".join(cells))
    print("=" * len(header) + "\n")


def save_results(results: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved evaluation results to: {out_path}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate an RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # apply the SAME env/agent overrides used to train this checkpoint, if given
    if args_cli.experiment_config is not None:
        apply_experiment_config(env_cfg, agent_cfg, args_cli.experiment_config)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # apply action delay wrapper: randomized per env if a range was given, fixed otherwise
    env = maybe_wrap_with_delay(env, env_cfg, args_cli)
    # se guarda la referencia ahora: despues queda tapado por RslRlVecEnvWrapper, y al final de la
    # corrida hace falta su mapa env -> lag para desglosar las metricas
    delay_wrapper = env if isinstance(env, ActionDelayWrapper) else None

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = getattr(runner.alg, "policy", None) or getattr(runner.alg, "actor_critic", None)

    # reset environment
    obs = env.get_observations()
    state = {
        "base_falls": 0,
        "shoulder_falls": 0,
        "episodes": 0,
        "base_falls_per_env": None,
        "shoulder_falls_per_env": None,
        "episodes_per_env": None,
        "foot_ids": None,
        "foot_names": None,
        "foot_touchdowns": None,
        "foot_contact_steps": None,
        "foot_touchdowns_per_env": None,
        "foot_contact_steps_per_env": None,
        "foot_z": None,
        "foot_in_contact": None,
        "arm_ids": None,
        "arm_names": None,
        "foot_body_ids": None,
        "foot_tip_offset": None,
        "knee_body_ids": None,
        "knee_names": None,
        "impact_sum_frames": [],
        "impact_count_frames": [],
        "knee_z": None,
        "knee_backward_mask": None,
        "knee_forward_mask": None,
        "arm_contact_steps": 0,
        "arm_contact_force_sum": 0.0,
        "arm_contact_force_max": 0.0,
        "joint_names": None,
        "torque_joint_ids": None,
        "torque_joint_names": None,
        "torque_rows": None,
        "torque_sat_steps": 0.0,
        "torque_abs_sum": 0.0,
        "torque_abs_max": None,
        "torque_steps": 0,
        "joint_dev_sum": 0.0,
        "joint_dev_per_joint_sum": 0.0,
        "joint_dev_steps": 0,
        "speed_sum": 0.0,
        "speed_steps": 0,
        "impact_force_sum": 0.0,
        "impact_force_sq_sum": 0.0,
        "impact_force_count": 0,
        "impact_force_max": 0.0,
        "joint_positions": None,
        "base_positions": None,
        "base_quats": None,
        "done_history": None,
        "has_velocity_command": None,
        "cmd_vels": None,
        "actual_vels": None,
    }
    # simulate environment for a fixed number of steps
    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, extras = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        evaluate_step(env, obs, actions, rewards, dones, extras, step, state)

    results = compute_results(env, state)
    if delay_wrapper is not None:
        results["action_delay"] = describe_delay_regime(delay_wrapper)
        per_lag = compute_per_lag_results(env, state, delay_wrapper)
        if per_lag:
            results["metrics_per_lag"] = per_lag
            print_per_lag_table(per_lag)
    print_dict(results)
    save_results(results, os.path.join(log_dir, "eval"))
    save_torques_csv(state, os.path.join(log_dir, "eval"))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
