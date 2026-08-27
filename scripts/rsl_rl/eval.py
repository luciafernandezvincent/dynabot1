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
parser.add_argument("--action-delay", type=int, default=1, help="Number of steps to delay actions (1 = no delay)")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of simulation steps to run the evaluation for.")
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

import dynabot1.tasks  # noqa: F401
from dynabot1.wrappers import ActionDelayWrapper


def evaluate_step(env, obs, actions, rewards, dones, extras, step: int, state: dict):
    """Called once per simulation step. Fill this in with per-step evaluation logic."""
    term_manager = env.unwrapped.termination_manager
    state["base_falls"] += int(term_manager.get_term("base_contact").sum().item())
    state["shoulder_falls"] += int(term_manager.get_term("shoulder_contact").sum().item())
    state["episodes"] += int(dones.sum().item())

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

    # altura de cada pie y si esta apoyado, para medir cuanto despega en el swing
    articulation_for_feet = env.unwrapped.scene.articulations["robot"]
    foot_z = articulation_for_feet.data.body_pos_w[:, state["foot_ids"], 2].detach().cpu().numpy()
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
    knee_z = articulation_for_feet.data.body_pos_w[:, state["arm_ids"], 2].detach().cpu().numpy()
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
        state["knee_z"], state["knee_backward_mask"], state["knee_forward_mask"], state["arm_names"] or []
    )

    # Calculate orientation metrics
    orientation_stability = compute_orientation_stability(state["base_quats"]) if state["base_quats"] else 0.0
    orientation_smoothness = (
        compute_orientation_smoothness(state["base_quats"], state["done_history"], dt=env.unwrapped.step_dt)
        if state["base_quats"]
        else 0.0
    )

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
    }


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

    # apply action delay wrapper if action_delay > 1
    if args_cli.action_delay > 1:
        env = ActionDelayWrapper(env, delay_steps=args_cli.action_delay)
        print(f"[INFO] Action delay enabled: {args_cli.action_delay} steps delay")

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
        "foot_ids": None,
        "foot_names": None,
        "foot_touchdowns": None,
        "foot_contact_steps": None,
        "foot_z": None,
        "foot_in_contact": None,
        "arm_ids": None,
        "arm_names": None,
        "knee_z": None,
        "knee_backward_mask": None,
        "knee_forward_mask": None,
        "arm_contact_steps": 0,
        "arm_contact_force_sum": 0.0,
        "arm_contact_force_max": 0.0,
        "joint_names": None,
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
    print_dict(results)
    save_results(results, os.path.join(log_dir, "eval"))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
