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

    foot_forces = torch.norm(contact_sensor.data.net_forces_w[:, state["foot_ids"], :], dim=-1)
    impact_forces = foot_forces[first_contact]
    if impact_forces.numel() > 0:
        state["impact_force_sum"] += impact_forces.sum().item()
        state["impact_force_sq_sum"] += (impact_forces**2).sum().item()
        state["impact_force_count"] += impact_forces.numel()
        state["impact_force_max"] = max(state["impact_force_max"], impact_forces.max().item())

    # Record joint positions for smoothness analysis
    articulation = env.unwrapped.scene.articulations["robot"]
    joint_pos = articulation.data.joint_pos.detach().cpu().numpy()  # Shape: (num_envs, num_joints)
    if state["joint_positions"] is None:
        state["joint_positions"] = []
    state["joint_positions"].append(joint_pos)

    # Record base velocity and orientation for path tracking and stability
    root_state = articulation.data.root_state_w  # (num_envs, 13)
    base_lin_vel = root_state[:, 7:10].detach().cpu().numpy()  # Linear velocity
    base_ang_vel = root_state[:, 10:13].detach().cpu().numpy()  # Angular velocity
    base_pos = root_state[:, :3].detach().cpu().numpy()  # Position
    base_quat = root_state[:, 3:7].detach().cpu().numpy()  # Quaternion

    if state["base_positions"] is None:
        state["base_positions"] = []
        state["base_lin_vels"] = []
        state["base_quats"] = []

    state["base_positions"].append(base_pos)
    state["base_lin_vels"].append(base_lin_vel)
    state["base_quats"].append(base_quat)


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


def quat_to_euler(quat):
    """Convert quaternion (x,y,z,w) to euler angles (roll, pitch, yaw) in radians."""
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

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
    roll_pitch = euler[:, 0, :2]  # Use first env, roll and pitch only

    # Stability = inverse of variance (lower variance = more stable)
    roll_pitch_variance = np.var(roll_pitch, axis=0).mean()

    # Return value from 0 to 1 (1 = perfectly stable)
    stability = 1.0 / (1.0 + roll_pitch_variance * 10)

    return float(stability)


def compute_orientation_smoothness(base_quats, dt=0.02):
    """Measure how smoothly orientation changes - no abrupt rotations."""
    if len(base_quats) < 3:
        return 0.0

    quats = np.array(base_quats)  # Shape: (num_steps, num_envs, 4)
    euler = quat_to_euler(quats)  # Shape: (num_steps, num_envs, 3)

    # Calculate angular velocity (derivative of euler angles)
    angular_vel = np.diff(euler, axis=0) / dt  # Shape: (num_steps-1, num_envs, 3)

    # Calculate angular acceleration (second derivative)
    angular_acc = np.diff(angular_vel, axis=0) / dt  # Shape: (num_steps-2, num_envs, 3)

    # Mean magnitude of angular acceleration
    acc_magnitude = np.mean(np.abs(angular_acc))

    # Smoothness: inverse of angular acceleration
    smoothness = 1.0 / (1.0 + acc_magnitude)

    return float(smoothness)


def compute_velocity_tracking_accuracy(base_lin_vels, cmd_vel_x_range=1.0, cmd_vel_y_range=1.0):
    """Measure how well actual velocity matches commanded velocity range."""
    if len(base_lin_vels) < 2:
        return 0.0

    vels = np.array(base_lin_vels)  # Shape: (num_steps, num_envs, 3)
    vel_magnitude = np.linalg.norm(vels[:, 0, :2], axis=1)  # XY velocity only

    # Expected velocity range (from commands)
    expected_max = np.sqrt(cmd_vel_x_range**2 + cmd_vel_y_range**2)

    # Accuracy: how close is actual velocity to expected range
    # 0 if never moving, 1 if consistent with commands
    mean_vel = np.mean(vel_magnitude)
    tracking_accuracy = min(mean_vel / expected_max, 1.0)

    return float(tracking_accuracy)


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

    # Calculate orientation metrics
    orientation_stability = compute_orientation_stability(state["base_quats"]) if state["base_quats"] else 0.0
    orientation_smoothness = compute_orientation_smoothness(state["base_quats"]) if state["base_quats"] else 0.0

    # Calculate velocity tracking
    velocity_tracking = compute_velocity_tracking_accuracy(state["base_lin_vels"]) if state["base_lin_vels"] else 0.0

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
        "impact_force_sum": 0.0,
        "impact_force_sq_sum": 0.0,
        "impact_force_count": 0,
        "impact_force_max": 0.0,
        "joint_positions": None,
        "base_positions": None,
        "base_lin_vels": None,
        "base_quats": None,
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
