import argparse
import os
import sys
import time


from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Run manual actuator response test.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Dyna1-GraphArtRes-v0")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--real-time", action="store_true", default=False)

#Nuevos arguments
parser.add_argument( "--robot_name", type=str, default="robot")
parser.add_argument("--leg_joint_names", nargs="+", required=True) #estos son de los que se generan graficos
parser.add_argument("--action_indices", nargs="+", type=int, required=True) #indica qué componentes del vector de acciones querés modificar
parser.add_argument("--test_mode", type=str, default="step", choices=["step", "sine", "sequence"])
parser.add_argument("--test_steps", type=int, default=1000) # para calc segundos de simulacion, el default es 5segs
parser.add_argument("--step_value", type=float, default=0.4)
parser.add_argument("--sine_amplitude", type=float, default=0.4)
parser.add_argument("--sine_frequency", type=float, default=0.5)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
import pandas as pd
import matplotlib.pyplot as plt

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from omni.isaac.core import SimulationContext

import dynabot1.tasks  # noqa: F401


def build_manual_action(t,num_envs, action_dim, device, action_indices, mode, step_value, sine_amplitude, sine_frequency, offset):
    actions = torch.zeros(num_envs, action_dim, device=device)

    if mode == "step":
        if t < 5.0:
            value = 0.0 - offset #priemro tendria que ir al 0 (considerando el initial pose) y desp paso al movimiento buscando
        else:
            value = step_value

        for action_idx in action_indices:
            actions[:, action_idx] = value

    elif mode == "sine":
        value = sine_amplitude * torch.sin(
            torch.tensor(2.0 * torch.pi * sine_frequency * t, device=device)
        )

        for action_idx in action_indices:
            actions[:, action_idx] = value

    elif mode == "sequence":
        if t < 1.0:
            value = 0.0
        elif t < 3.0:
            value = step_value
        elif t < 5.0:
            value = -step_value
        else:
            value = 0.0

        for action_idx in action_indices:
            actions[:, action_idx] = value

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return actions


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    initial_pos = {
        ".*_shoulder": 0.0,
        ".*shoulder_to_arm": - 0.79,
        ".*arm_to_hand": 1.5,
    } #ver que no sea hardcodeado

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    dt = env.unwrapped.step_dt
    robot = env.unwrapped.scene[args_cli.robot_name]

    print("[INFO] Available robot joints:")
    for i, name in enumerate(robot.joint_names):
        print(i, name)

    joint_ids, resolved_joint_names = robot.find_joints(args_cli.leg_joint_names)

    print("[INFO] Logging joints:")
    for name, jid in zip(resolved_joint_names, joint_ids):
        print(f"  {name}: joint_id={jid}")

    print("[INFO] Using action indices:")
    for idx in args_cli.action_indices:
        print(f"  action[{idx}]")

    obs = env.get_observations()

    logs = []
    timestep = 0

    ##### NUEVO: ANÁLISIS DE FRECUENCIA ANTES DEL BUCLE #####
    sim_context = SimulationContext.instance()
    
    physics_dt = sim_context.get_physics_dt()
    print("\n" + "="*50)
    print(f"[ANÁLISIS DE FRECUENCIA DE DATOS]")
    print(f"-> La física calcula a:    {1.0 / physics_dt:.1f} Hz (dt: {physics_dt} s)")
    print(f"-> Tu gráfico registrará a: {1.0 / dt:.1f} Hz (dt: {dt} s)")
    print("="*50 + "\n")

    while simulation_app.is_running():
        start_time = time.time()
      
        ##### CAMBIO AQUÍ: Reemplazamos t = timestep * dt por el tiempo del simulador #####
        t = sim_context.current_time
        #t = timestep * dt

        with torch.inference_mode():
            offset = 0.0
            for part, off in initial_pos.items():
                if part in resolved_joint_names[0]: #esto depende de que solo haya un joint name
                    offset = off
 
            step_value = ((args_cli.step_value*torch.pi/180) - offset)/0.25
            actions = build_manual_action(
                t=t,
                num_envs=env.unwrapped.num_envs,
                action_dim=env.num_actions,
                device=env.unwrapped.device,
                action_indices=args_cli.action_indices,
                mode=args_cli.test_mode,
                step_value=step_value,
                sine_amplitude=args_cli.sine_amplitude,
                sine_frequency=args_cli.sine_frequency,
                offset=offset
            )

            obs, _, dones, _ = env.step(actions)
            data = robot.data

            for joint_name, joint_id in zip(resolved_joint_names, joint_ids):
                row = {
                    "step": timestep,
                    "time": t,
                    "joint": joint_name,
                    "joint_id": int(joint_id),
                    "joint_pos": data.joint_pos[0, joint_id].item(),
                    "joint_vel": data.joint_vel[0, joint_id].item(),
                }
                if data.joint_pos_target is not None:
                    row["joint_pos_target"] = data.joint_pos_target[0, joint_id].item()
                if data.joint_vel_target is not None:
                    row["joint_vel_target"] = data.joint_vel_target[0, joint_id].item()
                if data.joint_effort_target is not None:
                    row["joint_effort_target"] = data.joint_effort_target[0, joint_id].item()
                if data.applied_torque is not None:
                    row["applied_torque"] = data.applied_torque[0, joint_id].item()
                if data.computed_torque is not None:
                    row["computed_torque"] = data.computed_torque[0, joint_id].item()
                for i in range(actions.shape[1]):
                    row[f"action_{i}"] = actions[0, i].item()
                logs.append(row)

        timestep += 1
        if timestep >= args_cli.test_steps:
            break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    output_dir = os.path.abspath("actuator_response_results")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(logs)

    joint_names = df["joint"].unique()
    if len(joint_names) == 1:
        csv_path = os.path.join(output_dir, f"actuator_response_{joint_names[0]}.csv")
    else:
        csv_path = os.path.join(output_dir, "actuator_response.csv")
    
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved CSV to: {csv_path}")

    for joint_name in joint_names:
        d = df[df["joint"] == joint_name]

        plt.figure(figsize=(10, 5))
        plt.plot(d["time"], d["joint_pos"], label="joint_pos")

        if "joint_pos_target" in d.columns:
            plt.plot(d["time"], d["joint_pos_target"], label="joint_pos_target")

        plt.xlabel("Time [s]")
        plt.ylabel("Position [rad]")
        plt.title(f"Position response - {joint_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{joint_name}_position_response.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(d["time"], d["joint_vel"], label="joint_vel")

        if "joint_vel_target" in d.columns:
            plt.plot(d["time"], d["joint_vel_target"], label="joint_vel_target")

        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [rad/s]")
        plt.title(f"Velocity response - {joint_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{joint_name}_velocity_response.png"), dpi=300)
        plt.close()

        torque_cols = []
        if "joint_effort_target" in d.columns:
            torque_cols.append("joint_effort_target")
        if "computed_torque" in d.columns:
            torque_cols.append("computed_torque")
        if "applied_torque" in d.columns:
            torque_cols.append("applied_torque")

        if torque_cols:
            plt.figure(figsize=(10, 5))

            for col in torque_cols:
                plt.plot(d["time"], d[col], label=col)

            plt.xlabel("Time [s]")
            plt.ylabel("Torque / effort [Nm]")
            plt.title(f"Torque response - {joint_name}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{joint_name}_torque_response.png"), dpi=300)
            plt.close()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

# base_to_front_right_shoulder front_right_shoulder_to_arm front_right_arm_to_hand