#!/usr/bin/env python3
"""
Example of training with action repetition/buffering.

This demonstrates how to use the ActionBufferWrapper to introduce
action delays/latency in simulation.

Usage:
    python3 train_with_action_buffer.py --task dynabot1_velocity --action-repeat 3
"""

import sys
import os
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "source/dynabot1"))

from dynabot1.tasks.manager_based.locomotion.velocity.action_buffer_wrapper import ActionBufferWrapper
from dynabot1.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab.app import AppLauncher
from isaaclab.envs import ManagerBasedRLEnv


def main():
    """Main training loop with action buffer."""
    import argparse

    parser = argparse.ArgumentParser(description="Train with action repetition.")
    parser.add_argument("--action-repeat", type=int, default=1, help="Number of times to repeat each action")
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of steps to run")

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Create environment
    env_cfg = LocomotionVelocityRoughEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Wrap with action buffer
    env = ActionBufferWrapper(env, action_repeat=args.action_repeat)

    print(f"🚀 Environment created with action repetition: {args.action_repeat}x")
    print(f"   (each action repeated {args.action_repeat} times)")

    # Run a few steps to test
    obs, _ = env.reset()
    print(f"\nRunning {args.num_steps} steps...")

    for step in range(args.num_steps):
        # Random actions
        actions = env.action_space.sample()

        # Step environment (action is repeated internally)
        obs, reward, done, truncated, info = env.step(actions)

        if step % 100 == 0:
            print(f"   Step {step}/{args.num_steps}: Avg reward = {reward.mean():.4f}")

        # Reset if done
        if done.any():
            obs, _ = env.reset()

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
