"""Wrapper to add action repetition/buffering for introducing latency in simulation."""

import torch
from isaaclab.envs import ManagerBasedRLEnv


class ActionBufferWrapper(ManagerBasedRLEnv):
    """
    Wraps a ManagerBasedRLEnv to repeat actions N times before accepting new actions.
    Useful for simulating latency or action delays.

    Example:
        env = ManagerBasedRLEnv(cfg=env_cfg)
        env = ActionBufferWrapper(env, action_repeat=3)

        # Now each action from policy is repeated 3 times before new action is sampled
    """

    def __init__(self, env: ManagerBasedRLEnv, action_repeat: int = 1):
        """
        Initialize action buffer wrapper.

        Args:
            env: The base environment to wrap
            action_repeat: Number of times to repeat each action (default: 1 = no repeat)
        """
        self.env = env
        self.action_repeat = action_repeat
        self.action_buffer = None
        self.repeat_counter = 0

        # Expose environment attributes
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.render_mode = env.render_mode
        self.metadata = env.metadata

    def reset(self, seed=None, options=None):
        """Reset environment and clear action buffer."""
        self.action_buffer = None
        self.repeat_counter = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        """
        Step environment with action repetition.

        Args:
            actions: Actions from policy

        Returns:
            observations, rewards, dones, truncated, info
        """
        # Store new actions on first call or when repeat counter resets
        if self.repeat_counter == 0:
            self.action_buffer = actions.clone() if isinstance(actions, torch.Tensor) else actions.copy()

        # Step environment with buffered action
        obs, reward, done, truncated, info = self.env.step(self.action_buffer)

        # Update counter
        self.repeat_counter += 1
        if self.repeat_counter >= self.action_repeat:
            self.repeat_counter = 0

        return obs, reward, done, truncated, info

    def set_action_repeat(self, repeat: int):
        """Change action repeat dynamically."""
        self.action_repeat = max(1, repeat)
        self.repeat_counter = 0

    def get_action_repeat(self):
        """Get current action repeat value."""
        return self.action_repeat

    # Delegate other methods to wrapped environment
    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)

    def close(self):
        """Close the wrapped environment."""
        if hasattr(self.env, 'close'):
            self.env.close()
