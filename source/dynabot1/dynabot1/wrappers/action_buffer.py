"""Wrapper to add action delay for introducing latency in simulation."""

import torch
from collections import deque


class ActionDelayWrapper:
    """
    Wraps any environment to introduce action delay/latency.

    New actions are queued but executed X steps later.
    Useful for simulating communication delay or processing latency.

    Example:
        env = gym.make(task, cfg=env_cfg)
        env = ActionDelayWrapper(env, delay_steps=3)

        # Action sent at step 0 is executed at step 3
        # Action sent at step 1 is executed at step 4
        # etc.
    """

    def __init__(self, env, delay_steps: int = 1, default_action="zeros"):
        """
        Initialize action delay wrapper.

        Args:
            env: The base environment to wrap
            delay_steps: Number of steps to delay actions (default: 1)
            default_action: What to send before first actions arrive
                - "zeros": Send zero action
                - "hold": Hold last action (requires first action to initialize)
        """
        self.env = env
        self.delay_steps = max(1, delay_steps)
        self.default_action = default_action
        self.action_queue = deque(maxlen=delay_steps)
        self.last_action = None

        # Expose common environment attributes
        if hasattr(env, 'observation_space'):
            self.observation_space = env.observation_space
        if hasattr(env, 'action_space'):
            self.action_space = env.action_space
        if hasattr(env, 'render_mode'):
            self.render_mode = env.render_mode
        if hasattr(env, 'metadata'):
            self.metadata = env.metadata

    def reset(self, seed=None, options=None):
        """Reset environment and clear action queue."""
        self.action_queue.clear()
        self.last_action = None
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        """
        Step environment with action delay.

        Args:
            actions: Actions from policy (will be executed delay_steps later)

        Returns:
            observations, rewards, dones, truncated, info
        """
        # Queue the new action
        self.action_queue.append(actions)

        # Get action to execute (from front of queue or default)
        if len(self.action_queue) == self.delay_steps:
            # Queue is full, execute oldest action
            action_to_execute = self.action_queue[0]  # oldest
            self.last_action = action_to_execute
        else:
            # Queue not full yet, use default action
            if self.default_action == "zeros":
                # Create zero action with same shape as input
                if isinstance(actions, torch.Tensor):
                    action_to_execute = torch.zeros_like(actions)
                else:
                    action_to_execute = (
                        actions * 0 if hasattr(actions, '__mul__') else None
                    )
            elif self.default_action == "hold" and self.last_action is not None:
                action_to_execute = self.last_action
            else:
                action_to_execute = self._create_zero_action(actions)

        # Step environment with delayed action
        obs, reward, done, truncated, info = self.env.step(action_to_execute)

        return obs, reward, done,truncated, info

    def _create_zero_action(self, action_template):
        """Create zero action with same shape as template."""
        if isinstance(action_template, torch.Tensor):
            return torch.zeros_like(action_template)
        elif hasattr(action_template, 'shape'):
            import numpy as np
            return np.zeros_like(action_template)
        else:
            return action_template * 0

    def set_delay_steps(self, delay: int):
        """Change delay dynamically."""
        self.delay_steps = max(1, delay)
        self.action_queue = deque(self.action_queue, maxlen=delay)

    def get_delay_steps(self):
        """Get current delay value."""
        return self.delay_steps

    def render(self):
        """Render environment if supported."""
        if hasattr(self.env, 'render'):
            return self.env.render()

    def close(self):
        """Close the wrapped environment."""
        if hasattr(self.env, 'close'):
            self.env.close()

    @property
    def unwrapped(self):
        """Return the base environment."""
        return self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env

    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)
