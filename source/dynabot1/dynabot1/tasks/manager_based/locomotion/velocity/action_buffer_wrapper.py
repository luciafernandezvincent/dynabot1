"""Wrapper to add action repetition/buffering for introducing latency in simulation."""

import torch


class ActionBufferWrapper:
    """
    Wraps any environment to repeat actions N times before accepting new actions.
    Useful for simulating latency or action delays.

    Example:
        env = gym.make(task, cfg=env_cfg)
        env = ActionBufferWrapper(env, action_repeat=3)

        # Now each action from policy is repeated 3 times before new action is sampled
    """

    def __init__(self, env, action_repeat: int = 1):
        """
        Initialize action buffer wrapper.

        Args:
            env: The base environment to wrap (any gymnasium-compatible environment)
            action_repeat: Number of times to repeat each action (default: 1 = no repeat)
        """
        self.env = env
        self.action_repeat = action_repeat
        self.action_buffer = None
        self.repeat_counter = 0

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
        """Reset environment and clear action buffer."""
        self.action_buffer = None
        self.repeat_counter = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        """
        Step environment with action repetition.

        Supports both single and vectorized (batch) actions.

        Args:
            actions: Actions from policy (can be single action or batch)

        Returns:
            observations, rewards, dones, truncated, info
        """
        # Store new actions on first call or when repeat counter resets
        if self.repeat_counter == 0:
            if isinstance(actions, torch.Tensor):
                self.action_buffer = actions.clone()
            else:
                self.action_buffer = actions.copy() if hasattr(actions, 'copy') else actions

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
