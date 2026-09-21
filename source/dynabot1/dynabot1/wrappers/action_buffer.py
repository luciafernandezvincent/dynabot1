"""Wrapper to add action delay for introducing latency in simulation."""

from __future__ import annotations

import torch


class ActionDelayWrapper:
    """
    Wraps a vectorized environment to introduce action delay/latency.

    New actions are buffered and executed some steps later. The delay can be the same for every
    environment, or sampled per-environment from a range so the policy learns to cope with a
    spread of latencies (domain randomization) instead of a single fixed value.

    Units: ``delay_steps`` and ``delay_range`` share one convention, where the applied lag is
    ``value - 1`` steps -- ``delay_steps=1`` is no delay and ``delay_steps=5`` means the action
    applied at step t is the one the policy produced at step t-4. This matches the original
    deque-based implementation, so runs recorded with ``--action-delay 5`` stay comparable.

    Example:
        env = ActionDelayWrapper(env, delay_steps=5)                            # every env lags 4
        env = ActionDelayWrapper(env, delay_range=(3, 8), action_history_length=4)  # lags 2..7
    """

    def __init__(
        self,
        env,
        delay_steps: int = 1,
        delay_range: tuple[int, int] | None = None,
        default_action: str = "zeros",
        action_history_length: int | None = None,
        resample_on_reset: bool = True,
        log_interval: int | None = None,
    ):
        """
        Initialize action delay wrapper.

        Args:
            env: The base environment to wrap
            delay_steps: Fixed delay for every env (ignored when delay_range is given)
            delay_range: (min, max) inclusive. Each env draws its own delay from this range,
                so the policy sees a mix of latencies instead of one. Same units as delay_steps.
            default_action: What to execute before the buffer has filled
                - "zeros": Send zero action
                - "hold": Hold the oldest action available so far
            action_history_length: Number of most recent *raw* (not-yet-executed) actions written
                into the trailing slot of obs["policy"] that mdp.last_action reserves. MUST match
                the history_length of the `actions` ObsTerm in the env cfg. With a fixed delay it
                defaults to delay_steps - 1 (the number of actions still pending); with
                delay_range it has no sensible default and must be passed explicitly.
            resample_on_reset: Draw a new delay for an env each time its episode restarts
            log_interval: Print the lag distribution every N steps (None disables)
        """
        self.env = env
        self.default_action = default_action
        self.resample_on_reset = resample_on_reset
        self.log_interval = log_interval
        self.step_count = 0

        if delay_range is not None:
            low, high = int(delay_range[0]), int(delay_range[1])
            if low < 1 or high < low:
                raise ValueError(f"delay_range must be (min, max) with 1 <= min <= max, got {delay_range}")
            self.delay_range = (low, high)
            self.max_lag = high - 1
            if action_history_length is None:
                raise ValueError(
                    "delay_range needs an explicit action_history_length: it must equal the "
                    "history_length of the `actions` ObsTerm in the env cfg. Guessing it from the "
                    "delay would overwrite the wrong number of observation dims."
                )
        else:
            self.delay_range = None
            self.max_lag = max(0, int(delay_steps) - 1)
            if action_history_length is None:
                action_history_length = self.max_lag
        self.action_history_length = max(0, int(action_history_length))

        # buffers are allocated lazily, on the first action, to inherit its shape/device/dtype
        self.buffer: torch.Tensor = None  # type: ignore[assignment]
        self.lags: torch.Tensor = None  # type: ignore[assignment]
        self.steps_since_reset: torch.Tensor = None  # type: ignore[assignment]
        self.head = 0

        # Expose common environment attributes
        if hasattr(env, "observation_space"):
            self.observation_space = env.observation_space
        if hasattr(env, "action_space"):
            self.action_space = env.action_space
        if hasattr(env, "render_mode"):
            self.render_mode = env.render_mode
        if hasattr(env, "metadata"):
            self.metadata = env.metadata

    # ------------------------------------------------------------------ internals

    @property
    def _buffer_len(self) -> int:
        """Enough slots for the largest lag and for the observation history, plus the newest entry."""
        return max(self.max_lag, self.action_history_length - 1, 0) + 1

    def _sample_lags(self, count: int, device) -> torch.Tensor:
        """Draw `count` lags in steps: from the configured range, or the single fixed value."""
        if self.delay_range is None:
            return torch.full((count,), self.max_lag, dtype=torch.long, device=device)
        low, high = self.delay_range
        # lag = value - 1, and randint's upper bound is exclusive, so [low-1, high) covers it
        return torch.randint(low - 1, high, (count,), dtype=torch.long, device=device)

    def _lazy_init(self, actions: torch.Tensor) -> None:
        num_envs = actions.shape[0]
        self.buffer = torch.zeros(
            (self._buffer_len, *actions.shape), dtype=actions.dtype, device=actions.device
        )
        self.lags = self._sample_lags(num_envs, actions.device)
        self.steps_since_reset = torch.zeros(num_envs, dtype=torch.long, device=actions.device)
        self.head = 0

    def _on_reset(self, reset_mask: torch.Tensor) -> None:
        """Clear the action history of envs that just restarted and give them a new delay.

        Without this an env keeps executing actions queued during its previous episode, which
        leak across the reset boundary.
        """
        self.buffer[:, reset_mask] = 0.0
        self.steps_since_reset[reset_mask] = 0
        if self.resample_on_reset and self.delay_range is not None:
            count = int(reset_mask.sum().item())
            self.lags[reset_mask] = self._sample_lags(count, self.lags.device)

    # ------------------------------------------------------------------ gym API

    def reset(self, seed=None, options=None):
        """Reset environment and clear the action buffer."""
        if self.buffer is not None:
            self.buffer.zero_()
            self.steps_since_reset.zero_()
            if self.resample_on_reset and self.delay_range is not None:
                self.lags = self._sample_lags(self.lags.shape[0], self.lags.device)
            self.head = 0
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._fix_action_history_obs(obs)
        return obs, info

    def step(self, actions):
        """
        Step environment with a per-environment action delay.

        Args:
            actions: Actions from the policy, shape (num_envs, action_dim)

        Returns:
            observations, rewards, terminated, truncated, info
        """
        if not isinstance(actions, torch.Tensor):
            raise TypeError(f"ActionDelayWrapper expects a torch.Tensor of actions, got {type(actions)}")
        if self.buffer is None:
            self._lazy_init(actions)

        # write the newest raw action at the head of the circular buffer
        self.head = (self.head + 1) % self._buffer_len
        self.buffer[self.head] = actions

        # how far back each env reads. "hold" clamps to what is actually available, so a warming-up
        # env repeats its oldest action; "zeros" reads past it into a slot that is still zero.
        if self.default_action == "hold":
            lags = torch.minimum(self.lags, self.steps_since_reset)
        else:
            lags = self.lags

        read_idx = (self.head - lags) % self._buffer_len
        env_idx = torch.arange(actions.shape[0], device=actions.device)
        action_to_execute = self.buffer[read_idx, env_idx]

        self.steps_since_reset += 1

        obs, reward, terminated, truncated, info = self.env.step(action_to_execute)

        # IsaacLab resets terminated envs inside step() and computes their obs AFTER the reset, so
        # the obs returned for them is already the first one of the NEW episode. Clear their history
        # first, so that obs doesn't carry actions from the episode that just ended.
        reset_mask = terminated.bool() | truncated.bool()
        if reset_mask.any():
            self._on_reset(reset_mask)

        # replace mdp.last_action's history (executed/delayed actions) with the raw actions the
        # policy has committed, so the obs stays a Markov state under the delay
        obs = self._fix_action_history_obs(obs)

        info["delayed_action"] = action_to_execute
        info["input_action"] = actions
        info["action_lags"] = self.lags

        self.step_count += 1
        if self.log_interval and self.step_count % self.log_interval == 0:
            print(f"[ActionDelay] step {self.step_count}: {self.describe_delays()}")

        return obs, reward, terminated, truncated, info

    def _fix_action_history_obs(self, obs):
        """Overwrite the trailing action-history slot of obs["policy"] with raw actions.

        mdp.last_action(history_length=N) fills that slot with the N most recent actions that
        reached env.action_manager -- but since this wrapper delays actions before they get there,
        that's the N most recent *executed* actions, not the ones the policy just committed and
        are still pending. Replace it with the latter, oldest first, zero-padded until the buffer
        fills up (the buffer starts and resets to zeros, so the padding is automatic).

        With a randomized delay this slot is what lets the policy infer its own latency: it sees
        what it commanded and, through joint_pos/joint_vel, what actually came out.
        """
        n = self.action_history_length
        if n <= 0 or self.buffer is None:
            return obs
        if not isinstance(obs, dict) or "policy" not in obs or not isinstance(obs["policy"], torch.Tensor):
            return obs

        # oldest -> newest, matching the layout mdp.last_action writes
        offsets = torch.arange(n - 1, -1, -1, device=self.buffer.device)
        hist_idx = (self.head - offsets) % self._buffer_len
        history = self.buffer[hist_idx]  # (n, num_envs, action_dim)
        raw_hist = history.permute(1, 0, 2).reshape(history.shape[1], -1)

        slice_len = raw_hist.shape[-1]
        policy_obs = obs["policy"]
        if slice_len > policy_obs.shape[-1]:
            raise ValueError(
                f"action_history_length={n} needs {slice_len} obs dims but obs['policy'] only has "
                f"{policy_obs.shape[-1]}. It must match the `actions` ObsTerm history_length."
            )
        obs["policy"] = torch.cat([policy_obs[..., :-slice_len], raw_hist], dim=-1)
        return obs

    # ------------------------------------------------------------------ inspection

    def describe_delays(self) -> str:
        """One-line summary of the current per-env lag distribution, for logging."""
        if self.lags is None:
            return "not initialized yet"
        lags = self.lags.float()
        kind = f"randomized {self.delay_range}" if self.delay_range else "fixed"
        return f"lag min={int(lags.min())} max={int(lags.max())} mean={lags.mean():.2f} steps ({kind})"

    def get_delay_steps(self):
        """Current lag per env, in steps."""
        return self.lags

    def render(self):
        """Render environment if supported."""
        if hasattr(self.env, "render"):
            return self.env.render()

    def close(self):
        """Close the wrapped environment."""
        if hasattr(self.env, "close"):
            self.env.close()

    @property
    def unwrapped(self):
        """Return the base environment."""
        return self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)
