"""Shared CLI plumbing for the action-delay wrapper.

Kept in one place because train/play/eval must agree on it: the wrapper overwrites the trailing
action-history slot of the observation, so if its length disagrees with the `actions` ObsTerm in
the env cfg, the observation is silently corrupted instead of failing loudly.
"""

from __future__ import annotations

import argparse

# NOTE: do NOT import dynabot1 at module level. train/play/eval import this file BEFORE AppLauncher
# starts Isaac Sim, and dynabot1 pulls in isaaclab_tasks/omni modules, which Isaac Sim rejects
# ("execute other Omniverse/Isaac Sim imports after instantiating SimulationApp").


def add_delay_args(parser: argparse.ArgumentParser) -> None:
    """Register the action-delay flags shared by train_delay.py / play_delay.py / eval.py."""
    parser.add_argument(
        "--action-delay", type=int, default=1, help="Number of steps to delay actions (1 = no delay)"
    )
    parser.add_argument(
        "--action-delay-range", type=str, default=None,
        help=(
            "Randomize the action delay per environment, as 'min,max' (inclusive, same units as "
            "--action-delay), e.g. '3,8'. Each env draws its own delay, resampled every episode. "
            "Overrides --action-delay when set."
        ),
    )
    parser.add_argument(
        "--action-delay-stratified", action="store_true",
        help=(
            "With --action-delay-range, split the envs evenly and deterministically over the "
            "range (env i gets the i-th delay, cycling) instead of drawing at random, and keep "
            "each env on its delay for the whole run. Use it for eval/play: the delay mix stops "
            "being run-to-run noise and every metric can be grouped exactly by lag. Training "
            "wants the random draw, so leave it off there."
        ),
    )
    parser.add_argument(
        "--action-history-length", type=int, default=None,
        help=(
            "Number of raw pending actions fed back into the observation. Must match the "
            "history_length of the `actions` ObsTerm; read from the env cfg when omitted."
        ),
    )


def parse_delay_range(text: str) -> tuple[int, int]:
    """Parse the 'min,max' form of --action-delay-range."""
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--action-delay-range must be 'min,max', got {text!r}")
    low, high = int(parts[0]), int(parts[1])
    if low < 1 or high < low:
        raise ValueError(f"--action-delay-range must satisfy 1 <= min <= max, got {text!r}")
    return low, high


def infer_action_history_length(env_cfg) -> int:
    """Read history_length off the `actions` ObsTerm so the wrapper matches the env cfg.

    Do NOT fall back to `delay_steps - 1` here: that only happens to be right when the delay
    equals the configured history length, and is silently wrong at every other delay -- which is
    exactly what a delay sweep does.
    """
    try:
        return int(env_cfg.observations.policy.actions.history_length)
    except AttributeError as exc:
        raise ValueError(
            "could not read observations.policy.actions.history_length from the env cfg; "
            "pass --action-history-length explicitly"
        ) from exc


def maybe_wrap_with_delay(env, env_cfg, args_cli, log=print):
    """Wrap `env` with the action delay configured on the CLI, if any. Returns the env."""
    from dynabot1.wrappers import ActionDelayWrapper  # imported late: needs Isaac Sim running

    history_length = (
        args_cli.action_history_length
        if getattr(args_cli, "action_history_length", None) is not None
        else infer_action_history_length(env_cfg)
    )

    delay_range_arg = getattr(args_cli, "action_delay_range", None)
    stratified = bool(getattr(args_cli, "action_delay_stratified", False))
    if delay_range_arg is not None:
        low, high = parse_delay_range(delay_range_arg)
        env = ActionDelayWrapper(
            env, delay_range=(low, high), action_history_length=history_length, stratified=stratified
        )
        log(
            f"[INFO] {'Stratified' if stratified else 'Randomized'} action delay: {low}..{high} "
            f"(lag {low - 1}..{high - 1} steps), action_history_length={history_length}"
        )
        if stratified:
            log(f"[INFO] Envs per lag: {env.lag_counts() or 'assigned on the first step'}")
    elif args_cli.action_delay > 1:
        env = ActionDelayWrapper(
            env, delay_steps=args_cli.action_delay, action_history_length=history_length
        )
        log(
            f"[INFO] Action delay: {args_cli.action_delay} "
            f"(lag {args_cli.action_delay - 1} steps), action_history_length={history_length}"
        )
    return env
