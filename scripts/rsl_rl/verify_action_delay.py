"""Verifica en la simulacion real que cada env aplica su propio delay aleatorio.

No confia en la contabilidad interna de ActionDelayWrapper: mide el lag de forma empirica desde el
registro propio de IsaacLab (``action_manager.action``, la accion que efectivamente entro a
``env.step``). Le manda acciones aleatorias y, para cada env, busca el corrimiento k que hace
coincidir EXACTAMENTE lo aplicado con lo enviado. Con acciones aleatorias continuas, una
coincidencia exacta solo ocurre en el lag verdadero.

Uso (corre en la maquina con Isaac Sim):
    python scripts/rsl_rl/verify_action_delay.py --task=Dyna1-Flat-v0 --num_envs 16 --delay_range 3,8
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify per-env randomized action delay in simulation.")
parser.add_argument("--task", type=str, default="Dyna1-Flat-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to check.")
parser.add_argument("--delay_range", type=str, default="3,8", help="'min,max', same units as --action-delay.")
parser.add_argument("--num_steps", type=int, default=80, help="Steps to record.")
parser.add_argument(
    "--action_scale", type=float, default=0.05,
    help="Amplitude of the random actions. Small, so the robots mostly stay up.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from collections import Counter

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import dynabot1.tasks  # noqa: F401
from dynabot1.wrappers import ActionDelayWrapper

from delay_args import infer_action_history_length, parse_delay_range  # isort: skip


def measure_lags(sent: torch.Tensor, applied: torch.Tensor, max_lag: int):
    """For each env, the shift k where applied[t] == sent[t-k] holds most often.

    Args:
        sent: (T, N, A) actions fed to the wrapper
        applied: (T, N, A) actions IsaacLab actually received
    Returns:
        (best_k, matches_at_best_k, possible_at_best_k), each of shape (N,)
    """
    num_steps, num_envs = sent.shape[0], sent.shape[1]
    counts = torch.zeros(num_envs, max_lag + 1, dtype=torch.long, device=sent.device)
    for k in range(max_lag + 1):
        equal = (applied[k:] == sent[: num_steps - k]).all(dim=-1)  # (T-k, N)
        counts[:, k] = equal.sum(dim=0)
    best_k = counts.argmax(dim=1)
    matches = counts.gather(1, best_k.unsqueeze(1)).squeeze(1)
    possible = num_steps - best_k
    return best_k, matches, possible


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    low, high = parse_delay_range(args_cli.delay_range)
    # resample_on_reset=False: keep each env's lag constant while we measure it
    wrapped = ActionDelayWrapper(
        env,
        delay_range=(low, high),
        action_history_length=infer_action_history_length(env_cfg),
        resample_on_reset=False,
    )
    wrapped.reset()

    action_manager = env.unwrapped.action_manager
    device = env.unwrapped.device
    num_envs, act_dim = args_cli.num_envs, action_manager.total_action_dim

    sent, applied, resets = [], [], []
    with torch.inference_mode():
        for _ in range(args_cli.num_steps):
            actions = args_cli.action_scale * torch.randn(num_envs, act_dim, device=device)
            _, _, terminated, truncated, _ = wrapped.step(actions)
            sent.append(actions.clone())
            applied.append(action_manager.action.clone())  # IsaacLab's own record, not the wrapper's
            resets.append((terminated | truncated).clone())

    sent, applied, resets = torch.stack(sent), torch.stack(applied), torch.stack(resets)
    configured = wrapped.lags.clone()
    measured, matches, possible = measure_lags(sent, applied, max_lag=high - 1)

    print("\n" + "=" * 72)
    print(f"Verificacion de delay aleatorio: {args_cli.task}, {num_envs} envs, {args_cli.num_steps} pasos")
    print(f"delay_range={low},{high}  ->  lag esperado en [{low - 1}, {high - 1}] pasos")
    print("=" * 72)
    print(f"{'env':>4} | {'lag config':>10} | {'lag medido':>10} | {'coincidencias':>14} | {'resets':>6} | ok")
    print("-" * 72)

    failures = 0
    for i in range(num_envs):
        ok = measured[i] == configured[i] and matches[i] > 0
        failures += int(not ok)
        print(
            f"{i:>4} | {int(configured[i]):>10} | {int(measured[i]):>10} | "
            f"{int(matches[i]):>6}/{int(possible[i]):<7} | {int(resets[:, i].sum()):>6} | "
            f"{'✓' if ok else '✗ MISMATCH'}"
        )

    distribution = Counter(int(lag) for lag in configured)
    distinct = len(distribution)
    in_range = all(low - 1 <= lag <= high - 1 for lag in distribution)

    print("-" * 72)
    print("Distribucion de lags: " + ", ".join(f"{lag}:{n}" for lag, n in sorted(distribution.items())))
    print(f"Valores distintos: {distinct}   (todos dentro del rango: {'si' if in_range else 'NO'})")
    print("=" * 72)

    passed = failures == 0 and distinct > 1 and in_range
    if passed:
        print(f"PASS: los {num_envs} envs aplican exactamente su propio lag, medido desde IsaacLab, "
              f"con {distinct} valores distintos.")
    else:
        if failures:
            print(f"FAIL: {failures} env(s) no aplican el lag configurado.")
        if distinct <= 1:
            print("FAIL: todos los envs tienen el mismo lag, el delay no se esta randomizando.")
        if not in_range:
            print("FAIL: hay lags fuera del rango pedido.")

    env.close()
    return 0 if passed else 1


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    sys.exit(exit_code)
