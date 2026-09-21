#!/usr/bin/env python3
"""Curva de robustez al delay: evalua UN checkpoint a varios delays FIJOS.

Entrenar con delay aleatorio hace a la politica robusta; esto lo mide. Para cada delay corre el
protocolo de evaluacion completo y registra el score, de modo que se vea a partir de que latencia
la marcha se degrada. Eso es lo que dice si la politica va a sobrevivir en el perro real, donde la
latencia no es un numero fijo elegido por nosotros.

El delay se barre FIJO a proposito: con delay aleatorio el sorteo se vuelve ruido en la metrica y
dos corridas identicas dan numeros distintos.

Uso:
    python research/delay_sweep.py --name exp_random_delay --delays 1,3,5,7,9,11
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiment import (  # noqa: E402
    EVAL_NUM_ENVS,
    EVAL_NUM_STEPS,
    EVAL_TIMEOUT_S,
    EXPERIMENT_NAME,
    REPO_ROOT,
    RUNS_DIR,
    SEED,
    TASK,
    run_command,
)
from score import score_results  # noqa: E402

DEFAULT_DELAYS = "1,3,5,7,9,11"
#: step_dt del entorno (decimation 4 x sim.dt 0.005), para reportar el lag en milisegundos.
STEP_DT_S = 0.02


def eval_at_delay(name: str, delay: int, seed: int, out_dir: Path, experiment_config: Path | None) -> dict | None:
    """Corre eval.py con un delay fijo y devuelve las metricas, o None si fallo."""
    cmd = [
        sys.executable, "scripts/rsl_rl/eval.py",
        f"--task={TASK}",
        "--headless",
        f"--load_run={name}",
        f"--num_envs={EVAL_NUM_ENVS}",
        f"--num_steps={EVAL_NUM_STEPS}",
        f"--seed={seed}",
        f"--action-delay={delay}",
    ]
    if experiment_config is not None:
        cmd.append(f"--experiment_config={experiment_config}")

    returncode, tail = run_command(cmd, out_dir / f"eval_delay{delay:02d}.log", EVAL_TIMEOUT_S)
    if returncode != 0:
        print(f"[ERROR] delay={delay} fallo (rc={returncode}). Ultimas lineas:\n{tail}")
        return None

    # eval.py siempre escribe al mismo path, asi que hay que apartarlo antes de la proxima vuelta
    produced = REPO_ROOT / "logs" / "rsl_rl" / EXPERIMENT_NAME / name / "eval" / "results.json"
    if not produced.exists():
        print(f"[ERROR] delay={delay}: no se genero {produced}")
        return None

    saved = out_dir / f"delay{delay:02d}.json"
    shutil.copy(produced, saved)
    with open(saved) as f:
        return json.load(f)


def write_table(name: str, rows: list[dict], out_dir: Path) -> None:
    """Escribe la curva como markdown, ordenada por delay creciente."""
    lines = [
        f"# Curva de robustez al delay - `{name}`",
        "",
        f"Protocolo: `{TASK}`, {EVAL_NUM_ENVS} envs x {EVAL_NUM_STEPS} pasos, seed {SEED}. "
        f"El lag aplicado es `delay - 1` pasos de {STEP_DT_S * 1000:.0f} ms.",
        "",
        "| delay | lag ms | score | vel_track | caidas/ep | zancada Hz | duty | impacto N | despeje mm |",
        "|-------|--------|-------|-----------|-----------|------------|------|-----------|------------|",
    ]
    for row in rows:
        m = row["metrics"]
        clearance = m.get("foot_clearance_peak_m")
        lines.append(
            "| {delay} | {lag_ms:.0f} | {score:.4f}{flag} | {vel:.3f} | {falls:.3f} | {stride} | "
            "{duty} | {impact:.1f} | {clearance} |".format(
                delay=row["delay"],
                lag_ms=(row["delay"] - 1) * STEP_DT_S * 1000,
                score=row["score"],
                flag="" if row["valid"] else " ⚠",
                vel=m.get("velocity_tracking_accuracy_0to1", float("nan")),
                falls=m.get("fall_rate_per_episode", float("nan")),
                stride=(f"{m['stride_frequency_hz_mean']:.2f}"
                        if m.get("stride_frequency_hz_mean") is not None else "-"),
                duty=f"{m['duty_factor_mean']:.2f}" if m.get("duty_factor_mean") is not None else "-",
                impact=m.get("impact_force_mean", float("nan")),
                clearance=f"{clearance * 1000:.1f}" if clearance is not None else "-",
            )
        )

    lines += ["", "⚠ = viola una restriccion dura de `research/score.py`.", ""]

    # punto de quiebre: primer delay cuyo score cae por debajo del 90% del mejor de la curva
    if rows:
        best = max(row["score"] for row in rows)
        broken = [row for row in rows if row["score"] < 0.9 * best]
        if broken:
            first = min(broken, key=lambda r: r["delay"])
            lines += [
                f"**Punto de quiebre**: a delay {first['delay']} "
                f"({(first['delay'] - 1) * STEP_DT_S * 1000:.0f} ms) el score cae a "
                f"{first['score']:.4f}, por debajo del 90% del mejor de la curva ({best:.4f}).",
                "",
            ]
        else:
            lines += [
                f"**Sin punto de quiebre** en el rango barrido: ningun delay baja del 90% del "
                f"mejor score ({best:.4f}). La politica aguanta toda la ventana probada.",
                "",
            ]

    path = out_dir / "delay_sweep.md"
    path.write_text("\n".join(lines))
    print(f"\n[INFO] Tabla: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evalua un checkpoint a varios delays fijos")
    parser.add_argument("--name", type=str, required=True, help="Nombre de la corrida a evaluar")
    parser.add_argument("--delays", type=str, default=DEFAULT_DELAYS,
                        help=f"Delays a probar, separados por coma (default: {DEFAULT_DELAYS})")
    parser.add_argument("--seed", type=int, default=SEED, help="Semilla (fija para comparar)")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML usado para entrenar (default: el config.resolved.yaml de la corrida)")
    args = parser.parse_args()

    log_dir = REPO_ROOT / "logs" / "rsl_rl" / EXPERIMENT_NAME / args.name
    if not log_dir.exists():
        print(f"[ERROR] No existe la corrida {log_dir}", file=sys.stderr)
        return 1

    experiment_config = args.config
    if experiment_config is None:
        resolved = RUNS_DIR / args.name / "config.resolved.yaml"
        experiment_config = resolved if resolved.exists() else None
    if experiment_config is not None:
        print(f"[INFO] Usando config de entrenamiento: {experiment_config}")

    delays = [int(part.strip()) for part in args.delays.split(",")]
    out_dir = RUNS_DIR / args.name / "delay_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.time()
    for i, delay in enumerate(delays, start=1):
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(delays)}] {args.name} @ delay={delay} "
              f"(lag {delay - 1} pasos = {(delay - 1) * STEP_DT_S * 1000:.0f} ms)")
        print("=" * 80)

        metrics = eval_at_delay(args.name, delay, args.seed, out_dir, experiment_config)
        if metrics is None:
            continue

        scored = score_results(metrics)
        rows.append({
            "delay": delay,
            "lag_steps": delay - 1,
            "lag_ms": (delay - 1) * STEP_DT_S * 1000,
            "score": scored["score"],
            "valid": scored["valid"],
            "metrics": metrics,
        })
        print(f"  -> score {scored['score']:.4f}"
              f"  vel_track {metrics.get('velocity_tracking_accuracy_0to1', float('nan')):.3f}"
              f"  caidas/ep {metrics.get('fall_rate_per_episode', float('nan')):.3f}")

    if not rows:
        print("[ERROR] Ninguna evaluacion produjo resultados", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r["delay"])
    with open(out_dir / "delay_sweep.json", "w") as f:
        json.dump({"name": args.name, "seed": args.seed, "rows": rows}, f, indent=2)
    write_table(args.name, rows, out_dir)

    print(f"[INFO] {len(rows)}/{len(delays)} evaluaciones en {round(time.time() - started, 1)} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
