#!/usr/bin/env python3
"""Runner de un experimento de autoresearch para Dyna1.

Encadena, con presupuesto fijo para que todos los experimentos sean comparables:

    train_delay.py (config YAML) -> eval.py (protocolo fijo) -> score.py -> results.jsonl -> RESULTS.md

Uso tipico:
    python research/run_experiment.py --config research/configs/exp_001_air_time.yaml \
        --name exp_001_air_time --notes "subir feet_air_time para pasos mas largos"

Solo se permiten overrides de pesos de rewards e hiperparametros de PPO. Cualquier YAML que
toque fisica (actuadores, masas, inercias, delays, dt, escena, eventos) se rechaza antes de
lanzar nada.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import format_breakdown, score_results  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = REPO_ROOT / "research"
RESULTS_JSONL = RESEARCH_DIR / "results.jsonl"
RESULTS_MD = RESEARCH_DIR / "RESULTS.md"
RUNS_DIR = RESEARCH_DIR / "runs"

# --------------------------------------------------------------------------------------
# Protocolo fijo. No cambiar entre experimentos: si cambia, los scores dejan de ser comparables.
# --------------------------------------------------------------------------------------
TASK = "Dyna1-Flat-v0"
EXPERIMENT_NAME = "anymal_d_flat"  # carpeta bajo logs/rsl_rl (viene del PPORunnerCfg)
TRAIN_ITERATIONS = 1500
TRAIN_NUM_ENVS = 4096
SEED = 42
EVAL_NUM_ENVS = 1000
EVAL_NUM_STEPS = 1000
VIDEO_NUM_ENVS = 10  # perros en pantalla
VIDEO_LENGTH_STEPS = 300  # 300 pasos * 0.02 s de step_dt = 6 s de video
VIDEO_TIMEOUT_S = 900
TRAIN_TIMEOUT_S = 5400
EVAL_TIMEOUT_S = 2400

# Prefijos de config permitidos. Todo lo demas se rechaza (regla dura: la fisica no se toca).
ALLOWED_PREFIXES = (
    "env.rewards",
    "env.episode_length_s",
    "agent.",
)
# Campos que el runner controla y el YAML no puede pisar (garantizan comparabilidad).
LOCKED_AGENT_FIELDS = ("max_iterations", "seed", "experiment_name", "resume", "load_run")


def deep_merge(base: dict, override: dict) -> dict:
    """Combina dos configs anidados; los valores de override ganan."""
    merged = json.loads(json.dumps(base))
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def flatten(config: dict, prefix: str = "") -> dict:
    """Aplana un dict anidado a claves con notacion de punto."""
    flat = {}
    for key, value in (config or {}).items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten(value, path))
        else:
            flat[path] = value
    return flat


def validate_config(config: dict) -> list[str]:
    """Devuelve la lista de violaciones del espacio de busqueda permitido."""
    problems = []
    for key in flatten(config):
        if not key.startswith(ALLOWED_PREFIXES):
            problems.append(f"'{key}' esta fuera del espacio permitido (solo env.rewards.*, env.episode_length_s, agent.*)")
        if key.startswith("agent.") and key.split(".", 2)[1] in LOCKED_AGENT_FIELDS:
            problems.append(f"'{key}' lo fija el runner y no puede venir del YAML")
    return problems


def resolve_config(config: dict, iterations: int, seed: int, out_path: Path) -> Path:
    """Escribe una copia del config con el presupuesto forzado y la devuelve.

    train_delay.py aplica el YAML DESPUES de --max_iterations, asi que el presupuesto solo
    queda garantizado si va dentro del propio YAML.
    """
    resolved = json.loads(json.dumps(config))  # copia profunda
    agent = resolved.setdefault("agent", {})
    agent["max_iterations"] = iterations
    agent["seed"] = seed
    agent["experiment_name"] = EXPERIMENT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)
    return out_path


def run_command(cmd: list[str], log_path: Path, timeout: int) -> tuple[int, str]:
    """Ejecuta un comando volcando stdout/stderr a un archivo. Devuelve (returncode, tail)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[CMD] {' '.join(cmd)}")
    print(f"[LOG] {log_path}")
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return 124, f"TIMEOUT tras {timeout} s"
    tail = "".join(open(log_path, errors="replace").readlines()[-25:])
    return returncode, tail


#: Archivos que definen el juez de la investigacion. Si cambian a mitad del camino, los scores
#: dejan de ser comparables, asi que el runner se planta.
JUDGE_FILES = ("research/score.py", "scripts/rsl_rl/eval.py")
JUDGE_PIN = RESEARCH_DIR / ".judge_hashes.json"


def judge_hashes() -> dict:
    """sha256 de los archivos que definen la metrica objetivo."""
    hashes = {}
    for relative in JUDGE_FILES:
        path = REPO_ROOT / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "MISSING"
    return hashes


def check_judge_integrity(accept_change: bool) -> list[str]:
    """Verifica que el juez no haya cambiado desde que arranco la investigacion.

    Devuelve la lista de archivos que cambiaron (vacia si esta todo bien). Con accept_change
    se re-fija la referencia, que es lo que corresponde cuando el humano cambia el criterio
    a proposito (y entonces conviene re-correr la baseline).
    """
    current = judge_hashes()
    if not JUDGE_PIN.exists() or accept_change:
        JUDGE_PIN.write_text(json.dumps(current, indent=2))
        return []
    pinned = json.loads(JUDGE_PIN.read_text())
    return [name for name, digest in current.items() if pinned.get(name) != digest]


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def append_record(record: dict) -> None:
    with open(RESULTS_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_records() -> list[dict]:
    if not RESULTS_JSONL.exists():
        return []
    records = []
    with open(RESULTS_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_results_table() -> None:
    """Regenera RESULTS.md ordenado por score descendente."""
    records = load_records()
    done = [r for r in records if r.get("status") == "ok"]
    failed = [r for r in records if r.get("status") != "ok"]
    done.sort(key=lambda r: r["score"], reverse=True)

    lines = [
        "# Resultados de autoresearch - Dyna1",
        "",
        "Generado por `research/run_experiment.py`. No editar a mano: se reescribe en cada experimento.",
        "",
        f"Protocolo: `{TASK}`, {TRAIN_ITERATIONS} iters x {TRAIN_NUM_ENVS} envs, seed {SEED}, "
        f"eval {EVAL_NUM_ENVS} envs x {EVAL_NUM_STEPS} pasos.",
        "",
        "| # | experimento | score | despeje mm | vel_track | ori_estab | ori_suav | mov_suav | impacto N | caidas/ep | zancada Hz | duty | notas |",
        "|---|-------------|-------|------------|-----------|-----------|----------|----------|-----------|-----------|------------|------|-------|",
    ]
    for rank, r in enumerate(done, start=1):
        m = r.get("metrics", {})
        stride = m.get("stride_frequency_hz_mean", m.get("movement_frequency_hz"))
        flag = "" if r.get("valid", True) else " ⚠"
        lines.append(
            "| {rank} | {name} | {score:.4f}{flag} | {clearance} | {vel:.3f} | {stab:.3f} | {smooth:.3f} | {mov:.3f} | "
            "{impact:.1f} | {falls:.3f} | {stride} | {duty} | {notes} |".format(
                rank=rank,
                name=r["name"],
                score=r["score"],
                flag=flag,
                clearance=(f"{m['foot_clearance_peak_m'] * 1000:.1f}"
                           if m.get("foot_clearance_peak_m") is not None else "-"),
                vel=m.get("velocity_tracking_accuracy_0to1", float("nan")),
                stab=m.get("orientation_stability_0to1", float("nan")),
                smooth=m.get("orientation_smoothness_0to1", float("nan")),
                mov=m.get("movement_smoothness", float("nan")),
                impact=m.get("impact_force_mean", float("nan")),
                falls=m.get("fall_rate_per_episode", float("nan")),
                stride=f"{stride:.2f}" if stride is not None else "-",
                duty=f"{m['duty_factor_mean']:.2f}" if m.get("duty_factor_mean") is not None else "-",
                notes=(r.get("notes") or "").replace("|", "/"),
            )
        )

    if failed:
        lines += ["", "## Experimentos fallidos", ""]
        for r in failed:
            lines.append(f"- `{r['name']}` ({r.get('timestamp', '?')}): {r.get('status')} - {r.get('error', '')[:200]}")

    lines += ["", "⚠ = viola una restriccion dura de `research/score.py` (no compite por el mejor).", ""]
    RESULTS_MD.write_text("\n".join(lines))
    print(f"[INFO] Tabla actualizada: {RESULTS_MD}")


def record_video(name: str, run_dir: Path, log_dir: Path, seed: int) -> tuple[str | None, float, str | None]:
    """Graba un video de la marcha con VIDEO_NUM_ENVS perros y lo deja en el dir del experimento.

    Corre despues del eval y antes de devolver el control, para que una cola secuencial nunca
    solape un play.py con el entrenamiento del experimento siguiente. Su resultado no afecta al
    score: si falla, se registra el error y el experimento sigue siendo valido.
    """
    video_cmd = [
        sys.executable, "scripts/rsl_rl/play.py",
        f"--task={TASK}",
        "--headless",
        "--video",
        f"--video_length={VIDEO_LENGTH_STEPS}",
        f"--load_run={name}",
        f"--num_envs={VIDEO_NUM_ENVS}",
        f"--seed={seed}",
    ]
    started = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    returncode, tail = run_command(video_cmd, run_dir / "video.log", VIDEO_TIMEOUT_S)
    elapsed = round(time.time() - started, 1)

    if returncode != 0:
        print(f"[WARN] No se pudo grabar el video (rc={returncode}); el experimento sigue siendo valido")
        return None, elapsed, tail[-500:]

    # play.py deja el mp4 en <log_dir>/videos/play/; se copia al dir del experimento con su nombre
    produced = sorted((log_dir / "videos" / "play").glob("*.mp4"))
    if not produced:
        print(f"[WARN] play.py termino ok pero no dejo ningun mp4 en {log_dir / 'videos' / 'play'}")
        return None, elapsed, "sin mp4 generado"

    destination = run_dir / f"{name}.mp4"
    destination.write_bytes(produced[-1].read_bytes())
    print(f"[INFO] Video: {destination}")
    return str(destination.relative_to(REPO_ROOT)), elapsed, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Corre un experimento de autoresearch (train + eval + score)")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML de overrides (rewards / PPO). Si se omite, corre con los defaults del codigo")
    parser.add_argument("--base", type=Path, default=None,
                        help="YAML base sobre el que se aplica --config (tipicamente el campeon vigente)")
    parser.add_argument("--name", type=str, default=None, help="Nombre del run (default: nombre del YAML)")
    parser.add_argument("--notes", type=str, default="", help="Hipotesis o cambio probado, para la tabla")
    parser.add_argument("--iterations", type=int, default=TRAIN_ITERATIONS, help="Presupuesto de entrenamiento")
    parser.add_argument("--num-envs", type=int, default=TRAIN_NUM_ENVS, help="Envs de entrenamiento")
    parser.add_argument("--seed", type=int, default=SEED, help="Semilla (fija para comparar)")
    parser.add_argument("--eval-only", action="store_true", help="Saltea el entrenamiento y evalua el checkpoint existente")
    parser.add_argument("--no-video", action="store_true", help="No grabar el video de la marcha al final")
    parser.add_argument("--video-only", action="store_true",
                        help="Solo grabar el video desde el checkpoint existente, sin entrenar ni evaluar")
    parser.add_argument("--dry-run", action="store_true", help="Valida el config e imprime los comandos, sin correr")
    parser.add_argument("--rebuild-table", action="store_true", help="Solo regenera RESULTS.md desde results.jsonl")
    parser.add_argument("--rescore", action="store_true",
                        help="Re-scorea todo results.jsonl con la version actual de score.py (usar tras recalibrar) y regenera la tabla")
    parser.add_argument("--accept-judge-change", action="store_true",
                        help="Acepta que score.py / eval.py cambiaron y re-fija la referencia (invalida comparaciones previas)")
    args = parser.parse_args()

    if args.video_only:
        if args.name is None:
            print("[ERROR] --video-only necesita --name", file=sys.stderr)
            return 1
        log_dir = REPO_ROOT / "logs" / "rsl_rl" / EXPERIMENT_NAME / args.name
        if not log_dir.exists():
            print(f"[ERROR] No existe la corrida {log_dir}", file=sys.stderr)
            return 1
        path, _, error = record_video(args.name, RUNS_DIR / args.name, log_dir, args.seed)
        return 0 if path else 1

    if args.rescore:
        # re-fijar el hash del juez ANTES de salir por este camino: si no, un --rescore
        # --accept-judge-change deja la referencia vieja y todo experimento posterior se planta
        check_judge_integrity(args.accept_judge_change)
        records = load_records()
        for r in records:
            if r.get("status") == "ok" and "metrics" in r:
                scored = score_results(r["metrics"])
                r["score"] = scored["score"]
                r["valid"] = scored["valid"]
                r["score_breakdown"] = scored
        with open(RESULTS_JSONL, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        write_results_table()
        print(f"[INFO] Re-scoreados {len(records)} registros con la version actual de score.py")
        return 0

    if args.rebuild_table:
        write_results_table()
        return 0

    if args.config is not None:
        args.config = args.config.resolve()
        if not args.config.exists():
            print(f"[ERROR] No existe el config: {args.config}", file=sys.stderr)
            return 1
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    elif args.name is None:
        print("[ERROR] Sin --config hay que dar un --name", file=sys.stderr)
        return 1
    else:
        config = {}  # defaults del codigo

    if args.base is not None:
        args.base = args.base.resolve()
        if not args.base.exists():
            print(f"[ERROR] No existe el base: {args.base}", file=sys.stderr)
            return 1
        config = deep_merge(load_yaml(args.base), config)
        print(f"[INFO] Config construido sobre la base {args.base.name}")

    changed = check_judge_integrity(args.accept_judge_change)
    if changed:
        print("[ERROR] Cambio el juez de la investigacion desde que arranco: " + ", ".join(changed), file=sys.stderr)
        print("  Los scores ya registrados dejan de ser comparables con los nuevos.", file=sys.stderr)
        print("  Si el cambio fue intencional: correr con --accept-judge-change y re-correr la baseline.", file=sys.stderr)
        return 1

    problems = validate_config(config)
    if problems:
        print("[ERROR] El config toca cosas fuera del espacio de busqueda permitido:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("  Recordatorio: la fisica del simulador (actuadores, masas, inercias, delays, dt) NO se toca.",
              file=sys.stderr)
        return 1

    name = args.name or args.config.stem
    run_dir = RUNS_DIR / name
    log_dir = REPO_ROOT / "logs" / "rsl_rl" / EXPERIMENT_NAME / name
    resolved_path = resolve_config(config, args.iterations, args.seed, run_dir / "config.resolved.yaml")

    train_cmd = [
        sys.executable, "scripts/rsl_rl/train_delay.py",
        f"--task={TASK}",
        "--headless",
        f"--num_envs={args.num_envs}",
        f"--name={name}",
        f"--seed={args.seed}",
        f"--experiment_config={resolved_path}",
    ]
    eval_cmd = [
        sys.executable, "scripts/rsl_rl/eval.py",
        f"--task={TASK}",
        "--headless",
        f"--load_run={name}",
        f"--num_envs={EVAL_NUM_ENVS}",
        f"--num_steps={EVAL_NUM_STEPS}",
        f"--seed={args.seed}",
    ]

    if args.dry_run:
        print("[DRY-RUN] config valido. Comandos que se ejecutarian:")
        print("  " + " ".join(train_cmd))
        print("  " + " ".join(eval_cmd))
        return 0

    record = {
        "name": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config_path": str(args.config.relative_to(REPO_ROOT)) if args.config else None,
        "base_path": str(args.base.relative_to(REPO_ROOT)) if args.base else None,
        "config": config,
        "notes": args.notes,
        "iterations": args.iterations,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "judge_hashes": judge_hashes(),
    }

    started = time.time()

    if not args.eval_only:
        returncode, tail = run_command(train_cmd, run_dir / "train.log", TRAIN_TIMEOUT_S)
        if returncode != 0:
            record.update({"status": "train_failed", "error": tail})
            append_record(record)
            write_results_table()
            print(f"[ERROR] Entrenamiento fallido (rc={returncode}). Ultimas lineas:\n{tail}", file=sys.stderr)
            return 1
    record["train_seconds"] = round(time.time() - started, 1)

    eval_started = time.time()
    returncode, tail = run_command(eval_cmd, run_dir / "eval.log", EVAL_TIMEOUT_S)
    if returncode != 0:
        record.update({"status": "eval_failed", "error": tail})
        append_record(record)
        write_results_table()
        print(f"[ERROR] Evaluacion fallida (rc={returncode}). Ultimas lineas:\n{tail}", file=sys.stderr)
        return 1
    record["eval_seconds"] = round(time.time() - eval_started, 1)

    if not args.no_video:
        video_path, video_seconds, video_error = record_video(name, run_dir, log_dir, args.seed)
        record["video_seconds"] = video_seconds
        if video_path:
            record["video_path"] = video_path
        if video_error:
            record["video_error"] = video_error

    results_path = log_dir / "eval" / "results.json"
    if not results_path.exists():
        record.update({"status": "no_results", "error": f"No se genero {results_path}"})
        append_record(record)
        write_results_table()
        print(f"[ERROR] No se encontro {results_path}", file=sys.stderr)
        return 1

    with open(results_path) as f:
        metrics = json.load(f)
    scored = score_results(metrics)

    record.update({
        "status": "ok",
        "metrics": metrics,
        "score": scored["score"],
        "valid": scored["valid"],
        "score_breakdown": scored,
        "results_path": str(results_path.relative_to(REPO_ROOT)),
    })
    append_record(record)
    write_results_table()

    # comparacion contra el mejor previo
    previous = [r for r in load_records()[:-1] if r.get("status") == "ok" and r.get("valid", True)]
    best_previous = max(previous, key=lambda r: r["score"], default=None)

    print("\n" + "=" * 80)
    print(format_breakdown(name, metrics, scored))
    if best_previous:
        delta = scored["score"] - best_previous["score"]
        veredicto = "MEJORA" if delta > 0 else "NO MEJORA"
        print(f"\nMejor previo: {best_previous['name']} = {best_previous['score']:.4f}  ->  {veredicto} ({delta:+.4f})")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
