#!/usr/bin/env python3
"""Corre la cola de experimentos de una noche de autoresearch para Dyna1, sin supervision.

Espera a que termine cualquier train_delay.py/eval.py en curso (para no competir por GPU con la
baseline), y despues corre la cola: para cada experimento, entrena sobre el campeon vigente
(research/configs/champion.yaml), evalua, scorea, y si mejora y es valido, actualiza el campeon.
Todo pasa por research/run_experiment.py, que ya tiene el protocolo fijo y las validaciones
(rechazo de configs que tocan fisica, verificacion de integridad del juez).

No es un agente: sigue la cola tal cual esta escrita, no decide nuevas hipotesis. Es la ejecucion
mecanica del backlog inicial de research/program.md para que no dependa de que la sesion del
agente siga viva toda la noche.

Uso:
    nohup /home/linar/miniconda3/envs/env_isaaclab/bin/python research/run_night.py \
        > research/runs/night.log 2>&1 &
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = REPO_ROOT / "research"
CONFIGS_DIR = RESEARCH_DIR / "configs"
CHAMPION_PATH = CONFIGS_DIR / "champion.yaml"
BASELINE_PATH = CONFIGS_DIR / "baseline.yaml"
RESULTS_JSONL = RESEARCH_DIR / "results.jsonl"
PYTHON = "/home/linar/miniconda3/envs/env_isaaclab/bin/python"

# Cola de la noche: (config, nombre, notas). Un factor por experimento (ver hipotesis en cada
# YAML). Se corren en este orden porque exp_001/002/003 atacan el problema mas grave medido
# (la baseline sigue el comando de velocidad peor que quedarse quieta).
QUEUE = [
    ("exp_026_airtime_5.yaml", "exp_026_airtime_5", "feet_air_time 5.0: mas amplitud"),
    ("exp_027_airtime3_plus_vert.yaml", "exp_027_airtime3_plus_vert", "air_time 3.0 + libertad vertical: los dos mejores juntos"),
    ("exp_028_airtime5_plus_vert.yaml", "exp_028_airtime5_plus_vert", "air_time 5.0 + libertad vertical: el mas agresivo"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_gpu_free(poll_s: int = 30) -> None:
    """Espera a que no haya train_delay.py ni eval.py corriendo (deja terminar la baseline)."""
    while True:
        out = subprocess.run(["pgrep", "-f", "train_delay.py|eval.py"], capture_output=True, text=True)
        if out.returncode != 0:  # pgrep sin matches -> rc 1
            return
        log("Esperando a que termine el proceso de train/eval en curso...")
        time.sleep(poll_s)


def load_records() -> list[dict]:
    """Deja solo el ULTIMO registro por nombre: un --name relanzado (bug corregido) reemplaza
    al intento anterior en vez de convivir con el en el historial."""
    if not RESULTS_JSONL.exists():
        return []
    by_name = {}
    for line in RESULTS_JSONL.read_text().splitlines():
        line = line.strip()
        if line:
            record = json.loads(line)
            by_name[record["name"]] = record
    return list(by_name.values())


def best_valid_score() -> float | None:
    records = [r for r in load_records() if r.get("status") == "ok" and r.get("valid", True)]
    if not records:
        return None
    return max(r["score"] for r in records)


def ensure_champion() -> None:
    """Si todavia no hay campeon, arranca de la baseline (o de los defaults si no existe)."""
    if CHAMPION_PATH.exists():
        return
    if BASELINE_PATH.exists():
        CHAMPION_PATH.write_text(BASELINE_PATH.read_text())
        log(f"Campeon inicial = baseline.yaml (copiado a {CHAMPION_PATH.name})")
    else:
        CHAMPION_PATH.write_text("# vacio: corre con los defaults del codigo\n")
        log("Campeon inicial vacio (defaults del codigo)")


def run_one(config_name: str, exp_name: str, notes: str) -> None:
    run_dir = RESEARCH_DIR / "runs" / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    console_log = run_dir / "console.log"

    score_before = best_valid_score()
    log(f"=== {exp_name}: {notes} ===")

    cmd = [
        PYTHON, str(RESEARCH_DIR / "run_experiment.py"),
        "--base", str(CHAMPION_PATH),
        "--config", str(CONFIGS_DIR / config_name),
        "--name", exp_name,
        "--notes", notes,
    ]
    with open(console_log, "w") as f:
        subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)

    records = load_records()
    record = next((r for r in reversed(records) if r["name"] == exp_name), None)
    if record is None:
        log(f"{exp_name}: no genero registro (revisar {console_log})")
        return
    if record.get("status") != "ok":
        log(f"{exp_name}: FALLO ({record.get('status')}). Ver {console_log}")
        return

    score = record["score"]
    valid = record.get("valid", True)
    if not valid:
        log(f"{exp_name}: score={score:.4f} pero INVALIDO ({'; '.join(record['score_breakdown']['violations'])}). No promueve.")
        return

    if score_before is None or score > score_before:
        resolved = run_dir / "config.resolved.yaml"
        lines = [
            ln for ln in resolved.read_text().splitlines()
            if not ln.strip().startswith(("max_iterations:", "seed:", "experiment_name:"))
        ]
        CHAMPION_PATH.write_text("\n".join(lines) + "\n")
        log(f"{exp_name}: score={score:.4f} MEJORA sobre {score_before}. Nuevo campeon.")
    else:
        log(f"{exp_name}: score={score:.4f}, no supera al campeon ({score_before:.4f}). Descartado.")


def main() -> int:
    log("Arrancando cola de autoresearch nocturna")
    wait_for_gpu_free()
    ensure_champion()

    for config_name, exp_name, notes in QUEUE:
        run_dir = RESEARCH_DIR / "runs" / exp_name
        if (run_dir / "console.log").exists() and any(r["name"] == exp_name for r in load_records()):
            log(f"{exp_name}: ya tiene un registro, se saltea (borrar research/runs/{exp_name} para re-correr)")
            continue
        run_one(config_name, exp_name, notes)

    best = best_valid_score()
    log(f"Cola terminada. Mejor score valido: {best}")
    log(f"Campeon final en {CHAMPION_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
