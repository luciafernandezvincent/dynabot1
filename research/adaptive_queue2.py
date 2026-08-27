#!/usr/bin/env python3
"""Cola de experimentos adaptativos bajo action-delay=5, tanda 2 (exp_034 a exp_037).

A diferencia de regen_delay5.py (que replicaba la cadena historica), esta cola prueba hipotesis
nuevas definidas durante la sesion: dos levers no probados (capacidad de red, entropy_coef) y dos
combos con el unico factor de reward que mejoro aislado (feet_air_time.threshold=0.12, exp_002).

Espera a que se libere la GPU (deja terminar lo que este corriendo) y corre cada uno en orden.
Genera el doc estructurado (research/gen_experiment_docs.py) despues de cada uno; la explicacion
CUALITATIVA detallada (research/save_explanation.py) la escribe el agente a mano al revisar cada
resultado, no este script.

Uso:
    nohup /home/linar/miniconda3/envs/env_isaaclab/bin/python research/adaptive_queue2.py \
        > research/runs/adaptive_queue2.log 2>&1 &
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = REPO_ROOT / "research"
CONFIGS_DIR = RESEARCH_DIR / "configs"
RESULTS_JSONL = RESEARCH_DIR / "results.jsonl"
PYTHON = "/home/linar/miniconda3/envs/env_isaaclab/bin/python"

QUEUE = [
    ("exp_034_bigger_net.yaml", "exp_034_bigger_net_delay5",
     "[delay5, adaptativo] lever nuevo: red grande [512,256,128] aislada, sin tocar rewards, ver si mas capacidad compensa el delay"),
    ("exp_035_airtime012_biggernet.yaml", "exp_035_airtime_biggernet_delay5",
     "[delay5, adaptativo] combo: threshold=0.12 (exp_002) + red grande (exp_034)"),
    ("exp_036_entropy_010.yaml", "exp_036_entropy_010_delay5",
     "[delay5, adaptativo] lever nuevo: entropy_coef 0.005->0.01 aislado, mas exploracion"),
    ("exp_037_airtime012_gait.yaml", "exp_037_airtime_gait_delay5",
     "[delay5, adaptativo] combo: threshold=0.12 (exp_002) + GaitReward w=5.0 (trote), exp_010 no ayudo sin delay"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_gpu_free(poll_s: int = 20) -> None:
    while True:
        out = subprocess.run(["pgrep", "-f", "train_delay.py|eval.py|play_delay.py"],
                              capture_output=True, text=True)
        if out.returncode != 0:
            return
        log("Esperando a que termine el proceso de train/eval/video en curso...")
        time.sleep(poll_s)


def already_done(name: str) -> bool:
    if not RESULTS_JSONL.exists():
        return False
    for line in RESULTS_JSONL.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("name") == name and record.get("status") == "ok":
            return True
    return False


def main() -> int:
    log(f"Arrancando tanda 2 de {len(QUEUE)} experimentos adaptativos con action-delay=5")
    for config_name, name, notes in QUEUE:
        if already_done(name):
            log(f"{name}: ya tiene un registro OK, se saltea")
            continue
        wait_for_gpu_free()
        log(f"=== {name}: {notes} ===")
        cmd = [
            PYTHON, str(RESEARCH_DIR / "run_experiment.py"),
            "--base", str(CONFIGS_DIR / "baseline.yaml"),
            "--config", str(CONFIGS_DIR / config_name),
            "--name", name,
            "--notes", notes,
        ]
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            log(f"{name}: FALLO (returncode={result.returncode}), sigo con el siguiente")
        subprocess.run([PYTHON, str(RESEARCH_DIR / "gen_experiment_docs.py")], cwd=REPO_ROOT)
    log("Tanda 2 completa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
