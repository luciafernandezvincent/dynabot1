#!/usr/bin/env python3
"""Regenera la cadena de experimentos de research bajo action-delay=5 (el delay real de los
actuadores; las corridas anteriores se entrenaron sin delay por el bug de run_experiment.py ya
corregido: no propagaba --action-delay a eval_cmd).

Reproduce, en orden, la misma secuencia de (base, config, notas) que quedo registrada en
results.jsonl ANTES del fix, pero corriendo sobre el run_experiment.py ya parcheado (train, eval
y video usan --action-delay=5). Cada corrida se nombra "<exp>_delay5" para no pisar los
checkpoints/videos viejos (quedan como referencia para comparar antes/despues del delay).

Se deja afuera "smoke_revert" (test de humo del pipeline, no una hipotesis de reward) y los 3
intentos fallidos de exp_008_bigger_net (solo se replica el 4to, ya con el pipeline corregido).
baseline_ar no esta en esta cola porque ya se corrio a mano como "baseline_delay5"
(research/configs/baseline.yaml, sin base).

Uso:
    nohup /home/linar/miniconda3/envs/env_isaaclab/bin/python research/regen_delay5.py \
        > research/runs/regen_delay5.log 2>&1 &
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
RESULTS_JSONL = RESEARCH_DIR / "results.jsonl"
PYTHON = "/home/linar/miniconda3/envs/env_isaaclab/bin/python"
NAME_SUFFIX = "_delay5"

# (base_config_filename, config_filename, notas historicas) en el mismo orden en que corrieron.
QUEUE = [
    ("baseline.yaml", "exp_001_clearance_on.yaml",
     "activar foot_clearance w=0.5 target 5cm"),
    ("baseline.yaml", "exp_002_airtime_threshold_fix.yaml",
     "CAUSA RAIZ: threshold 0.5->0.12s aislado vs baseline (sin clearance)"),
    ("champion.yaml", "exp_009_long_swing.yaml",
     "swing mas largo via clearance w=1.0 target 8cm (el umbral de air_time resulto contraproducente)"),
    ("champion.yaml", "exp_011_symmetry.yaml",
     "air_time_variance w=-1.0: la asimetria empeora al subir el despeje"),
    ("champion.yaml", "exp_004_clearance_strong.yaml",
     "clearance w=1.5 target 5cm: mas peso con objetivo alcanzable"),
    ("champion.yaml", "exp_010_trot_gait.yaml",
     "GaitReward w=5.0, sincronizar diagonales (trote real)"),
    ("champion.yaml", "exp_008_bigger_net.yaml",
     "red actor/critic [512,256,128] en vez de [128,128,128], sobre el campeon"),
    ("champion.yaml", "exp_015_joint_dev_soft.yaml",
     "joint_deviation_l1 w=-0.05: mantener juntas cerca de la default"),
    ("champion.yaml", "exp_016_joint_dev_strong.yaml",
     "joint_deviation_l1 w=-0.2: version agresiva"),
    ("champion.yaml", "exp_017_joint_dev_very_strong.yaml",
     "joint_deviation w=-0.5: aun menos flexionado"),
    ("champion.yaml", "exp_019_shoulder_dev.yaml",
     "joint_deviation SOLO en hombros w=-0.5: postura recta sin achicar el paso"),
    ("champion.yaml", "exp_020_shoulder_dev_strong.yaml",
     "joint_deviation solo hombros w=-1.0 (el doble de exp_019)"),
    ("champion.yaml", "exp_021_shoulder_plus_swing.yaml",
     "hombros w=-1.0 + feet_air_time w=1.5: postura recta con swing mas largo"),
    ("champion.yaml", "exp_022_airtime_max.yaml",
     "feet_air_time 3.0: mas vuelo, paso mas largo"),
    ("champion.yaml", "exp_023_vertical_freedom.yaml",
     "lin_vel_z_l2 -0.5: permitir arco vertical"),
    ("champion.yaml", "exp_024_free_actions.yaml",
     "action_rate_l2 -0.03: excursiones articulares mas grandes"),
    ("champion.yaml", "exp_025_clearance_up.yaml",
     "foot_clearance target 7cm con air_time alto"),
    ("champion.yaml", "exp_026_airtime_5.yaml",
     "feet_air_time 5.0: mas amplitud"),
    ("champion.yaml", "exp_027_airtime3_plus_vert.yaml",
     "air_time 3.0 + libertad vertical: los dos mejores juntos"),
    ("champion.yaml", "exp_028_airtime5_plus_vert.yaml",
     "air_time 5.0 + libertad vertical: el mas agresivo"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_gpu_free(poll_s: int = 30) -> None:
    while True:
        out = subprocess.run(["pgrep", "-f", "train_delay.py|eval.py|play_delay.py"],
                              capture_output=True, text=True)
        if out.returncode != 0:  # pgrep sin matches -> rc 1
            return
        log("Esperando a que termine el proceso de train/eval/video en curso...")
        time.sleep(poll_s)


def already_done(name: str) -> bool:
    if not RESULTS_JSONL.exists():
        return False
    for line in RESULTS_JSONL.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get("name") == name and record.get("status") == "ok":
            return True
    return False


def main() -> int:
    log(f"Arrancando regeneracion de {len(QUEUE)} experimentos con action-delay=5")
    for base_name, config_name, notes in QUEUE:
        name = config_name[:-len(".yaml")] + NAME_SUFFIX
        if already_done(name):
            log(f"{name}: ya tiene un registro OK, se saltea (borrar research/runs/{name} para re-correr)")
            continue
        wait_for_gpu_free()
        log(f"=== {name} (base={base_name}) : {notes} ===")
        cmd = [
            PYTHON, str(RESEARCH_DIR / "run_experiment.py"),
            "--base", str(CONFIGS_DIR / base_name),
            "--config", str(CONFIGS_DIR / config_name),
            "--name", name,
            "--notes", f"[delay5, replica de {config_name[:-len('.yaml')]}] {notes}",
        ]
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            log(f"{name}: FALLO (returncode={result.returncode}), sigo con el siguiente")
        subprocess.run([PYTHON, str(RESEARCH_DIR / "gen_experiment_docs.py")], cwd=REPO_ROOT)
    log("Regeneracion completa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
