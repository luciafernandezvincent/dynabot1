#!/usr/bin/env python3
"""Companion de una corrida de regen_delay5.py ya en marcha (que no puede leer en caliente el
llamado a gen_experiment_docs.py agregado despues de arrancarla). Mientras regen_delay5.py siga
vivo, cada vez que resultados.jsonl cambia corre gen_experiment_docs.py; hace una pasada final al
detectar que termino y se cierra solo.

Uso:
    nohup /home/linar/miniconda3/envs/env_isaaclab/bin/python research/watch_and_doc.py \
        > research/runs/watch_and_doc.log 2>&1 &
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = REPO_ROOT / "research"
RESULTS_JSONL = RESEARCH_DIR / "results.jsonl"
PYTHON = "/home/linar/miniconda3/envs/env_isaaclab/bin/python"
POLL_S = 20


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def regen_running() -> bool:
    out = subprocess.run(["pgrep", "-f", "regen_delay5.py"], capture_output=True, text=True)
    return out.returncode == 0


def gen_docs() -> None:
    subprocess.run([PYTHON, str(RESEARCH_DIR / "gen_experiment_docs.py")], cwd=REPO_ROOT)


def main() -> int:
    log("Arrancando watcher de docs para la corrida de regen_delay5.py en curso")
    last_mtime = RESULTS_JSONL.stat().st_mtime if RESULTS_JSONL.exists() else None
    while regen_running():
        time.sleep(POLL_S)
        mtime = RESULTS_JSONL.stat().st_mtime if RESULTS_JSONL.exists() else None
        if mtime != last_mtime:
            last_mtime = mtime
            log("results.jsonl cambio, regenerando docs")
            gen_docs()
    log("regen_delay5.py ya no esta corriendo, ultima pasada de docs y cierro")
    gen_docs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
