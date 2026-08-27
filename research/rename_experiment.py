#!/usr/bin/env python3
"""Renombra una corrida YA TERMINADA: mueve research/runs/<old> y logs/rsl_rl/anymal_d_flat/<old>,
actualiza results.jsonl y research/experiment_docs/, y opcionalmente renombra su config.

NO usar sobre una corrida que todavia esta entrenando/evaluando: los paths ya quedaron resueltos
dentro del proceso vivo (env_cfg.log_dir, etc.) y renombrar el directorio por debajo lo rompe.

Uso:
    python research/rename_experiment.py <nombre_viejo> <nombre_nuevo> \
        [--config-old <archivo.yaml> --config-new <archivo.yaml>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = REPO_ROOT / "research"
RESULTS_JSONL = RESEARCH_DIR / "results.jsonl"
RUNS_DIR = RESEARCH_DIR / "runs"
DOCS_DIR = RESEARCH_DIR / "experiment_docs"
LOGS_DIR = REPO_ROOT / "logs" / "rsl_rl" / "anymal_d_flat"
CONFIGS_DIR = RESEARCH_DIR / "configs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old_name")
    ap.add_argument("new_name")
    ap.add_argument("--config-old")
    ap.add_argument("--config-new")
    args = ap.parse_args()
    old, new = args.old_name, args.new_name

    old_log, new_log = LOGS_DIR / old, LOGS_DIR / new
    if old_log.exists():
        old_log.rename(new_log)
        print(f"[OK] {old_log} -> {new_log}")

    old_run, new_run = RUNS_DIR / old, RUNS_DIR / new
    if old_run.exists():
        old_run.rename(new_run)
        old_mp4 = new_run / f"{old}.mp4"
        if old_mp4.exists():
            old_mp4.rename(new_run / f"{new}.mp4")
        print(f"[OK] {old_run} -> {new_run}")

    if args.config_old and args.config_new:
        co, cn = CONFIGS_DIR / args.config_old, CONFIGS_DIR / args.config_new
        if co.exists():
            co.rename(cn)
            print(f"[OK] {co} -> {cn}")

    lines = RESULTS_JSONL.read_text().splitlines()
    out = []
    changed = 0
    for line in lines:
        if not line.strip():
            out.append(line)
            continue
        rec = json.loads(line)
        if rec.get("name") == old:
            rec["name"] = new
            for key in ("video_path", "results_path"):
                if rec.get(key):
                    rec[key] = rec[key].replace(old, new)
            if args.config_old and rec.get("config_path", "").endswith(args.config_old):
                rec["config_path"] = rec["config_path"].replace(args.config_old, args.config_new)
            changed += 1
        out.append(json.dumps(rec))
    RESULTS_JSONL.write_text("\n".join(out) + "\n")
    print(f"[OK] results.jsonl: {changed} linea(s) actualizada(s)")

    old_doc, new_doc = DOCS_DIR / f"{old}.yaml", DOCS_DIR / f"{new}.yaml"
    if old_doc.exists():
        new_doc.write_text(old_doc.read_text().replace(old, new))
        old_doc.unlink()
        print(f"[OK] {old_doc} -> {new_doc}")

    print("Listo. Correr: python research/run_experiment.py --rebuild-table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
