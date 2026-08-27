#!/usr/bin/env python3
"""Genera, por cada experimento "*_delay5" registrado en results.jsonl, un .yaml legible en
research/experiment_docs/ con: los hiperparametros resueltos, la explicacion (notas originales
de por que se eligio esa variacion) y el experimento contra el que se comparo (compared_to).

Es idempotente: solo lee results.jsonl (y el header de champion.yaml para resolver a que
experimento equivale "champion.yaml" como base) y reescribe los .yaml. Se puede correr en
cualquier momento durante research/regen_delay5.py para ir viendo los docs de lo que ya termino,
o al final para generarlos todos.

Uso:
    python research/gen_experiment_docs.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = REPO_ROOT / "research"
CONFIGS_DIR = RESEARCH_DIR / "configs"
RESULTS_JSONL = RESEARCH_DIR / "results.jsonl"
OUT_DIR = RESEARCH_DIR / "experiment_docs"

CHAMPION_HEADER_RE = re.compile(r"Actual:\s*([A-Za-z0-9_]+),\s*score")


def load_records() -> list[dict]:
    """Ultimo registro por nombre, en orden de primera aparicion."""
    by_name: dict[str, dict] = {}
    order: list[str] = []
    for line in RESULTS_JSONL.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record["name"] not in by_name:
            order.append(record["name"])
        by_name[record["name"]] = record
    return [by_name[n] for n in order]


def champion_reference() -> str | None:
    """Nombre del experimento (sin sufijo) cuyo config resuelto quedo grabado como champion.yaml."""
    path = CONFIGS_DIR / "champion.yaml"
    if not path.exists():
        return None
    for line in path.read_text().splitlines()[:5]:
        m = CHAMPION_HEADER_RE.search(line)
        if m:
            return m.group(1)
    return None


def reference_for(record: dict, champion_name: str | None) -> str | None:
    base_path = record.get("base_path")
    if not base_path:
        return None
    base_name = Path(base_path).name
    if base_name == "baseline.yaml":
        return "baseline_delay5"
    if base_name == "champion.yaml" and champion_name:
        return f"{champion_name}_delay5"
    return base_name


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    champion_name = champion_reference()
    written = 0
    for record in load_records():
        name = record["name"]
        if not name.endswith("_delay5") or record.get("status") != "ok":
            continue
        doc = {
            "experiment": name,
            "score": record.get("score"),
            "valid": record.get("valid", True),
            "action_delay": record.get("action_delay"),
            "compared_to": reference_for(record, champion_name),
            "explicacion": record.get("notes"),
            "config_source": record.get("config_path"),
            "hyperparameters": record.get("config"),
        }
        out_path = OUT_DIR / f"{name}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(doc, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
        written += 1
        print(f"[OK] {out_path}")
    print(f"Listo: {written} docs en {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
