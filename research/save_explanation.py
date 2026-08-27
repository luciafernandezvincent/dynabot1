#!/usr/bin/env python3
"""Guarda una explicacion detallada (score global + lectura metrica por metrica de eval.py,
por que se eligio el cambio, por que funciono o no) dentro del results.json de la corrida.

No toca eval.py ni score.py (son los "jueces": cambiar su contenido cambia su hash y
run_experiment.py invalida las comparaciones ya hechas, ver judge_hashes en results.jsonl).
Este script solo le agrega una clave extra ("_explicacion_detallada") al results.json que
eval.py ya escribio, para que quede junto a las metricas crudas sin alterar el archivo que
score.py usa para puntuar.

Uso:
    python research/save_explanation.py <nombre_corrida> <archivo_con_el_texto>
    echo "texto..." | python research/save_explanation.py <nombre_corrida> -
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_NAME = "anymal_d_flat"


def main() -> int:
    if len(sys.argv) != 3:
        print("uso: save_explanation.py <nombre_corrida> <archivo|->", file=sys.stderr)
        return 1
    name, src = sys.argv[1], sys.argv[2]
    text = sys.stdin.read() if src == "-" else Path(src).read_text()
    results_path = REPO_ROOT / "logs" / "rsl_rl" / EXPERIMENT_NAME / name / "eval" / "results.json"
    if not results_path.exists():
        print(f"[ERROR] no existe {results_path}", file=sys.stderr)
        return 1
    data = json.loads(results_path.read_text())
    data["_explicacion_detallada"] = text.strip()
    results_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"[OK] guardado en {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
