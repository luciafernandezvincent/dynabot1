#!/usr/bin/env python3
"""Score compuesto para los experimentos de autoresearch de Dyna1.

Convierte el ``results.json`` que produce ``scripts/rsl_rl/eval.py`` en un unico numero
comparable entre experimentos. Este archivo lo edita el HUMANO (define que es "mejor");
el agente de autoresearch NO debe modificarlo: cambiar el criterio a mitad de la
investigacion hace que los scores dejen de ser comparables.

Diseno del score
----------------
Las metricas de eval.py viven en escalas muy distintas (velocity_tracking ~0.21,
orientation_stability ~0.04, movement_smoothness ~0.98 saturada). Sumarlas con pesos
directamente hace que unas dominen y otras no discriminen nada.

Por eso cada metrica se mapea con una referencia tomada de la corrida ``baseline``:

    termino = ref / (ref + x)      para cantidades donde menos es mejor (aceleracion, impacto)
    termino = x / (x + ref)        para cantidades donde mas es mejor (seguimiento de velocidad)

Ambas formas valen **0.5 exactamente en la baseline**, tienden a 1.0 en el optimo y a 0.0 en
el peor caso. Lectura directa del score total:

    ~0.5  = igual que la baseline
    >0.5  = mejor que la baseline
    <0.5  = peor que la baseline

Gate de locomocion
------------------
Medido empiricamente (21/08, con una version del codigo anterior a la actual): una politica SIN
ENTRENAR, que deja las cuatro patas apoyadas el 99.5% del tiempo, saco mejor estabilidad, suavidad,
impacto e incluso mejor seguimiento de velocidad que una corrida entrenada CUYO results.json
resulto estar desactualizado (train.py de una version vieja del codigo, antes de que cambiaran los
pesos de reward). La baseline entrenada con el codigo actual (`baseline_ar`, 21/08/2026) camina
bien de entrada: tracking 0.972, zancada 3.07 Hz, duty factor 0.47. El gate se deja igual porque
sigue siendo una red de seguridad correcta ante cualquier config futura que induzca el mismo
minimo local (quedarse quieto), no porque siga siendo necesario para la baseline actual.

Uso:
    python research/score.py logs/rsl_rl/anymal_d_flat/baseline/eval/results.json
    python research/score.py logs/rsl_rl/anymal_d_flat/*/eval/results.json --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# --------------------------------------------------------------------------------------
# Definicion del objetivo (editar aca para cambiar que se considera "mejor")
# --------------------------------------------------------------------------------------

#: Peso de cada termino del score. Suman 1.0.
WEIGHTS = {
    "velocity_tracking": 0.40,  # sigue el comando de velocidad (lo mas importante)
    "orientation_stability": 0.20,  # roll/pitch constantes: no cabecea ni se balancea
    "orientation_smoothness": 0.15,  # sin rotaciones bruscas del cuerpo
    "movement_smoothness": 0.10,  # aceleraciones articulares bajas (menos desgaste del servo)
    "impact": 0.10,  # pisadas suaves (protege el hardware real)
    "gait": 0.05,  # marcha en rango de frecuencia / duty factor razonable
}

# --------------------------------------------------------------------------------------
# Referencias: valores de la corrida 'baseline_ar' (logs/rsl_rl/anymal_d_flat/baseline_ar,
# 21/08/2026), entrenada con el protocolo actual del runner (1500 iters x 4096 envs, seed 42)
# y los defaults del codigo TAL COMO ESTA HOY. Cada termino vale 0.5 cuando la metrica iguala
# su referencia. Si algun dia se re-define la baseline, actualizar estos numeros y re-scorear
# todo el historico con:  python research/run_experiment.py --rebuild-table
# --------------------------------------------------------------------------------------
REF_VELOCITY_TRACKING = 0.9718  # velocity_tracking_accuracy_0to1 (mas es mejor)
REF_ORIENTATION_VARIANCE = 0.001118  # varianza de roll/pitch en rad^2 (menos es mejor)
REF_ANGULAR_ACC = 7.5271  # aceleracion angular media de la base, rad/s^2 (menos es mejor)
REF_JOINT_ACC = 0.013605  # aceleracion articular media, rad/paso^2 (menos es mejor)
REF_IMPACT_FORCE = 74.77  # fuerza media de pisada en N (menos es mejor)

#: Banda objetivo de frecuencia de zancada por pata, en Hz (ni arrastrar las patas ni vibrar).
STRIDE_BAND_HZ = (1.5, 3.5)

#: Banda objetivo de duty factor (fraccion del tiempo con la pata apoyada). ~0.5 = trote.
DUTY_BAND = (0.35, 0.65)

#: Gate de locomocion. El score se multiplica por este factor, que vale 1.0 mientras el robot
#: camine y decae linealmente a 0.0 en el caso degenerado (patas siempre apoyadas / sin pasos).
DUTY_GATE_FULL = 0.70  # duty factor hasta el cual no hay penalizacion
DUTY_GATE_ZERO = 0.90  # duty factor a partir del cual el score se anula
STRIDE_GATE_FULL_HZ = 1.0  # frecuencia de zancada a partir de la cual no hay penalizacion
STRIDE_GATE_ZERO_HZ = 0.2  # frecuencia por debajo de la cual el score se anula

#: Referencia medida (smoke test de 5 iteraciones): seguimiento de velocidad de una politica que
#: se queda quieta. Cualquier politica util tiene que superarlo; sirve para leer si un experimento
#: realmente camina mejor. Muy por debajo de la baseline actual (0.972), asi que en la practica el
#: gate de locomocion no deberia activarse salvo que algo salga mal.
STANDSTILL_VELOCITY_TRACKING = 0.257

#: Penalizacion lineal por caidas: score -= FALL_PENALTY * fall_rate_per_episode.
FALL_PENALTY = 2.0

#: Restricciones duras: si se violan, el experimento se marca invalido. Se registra igual,
#: pero no compite por ser el mejor.
MAX_FALL_RATE = 0.15  # caidas por episodio
MAX_IMPACT_MEAN = 250.0  # N
MAX_DUTY_FACTOR = 0.85  # por encima de esto el robot no esta caminando
MIN_STRIDE_HZ = 0.8  # por debajo de esto casi no da pasos


def _lower_is_better(value: float, ref: float) -> float:
    """ref / (ref + x): vale 0.5 en la referencia, 1.0 en el optimo (x=0), 0.0 cuando x -> inf."""
    if not math.isfinite(value) or value < 0:
        return 0.0
    return float(ref / (ref + value))


def _higher_is_better(value: float, ref: float) -> float:
    """x / (x + ref): vale 0.5 en la referencia, tiende a 1.0 cuando x crece."""
    if not math.isfinite(value) or value <= 0:
        return 0.0
    return float(value / (value + ref))


def _invert_smoothness(smoothness: float) -> float:
    """Recupera la cantidad fisica x a partir de una metrica del tipo 1/(1+x) de eval.py."""
    if smoothness <= 0:
        return math.inf
    return 1.0 / smoothness - 1.0


def _ramp(value: float, zero_at: float, full_at: float) -> float:
    """Rampa lineal en [0, 1]: 0.0 en zero_at, 1.0 en full_at (funciona en ambas direcciones)."""
    if zero_at == full_at:
        return 1.0
    fraction = (value - zero_at) / (full_at - zero_at)
    return float(min(1.0, max(0.0, fraction)))


def _locomotion_gate(raw: dict) -> float:
    """Factor en [0, 1] que anula el score de politicas que en realidad no caminan."""
    gate = 1.0
    if "duty_factor" in raw:
        gate *= _ramp(raw["duty_factor"], DUTY_GATE_ZERO, DUTY_GATE_FULL)
    if "stride_hz" in raw:
        gate *= _ramp(raw["stride_hz"], STRIDE_GATE_ZERO_HZ, STRIDE_GATE_FULL_HZ)
    return gate


def _band_score(value: float, low: float, high: float) -> float:
    """1.0 dentro de la banda [low, high], decayendo suavemente fuera de ella."""
    if low <= value <= high:
        return 1.0
    width = high - low
    distance = low - value if value < low else value - high
    return float(1.0 / (1.0 + distance / width))


def _terms_from_results(results: dict) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Normaliza las metricas crudas de eval.py a terminos en [0, 1].

    Devuelve (terminos, cantidades fisicas subyacentes, metricas ausentes).
    """
    terms: dict[str, float] = {}
    raw: dict[str, float] = {}
    missing: list[str] = []

    if "velocity_tracking_accuracy_0to1" in results:
        value = float(results["velocity_tracking_accuracy_0to1"])
        raw["velocity_tracking"] = value
        terms["velocity_tracking"] = _higher_is_better(value, REF_VELOCITY_TRACKING)
    else:
        missing.append("velocity_tracking_accuracy_0to1")

    # eval.py: stability = 1/(1 + var*10)  ->  var = (1/stability - 1)/10
    if "orientation_stability_0to1" in results:
        variance = _invert_smoothness(float(results["orientation_stability_0to1"])) / 10.0
        raw["orientation_variance_rad2"] = variance
        terms["orientation_stability"] = _lower_is_better(variance, REF_ORIENTATION_VARIANCE)
    else:
        missing.append("orientation_stability_0to1")

    # eval.py: smoothness = 1/(1 + |aceleracion angular|)
    if "orientation_smoothness_0to1" in results:
        angular_acc = _invert_smoothness(float(results["orientation_smoothness_0to1"]))
        raw["angular_acc_rad_s2"] = angular_acc
        terms["orientation_smoothness"] = _lower_is_better(angular_acc, REF_ANGULAR_ACC)
    else:
        missing.append("orientation_smoothness_0to1")

    # eval.py: smoothness = 1/(1 + |aceleracion articular|)
    if "movement_smoothness" in results:
        joint_acc = _invert_smoothness(float(results["movement_smoothness"]))
        raw["joint_acc"] = joint_acc
        terms["movement_smoothness"] = _lower_is_better(joint_acc, REF_JOINT_ACC)
    else:
        missing.append("movement_smoothness")

    if "impact_force_mean" in results:
        impact = float(results["impact_force_mean"])
        raw["impact_force_n"] = impact
        terms["impact"] = _lower_is_better(impact, REF_IMPACT_FORCE)
    else:
        missing.append("impact_force_mean")

    # Marcha: frecuencia de zancada y duty factor dentro de bandas razonables.
    # Ojo: los results.json anteriores a la version actual de eval.py traen 'movement_frequency_hz',
    # que NO es equivalente (baseline dice 4.8 y baseline_mass_random 0.05 para marchas parecidas).
    # No se usa como alias: esos runs quedan marcados como no comparables, que es lo honesto.
    stride = results.get("stride_frequency_hz_mean")
    duty = results.get("duty_factor_mean")
    gait_parts = []
    if stride is not None:
        raw["stride_hz"] = float(stride)
        gait_parts.append(_band_score(float(stride), *STRIDE_BAND_HZ))
    else:
        missing.append("stride_frequency_hz_mean")
    if duty is not None:
        raw["duty_factor"] = float(duty)
        gait_parts.append(_band_score(float(duty), *DUTY_BAND))
    if gait_parts:
        terms["gait"] = float(sum(gait_parts) / len(gait_parts))

    return terms, raw, missing


def score_results(results: dict) -> dict:
    """Calcula el score compuesto y su desglose a partir de un dict de results.json."""
    terms, raw, missing = _terms_from_results(results)

    # Los pesos se renormalizan sobre los terminos disponibles para que el numero siga siendo
    # legible, pero un run al que le falten metricas se marca invalido: no es comparable.
    total_weight = sum(WEIGHTS[name] for name in terms)
    contributions = (
        {name: WEIGHTS[name] / total_weight * value for name, value in terms.items()} if total_weight > 0 else {}
    )
    base_score = sum(contributions.values())

    # el gate multiplica ANTES de la penalizacion por caidas: una politica que no camina
    # no puede compensar con estabilidad ni con pisadas suaves
    gate = _locomotion_gate(raw)
    fall_rate = float(results.get("fall_rate_per_episode", 0.0))
    fall_penalty = FALL_PENALTY * fall_rate
    score = base_score * gate - fall_penalty

    violations = []
    if missing:
        violations.append(f"faltan metricas en results.json: {', '.join(missing)}")
    if fall_rate > MAX_FALL_RATE:
        violations.append(f"fall_rate_per_episode={fall_rate:.3f} > {MAX_FALL_RATE}")
    impact_mean = float(results.get("impact_force_mean", 0.0))
    if impact_mean > MAX_IMPACT_MEAN:
        violations.append(f"impact_force_mean={impact_mean:.1f} N > {MAX_IMPACT_MEAN} N")
    if "duty_factor" in raw and raw["duty_factor"] > MAX_DUTY_FACTOR:
        violations.append(f"duty_factor={raw['duty_factor']:.3f} > {MAX_DUTY_FACTOR}: no esta caminando")
    if "stride_hz" in raw and raw["stride_hz"] < MIN_STRIDE_HZ:
        violations.append(f"stride_frequency={raw['stride_hz']:.2f} Hz < {MIN_STRIDE_HZ} Hz: casi no da pasos")

    return {
        "score": float(score),
        "base_score": float(base_score),
        "locomotion_gate": float(gate),
        "fall_penalty": float(fall_penalty),
        "valid": len(violations) == 0,
        "terms": terms,
        "raw_quantities": raw,
        "contributions": contributions,
        "missing_metrics": missing,
        "violations": violations,
    }


def format_breakdown(name: str, results: dict, scored: dict) -> str:
    """Arma el desglose legible del score de un experimento."""
    lines = [name, f"  score = {scored['score']:.4f}" + ("" if scored["valid"] else "   [INVALIDO]")]
    for term in WEIGHTS:
        if term in scored["terms"]:
            lines.append(
                f"    {term:<24} norm={scored['terms'][term]:.3f}  aporte={scored['contributions'][term]:+.4f}"
            )
    gate = scored["locomotion_gate"]
    if gate < 1.0:
        lines.append(
            f"    {'gate de locomocion':<24} x{gate:.3f}  (duty={scored['raw_quantities'].get('duty_factor', float('nan')):.3f},"
            f" zancada={scored['raw_quantities'].get('stride_hz', float('nan')):.2f} Hz)"
        )
    lines.append(
        f"    {'fall_rate (penalizacion)':<24} raw={results.get('fall_rate_per_episode', 0.0):.4f}"
        f"  aporte={-scored['fall_penalty']:+.4f}"
    )
    for violation in scored["violations"]:
        lines.append(f"    [violacion] {violation}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calcula el score compuesto de uno o mas results.json")
    parser.add_argument("results", nargs="+", type=Path, help="Rutas a results.json generados por eval.py")
    parser.add_argument("--json", action="store_true", help="Salida en JSON en vez de texto")
    args = parser.parse_args()

    output = {}
    for path in args.results:
        if not path.exists():
            print(f"[ERROR] No existe: {path}", file=sys.stderr)
            continue
        with open(path) as f:
            results = json.load(f)
        # logs/rsl_rl/<experiment>/<run>/eval/results.json -> <run>
        name = path.parent.parent.name
        scored = score_results(results)
        output[name] = scored
        if not args.json:
            print(format_breakdown(name, results, scored))
            print()

    if args.json:
        print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
