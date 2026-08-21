# Resultados de autoresearch - Dyna1

Generado por `research/run_experiment.py`. No editar a mano: se reescribe en cada experimento.

Protocolo: `Dyna1-Flat-v0`, 1500 iters x 4096 envs, seed 42, eval 1000 envs x 1000 pasos.

| # | experimento | score | despeje mm | vel_track | ori_estab | ori_suav | mov_suav | impacto N | caidas/ep | zancada Hz | duty | notas |
|---|-------------|-------|------------|-----------|-----------|----------|----------|-----------|-----------|------------|------|-------|
| 1 | exp_001_clearance_on | 0.5837 | 40.6 | 0.975 | 0.995 | 0.138 | 0.986 | 50.9 | 0.003 | 2.56 | 0.50 | activar foot_clearance w=0.5 target 5cm |
| 2 | baseline_ar | 0.4790 | 22.9 | 0.972 | 0.989 | 0.117 | 0.987 | 74.8 | 0.023 | 3.07 | 0.47 | baseline de referencia, defaults del codigo |

⚠ = viola una restriccion dura de `research/score.py` (no compite por el mejor).
