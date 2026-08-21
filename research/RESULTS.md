# Resultados de autoresearch - Dyna1

Generado por `research/run_experiment.py`. No editar a mano: se reescribe en cada experimento.

Protocolo: `Dyna1-Flat-v0`, 1500 iters x 4096 envs, seed 42, eval 1000 envs x 1000 pasos.

| # | experimento | score | despeje mm | vel_track | ori_estab | ori_suav | mov_suav | impacto N | caidas/ep | zancada Hz | duty | notas |
|---|-------------|-------|------------|-----------|-----------|----------|----------|-----------|-----------|------------|------|-------|
| 1 | exp_001_clearance_on | 0.5837 | 40.6 | 0.975 | 0.995 | 0.138 | 0.986 | 50.9 | 0.003 | 2.56 | 0.50 | activar foot_clearance w=0.5 target 5cm |
| 2 | exp_011_symmetry | 0.5768 | 39.8 | 0.975 | 0.995 | 0.134 | 0.986 | 57.9 | 0.003 | 2.62 | 0.47 | air_time_variance w=-1.0: la asimetria empeora al subir el despeje (3.2x -> 3.8x) |
| 3 | exp_004_clearance_strong | 0.5728 | 28.6 | 0.974 | 0.996 | 0.115 | 0.984 | 49.1 | 0.002 | 3.12 | 0.47 | clearance w=1.5 target 5cm: mas peso con objetivo alcanzable (exp_009 mostro que target lejano satura) |
| 4 | exp_009_long_swing | 0.5694 | 33.3 | 0.974 | 0.994 | 0.138 | 0.986 | 54.3 | 0.003 | 2.73 | 0.49 | swing mas largo via clearance w=1.0 target 8cm (el umbral de air_time resulto contraproducente) |
| 5 | baseline_ar | 0.4790 | 22.9 | 0.972 | 0.989 | 0.117 | 0.987 | 74.8 | 0.023 | 3.07 | 0.47 | baseline de referencia, defaults del codigo |
| 6 | exp_002_airtime_threshold_fix | 0.4459 | 23.9 | 0.971 | 0.988 | 0.108 | 0.986 | 65.4 | 0.036 | 3.96 | 0.50 | CAUSA RAIZ: threshold 0.5->0.12s aislado vs baseline (sin clearance) |

⚠ = viola una restriccion dura de `research/score.py` (no compite por el mejor).
