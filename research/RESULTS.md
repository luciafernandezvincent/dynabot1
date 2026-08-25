# Resultados de autoresearch - Dyna1

Generado por `research/run_experiment.py`. No editar a mano: se reescribe en cada experimento.

Protocolo: `Dyna1-Flat-v0`, 1500 iters x 4096 envs, seed 42, eval 1000 envs x 1000 pasos.

| # | experimento | score | despeje mm | vel_track | ori_estab | ori_suav | mov_suav | impacto N | caidas/ep | zancada Hz | duty | notas |
|---|-------------|-------|------------|-----------|-----------|----------|----------|-----------|-----------|------------|------|-------|
| 1 | exp_001_clearance_on | 0.5837 | 40.6 | 0.975 | 0.995 | 0.138 | 0.986 | 50.9 | 0.003 | 2.56 | 0.50 | activar foot_clearance w=0.5 target 5cm |
| 2 | exp_011_symmetry | 0.5768 | 39.8 | 0.975 | 0.995 | 0.134 | 0.986 | 57.9 | 0.003 | 2.62 | 0.47 | air_time_variance w=-1.0: la asimetria empeora al subir el despeje (3.2x -> 3.8x) |
| 3 | exp_019_shoulder_dev | 0.5765 | 41.2 | 0.973 | 0.996 | 0.111 | 0.982 | 60.9 | 0.000 | 3.84 | 0.42 | joint_deviation SOLO en hombros w=-0.5: postura recta sin achicar el paso |
| 4 | exp_008_bigger_net | 0.5764 | 35.8 | 0.976 | 0.996 | 0.130 | 0.986 | 55.4 | 0.004 | 2.52 | 0.47 | 4to intento: autocolision revertida, pipeline correcto, medicion real de la red grande |
| 5 | exp_004_clearance_strong | 0.5728 | 28.6 | 0.974 | 0.996 | 0.115 | 0.984 | 49.1 | 0.002 | 3.12 | 0.47 | clearance w=1.5 target 5cm: mas peso con objetivo alcanzable (exp_009 mostro que target lejano satura) |
| 6 | exp_009_long_swing | 0.5694 | 33.3 | 0.974 | 0.994 | 0.138 | 0.986 | 54.3 | 0.003 | 2.73 | 0.49 | swing mas largo via clearance w=1.0 target 8cm (el umbral de air_time resulto contraproducente) |
| 7 | exp_016_joint_dev_strong | 0.5670 | 11.8 | 0.980 | 0.999 | 0.149 | 0.985 | 56.7 | 0.002 | 3.51 | 0.47 | joint_deviation_l1 w=-0.2: version agresiva, mantener juntas cerca de la default |
| 8 | exp_017_joint_dev_very_strong | 0.5590 | 10.3 | 0.982 | 0.999 | 0.136 | 0.983 | 65.7 | 0.000 | 4.54 | 0.39 | joint_deviation w=-0.5: aun menos flexionado (pedido del usuario) |
| 9 | exp_010_trot_gait | 0.5574 | 38.6 | 0.965 | 0.997 | 0.109 | 0.980 | 82.7 | 0.001 | 4.84 | 0.50 | GaitReward w=5.0, sincronizar diagonales (trote real). Cambio de via tras 3 intentos fallidos de empujar clearance mas fuerte |
| 10 | exp_020_shoulder_dev_strong | 0.5554 | 40.3 | 0.971 | 0.995 | 0.105 | 0.982 | 64.3 | 0.002 | 4.45 | 0.41 | joint_deviation solo hombros w=-1.0 (el doble de exp_019) |
| 11 | exp_021_shoulder_plus_swing | 0.5511 | 35.4 | 0.968 | 0.994 | 0.093 | 0.982 | 69.7 | 0.000 | 3.26 | 0.36 | hombros w=-1.0 + feet_air_time w=1.5: postura recta con swing mas largo |
| 12 | exp_024_free_actions | 0.5435 | 29.9 | 0.968 | 0.995 | 0.081 | 0.978 | 75.0 | 0.000 | 3.34 | 0.34 | action_rate_l2 -0.03: excursiones articulares mas grandes |
| 13 | exp_025_clearance_up | 0.5400 | 30.0 | 0.967 | 0.994 | 0.091 | 0.982 | 71.3 | 0.001 | 3.11 | 0.36 | foot_clearance target 7cm con air_time alto |
| 14 | exp_015_joint_dev_soft | 0.5378 | 28.3 | 0.973 | 0.994 | 0.135 | 0.986 | 54.1 | 0.014 | 2.80 | 0.49 | joint_deviation_l1 w=-0.05: mantener juntas cerca de la default (mejor fuerza en el robot real) |
| 15 | exp_022_airtime_max | 0.5352 | 34.5 | 0.961 | 0.992 | 0.092 | 0.983 | 80.6 | 0.000 | 2.46 | 0.34 | feet_air_time 3.0: mas vuelo, paso mas largo |
| 16 | exp_027_airtime3_plus_vert | 0.5293 | 45.5 | 0.958 | 0.991 | 0.091 | 0.982 | 100.1 | 0.001 | 2.09 | 0.28 | air_time 3.0 + libertad vertical: los dos mejores juntos |
| 17 | exp_026_airtime_5 | 0.5255 | 37.7 | 0.956 | 0.990 | 0.090 | 0.983 | 91.3 | 0.001 | 2.00 | 0.33 | feet_air_time 5.0: mas amplitud |
| 18 | exp_023_vertical_freedom | 0.5220 | 38.1 | 0.963 | 0.990 | 0.084 | 0.982 | 84.9 | 0.002 | 2.74 | 0.32 | lin_vel_z_l2 -0.5: permitir arco vertical |
| 19 | exp_028_airtime5_plus_vert | 0.5199 | 53.9 | 0.953 | 0.988 | 0.109 | 0.983 | 93.4 | 0.006 | 2.11 | 0.28 | air_time 5.0 + libertad vertical: el mas agresivo |
| 20 | baseline_ar | 0.4790 | 22.9 | 0.972 | 0.989 | 0.117 | 0.987 | 74.8 | 0.023 | 3.07 | 0.47 | baseline de referencia, defaults del codigo |
| 21 | exp_002_airtime_threshold_fix | 0.4459 | 23.9 | 0.971 | 0.988 | 0.108 | 0.986 | 65.4 | 0.036 | 3.96 | 0.50 | CAUSA RAIZ: threshold 0.5->0.12s aislado vs baseline (sin clearance) |
| 22 | smoke_revert | 0.0000 ⚠ | 8.7 | 0.257 | 1.000 | 0.633 | 1.000 | 5.3 | 0.000 | 0.89 | 0.98 |  |

⚠ = viola una restriccion dura de `research/score.py` (no compite por el mejor).
