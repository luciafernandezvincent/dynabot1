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
| 12 | exp_053_combo_epochs_delay5 | 0.5467 | 15.0 | 0.963 | 0.998 | 0.108 | 0.983 | 75.9 | 0.001 | 4.73 | 0.39 | [delay5, adaptativo, 20/20] combo exp_046 + num_learning_epochs 5->8 (lever de estabilidad de PPO, mas seguro que entropy_coef que crasheo) |
| 13 | exp_024_free_actions | 0.5435 | 29.9 | 0.968 | 0.995 | 0.081 | 0.978 | 75.0 | 0.000 | 3.34 | 0.34 | action_rate_l2 -0.03: excursiones articulares mas grandes |
| 14 | exp_040_triple_combo_delay5 | 0.5406 | 21.8 | 0.957 | 0.998 | 0.095 | 0.982 | 67.5 | 0.001 | 5.30 | 0.42 | [delay5, adaptativo, 7/20] re-eval para traer el desglose por pata/direccion nuevo |
| 15 | exp_025_clearance_up | 0.5400 | 30.0 | 0.967 | 0.994 | 0.091 | 0.982 | 71.3 | 0.001 | 3.11 | 0.36 | foot_clearance target 7cm con air_time alto |
| 16 | exp_057_vertical_freedom_delay5 | 0.5393 | 17.8 | 0.959 | 0.998 | 0.103 | 0.983 | 83.5 | 0.002 | 4.58 | 0.36 | [delay5, tanda2, 1/20] aflojar lin_vel_z_l2 (-2.0->-0.5) sobre el nuevo campeon (exp_053), pedido del usuario para dar mas libertad vertical a la base |
| 17 | exp_015_joint_dev_soft | 0.5378 | 28.3 | 0.973 | 0.994 | 0.135 | 0.986 | 54.1 | 0.014 | 2.80 | 0.49 | joint_deviation_l1 w=-0.05: mantener juntas cerca de la default (mejor fuerza en el robot real) |
| 18 | exp_046_combo_jointdev015_delay5 | 0.5375 | 17.0 | 0.955 | 0.997 | 0.095 | 0.983 | 70.2 | 0.001 | 4.86 | 0.41 | [delay5, adaptativo, 13/20] combo exp_040 con joint_deviation -0.1->-0.15 (se abandonan levers de PPO, entropy_coef crasheo en exp_045) |
| 19 | exp_047_combo_threshold010_delay5 | 0.5372 | 15.5 | 0.961 | 0.998 | 0.105 | 0.983 | 71.1 | 0.002 | 4.66 | 0.40 | [delay5, adaptativo, 14/20] combo con joint_dev=-0.15 (exp_046) + threshold 0.12->0.10, buscando un optimo conjunto mejor |
| 20 | exp_002_airtime_threshold_fix_delay5 | 0.5368 | 32.2 | 0.935 | 0.995 | 0.091 | 0.982 | 55.1 | 0.004 | 5.56 | 0.46 | [delay5, replica de exp_002_airtime_threshold_fix] CAUSA RAIZ: threshold 0.5->0.12s aislado vs baseline (sin clearance) |
| 21 | exp_022_airtime_max | 0.5352 | 34.5 | 0.961 | 0.992 | 0.092 | 0.983 | 80.6 | 0.000 | 2.46 | 0.34 | feet_air_time 3.0: mas vuelo, paso mas largo |
| 22 | exp_054_combo_epochs10_delay5 | 0.5342 | 16.6 | 0.960 | 0.998 | 0.102 | 0.983 | 73.0 | 0.004 | 4.71 | 0.40 | [delay5, adaptativo, 12/15 desde la instruccion] barrido: num_learning_epochs 8->10 (exp_053 con 8 fue el mejor hasta ahora) |
| 23 | exp_009_long_swing_delay5 | 0.5341 | 37.5 | 0.941 | 0.995 | 0.084 | 0.978 | 70.5 | 0.002 | 4.75 | 0.40 | [delay5, replica de exp_009_long_swing] swing mas largo via clearance w=1.0 target 8cm (el umbral de air_time resulto contraproducente) |
| 24 | exp_042_combo_variance_delay5 | 0.5332 | 20.9 | 0.956 | 0.997 | 0.102 | 0.983 | 69.3 | 0.004 | 4.97 | 0.42 | [delay5, adaptativo, 9/20] mejor combo (exp_040) + air_time_variance=-1.0 (exp_011 historico), motivado por la asimetria de front_right_arm_link vista en el desglose por pata de exp_040 |
| 25 | exp_033_airtime_jointdev_combo_delay5 | 0.5316 | 18.6 | 0.958 | 0.997 | 0.105 | 0.983 | 64.7 | 0.002 | 5.03 | 0.42 | [delay5, adaptativo] combo threshold=0.12+joint_dev=-0.1, re-evaluado con knee_clearance calibrado |
| 26 | exp_063_stance_time_minibatches_delay5 | 0.5310 | 19.1 | 0.950 | 0.997 | 0.093 | 0.982 | 66.8 | 0.003 | 5.26 | 0.43 | [delay5, tanda2, 7/20] config de exp_059 + num_mini_batches 4->8 (gradientes mas estables), buscando conservar knee_height=0.086 con menos caidas |
| 27 | exp_060_stance_time_03_delay5 | 0.5306 | 17.2 | 0.949 | 0.997 | 0.101 | 0.983 | 70.1 | 0.003 | 4.90 | 0.42 | [delay5, tanda2, 4/20] stance_time_reward w=0.5->0.3 (exp_059 mejoro knee_height pero con mas caidas) |
| 28 | exp_059_stance_time_delay5 | 0.5305 | 16.8 | 0.956 | 0.998 | 0.102 | 0.983 | 71.1 | 0.005 | 4.95 | 0.42 | [delay5, tanda2, 3/20] reward nuevo stance_time_reward (target=0.15s), aislado sobre el campeon, para uniformidad y marcha mas lenta (pedido del usuario) |
| 29 | exp_048_combo_minheight015_delay5 | 0.5302 | 16.2 | 0.960 | 0.997 | 0.105 | 0.984 | 68.3 | 0.003 | 4.79 | 0.40 | [delay5, adaptativo, 15/20] combo exp_046 con knee_clearance min_height 0.11->0.15 (mas ambicioso), para ver si empuja mas alla de la meseta actual |
| 30 | exp_027_airtime3_plus_vert | 0.5293 | 45.5 | 0.958 | 0.991 | 0.091 | 0.982 | 100.1 | 0.001 | 2.09 | 0.28 | air_time 3.0 + libertad vertical: los dos mejores juntos |
| 31 | exp_030_airtime_008_delay5 | 0.5285 | 28.8 | 0.940 | 0.995 | 0.088 | 0.981 | 57.7 | 0.003 | 5.62 | 0.47 | [delay5, adaptativo] barrido: feet_air_time.threshold 0.12->0.08 aislado, a partir del mejor resultado (exp_002) |
| 32 | exp_039_knee_clearance_std_delay5 | 0.5284 | 25.4 | 0.948 | 0.995 | 0.094 | 0.982 | 66.4 | 0.000 | 4.69 | 0.41 | [delay5, adaptativo, 6/20] barrido: knee_clearance std 0.05->0.005 (exp_038 con std=0.05 no fue sensible, exp casi siempre ~1.0) |
| 33 | exp_049_combo_shoulder_delay5 | 0.5269 | 32.3 | 0.947 | 0.996 | 0.092 | 0.982 | 64.2 | 0.003 | 5.11 | 0.44 | [delay5, adaptativo, 16/20] combo: threshold=0.12 + knee_clearance + joint_deviation_shoulder=-0.5 (historico exp_019) en vez del generico |
| 34 | exp_064_threshold018_delay5 | 0.5264 | 14.3 | 0.957 | 0.997 | 0.101 | 0.983 | 72.9 | 0.003 | 4.79 | 0.40 | [delay5, tanda2, 8/20] threshold 0.12->0.18 (marcha mas lenta) sobre el campeon con epochs=8, reintento de exp_031 bajo entrenamiento mas estable |
| 35 | exp_026_airtime_5 | 0.5255 | 37.7 | 0.956 | 0.990 | 0.090 | 0.983 | 91.3 | 0.001 | 2.00 | 0.33 | feet_air_time 5.0: mas amplitud |
| 36 | exp_058_variance_epochs8_delay5 | 0.5241 | 17.2 | 0.961 | 0.998 | 0.107 | 0.983 | 71.4 | 0.005 | 4.80 | 0.41 | [delay5, tanda2, 2/20] air_time_variance=-1.0 sobre el campeon nuevo (con epochs=8), pedido del usuario de movimiento mas uniforme; reintento de exp_042 con entrenamiento mas estable |
| 37 | exp_056_combo_epochs_seed7_delay5 | 0.5231 | 15.9 | 0.960 | 0.997 | 0.105 | 0.983 | 74.0 | 0.005 | 4.75 | 0.40 | [delay5, adaptativo, 15/15 desde la instruccion] exp_053 (el mejor) EXACTO con semilla distinta (7), chequeo final de robustez antes de cerrar la tanda |
| 38 | exp_050_combo_doflimits_delay5 | 0.5226 | 15.0 | 0.958 | 0.997 | 0.101 | 0.983 | 77.0 | 0.001 | 5.03 | 0.40 | [delay5, adaptativo, 17/20] combo exp_046 + dof_pos_limits=-1.0 (nunca probado, podria evitar flexion extrema) |
| 39 | exp_023_vertical_freedom | 0.5220 | 38.1 | 0.963 | 0.990 | 0.084 | 0.982 | 84.9 | 0.002 | 2.74 | 0.32 | lin_vel_z_l2 -0.5: permitir arco vertical |
| 40 | exp_035_airtime_biggernet_delay5 | 0.5201 | 32.0 | 0.944 | 0.993 | 0.087 | 0.982 | 53.6 | 0.002 | 5.33 | 0.47 | [delay5, adaptativo, 1/20] combo: threshold=0.12 (exp_002) + red grande (exp_034), con knee_clearance ya calibrado |
| 41 | exp_028_airtime5_plus_vert | 0.5199 | 53.9 | 0.953 | 0.988 | 0.109 | 0.983 | 93.4 | 0.006 | 2.11 | 0.28 | air_time 5.0 + libertad vertical: el mas agresivo |
| 42 | exp_055_combo_epochs6_delay5 | 0.5195 | 16.2 | 0.958 | 0.997 | 0.102 | 0.983 | 72.7 | 0.006 | 4.92 | 0.40 | [delay5, adaptativo, 13/15 desde la instruccion] acotar el optimo de num_learning_epochs entre 5 y 8 (exp_053 con 8 fue el mejor, 10 peor en exp_054) |
| 43 | exp_065_stance_plus_kneeweight_delay5 | 0.5193 | 18.7 | 0.943 | 0.996 | 0.088 | 0.983 | 67.5 | 0.003 | 5.05 | 0.42 | [delay5, tanda2, 9/20] stance_time (exp_059) + knee_clearance weight 0.5->1.0, buscando sinergia entre los dos factores de levantar rodilla |
| 44 | exp_016_joint_dev_strong_delay5 | 0.5189 | 18.3 | 0.951 | 0.996 | 0.102 | 0.983 | 67.0 | 0.008 | 4.61 | 0.42 | [delay5] re-evaluado con knee_clearance calibrado, para medir separacion real yendo para atras |
| 45 | baseline_delay5 | 0.5188 | 27.2 | 0.939 | 0.995 | 0.085 | 0.979 | 69.3 | 0.002 | 4.96 | 0.40 | baseline con action-delay=5, re-evaluado con knee_clearance calibrado (26/8) |
| 46 | exp_043_knee_weight_15_delay5 | 0.5174 | 20.1 | 0.951 | 0.996 | 0.095 | 0.982 | 70.5 | 0.007 | 5.09 | 0.41 | [delay5, adaptativo, 10/20] combo exp_040 con knee_clearance w=0.5->1.5 (nunca se probo mas fuerte con la v2 que funciona; air_time_variance en exp_042 no resolvio la asimetria) |
| 47 | exp_032_joint_dev_010_delay5 | 0.5134 | 31.7 | 0.910 | 0.992 | 0.075 | 0.982 | 58.1 | 0.004 | 4.85 | 0.46 | [delay5, adaptativo] barrido: joint_deviation w=-0.2->-0.1 aislado (exp_029 con -0.2 dio 0.5081) |
| 48 | exp_062_stance_time_target012_delay5 | 0.5127 | 17.7 | 0.959 | 0.996 | 0.088 | 0.982 | 74.4 | 0.005 | 4.84 | 0.40 | [delay5, tanda2, 6/20] stance_time target 0.15->0.12 (menos ambicioso), weight/std de exp_059 |
| 49 | exp_034_bigger_net_delay5 | 0.5115 | 27.6 | 0.947 | 0.992 | 0.099 | 0.982 | 63.9 | 0.003 | 4.49 | 0.43 | [delay5, adaptativo] red grande aislada, re-evaluado con knee_clearance calibrado |
| 50 | exp_052_combo_jointdev015_seed7_delay5 | 0.5106 | 27.4 | 0.917 | 0.994 | 0.085 | 0.983 | 56.4 | 0.010 | 5.29 | 0.47 | [delay5, adaptativo, 19/20] exp_046 EXACTO con semilla distinta (7), para chequear si es mas robusto a la semilla que exp_040 (que en exp_044 vario mucho) |
| 51 | exp_029_joint_dev_isolated_delay5 | 0.5081 | 21.1 | 0.935 | 0.995 | 0.085 | 0.983 | 73.0 | 0.007 | 4.59 | 0.41 | [delay5, adaptativo] joint_deviation w=-0.2 AISLADO contra baseline (sin foot_clearance de exp_001), para separar el efecto de exp_016_delay5 |
| 52 | exp_061_stance_time_std02_delay5 | 0.5079 | 19.9 | 0.937 | 0.996 | 0.084 | 0.983 | 66.2 | 0.008 | 5.08 | 0.43 | [delay5, tanda2, 5/20] stance_time std 0.01->0.02, weight de vuelta a 0.5 (bajar el peso en exp_060 perdio el efecto en knee_height) |
| 53 | exp_031_airtime_020_delay5 | 0.5068 | 33.2 | 0.943 | 0.991 | 0.086 | 0.981 | 59.4 | 0.008 | 5.33 | 0.45 | [delay5, adaptativo] barrido: feet_air_time.threshold 0.12->0.20 aislado, confirmar si 0.12 es pico local (0.08 en exp_030 salio peor) |
| 54 | exp_038_knee_clearance_reward_delay5 | 0.5062 | 28.9 | 0.929 | 0.993 | 0.083 | 0.981 | 70.6 | 0.004 | 4.85 | 0.40 | [delay5, adaptativo, 5/20] knee_clearance rediseñado (exp(-x/std) acotado, patron foot_clearance_reward), w=0.5 |
| 55 | exp_036_knee_clearance_delay5 | 0.5032 | 29.5 | 0.945 | 0.993 | 0.091 | 0.981 | 72.7 | 0.007 | 4.65 | 0.41 | [delay5, adaptativo, 3/20] reward nuevo knee_clearance w=-5.0 min_height=0.11, aislado, ataca directo el gateo para atras |
| 56 | exp_051_combo_orientation_soft_delay5 | 0.5032 | 16.7 | 0.958 | 0.995 | 0.102 | 0.983 | 73.1 | 0.008 | 4.85 | 0.40 | [delay5, adaptativo, 18/20] combo exp_046 + flat_orientation_l2 -5.0->-1.0 (mas libertad de inclinar el cuerpo) |
| 57 | exp_037_knee_clearance_050_delay5 | 0.4897 | 31.3 | 0.936 | 0.992 | 0.083 | 0.981 | 67.0 | 0.011 | 4.89 | 0.41 | [delay5, adaptativo, 4/20] barrido: knee_clearance w=-5.0->-50.0 (exp_036 con -5.0 no tuvo efecto medible) |
| 58 | exp_001_clearance_on_delay5 | 0.4823 | 38.9 | 0.890 | 0.986 | 0.067 | 0.980 | 58.9 | 0.011 | 4.14 | 0.46 | [delay5, replica de exp_001_clearance_on] activar foot_clearance w=0.5 target 5cm |
| 59 | exp_041_knee_backward_gated_delay5 | 0.4820 | 41.3 | 0.903 | 0.990 | 0.066 | 0.981 | 57.2 | 0.012 | 4.92 | 0.48 | [delay5, adaptativo, 8/20] knee_clearance v3: filtro por direccion del comando (solo para atras) en vez de por velocidad del segmento, mismo weight/std que exp_039 |
| 60 | baseline_ar | 0.4790 | 22.9 | 0.972 | 0.989 | 0.117 | 0.987 | 74.8 | 0.023 | 3.07 | 0.47 | baseline de referencia, defaults del codigo |
| 61 | exp_044_combo_seed123_delay5 | 0.4761 | 29.6 | 0.920 | 0.992 | 0.086 | 0.982 | 60.4 | 0.022 | 5.20 | 0.46 | [delay5, adaptativo, 11/20] exp_040 EXACTO con semilla distinta (123), para saber si la asimetria de pata es ruido de entrenamiento (reintento, el primero fallo por config desincronizado) |
| 62 | exp_002_airtime_threshold_fix | 0.4459 | 23.9 | 0.971 | 0.988 | 0.108 | 0.986 | 65.4 | 0.036 | 3.96 | 0.50 | CAUSA RAIZ: threshold 0.5->0.12s aislado vs baseline (sin clearance) |
| 63 | smoke_revert | 0.0000 ⚠ | 8.7 | 0.257 | 1.000 | 0.633 | 1.000 | 5.3 | 0.000 | 0.89 | 0.98 |  |

## Experimentos fallidos

- `exp_011_symmetry_delay5` (2026-08-25T16:37:17): train_failed -        Episode_Termination/base_contact: 0.0010
   Episode_Termination/shoulder_contact: 0.0102
--------------------------------------------------------------------------------
                       
- `exp_045_combo_entropy_delay5` (2026-08-26T17:54:46): train_failed - |=============================================================================================|
| OS: 24.04.4 LTS (Noble Numbat) ubuntu, Version: 24.04.4, Kernel: 7.0.0-30-generic
| Processor: Intel(R

⚠ = viola una restriccion dura de `research/score.py` (no compite por el mejor).
