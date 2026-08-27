# autoresearch — Dyna1

Investigación autónoma sobre la marcha de Dyna1 en Isaac Lab. Este archivo lo edita el HUMANO
(define cómo trabaja el agente); el agente lo lee al empezar y lo sigue al pie de la letra.

Adaptación de https://github.com/karpathy/autoresearch (clonado en `autoresearch/` como referencia)
al caso de locomoción: en vez de editar `train.py` y minimizar `val_bpb`, acá se escribe un YAML de
configuración por experimento y se maximiza el score de `research/score.py`.

## Setup

Antes de arrancar el loop:

1. **Entorno**: todo corre con el Python de conda `env_isaaclab`:
   `/home/linar/miniconda3/envs/env_isaaclab/bin/python`. No instalar paquetes nuevos.
2. **Rama**: trabajar en una rama dedicada, `autoresearch-<tag>` (ej. `autoresearch-ago21`).
   OJO: ya existe una rama llamada `autoresearch`, así que git **rechaza** cualquier rama
   `autoresearch/<tag>` (no puede haber una ref que sea a la vez archivo y directorio). Usar guion.
3. **Leer para tener contexto** (el resto del repo NO hace falta):
   - `research/program.md` (este archivo)
   - `research/score.py` — la métrica objetivo. Solo lectura.
   - `research/configs/baseline.yaml` — de qué valores se parte.
   - `research/RESULTS.md` — qué se probó hasta ahora y qué funcionó.
   - `source/dynabot1/dynabot1/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` — qué
     significa cada término de reward.
4. **Primer experimento SIEMPRE la baseline**, para tener la referencia con el protocolo actual:
   `python research/run_experiment.py --name baseline_ar --notes "baseline de referencia"`
   (los `results.json` viejos de `logs/` NO son comparables: los generó una versión anterior de
   `eval.py` con otras métricas).
   **Ya está hecha** (21/08/2026): si `research/RESULTS.md` ya tiene la fila `baseline_ar`, saltear
   este paso y arrancar el loop directamente por `exp_001`.
5. Copiar el config ganador a `research/configs/champion.yaml` — es la base sobre la que se apilan
   los experimentos siguientes.

## Experimentación

Cada experimento es: entrenar con presupuesto fijo → evaluar con protocolo fijo → scorear → registrar.
Un solo comando hace todo:

```bash
NAME=exp_007_mi_idea
mkdir -p research/runs/$NAME   # sin esto el redirect falla: la carpeta todavía no existe
/home/linar/miniconda3/envs/env_isaaclab/bin/python research/run_experiment.py \
  --base research/configs/champion.yaml \
  --config research/configs/$NAME.yaml \
  --name $NAME \
  --notes "hipótesis en una línea" \
  > research/runs/$NAME/console.log 2>&1
```

Tarda ~40 min. Es normal que el comando parezca colgado: está entrenando.

**Presupuesto fijo** (no negociable, es lo que hace comparables los experimentos): 1500 iteraciones
× 4096 envs, seed 42, evaluación con 1000 envs × 1000 pasos. Medido en la RTX 5070 Ti de esta
máquina: **1.42 s/iteración → ~36 min de entrenamiento + ~3 min de evaluación ≈ 40 min por
experimento**. O sea ~1.5 por hora, **~12-15 en una noche de 8-10 h**.

Si hicieran falta más experimentos por noche, la única perilla aceptable es bajar el presupuesto
(p.ej. 750 iteraciones ≈ 20 min → ~25 experimentos), pero hay que decidirlo **antes** de arrancar y
volver a correr la baseline: mezclar presupuestos invalida las comparaciones.

**Memoria**: con 4096 envs el uso de VRAM es ~15.1 GB de 16.3 GB. Queda poco margen, así que un
config que agrande mucho las redes puede terminar en OOM. Si pasa, se registra como fallo y se sigue.

**Qué PODÉS hacer:**
- Escribir un YAML nuevo en `research/configs/exp_NNN_<slug>.yaml` con **solo el delta** respecto de
  la base (no copiar el config entero). Espacio permitido:
  - `env.rewards.<termino>.weight` — pesos de recompensa
  - `env.rewards.<termino>.params.<param>` — p.ej. `feet_air_time.params.threshold`
  - `env.episode_length_s`
  - `agent.algorithm.*` — learning_rate, entropy_coef, gamma, lam, clip_param, desired_kl, epochs…
  - `agent.policy.*` — dimensiones de las redes, activación, init_noise_std
  - `agent.num_steps_per_env`
- Poner un término en `null` para desactivarlo (p.ej. `undesired_contacts: null`).

**Qué NO podés hacer (regla dura del usuario):**
- **Tocar la física del simulador**: actuadores (stiffness, damping, armature, effort/velocity limit),
  masas, inercias, fricción, restitución, delays del `DelayedPDActuator`, `sim.dt`, decimation,
  terreno, sensores o eventos de randomización. Nada de `dynabot.py` ni de `env.scene`, `env.sim`,
  `env.events`, `env.actions`, `env.observations`, `env.commands`.
  El runner **rechaza** cualquier YAML que toque eso, antes de lanzar nada.
- Modificar `scripts/rsl_rl/eval.py` ni `research/score.py`: son el juez. Si el juez cambia a mitad
  de la investigación, los scores dejan de ser comparables. **Está verificado por hash**: el runner
  se planta y no corre nada si alguno de los dos cambió (`research/.judge_hashes.json`). No intentes
  sortearlo con `--accept-judge-change`: esa bandera es del humano.
- Cambiar el protocolo (`--iterations`, `--num-envs`, `--seed`, envs/pasos de evaluación).
- Editar código Python del entorno o de las tareas. Esta investigación es **solo de configuración**.
- Instalar paquetes.

## La métrica

`research/score.py` combina las métricas de `eval.py` en un número. Cada término está normalizado
contra la corrida baseline, así que se lee directo:

- **~0.5 = igual que la baseline**, **>0.5 = mejor**, **<0.5 = peor**.
- Pesos: seguimiento de velocidad 0.40, estabilidad de orientación 0.20, suavidad de orientación 0.15,
  suavidad de movimiento 0.10, fuerza de impacto 0.10, marcha (frecuencia/duty factor) 0.05.
- Penalización por caídas: `-2.0 × caídas_por_episodio`.
- **Gate de locomoción**: el score se multiplica por un factor que cae a 0 si el perro no camina de
  verdad (duty factor > 0.9 o menos de 0.2 pasos/s). Está medido: una política **sin entrenar**, con
  las 4 patas apoyadas el 99.5% del tiempo, saca mejor estabilidad, suavidad, impacto y hasta mejor
  seguimiento de velocidad (0.257) que la baseline entrenada (0.211). Sin el gate, el óptimo del
  score sería un perro paralizado. Si ves un score alto, **mirá siempre el duty factor y la
  frecuencia de zancada** antes de cantar victoria.
- Un run marcado `[INVALIDO]` (viola una restricción dura) **no puede ser campeón**, por más score
  que tenga.

## Salida de un experimento

`run_experiment.py` imprime al final el desglose y lo compara contra el mejor previo:

```
================================================================================
exp_007_mi_idea
  score = 0.5534
    velocity_tracking        norm=0.586  aporte=+0.2343
    orientation_stability    norm=0.611  aporte=+0.1222
    ...
    fall_rate (penalizacion) raw=0.0020  aporte=-0.0040

Mejor previo: baseline_ar = 0.4852  ->  MEJORA (+0.0682)
================================================================================
```

Todo queda registrado automáticamente en `research/results.jsonl` (append-only, una línea por
experimento con config + métricas + score) y en `research/RESULTS.md` (tabla ordenada por score).

**Higiene de contexto**: los logs de entrenamiento son enormes. Nunca hagas `cat` de
`research/runs/<name>/train.log`. Si algo falla, `tail -n 50`.

## El loop

LOOP FOREVER:

1. Leer `research/RESULTS.md` para ver el estado: campeón vigente, qué se probó, qué falló.
2. Elegir **un solo cambio** (un factor por experimento; combinar recién cuando dos cambios
   individuales ya demostraron ayudar por separado). Escribir la hipótesis en una línea.
3. Escribir `research/configs/exp_NNN_<slug>.yaml` con el delta.
4. Correr el experimento (comando de arriba), redirigiendo la salida a un archivo.
5. Leer el desglose del final: `tail -n 20 research/runs/<name>/console.log`.
6. Si el run falló, `tail -n 50 research/runs/<name>/train.log` para ver el traceback. Si es una
   tontería (typo en el YAML), arreglar y reintentar una vez. Si la idea es inviable, registrar y
   seguir.
7. **Guardar SIEMPRE una explicación detallada**, no solo el score. Comparando contra el run
   anterior/base directo (no solo contra el campeón global), explicar: qué cambió exactamente, qué
   se buscaba con ese cambio, y por qué salió mejor o peor — citando las métricas concretas de
   `eval.py` que lo explican (`fall_rate_per_episode`, `orientation_stability_0to1`,
   `foot_clearance_peak_m`, `stride_frequency_hz_mean`, `duty_factor_mean`, `joint_deviation_mean_rad`,
   etc., no solo el score compuesto). Guardarla con:
   ```bash
   python research/save_explanation.py <nombre_corrida> <archivo_con_el_texto>
   ```
   Esto la mete como clave `_explicacion_detallada` en `logs/rsl_rl/anymal_d_flat/<nombre>/eval/results.json`,
   sin tocar `eval.py`/`score.py` (los jueces no se tocan). Un score solo, sin esta explicación, no
   sirve para decidir el siguiente experimento ni para que alguien entienda despues por qué se tomó
   ese camino.
8. **Decidir**: si el score mejoró Y el run es válido → copiar el config resuelto
   (`research/runs/<name>/config.resolved.yaml`, sacándole las claves `max_iterations`, `seed` y
   `experiment_name` que fija el runner) a `research/configs/champion.yaml`. Si no mejoró, el
   campeón queda como está y el experimento igual queda registrado.
9. Commitear los configs y `RESULTS.md` (los checkpoints de `logs/` no se commitean).
10. Volver a 1.

**NUNCA PARES**: una vez arrancado el loop, no preguntes si seguir. El humano puede estar durmiendo
y espera encontrarse los resultados a la mañana. Si te quedás sin ideas, pensá más: releé
`velocity_env_cfg.py` para ver qué términos no tocaste, mirá los términos de reward que están en 0
(`dof_pos_limits`), revisá qué métrica es la que más lejos está de su óptimo, combiná near-misses.
El loop corre hasta que el humano lo interrumpa.

**Criterio de simplicidad**: ante resultados iguales, gana el config más simple. Un delta chico que
requiere retocar seis pesos a la vez no vale la pena; volver a los defaults y obtener lo mismo es una
victoria.

## Estado conocido al arrancar (21/08/2026)

**Corrección importante**: el primer diagnóstico ("la baseline camina peor que quedarse quieta")
se basó en `logs/rsl_rl/anymal_d_flat/baseline/eval/results.json`, una corrida del **19/08**
entrenada con una versión del código anterior a los pesos de reward actuales — no era comparable.
La baseline real (`baseline_ar`, entrenada el 21/08 con el protocolo del runner y los pesos
actuales del código) **camina bien de entrada**:

| métrica | valor |
|---|---|
| velocity_tracking_accuracy_0to1 | 0.972 |
| stride_frequency_hz_mean | 3.07 (dentro de la banda 1.5–3.5) |
| duty_factor_mean | 0.47 (trote sano, banda 0.35–0.65) |
| fall_rate_per_episode | 0.023 (23/1000 episodios, casi todo `shoulder_contact`) |
| impact_force_mean | 74.8 N |
| movement_smoothness | 0.987 (norm 0.583 — el término con más margen) |
| orientation_smoothness_0to1 | 0.117 (norm 0.725) |

`research/score.py` fue recalibrado contra esta baseline correcta (score de referencia ≈ 0.479,
no 0.5 exacto por la penalización de caídas y el término de marcha ya saturado en 1.0). El score
histórico se recalculó con `--rebuild-table`.

Con esta baseline, **el objetivo ya no es "lograr que camine"** sino refinar una marcha que
funciona: menos caídas de hombro, pisadas más suaves (menos fuerza de impacto), menos aceleración
articular y angular. El gate de locomoción se deja como red de seguridad para cualquier config
futura que induzca el mínimo local de quedarse quieto, aunque no debería activarse con esta
baseline.

**El problema principal, medido el 21/08**: el perro **casi no levanta las patas**. Agregada la
métrica `foot_clearance_*` a `eval.py` (mide la altura del pie en swing relativa a su altura en
apoyo, no la absoluta, porque el origen del `hand_link` no está en la planta):

| | valor |
|---|---|
| despeje medio | **1.9 mm** |
| despeje pico (p95) | **22.9 mm** |
| trasera izquierda, medio | **−2.7 mm** (arrastra: en swing está por debajo de su altura de apoyo) |
| altura de base | 300 mm |

`duty_factor` (0.47) y `stride_frequency` (3.07 Hz) **no detectan esto**, porque solo miran el
sensor de contacto: un pie que despega 2 mm da los mismos números que uno que despega 5 cm. Por eso
la marcha parecía sana en la tabla.

Cambios hechos en consecuencia:
- `mdp/rewards.py`: agregado `foot_clearance_reward` (adaptado de la config de Spot de Isaac Lab),
  cableado en `velocity_env_cfg.py` con **peso 0** para no alterar la baseline, disponible para
  tunear vía YAML (`env.rewards.foot_clearance.weight` / `.params.target_height`).
- `eval.py`: agregadas `foot_clearance_mean_m`, `foot_clearance_peak_m` y sus versiones por pata.
- `score.py`: agregado el término `foot_clearance` con peso **0.15** (referencia: el pico de la
  baseline, 22.9 mm). Sin esto, el loop descartaría los experimentos que arreglan el problema,
  porque cualquier ganancia de despeje con una pérdida mínima en otro término daría "NO MEJORA".

**Causa raíz sospechada (medida el 21/08)**: `feet_air_time` calcula
`(last_air_time - threshold) * first_contact` con peso **positivo** (+0.5) y `threshold = 0.5 s`.
La fase de vuelo real dura ~172 ms (zancada 3.07 Hz, duty 0.47), o sea **2.9x por debajo del
umbral**, así que el paréntesis es siempre negativo: el término **cobra ~-0.05 por episodio en vez
de premiar**. Está penalizando cada pisada, y la respuesta óptima de la política es dar pasos
cortos y rasantes — exactamente el arrastre observado. Verificado en los logs de entrenamiento:

```
Episode_Reward/feet_air_time: -0.0566   (baseline_ar)
Episode_Reward/feet_air_time: -0.0534   (exp_001)
```

Esto es un bug de configuración, no una preferencia: el docstring de la función dice que su
propósito es "ensure that the robot lifts its feet off the ground and takes steps".

Backlog para las primeras noches (una hipótesis por experimento):
  1. `foot_clearance` w=0.5, `target_height` 5 cm — activar el término de altura.
  2. **`feet_air_time.params.threshold` 0.5 → 0.12 s** — invierte el signo del término y recién
     ahí premia el vuelo. Se prueba aislado, sin clearance, para atribuir el efecto.
  3. Umbral corregido + clearance activado — la combinación que debería dar la mejor zancada.
  4. `foot_clearance` w=1.5, 5 cm — variante agresiva.
  5. `foot_clearance` w=1.0, 3 cm — objetivo más modesto por si 5 cm satura.
  6. `foot_clearance` w=1.0 + `feet_air_time` w=1.0 — altura y duración con pesos altos.
  7. `undesired_contacts` -1.0 → -3.0 — contener caídas de hombro, que pueden empeorar al pasar
     más tiempo en apoyo de tres patas.
  8. Red más grande `[512,256,128]`. VRAM: la baseline usa ~4.4 GB de 16.3 GB, hay margen.

`research/run_night.py` ya tiene esta cola cargada y lista para correr sin supervisión.

**Al mirar resultados**: un score alto con `foot_clearance_peak_m` bajo significa que el
experimento mejoró otras cosas sin resolver el problema de fondo. Mirá siempre esa columna, y
mirá `Episode_Reward/<termino>` en el log de entrenamiento para confirmar que cada término aporta
con el signo que se espera.


## Resultados y lecciones (21/08/2026)

| experimento | score | despeje pico | swing | zancada | caídas/ep |
|---|---|---|---|---|---|
| `baseline_ar` | 0.4790 | 22.9 mm | 172 ms | 3.07 Hz | 0.023 |
| `exp_001_clearance_on` | **0.5837** | **40.6 mm** | **197 ms** | 2.56 Hz | **0.003** |
| `exp_002_airtime_threshold_fix` | 0.4459 | 23.9 mm | 127 ms | 3.96 Hz | 0.036 |

**Lección 1 — `foot_clearance` es la palanca que funciona.** Activarlo (w=0.5, 5 cm) subió el
despeje 77%, estiró el swing, y de yapa bajó caídas 87% e impacto 32%, todo sin costar
seguimiento de velocidad (0.972 → 0.975). Es el campeón vigente.

**Lección 2 — bajar el `threshold` de `feet_air_time` es CONTRAPRODUCENTE, al revés de lo que
predije.** El razonamiento era: la tasa del término es `(1-D) - thr*f`, así que con `thr` menor
la política debería preferir pasos largos. Falso. El término **nunca cruza a positivo** (se queda
en -0.0057): estando en zona negativa, un `thr` más chico abarata el castigo por pisada, así que
la política **pisa más seguido**. Resultado: swing 197 → 127 ms, zancada 3.96 Hz, caídas +56%.
El gradiente va al revés del análisis en el límite. **No volver a bajar ese umbral** sin antes
conseguir que el término sea positivo (haría falta un swing > threshold, que a estas frecuencias
no ocurre).

**Lección 3 — pedir más despeje da menos: el reward satura.** `exp_009` subió `target_height`
de 5 a 8 cm y el despeje BAJÓ (40.6 → 33.3 mm), score 0.5837 → 0.5694. El término es
`exp(-error²/std)` con `std=0.05`: con un objetivo de 8 cm y un pie que llega a 3, el error al
cuadrado aplana la exponencial y el gradiente se desvanece. El objetivo tiene que quedar dentro
del rango alcanzable para que discrimine. La vía correcta es subir el PESO con objetivo
alcanzable (`exp_004`: w=1.5, 5 cm), no alejar el objetivo.

**Lección 4 — la asimetría empeora cuanto más levanta las patas.** Ratio entre la pata que más
despega y la que menos: 3.2x (baseline) → 3.8x (campeón) → 3.9x (`exp_009`). Ninguna métrica del
score lo penaliza todavía; `exp_011` (`air_time_variance`) pasa a ser prioritario.

**Cómo leer esto para el resto de la cola**: el swing se estira por la vía de la ALTURA
(`foot_clearance.target_height`), no por la vía del tiempo (`feet_air_time`). Un arco más alto
toma más tiempo por geometría.


**Lección 5 — `w=0.5, target=5cm` (el campeón) es un óptimo local afinado, no un punto de
partida tímido.** Tres intentos distintos de "empujar más fuerte" en cualquier dirección dieron
MENOS despeje, no más:

| cambio probado | despeje pico | score |
|---|---|---|
| campeón (w=0.5, target 5cm) | **40.6 mm** | **0.5837** |
| `exp_002`: threshold de air_time más bajo | 23.9 mm | 0.4459 |
| `exp_009`: target 8cm (en vez de 5) | 33.3 mm | 0.5694 |
| `exp_004`: peso 1.5 (en vez de 0.5) | 28.6 mm | 0.5728 |

En `exp_004`, `orientation_stability` sí mejoró (0.752 vs 0.698 del campeón), a costa de casi
todo el resto — la política parece priorizar el clearance sobre la coordinación general cuando
el gradiente empuja más fuerte, y sale peor en conjunto. No insistir con subir peso/target de
`foot_clearance` sin evidencia nueva; explorar otras vías (gait, contactos, hiperparámetros).


**Lección 6 — `GaitReward` sincroniza pero acelera, no da "elegante".** w=5.0 sobre el campeón:
score 0.5574 (no mejora), zancada 2.56 → **4.84 Hz**, swing 197 → **103 ms**, impacto 50.9 →
**82.7 N** (peor que la baseline original, 74.8 N). Caídas sí bajaron a casi cero (0.001). El
término premia que las diagonales tengan el mismo timing, y la forma más fácil de lograr eso es
una zancada corta y rápida (menos ventana para que se desalinee), no necesariamente un paso lento
y amplio. Sincronía y "marcha suelta" no son lo mismo — no combinar `gait` con peso alto si el
objetivo es soltura, capaz sirve con peso bajo solo para prolijidad de timing.

Van 5 experimentos sobre el campeón (`exp_002`, `_004`, `_009`, `_010`, `_011`) y ninguno lo
superó. El campeón (`foot_clearance` w=0.5, target 5cm, solo) sigue siendo difícil de mejorar por
esta familia de rewards. Vías no probadas aún: `exp_005` (target más bajo, 3cm, con w=1.0),
hiperparámetros de PPO, o combinar clearance bajo + air_time_variance bajo (sin gait).


## Bug encontrado: la red grande nunca se entreno (21/08/2026)

`exp_008` (primer intento) uso `agent.policy.actor_hidden_dims` / `critic_hidden_dims`, que en
esta version de `rsl-rl-lib` (5.0.1, >=4.0.0) esta **deprecado**. El shim de compatibilidad
`handle_deprecated_rsl_rl_cfg` corre ANTES de aplicar el YAML del experimento: usa `policy` para
poblar los campos reales `actor`/`critic`, y despues **vacia `policy`**. El YAML terminaba
seteando un campo ya descartado, sin error ni warning visible.

Confirmado con evidencia dura: el checkpoint `model_1499.pt` de ese primer intento es **byte-
identico** (mismo MD5) al del campeon — entreno con la red chica de siempre, no con `[512,256,128]`
como decia el config. El score idéntico a 4 decimales debería haber sido la primera señal.

**Campo correcto**: `agent.actor.hidden_dims` / `agent.critic.hidden_dims`. Corregido en
`exp_008_bigger_net.yaml` y relanzado. `baseline.yaml` y `champion.yaml` tenían el mismo bloque
`policy:` muerto — no afectaba resultados porque coincidía por casualidad con el default real
([128,128,128]), pero se limpió para que nadie confíe en ese campo a futuro.

**Chequeo para la próxima vez que un experimento no mueva la aguja**: comparar el MD5 del
checkpoint contra el del campeón. Si son iguales, el override no se aplicó — no es que el cambio
no importe.


## Excepcion puntual: autocolision habilitada (21/08/2026)

El usuario observo en el video del campeon que, caminando hacia atras, la pata casi toca el muslo.
Causa: `enabled_self_collisions=False` en `dynabot.py` — el simulador no genera fuerza de contacto
cuando una pata atraviesa el cuerpo, asi que `undesired_contacts` (ya cableado, peso -1.0 sobre
`.*arm_link`) nunca podia penalizarlo por mas que el pliegue existiera.

Es un parametro de fisica/colision, fuera del espacio de configuracion permitido — se le pregunto
al usuario explicitamente y autorizo esta UNICA excepcion, razonando que a diferencia de masa,
actuadores o delay, la autocolision no cambia la dinamica del robot real (el cuerpo real siempre
choca consigo mismo; lo que fallaba era que el simulador no lo veia). Cambio hecho en
`dynabot.py`, documentado inline, en `DYNABOT_1_CFG` y `DYNABOT_1_WITH_DELAY_CFG`.

**Esto es un cambio de regimen, no un experimento mas.** A partir de `exp_013_self_collision`, los
scores no son directamente comparables contra `baseline_ar`/`exp_001`-`012` (todos entrenados con
autocolision apagada). Riesgo conocido: puede aparecer contacto nuevo en gestos que antes pasaban
gratis (base, hombro contra el propio cuerpo), asi que es esperable ver mas caidas de lo usual
mientras la politica aprende a evitar el contacto real que antes no existia.

No volver a tocar ningun otro parametro de fisica sin pedirlo explicitamente: esta excepcion es
puntual, no un precedente general.


## Bug encontrado: eval.py y play.py no aplicaban el YAML del experimento (21/08/2026)

Tras corregir el campo `agent.policy.*` (arriba), `exp_008` volvio a fallar: el checkpoint SI se
entreno con la red grande (MD5 distinto, confirmado), pero `eval.py` crasheaba al cargarlo:

```
RuntimeError: Error(s) in loading state_dict for MLPModel:
	size mismatch for mlp.0.weight: copying a param with shape torch.Size([512, 48]) from
	checkpoint, the shape in current model is torch.Size([128, 48]).
```

**Causa raiz**: `apply_experiment_config` (que aplica el YAML del experimento) solo estaba
cableado en `train_delay.py`. `eval.py` y `play.py` reconstruyen `agent_cfg` desde el default de
la tarea (`AnymalDFlatPPORunnerCfg`, red `[128,128,128]`) y JAMAS leen el YAML — no tienen forma
de saber que el checkpoint que estan por cargar tiene otra arquitectura.

**Por que no lo detectamos antes**: para experimentos de pesos de reward e hiperparametros de PPO
(`exp_001`-`_011`), esto es inofensivo — esos campos no cambian la FORMA del checkpoint, asi que
`eval.py`/`play.py` cargan igual aunque no reciban el YAML. Solo explota con cambios de
arquitectura (`actor`/`critic.hidden_dims`).

**Como se coló un resultado falso en la tabla**: el primer intento roto de `exp_008` (campo
deprecado, entreno la red chica sin darse cuenta) tuvo un eval "exitoso" -porque cargaba una red
chica en un checkpoint chico, todo consistente, aunque por la razon equivocada- y quedo con
`status: ok, score: 0.5837` en `results.jsonl`. El SEGUNDO intento (red grande de verdad) fallo en
eval, pero `write_results_table` no descartaba entradas viejas con el mismo nombre: la tabla
seguia mostrando el primer resultado como si fuera valido, dandole credito a la red grande por un
numero que en realidad era de la red chica.

**Arreglado**:
- `scripts/rsl_rl/experiment_config.py`: modulo nuevo, `apply_overrides`/`apply_experiment_config`
  sacados de `train_delay.py` para poder compartirlos.
- `eval.py` y `play.py`: agregado `--experiment_config`, aplicado en el mismo punto del flujo que
  `train_delay.py` (despues de `handle_deprecated_rsl_rl_cfg`).
- `run_experiment.py`: pasa `--experiment_config={resolved_path}` tambien a `eval_cmd` y al
  `video_cmd` (incluso en `--video-only`, buscando el `config.resolved.yaml` del propio run).
- `load_records()` (en `run_experiment.py` Y en `run_night.py`, tenian copias separadas): ahora
  deja solo el ULTIMO registro por nombre. Un `--name` relanzado reemplaza al intento anterior en
  la tabla en vez de convivir con el.

**Chequeo para la proxima vez que algo no se explica**: si un experimento "no mueve la aguja" y
cambia arquitectura de red, comparar el MD5 del checkpoint contra el campeon antes de confiar en
el numero. Si un experimento se corre dos veces con el mismo nombre, mirar `research/RESULTS.md`
la seccion de fallidos ademas de la tabla principal.


## Autocolision revertida: bloqueaba TODO entrenamiento (21/08/2026)

`exp_008` (red grande) dio un resultado catastrofico: `fall_rate=1.0`, `base_falls` EXACTAMENTE
igual a `episodes_completed` (1,000,000), `shoulder_falls=0`, `stride_frequency=0`, y
`Mean episode length: 1.00` **desde la primera linea del log de entrenamiento**, no como algo que
se degrada con el tiempo. Eso descarta que sea un problema de la red grande: es geometrico, pasa
en el paso 0, para el 100% de los envs, independiente de que politica corra.

Causa: la autocolision (habilitada un rato antes, ver seccion anterior) probablemente hace que la
pose de reposo por defecto (`shoulder_to_arm=-0.79`, `arm_to_hand=1.5`, plegada) se auto-toque
contra `base_link`. Esa pose nunca se habia validado contra autocolision porque siempre estuvo
apagada. Con self-collision=True, ese contacto se registra como `base_contact` (termination) en
el instante del reset, para todos los robots, siempre.

**Revertido a `enabled_self_collisions=False`.** Bloqueaba cualquier entrenamiento, no solo el de
`exp_008` que estaba corriendo en ese momento — de no revertirlo, `exp_013` (que iba a probar
exactamente esto de forma aislada) hubiera dado el mismo resultado catastrofico, y cualquier otra
cosa que se intentara despues tambien.

**exp_008 queda sin dato valido sobre la red grande**: su fracaso midio el efecto de la
autocolision sobre la pose de reposo, no el de la arquitectura. Hay que relanzarlo (4ta vez) con
autocolision apagada para tener una medicion real.

**El pliegue de pata en marcha hacia atras vuelve al espacio de reward shaping**: `dof_pos_limits`
(ya cableado, peso 0) penaliza que las juntas se acerquen a sus limites blandos. No detecta
contacto fisico como hubiera hecho `undesired_contacts` con autocolision, pero desalienta la
flexion extrema sin tocar fisica ni colision. `exp_014_dof_pos_limits.yaml`, peso -1.0.

**Leccion para la proxima vez que se toque algo fisico/geometrico**: antes de lanzar un
entrenamiento largo sobre un cambio de esta naturaleza, correr un diagnostico corto (unas pocas
iteraciones) para ver si sobrevive el primer reset. Hubiera costado 2 minutos en vez de un
entrenamiento completo mal atribuido.
