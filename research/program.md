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
7. **Decidir**: si el score mejoró Y el run es válido → copiar el config resuelto
   (`research/runs/<name>/config.resolved.yaml`, sacándole las claves `max_iterations`, `seed` y
   `experiment_name` que fija el runner) a `research/configs/champion.yaml`. Si no mejoró, el
   campeón queda como está y el experimento igual queda registrado.
8. Commitear los configs y `RESULTS.md` (los checkpoints de `logs/` no se commitean).
9. Volver a 1.

**NUNCA PARES**: una vez arrancado el loop, no preguntes si seguir. El humano puede estar durmiendo
y espera encontrarse los resultados a la mañana. Si te quedás sin ideas, pensá más: releé
`velocity_env_cfg.py` para ver qué términos no tocaste, mirá los términos de reward que están en 0
(`dof_pos_limits`), revisá qué métrica es la que más lejos está de su óptimo, combiná near-misses.
El loop corre hasta que el humano lo interrumpa.

**Criterio de simplicidad**: ante resultados iguales, gana el config más simple. Un delta chico que
requiere retocar seis pesos a la vez no vale la pena; volver a los defaults y obtener lo mismo es una
victoria.

## Estado conocido al arrancar (21/08/2026)

- La baseline entrenada sigue el comando de velocidad **peor que quedarse quieta** (0.211 vs 0.257).
  Ese es el problema central: la señal de tarea está siendo aplastada por las penalizaciones.
- Sospechosos principales, verificados en `params/env.yaml` de un run real:
  `flat_orientation_l2 = -5.0` (enorme frente a `track_lin_vel_xy_exp = 1.5`), `action_rate_l2 = -0.1`,
  `feet_air_time = 0.5` con `threshold = 0.5 s` (premia pasos larguísimos, favorece arrastrar).
- Backlog de ideas para las primeras noches (una por experimento):
  1. Subir la señal de tarea: `track_lin_vel_xy_exp` 1.5 → 3.0.
  2. Aflojar la penalización de orientación: `flat_orientation_l2` -5.0 → -0.5.
  3. Aflojar `action_rate_l2` -0.1 → -0.04.
  4. `feet_air_time.params.threshold` 0.5 → 0.25 (pasos más frecuentes, menos arrastre).
  5. Permitir movimiento vertical: `lin_vel_z_l2` -2.0 → -1.0.
  6. Más exploración: `entropy_coef` 0.005 → 0.01.
  7. `learning_rate` 1e-3 → 5e-4.
  8. Red más grande: `actor_hidden_dims` [128,128,128] → [512,256,128] (la config "rough" ya la usa).
