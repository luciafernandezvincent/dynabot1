# Comandos de entrenamiento y simulación (RSL-RL)

Todos los comandos se ejecutan desde la raíz del repo (`dynabot1/`).

## 1. Entrenar

### 1.1 Entrenamiento con delay de acciones (`train_delay.py`)

```bash
python scripts/rsl_rl/train_delay.py \
  --task=Dyna1-Flat-v0 \
  --headless \
  --num_envs=4096 \
  --name=slower_gait \
  --experiment_config=scripts/rsl_rl/experiment_configs/slower_gait.yaml >> scripts/rsl_rl/logs.txt
```

- `--task`: nombre de la tarea (ej. `Dyna1-Flat-v0`).
- `--headless`: corre sin visualización (recomendado para entrenar rápido).
- `--num_envs`: cantidad de entornos paralelos.
- `--name`: nombre de la carpeta de la corrida dentro de `logs/rsl_rl/anymal_d_flat/`. Si no se pasa, se usa timestamp (ej. `2026-08-20_16-18-55`).
- `--experiment_config`: archivo `.yaml` con overrides de `env`/`agent` (rewards, hiperparámetros, red, etc.). Ver ejemplos en [scripts/rsl_rl/experiment_configs/](scripts/rsl_rl/experiment_configs/) (`baseline.yaml`, `slower_gait.yaml`, `smooth_gait.yaml`, `example.yaml` documenta el formato).
- `--action-delay`: cantidad de pasos de delay en las acciones (default `1` = sin delay).
- `>> scripts/rsl_rl/logs.txt`: agrega el output al log de texto (histórico de entrenamientos).

Otros flags disponibles: `--seed`, `--max_iterations`, `--resume`, `--load_run`, `--checkpoint`, `--run_name`, `--video`.

### 1.2 Entrenamiento estándar sin delay (`train.py`)

`train.py` **no** acepta `--name`, `--experiment_config` ni `--action-delay`: esos tres flags son
exclusivos de `train_delay.py`. Pasárselos hace fallar el comando (Hydra intenta parsearlos como
overrides y no puede). Los flags disponibles son `--task`, `--num_envs`, `--seed`, `--max_iterations`,
`--agent`, `--distributed`, `--video`, `--video_length`, `--video_interval`, `--export_io_descriptors`,
más los de RSL-RL (`--resume`, `--load_run`, `--checkpoint`, `--run_name`, `--experiment_name`, `--logger`).

```bash
python scripts/rsl_rl/train.py \
  --task=Dyna1-Flat-v0 \
  --headless \
  --num_envs=4096
```

La carpeta de la corrida sale de un timestamp. Si necesitás nombre propio o un YAML de overrides,
usá `train_delay.py` sin `--action-delay` (por defecto es `1`, o sea sin delay).

Las corridas quedan guardadas en `logs/rsl_rl/anymal_d_flat/<nombre_o_timestamp>/`, con los checkpoints (`model_*.pt`), configs (`params/`), TensorBoard (`events.out.tfevents...`) y política exportada (`exported/`).

## 2. Simular / visualizar (play)

### 2.1 Play normal (`play.py`)

```bash
python scripts/rsl_rl/play.py --task=Dyna1-Flat-v0 --num_envs=32 --load_run {nombre_carpeta}
```

- `--load_run`: nombre de la carpeta de la corrida a cargar, dentro de `logs/rsl_rl/anymal_d_flat/` (ej. `slower_gait`, `baseline_mass_random`, o un timestamp como `2026-08-11_15-21-54`).
- `--num_envs`: cantidad de entornos a simular (usar un número chico para visualizar, ej. `32`).
- Otros flags: `--checkpoint` (elegir un `model_N.pt` específico en vez del último), `--seed`, `--real-time`, `--video`.

### 2.2 Play con delay de acciones (`play_delay.py`)

```bash
python scripts/rsl_rl/play_delay.py --task=Dyna1-Flat-v0 --num_envs=32 --action-delay 5 --load_run {nombre_carpeta}
```

- Igual que `play.py`, agregando `--action-delay` para simular con el mismo delay (o distinto) al usado en entrenamiento.

## 3. Evaluar (`eval.py`)

```bash
python scripts/rsl_rl/eval.py --task=Dyna1-Flat-v0 --num_envs=32 --action-delay 5 --load_run {nombre_carpeta} --num_steps 1000
```

- `--num_steps`: cantidad de pasos de simulación para la evaluación (default `1000`).
- Genera resultados en `logs/rsl_rl/anymal_d_flat/<nombre_carpeta>/eval/results.json`.

## Referencia rápida de carpetas

- Configs de experimento: [scripts/rsl_rl/experiment_configs/](scripts/rsl_rl/experiment_configs/)
- Log de texto acumulado de entrenamientos: [scripts/rsl_rl/logs.txt](scripts/rsl_rl/logs.txt)
- Corridas guardadas: `logs/rsl_rl/anymal_d_flat/<nombre_o_timestamp>/`
