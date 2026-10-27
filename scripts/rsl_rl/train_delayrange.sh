#!/usr/bin/env bash
# Entrena con delay sorteado por env en 5..8, sobre la receta de exp_414_ang_vel_xy.
# Uso:  bash scripts/rsl_rl/train_delayrange.sh [nombre]
# El nombre por defecto se puede pisar con el primer argumento.
set -euo pipefail
cd "$(dirname "$0")/../.."

NAME="${1:-move_sea_delayrange}"
CFG=research/runs/delay_range/exp_414_ang_vel_xy/config.resolved.yaml

# OJO: ese config trae experiment_name=anymal_d_flat/delay_range, asi que el checkpoint queda en
#      logs/rsl_rl/anymal_d_flat/delay_range/$NAME/   (no en anymal_d_flat/$NAME)
echo "[*] entrenando '$NAME' con delay 5..8 — checkpoints en logs/rsl_rl/anymal_d_flat/delay_range/$NAME"

python scripts/rsl_rl/train_delay.py \
  --task=Dyna1-Flat-v0 \
  --headless \
  --num_envs=4096 \
  --name="$NAME" \
  --action-delay-range 5,8 \
  --experiment_config="$CFG" 2>&1 | tee -a scripts/rsl_rl/logs.txt
