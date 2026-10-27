#!/usr/bin/env bash
# Reproduce un checkpoint de la tanda delay_range (red [512,256,128], delay sorteado por env).
#
# Uso:  bash scripts/rsl_rl/play_exp414.sh [corrida] [rango_delay] [task]
#   corrida      default nomov_delayrange   (cualquier carpeta de logs/.../anymal_d_flat/delay_range/)
#   rango_delay  default 5,8                (mismo formato que --action-delay-range; "5,5" = fijo en 5)
#   task         default Dyna1-Flat-v0      (usar Dyna1-GraphArtRes-v0 para las corridas "colgado")
#
# Ejemplos:
#   bash scripts/rsl_rl/play_exp414.sh
#   bash scripts/rsl_rl/play_exp414.sh nomovcolgado_delayrange 5,8 Dyna1-GraphArtRes-v0
#   bash scripts/rsl_rl/play_exp414.sh exp_414_ang_vel_xy 7,7
set -euo pipefail
cd "$(dirname "$0")/../.."

RUN="${1:-nomov_delayrange}"
RANGE="${2:-5,8}"
TASK="${3:-Dyna1-Flat-v0}"

# Este config hace DOS cosas imprescindibles y por eso se usa el resuelto completo, no el recortado:
#   - agent.actor/critic.hidden_dims = [512,256,128]  -> sin esto, "size mismatch" al cargar
#   - agent.experiment_name = anymal_d_flat/delay_range -> es la carpeta donde viven estos
#     checkpoints; el recortado exp414.yaml no la tiene y busca en anymal_d_flat/ a secas.
CFG=research/runs/delay_range/exp_414_ang_vel_xy/config.resolved.yaml

echo "[*] play: run=$RUN  delay=$RANGE  task=$TASK"

# --action-delay-stratified reparte los envs entre los delays del rango y cada uno conserva el
# suyo, asi se ven los cuatro lags a la vez en la misma corrida en vez de un sorteo que cambia.
python scripts/rsl_rl/play_delay.py \
  --task="$TASK" \
  --num_envs=32 \
  --action-delay-range "$RANGE" \
  --action-delay-stratified \
  --real-time \
  --load_run "$RUN" \
  --experiment_config="$CFG"
