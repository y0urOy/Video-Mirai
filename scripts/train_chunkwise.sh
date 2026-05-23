#!/usr/bin/env bash
# Foresight training, chunk-wise (3 frames per block).
# Single-node launch — adjust --nproc_per_node to your GPU count.
# Requires a Causal-Forcing / Self-Forcing warmstart checkpoint set in the YAML
# under `generator_ckpt`. See README §"Pretrained warm-start" for details.

set -euo pipefail

CONFIG=configs/video_mirai_dmd_chunkwise.yaml
LOGDIR=${LOGDIR:-logs/foresight_chunkwise}

torchrun \
  --standalone \
  --nproc_per_node=${GPUS:-8} \
  train.py \
  --config_path "$CONFIG" \
  --logdir       "$LOGDIR"
