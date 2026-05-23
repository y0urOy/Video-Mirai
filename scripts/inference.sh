#!/usr/bin/env bash
# Short-video T2V inference (5 s, 81 frames @ 480x832).
# Loads a trained checkpoint and runs CausalInferencePipeline.

set -euo pipefail

CONFIG=${CONFIG:-configs/video_mirai_dmd_chunkwise.yaml}
# trainer writes to ${logdir}/checkpoint_model_<step>/model.pt — point CKPT at one of those.
CKPT=${CKPT:-logs/foresight_chunkwise/checkpoint_model_000100/model.pt}
PROMPTS=${PROMPTS:-prompts/demos.txt}
OUT=${OUT:-samples/foresight_chunkwise}

torchrun \
  --standalone \
  --nproc_per_node=${GPUS:-1} \
  inference.py \
  --config_path     "$CONFIG" \
  --checkpoint_path "$CKPT" \
  --data_path       "$PROMPTS" \
  --output_folder   "$OUT" \
  --num_output_frames 21 \
  --num_samples 1 \
  --seed 0
