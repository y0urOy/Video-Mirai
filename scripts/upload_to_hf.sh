#!/usr/bin/env bash
# Upload the Video-Mirai checkpoint and model card to the Hugging Face Hub.
#
# Prereqs (one-time):
#   pip install -U "huggingface_hub[cli]"
#   hf auth login          # paste a WRITE token from https://huggingface.co/settings/tokens
#
# Usage:
#   REPO_ID=y0urOy/Video-Mirai \
#   CKPT=/work/go67/o67002/Casual-Forcing-Foresight/logs/causal_forcing_dmd_foresight_chunkwise_wan_teacher_mean_pool_projector_every_step/checkpoint_model_000100/model.pt \
#   bash scripts/upload_to_hf.sh
#
# Override CARD to point at a different model card path if desired.

set -euo pipefail

REPO_ID=${REPO_ID:-y0urOy/Video-Mirai}
CKPT=${CKPT:?set CKPT to the .pt checkpoint path}
CARD=${CARD:-huggingface_model_card.md}

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2; exit 1
fi
if [[ ! -f "$CARD" ]]; then
  echo "Model card not found: $CARD" >&2; exit 1
fi

# Install hf CLI if missing.
if ! command -v hf >/dev/null 2>&1; then
  echo "[upload_to_hf] installing huggingface_hub[cli] ..."
  pip install -U "huggingface_hub[cli]"
fi

# Verify auth.
if ! hf auth whoami >/dev/null 2>&1; then
  echo "[upload_to_hf] not logged in. Run: hf auth login" >&2
  echo "  Get a WRITE token from https://huggingface.co/settings/tokens" >&2
  exit 1
fi

echo "[upload_to_hf] target repo: $REPO_ID"
echo "[upload_to_hf] checkpoint  : $CKPT ($(du -h "$CKPT" | cut -f1))"
echo "[upload_to_hf] model card  : $CARD"

# Create the repo (no-op if it already exists).
hf repo create "$REPO_ID" --type model --exist-ok

# Upload the model card as README.md (rendered on the repo's main page).
hf upload "$REPO_ID" "$CARD" README.md --repo-type model

# Upload the checkpoint. Resumable; safe to re-run after interruption.
hf upload "$REPO_ID" "$CKPT" model.pt --repo-type model

echo "[upload_to_hf] done. https://huggingface.co/$REPO_ID"
