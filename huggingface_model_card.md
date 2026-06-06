---
license: apache-2.0
tags:
- video-generation
- text-to-video
- autoregressive
- diffusion
- causal
- wan
pipeline_tag: text-to-video
---

# Video-Mirai

**Video-Mirai: Autoregressive Video Diffusion Models Need Foresight**

- 📄 Paper: https://arxiv.org/abs/2606.03971
- 💻 Code: https://github.com/y0urOy/Video-Mirai
- 🌐 Project page: https://y0urOy.github.io/Video-Mirai/

Video-Mirai closes the representation-level planning gap of causal video generators by letting future segments supervise the current causal state, only at training time. At inference the foresight encoder and predictor are discarded; the deployed generator keeps its causal architecture, FLOPs, and KV-cache behavior identical to the baseline.

## Files

| File | What it is |
|---|---|
| `model.pt` | Trained Video-Mirai foresight checkpoint (chunk-wise Causal-Forcing + DMD + foresight loss). Contains the student generator, EMA weights, and trainer state. Pass to `inference.py` via `--checkpoint_path`. |

## Quick start

```bash
# 1. Clone the code repo
git clone https://github.com/y0urOy/Video-Mirai.git
cd Video-Mirai
pip install -r requirements.txt

# 2. Download the Wan2.1 backbone (required by the model)
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
hf download Wan-AI/Wan2.1-T2V-14B  --local-dir wan_models/Wan2.1-T2V-14B

# 3. Download this checkpoint
hf download y0urOy/Video-Mirai model.pt --local-dir checkpoints

# 4. Run inference
CKPT=checkpoints/model.pt bash scripts/inference.sh
```

The trainer writes checkpoints in this layout, so once downloaded the file is loaded directly by `torch.load`:

```python
import torch
state_dict = torch.load("checkpoints/model.pt", map_location="cpu")
# Keys: 'generator' (student weights), 'generator_ema' (EMA weights), and optimizer/scheduler state.
# inference.py picks 'generator_ema' when launched with --use_ema, otherwise 'generator'.
```

See [`inference.py`](https://github.com/y0urOy/Video-Mirai/blob/main/inference.py) and [`scripts/inference.sh`](https://github.com/y0urOy/Video-Mirai/blob/main/scripts/inference.sh) for the full inference entry point.

## Method (short)

The causal generator rolls out causally under the same mask used at inference. A frozen foresight encoder (the same Wan2.1-14B used by the DMD score teacher) reads the completed rollout including future segments and produces future-aware feature targets. A lightweight predictor maps each causal hidden state to its fused target via a cosine loss. After training, the foresight encoder and predictor are discarded. The deployed generator keeps its causal architecture, FLOPs, and KV-cache behavior unchanged.

See the [paper](https://arxiv.org/abs/2606.03971) and [project page](https://y0urOy.github.io/Video-Mirai/) for the full method, ablations, and qualitative results.

## License

Apache 2.0. The Wan2.1 backbone weights this model depends on are released by the Wan-AI team under their own license; please review and comply with their terms when redistributing.

## Citation

```bibtex
@article{yu2026videomirai,
  title={Video-Mirai: Autoregressive Video Diffusion Models Need Foresight},
  author={Yu, Yonghao and Huang, Lang and Li, Runyi and Wang, Zerun and Yamasaki, Toshihiko},
  journal={arXiv preprint arXiv:2606.03971},
  year={2026}
}
```
