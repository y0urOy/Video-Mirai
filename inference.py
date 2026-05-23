"""Short-video T2V inference for the distilled Video-Mirai generator.

Loads a config + foresight checkpoint, runs CausalInferencePipeline over the
prompts in --data_path, and writes mp4 files to --output_folder. Supports
single-GPU and torchrun multi-GPU sharded inference.
"""

import argparse
import os
import json

import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import write_video
from tqdm import tqdm

from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset
from utils.misc import set_seed
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller


parser = argparse.ArgumentParser()
parser.add_argument("--config_path",     type=str, required=True, help="Path to the YAML config file")
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained checkpoint (.pt)")
parser.add_argument("--data_path",       type=str, required=True, help="Path to a prompt list (one prompt per line)")
parser.add_argument("--output_folder",   type=str, required=True, help="Directory to write generated mp4s")
parser.add_argument("--num_output_frames", type=int, default=21, help="Number of latent frames to roll out")
parser.add_argument("--use_ema",  action="store_true", help="Load EMA weights from the checkpoint")
parser.add_argument("--seed",     type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per prompt")
parser.add_argument("--extended_prompt_path", type=str, default=None,
                    help="Optional: longer-form prompts used in lieu of the short list at sampling time")
args = parser.parse_args()

# Distributed init (torchrun) — optional
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1

set_seed(args.seed)

# Match gpu reference to this rank's device
import demo_utils.memory as _mem
_mem.gpu = device
gpu = device

print(f"Free VRAM {get_cuda_free_memory_gb(gpu)} GB")
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

pipeline = CausalInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    key = "generator_ema" if args.use_ema else "generator"
    if isinstance(state_dict, dict) and key in state_dict:
        gen_sd = state_dict[key]
    elif isinstance(state_dict, dict) and "generator" in state_dict:
        if args.use_ema:
            print(f"[warn] '{key}' not in checkpoint; falling back to 'generator'.")
        gen_sd = state_dict["generator"]
    else:
        # Flat state_dict (e.g. the released causal_ode.pt warm-start)
        if args.use_ema:
            print(f"[warn] '{key}' not found; checkpoint looks flat — loading top level as state_dict.")
        gen_sd = state_dict

    try:
        pipeline.generator.load_state_dict(gen_sd)
    except RuntimeError:
        fixed = {}
        for k, v in gen_sd.items():
            if k.startswith("model._fsdp_wrapped_module."):
                k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
            fixed[k] = v
        pipeline.generator.load_state_dict(fixed, strict=False)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)
if dist.is_initialized():
    dist.barrier()


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    if isinstance(batch_data, dict):
        batch = batch_data
    else:
        batch = batch_data[0]

    prompt = batch["prompts"][0]
    extended_prompt = batch["extended_prompts"][0] if "extended_prompts" in batch else None

    for seed_idx in range(args.num_samples):
        set_seed(args.seed + seed_idx)

        if args.num_samples > 1:
            output_path = os.path.join(args.output_folder, f"{prompt[:300]}-{seed_idx}.mp4")
        else:
            output_path = os.path.join(args.output_folder, f"{prompt[:300]}.mp4")
        if os.path.exists(output_path):
            print("Video already generated — skip.")
            continue

        prompts = [extended_prompt] if extended_prompt is not None else [prompt]
        sampled_noise = torch.randn(
            [1, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16,
        )

        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=None,
        )
        current_video = rearrange(video, "b t c h w -> b t h w c").cpu()
        video = 255.0 * current_video

        pipeline.vae.model.clear_cache()
        write_video(output_path, video[0], fps=16)
