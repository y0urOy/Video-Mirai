"""Entry point for Video-Mirai foresight training (score distillation)."""

import argparse
import os
from omegaconf import OmegaConf
import wandb

from trainer import ScoreDistillationTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to YAML config under configs/")
    parser.add_argument("--logdir", type=str, default="",
                        help="Directory to write checkpoints / samples")
    parser.add_argument("--wandb-save-dir", type=str, default="",
                        help="Directory for wandb local files")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--resume", type=str, nargs="?", const="auto", default=None,
                        help="Resume from checkpoint dir (no value = auto-detect under logdir)")

    # Optional CLI overrides for the main foresight knobs
    parser.add_argument("--foresight_weight", type=float, default=None,
                        help="Override config.foresight_weight")
    parser.add_argument("--foresight_delta", type=int, default=None,
                        help="Override config.foresight_delta")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    if args.foresight_weight is not None:
        config.foresight_weight = args.foresight_weight
    if args.foresight_delta is not None:
        config.foresight_delta = args.foresight_delta

    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    config.resume_ckpt = args.resume
    config.tf = False  # teacher-forcing path is not used in this release
    config.config_name = os.path.basename(args.config_path).split(".")[0]
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb

    if config.get("trainer", "score_distillation") != "score_distillation":
        raise ValueError(
            f"Only trainer=score_distillation is shipped in this release; "
            f"got {config.trainer!r}."
        )

    trainer = ScoreDistillationTrainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
