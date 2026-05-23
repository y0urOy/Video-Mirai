import gc
import logging
from utils.dataset import cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
from model import DMD
import torch
import wandb
import time
import os


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=False
        )


        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        # FSDP wrap MiraiProjector if present (for Foresight-Forcing)
        if hasattr(self.model, 'mirai_projector') and self.model.mirai_projector is not None:
            self.model.mirai_projector = fsdp_wrap(
                self.model.mirai_projector,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy="size"
            )
            if self.is_main_process:
                print(f"Foresight-Forcing enabled: MiraiProjector FSDP-wrapped")

        # FSDP wrap dim projection layer if present (for Wan teacher with dim mismatch)
        if hasattr(self.model, 'foresight_dim_proj') and self.model.foresight_dim_proj is not None:
            self.model.foresight_dim_proj = fsdp_wrap(
                self.model.foresight_dim_proj,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy="size"
            )
            if self.is_main_process:
                print(f"Foresight dim projection FSDP-wrapped")

        # Collect generator parameters; separate foresight params if lr_foresight is set
        generator_params = [param for param in self.model.generator.parameters() if param.requires_grad]

        foresight_params = []
        if hasattr(self.model, 'mirai_projector') and self.model.mirai_projector is not None:
            mirai_params = [param for param in self.model.mirai_projector.parameters() if param.requires_grad]
            foresight_params.extend(mirai_params)
            if self.is_main_process:
                print(f"MiraiProjector has {sum(p.numel() for p in mirai_params)} trainable parameters")
        if hasattr(self.model, 'foresight_dim_proj') and self.model.foresight_dim_proj is not None:
            dim_proj_params = [param for param in self.model.foresight_dim_proj.parameters() if param.requires_grad]
            foresight_params.extend(dim_proj_params)
            if self.is_main_process:
                print(f"Foresight dim proj has {sum(p.numel() for p in dim_proj_params)} trainable parameters")

        lr_foresight = getattr(config, 'lr_foresight', None)
        self.projector_every_step = getattr(config, 'projector_every_step', False)
        self.model.cache_foresight_pairs = self.projector_every_step

        if self.projector_every_step and foresight_params:
            # Projector gets its own optimizer, updated every step via replay
            self.generator_optimizer = torch.optim.AdamW(
                generator_params,
                lr=config.lr,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay
            )
            foresight_lr = lr_foresight if lr_foresight is not None else config.lr
            self.projector_optimizer = torch.optim.AdamW(
                foresight_params,
                lr=foresight_lr,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay
            )
            if self.is_main_process:
                print(f"Projector every-step mode: projector_optimizer lr={foresight_lr}, "
                      f"generator_optimizer lr={config.lr}")
        else:
            self.projector_optimizer = None
            if lr_foresight is not None and foresight_params:
                param_groups = [
                    {'params': generator_params, 'lr': config.lr},
                    {'params': foresight_params, 'lr': lr_foresight},
                ]
                if self.is_main_process:
                    print(f"Using separate lr_foresight={lr_foresight} for projector params (generator lr={config.lr})")
            else:
                param_groups = generator_params + foresight_params

            self.generator_optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.lr,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay
            )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Free CPU memory after FSDP wrapping
        gc.collect()

        # Step 3: Initialize the dataloader
        dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=2)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            full_state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in full_state_dict:
                gen_sd = full_state_dict["generator"]
                fixed = {}
                for k, v in gen_sd.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                gen_sd = fixed
            elif "model" in full_state_dict:
                gen_sd = full_state_dict["model"]
            elif "generator_ema" in full_state_dict:
                gen_sd = full_state_dict["generator_ema"]
                fixed = {}
                for k, v in gen_sd.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                gen_sd = fixed
            else:
                gen_sd = full_state_dict
            self.model.generator.load_state_dict(
                gen_sd, strict=True
            )

            # Optional: inherit generator_ema shadow from checkpoint so EMA is
            # available (and usable as foresight teacher) from step 0 instead of
            # being rebuilt from scratch at ema_start_step.
            self._ema_inherited = False
            if getattr(config, "inherit_ema_from_ckpt", False):
                if self.generator_ema is None:
                    if self.is_main_process:
                        print("[WARN] inherit_ema_from_ckpt=true but EMA is disabled (ema_weight<=0); skipping.")
                elif "generator_ema" in full_state_dict:
                    ema_sd = full_state_dict["generator_ema"]
                    fixed = {}
                    for k, v in ema_sd.items():
                        if k.startswith("model._fsdp_wrapped_module."):
                            k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                        fixed[k] = v
                    try:
                        self.generator_ema.load_state_dict(fixed)
                        self._ema_inherited = True
                        if self.is_main_process:
                            print("Inherited generator_ema from checkpoint; EMA will continue updating from step 0.")
                    except Exception as e:
                        if self.is_main_process:
                            print(f"[WARN] inherit_ema_from_ckpt=true but load failed: {e}")
                else:
                    if self.is_main_process:
                        print("[WARN] inherit_ema_from_ckpt=true but no 'generator_ema' key in checkpoint.")

            # Option: load projector weights from ODE checkpoint (config: load_projector_from_ckpt)
            if getattr(config, "load_projector_from_ckpt", False):
                if "mirai_projector" in full_state_dict and \
                        hasattr(self.model, 'mirai_projector') and self.model.mirai_projector is not None:
                    self.model.mirai_projector.load_state_dict(
                        full_state_dict["mirai_projector"], strict=False)
                    if self.is_main_process:
                        print("Loaded mirai_projector weights from checkpoint")
                else:
                    if self.is_main_process:
                        print("Warning: load_projector_from_ckpt=true but mirai_projector not found in checkpoint or model")
                # Support both key names: "foresight_dim_proj" (ODE online) and "teacher_dim_proj" (ODE offline)
                _dim_proj_key = next(
                    (k for k in ("foresight_dim_proj", "teacher_dim_proj") if k in full_state_dict), None)
                if _dim_proj_key is not None and \
                        hasattr(self.model, 'foresight_dim_proj') and self.model.foresight_dim_proj is not None:
                    ckpt_sd = full_state_dict[_dim_proj_key]
                    model_sd = self.model.foresight_dim_proj.state_dict()
                    shape_ok = all(
                        k in model_sd and ckpt_sd[k].shape == model_sd[k].shape
                        for k in ckpt_sd
                    )
                    if shape_ok:
                        self.model.foresight_dim_proj.load_state_dict(ckpt_sd, strict=False)
                        if self.is_main_process:
                            print("Loaded foresight_dim_proj weights from checkpoint")
                    else:
                        if self.is_main_process:
                            ckpt_shapes = {k: tuple(v.shape) for k, v in ckpt_sd.items()}
                            model_shapes = {k: tuple(v.shape) for k, v in model_sd.items()}
                            print(f"Warning: skipping foresight_dim_proj load — shape mismatch "
                                  f"(ckpt={ckpt_shapes}, model={model_shapes}). "
                                  f"foresight_dim_proj will be randomly initialized.")

            del full_state_dict

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference.
        # Skip when EMA was just inherited from checkpoint — we want it live from step 0.
        if self.step < config.ema_start_step and not getattr(self, "_ema_inherited", False):
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        if self.is_main_process:
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
            print(f"Effective batch size per step: {config.batch_size * self.world_size * self.gradient_accumulation_steps}")
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        # Always save the raw (non-EMA) generator under "generator".
        # Collective op; must be called on all ranks.
        raw_generator_state_dict = fsdp_state_dict(self.model.generator)
        state_dict = {
            "generator": raw_generator_state_dict,
        }
        # Additionally save the EMA-smoothed generator under "generator_ema"
        # when EMA is active. inference.py picks this key with --use_ema.
        # When EMA was inherited from ckpt it is meaningful from step 0, so we
        # save it regardless of ema_start_step in that case.
        if self.generator_ema is not None and (
            self.config.ema_start_step < self.step
            or getattr(self, "_ema_inherited", False)
        ):
            # EMA state_dict() is a collective op (allgather inside)
            state_dict["generator_ema"] = self.generator_ema.state_dict()

        # Save MiraiProjector if present (for Foresight-Forcing)
        # Must use fsdp_state_dict() since these modules are FSDP-wrapped
        if hasattr(self.model, 'mirai_projector') and self.model.mirai_projector is not None:
            state_dict["mirai_projector"] = fsdp_state_dict(self.model.mirai_projector)
        if hasattr(self.model, 'foresight_dim_proj') and self.model.foresight_dim_proj is not None:
            state_dict["foresight_dim_proj"] = fsdp_state_dict(self.model.foresight_dim_proj)

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

        if dist.is_initialized():
            dist.barrier()

    def save_critic(self):
        print("Start gathering distributed model states...")
        
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        
        state_dict = critic_state_dict

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))
            
    def fwdbwd_one_step(self, batch, train_generator, clean_latent=None):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            # clean_latent = None #original code here
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
            gt_latent = batch["ode_latent"][:, -1].to(device=self.device, dtype=self.dtype)
        else:
            # clean_latent = None #original code here
            image_latent = None
            # GT latent for foresight teacher (if available)
            if "ode_latent" in batch:
                gt_latent = batch["ode_latent"][:, -1].to(device=self.device, dtype=self.dtype)
            else:
                gt_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            ema_generator = None
            foresight_enabled = getattr(self.config, "foresight_weight", 0.0) > 0.0

            # Foresight start-step gate: disable foresight loss until step >= foresight_start_step.
            foresight_start_step = getattr(self.config, "foresight_start_step", 0)
            if foresight_enabled and self.step < foresight_start_step:
                if self.is_main_process and self.step == 0:
                    print(f"Foresight-Forcing: foresight_start_step={foresight_start_step} — "
                          f"foresight loss disabled until step {foresight_start_step}.")
                foresight_enabled = False
            elif (foresight_enabled and foresight_start_step > 0
                  and self.is_main_process and self.step == foresight_start_step):
                print(f"Foresight-Forcing: reached foresight_start_step={foresight_start_step}, "
                      f"enabling foresight loss now.")

            if foresight_enabled:
                # Reuse the frozen bidirectional Wan-14B (real_score) as the foresight teacher.
                ema_generator = self.model.real_score
                if self.is_main_process and self.step == 0:
                    print("Foresight-Forcing: using bidirectional Wan (real_score) as teacher")

            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
                ema_generator=ema_generator,
                gt_latent=gt_latent,
            )

            # Scale loss for gradient accumulation
            scaled_generator_loss = generator_loss / self.gradient_accumulation_steps
            scaled_generator_loss.backward()

            generator_log_dict.update({"generator_loss": generator_loss})  # log unscaled loss

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        # Scale loss for gradient accumulation
        scaled_critic_loss = critic_loss / self.gradient_accumulation_steps
        scaled_critic_loss.backward()

        critic_log_dict.update({"critic_loss": critic_loss})  # log unscaled loss

        return critic_log_dict


    def train(self):
        start_step = self.step
        max_steps = getattr(self.config, "max_steps", float("inf"))
       
        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                if self.projector_optimizer is not None:
                    # Clear cached foresight pairs from previous generator step
                    self.model.clear_foresight_cache()
                    self.projector_optimizer.zero_grad(set_to_none=True)

                for accum_idx in range(self.gradient_accumulation_steps):
                    batch = next(self.dataloader)
                    generator_log_dict = self.fwdbwd_one_step(batch, True)

                # Grad clipping after all accumulation steps
                generator_grad_norm = self.model.generator.clip_grad_norm_(
                    self.max_grad_norm_generator)
                if hasattr(self.model, 'mirai_projector') and self.model.mirai_projector is not None:
                    self.model.mirai_projector.clip_grad_norm_(self.max_grad_norm_generator)
                if hasattr(self.model, 'foresight_dim_proj') and self.model.foresight_dim_proj is not None:
                    self.model.foresight_dim_proj.clip_grad_norm_(self.max_grad_norm_generator)
                generator_log_dict["generator_grad_norm"] = generator_grad_norm

                self.generator_optimizer.step()
                if self.projector_optimizer is not None:
                    self.projector_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)

            for accum_idx in range(self.gradient_accumulation_steps):
                batch = next(self.dataloader)
                critic_log_dict = self.fwdbwd_one_step(batch, False)

            # Grad clipping after all accumulation steps
            critic_grad_norm = self.model.fake_score.clip_grad_norm_(
                self.max_grad_norm_critic)
            critic_log_dict["critic_grad_norm"] = critic_grad_norm

            self.critic_optimizer.step()

            # Projector replay: update projector on non-generator steps using cached pairs
            replay_foresight_loss_val = None
            if self.projector_optimizer is not None and not TRAIN_GENERATOR:
                # Release critic activations before replay to free GPU memory
                torch.cuda.empty_cache()
                self.projector_optimizer.zero_grad(set_to_none=True)
                # replay_foresight_loss() does per-pair backward internally to avoid OOM
                replay_loss_val = self.model.replay_foresight_loss()
                if replay_loss_val is not None:
                    if hasattr(self.model, 'mirai_projector') and self.model.mirai_projector is not None:
                        self.model.mirai_projector.clip_grad_norm_(self.max_grad_norm_generator)
                    if hasattr(self.model, 'foresight_dim_proj') and self.model.foresight_dim_proj is not None:
                        self.model.foresight_dim_proj.clip_grad_norm_(self.max_grad_norm_generator)
                    self.projector_optimizer.step()
                    replay_foresight_loss_val = replay_loss_val.detach()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )
                    # Log foresight loss if available
                    if "foresight_loss" in generator_log_dict:
                        foresight_val = generator_log_dict["foresight_loss"]
                        if torch.is_tensor(foresight_val):
                            foresight_val = foresight_val.mean().item()
                        wandb_loss_dict["foresight_loss"] = foresight_val

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                # Log replay foresight loss (projector-only update on non-generator steps)
                if replay_foresight_loss_val is not None:
                    wandb_loss_dict["replay_foresight_loss"] = replay_foresight_loss_val.mean().item()

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            # Exit after reaching max_steps
            if self.step >= max_steps:
                if self.is_main_process:
                    logging.info(f"Reached max_steps={max_steps} at step {self.step}, saving and exiting...")
                if not self.config.no_save:
                    torch.cuda.empty_cache()
                    self.save()
                dist.barrier()
                break