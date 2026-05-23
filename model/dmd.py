from pipeline import SelfForcingTrainingPipeline
import torch.nn.functional as F
from typing import Optional, Tuple
import torch

from model.base import SelfForcingModel
from utils.wan_wrapper import WanDiffusionWrapper
from wan.modules.model import MiraiForesightDiTBlock


class DMD(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        
        Supports Foresight-Forcing: EMA-guided temporal feature distillation
        where current block features are projected to predict future block features.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: SelfForcingTrainingPipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None
        
        # Foresight-Forcing parameters
        self.foresight_weight = getattr(args, "foresight_weight", 0.0)
        self.foresight_layer = getattr(args, "foresight_layer", 16)
        self.foresight_delta = getattr(args, "foresight_delta", 1)
        self.foresight_projector_type = getattr(args, "foresight_projector_type", "simple")
        self.foresight_loss_type = getattr(args, "foresight_loss_type", "cosine")
        self.projection_layers_num = getattr(args, "projection_layers_num", 1)
        self.foresight_include_current = getattr(args, "foresight_include_current", False)
        self.foresight_virtual_blocks = getattr(args, "foresight_virtual_blocks", False)
        self.block_wise_using_frame_wise_foresight = getattr(args, "block_wise_using_frame_wise_foresight", False)
        self.foresight_skip = getattr(args, "foresight_skip", 0)
        self.foresight_spatial_offset = getattr(args, "foresight_spatial_offset", False)
        self.foresight_spatial_offset_max = getattr(args, "foresight_spatial_offset_max", 3)
        self.foresight_delta_mean_pool = getattr(args, "foresight_delta_mean_pool", False)
        self.foresight_online_mean_pool = getattr(args, "foresight_online_mean_pool", False)
        self.enable_foresight = self.foresight_weight > 0.0
        
        print(f"Foresight-Forcing Settings: loss_type={self.foresight_loss_type}, "
              f"projector={self.foresight_projector_type}, weight={self.foresight_weight}")
        
        # Initialize MiraiProjector for Foresight-Forcing
        if self.enable_foresight:
            hidden_dim = self.generator.model.dim  # 1536 for 1.3B, 5120 for 14B
            projector_max_delta = self.foresight_skip + self.foresight_delta

            # Only the DiT projector is shipped in this release.
            assert self.foresight_projector_type == "dit", (
                f"This release only supports foresight_projector_type='dit', "
                f"got {self.foresight_projector_type!r}."
            )
            ffn_ratio = getattr(args, "foresight_ffn_ratio", 4)
            num_heads = getattr(self.generator.model, 'num_heads', hidden_dim // 64)
            self.mirai_projector = MiraiForesightDiTBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=hidden_dim * ffn_ratio,
                num_layers=self.projection_layers_num,
                max_delta=projector_max_delta,
                qk_norm=True,
                use_cross_attn=False,
                timestep_cond=False,
                max_timestep=self.num_train_timestep,
                gate_init_value=0.0,
                delta_cond=False,
            )
            print(f"Using MiraiForesightDiTBlock with num_layers={self.projection_layers_num}, "
                  f"max_delta={projector_max_delta}, num_heads={num_heads}, ffn_ratio={ffn_ratio}")

            self.mirai_projector = self.mirai_projector.to(device=device, dtype=self.dtype)
            self.mirai_projector.requires_grad_(True)

            # Reuse the frozen DMD real-score (Wan-14B) as the bidirectional foresight teacher.
            # Add a dim projection if student hidden dim != teacher hidden dim.
            self.foresight_dim_proj = None
            teacher_dim = self.real_score.model.dim  # 5120 for 14B
            if hidden_dim != teacher_dim:
                self.foresight_dim_proj = torch.nn.Linear(hidden_dim, teacher_dim, bias=False).to(
                    device=device, dtype=self.dtype)
                self.foresight_dim_proj.requires_grad_(True)
                print(f"Foresight dim projection: {hidden_dim} -> {teacher_dim}")
        else:
            self.mirai_projector = None
            self.foresight_dim_proj = None



    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep
        )

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_image - pred_real_image)

        # TODO: Change the normalizer for causal teacher
        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict
            )

        if gradient_mask is not None:
            # Useless if we set always 21 latent frames
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        ema_generator=None,  # EMA model for foresight distillation
        gt_latent: torch.Tensor = None,  # GT latent for teacher
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss + optional Foresight loss.
        """
        # Step 1: Unroll generator to obtain fake videos (and optionally foresight loss)
        result = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            ema_generator=ema_generator,
            gt_latent=gt_latent,
        )
        
        if self.enable_foresight and len(result) == 5:
            pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to, foresight_loss = result
        else:
            pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = result
            foresight_loss = None

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to
        )
        
        # Step 3: Combine DMD loss and Foresight loss
        total_loss = dmd_loss
        if foresight_loss is not None:
            total_loss = total_loss + foresight_loss
            dmd_log_dict["foresight_loss"] = foresight_loss.detach()
        else:
            dmd_log_dict["foresight_loss"] = torch.tensor(0.0, device=dmd_loss.device)

        return total_loss, dmd_log_dict

    def replay_foresight_loss(self):
        """Replay cached foresight pairs through projector for projector-only update."""
        if self.inference_pipeline is not None:
            return self.inference_pipeline.replay_foresight_loss()
        return None

    def clear_foresight_cache(self):
        """Clear cached foresight pairs to free memory."""
        if self.inference_pipeline is not None:
            self.inference_pipeline.cached_foresight_pairs = []

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent
            )

        # Step 2: Compute the fake prediction
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True
        )

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )

        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        # Step 5: Debugging Log
        critic_log_dict = {
            "critic_timestep": critic_timestep.detach()
        }

        return denoising_loss, critic_log_dict

    @torch.no_grad()
    def _prepare_generator_input(self, ode_latent: torch.Tensor, tf=False, causal = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories,
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - ode_latent: a tensor containing the whole ODE sampling trajectories [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
        Output:
            - noisy_input: a tensor containing the selected latent [batch_size, num_frames, num_channels, height, width].
            - timestep: a tensor containing the corresponding timestep [batch_size].
        """
        batch_size, num_denoising_steps, num_frames, num_channels, height, width = ode_latent.shape

        # Step 1: Randomly choose a timestep for each frame
        uniform_timestep = tf or (not causal)
        print(f'uniform_timestep is {uniform_timestep}')
        index = self._get_timestep(
            0,
            len(self.denoising_step_list),
            batch_size,
            num_frames,
            self.num_frame_per_block,
            uniform_timestep=uniform_timestep
        )
        print(f'before self._process_timestep(index), index is {index}')
        if self.args.i2v:
            index[:, 0] = len(self.denoising_step_list) - 1

        noisy_input = torch.gather(
            ode_latent, dim=1,
            index=index.reshape(batch_size, 1, num_frames, 1, 1, 1).expand(
                -1, -1, -1, num_channels, height, width).to(self.device)
        ).squeeze(1)

        timestep = self.denoising_step_list[index].to(self.device)
        print(f'index is {index}, timestep is {timestep}')
    

        return noisy_input, timestep