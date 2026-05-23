from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
import torch.distributed as dist


class SelfForcingTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 # Foresight-Forcing parameters
                 ema_generator: Optional[WanDiffusionWrapper] = None,
                 mirai_projector: Optional[torch.nn.Module] = None,
                 foresight_layer: int = 16,
                 foresight_student_layer: Optional[int] = None,
                 foresight_teacher_layer: Optional[int] = None,
                 foresight_delta: int = 1,
                 foresight_weight: float = 1.0,
                 foresight_projector_type: str = "simple",
                 foresight_loss_type: str = "cosine",
                 foresight_include_current: bool = False,
                 foresight_virtual_blocks: bool = False,
                 block_wise_using_frame_wise_foresight: bool = False,
                 foresight_skip: int = 0,
                 foresight_spatial_offset: bool = False,
                 foresight_spatial_offset_max: int = 3,
                 max_replay_pairs: int = 0,
                 foresight_delta_mean_pool: bool = False,
                 foresight_online_mean_pool: bool = False,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length

        # Foresight-Forcing parameters
        self.ema_generator = ema_generator
        self.mirai_projector = mirai_projector
        self.foresight_layer = foresight_layer
        
        # Support cross-layer alignment (Student shallow → Teacher deep)
        if foresight_student_layer is not None and foresight_teacher_layer is not None:
            self.student_foresight_layer = foresight_student_layer
            self.teacher_foresight_layer = foresight_teacher_layer
        else:
            self.student_foresight_layer = foresight_layer
            self.teacher_foresight_layer = foresight_layer
        
        self.foresight_delta = foresight_delta
        self.foresight_weight = foresight_weight
        self.foresight_projector_type = foresight_projector_type
        self.foresight_loss_type = foresight_loss_type
        self.foresight_include_current = foresight_include_current
        self.foresight_virtual_blocks = foresight_virtual_blocks
        self.block_wise_using_frame_wise_foresight = block_wise_using_frame_wise_foresight
        self.foresight_skip = foresight_skip
        self.foresight_spatial_offset = foresight_spatial_offset
        self.foresight_spatial_offset_max = foresight_spatial_offset_max
        # Spatial token grid dimensions (H=60//patch_size=2, W=104//patch_size=2)
        self.foresight_H_tok = 30
        self.foresight_W_tok = 52
        self.foresight_dim_proj = None  # set externally when using Wan teacher with dim mismatch
        self.enable_foresight = (ema_generator is not None and mirai_projector is not None)

        # Cached foresight pairs for projector replay (used when projector_every_step=True)
        self.cached_foresight_pairs = []
        self.cache_foresight_pairs = False  # set True externally when projector_every_step is enabled
        self.foresight_delta_mean_pool = foresight_delta_mean_pool
        self.foresight_online_mean_pool = foresight_online_mean_pool
        self.max_replay_pairs = max_replay_pairs  # 0 = no limit, >0 = random sample N pairs per replay

        # EMA KV caches (initialized lazily)
        self.ema_kv_cache = None
        self.ema_kv_cache2 = None
        self.ema_crossattn_cache = None

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            # In our training, self.last_step_only is False
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            clean_image_or_video: torch.Tensor = None, # same shape as noise
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None: # Never met
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        # In out training, self.independent_first_frame is False
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21

        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
            
            if True:
                noisy_input = noise[
                    :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

                # Step 3.1: Spatial denoising loop
                # Such a loop corresponds to the truncated denoising algorithm:
                #    T -> \tau_1 -> \tau_2 ->...-> \tau —— enable grad ——> 0
                # For many-step model, we certainly cannot use this method, but for 4-step DMD, 
                # we can inherit it for a fair comaprison. Note that as long as the conditions 
                # are clean GT rather than self-generated frames, we can perform TF. So this 
                # method does not conflict with TF in the frame- dimension.
                for index, current_timestep in enumerate(self.denoising_step_list):
                    # self.same_step_across_blocks is True
                    if self.same_step_across_blocks:
                        exit_flag = (index == exit_flags[0])
                    else:
                        exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)
                    timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64) * current_timestep

                    if not exit_flag:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                            next_timestep = self.denoising_step_list[index + 1]
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                            ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        # for getting real output
                        # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                        if current_start_frame < start_gradient_frame_index: # Always True as long as we train 21 latent frames
                            with torch.no_grad():
                                _, denoised_pred = self.generator(
                                    noisy_image_or_video=noisy_input,
                                    conditional_dict=conditional_dict,
                                    timestep=timestep,
                                    kv_cache=self.kv_cache1,
                                    crossattn_cache=self.crossattn_cache,
                                    current_start=current_start_frame * self.frame_seq_length
                                )
                        else: # enable grad
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                        break
                    
            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks: # Useless, never met
            denoised_timestep_from, denoised_timestep_to = None, None
        # T -> \tau_1 -> \tau_2 ->...-> \tau —— enable grad ——> 0
        # denoised_timestep_from = \tau
        # denoised_timestep_to = next timestep smaller than \tau
        # These are just engineering tricks
        # to align DMD timestep sampling with the actual denoising range used by the generator
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            # corner case when \tau is the smallest non-zero timestep
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step: # False
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _initialize_ema_kv_cache(self, batch_size, dtype, device):
        """
        Initialize KV cache for EMA model (used in Foresight-Forcing).
        Extra space is allocated for virtual future blocks.
        """
        if not self.enable_foresight:
            return

        if self.foresight_virtual_blocks:
            extra_frames = self.foresight_delta * self.num_frame_per_block
            ema_cache_size = self.kv_cache_size + extra_frames * self.frame_seq_length
        else:
            ema_cache_size = self.kv_cache_size

        ema_kv_cache = []
        for _ in range(self.num_transformer_blocks):
            ema_kv_cache.append({
                "k": torch.zeros([batch_size, ema_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, ema_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.ema_kv_cache = ema_kv_cache

    def _initialize_ema_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize cross-attention cache for EMA model.
        """
        if not self.enable_foresight:
            return

        ema_crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            ema_crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.ema_crossattn_cache = ema_crossattn_cache

    def replay_foresight_loss(self):
        """
        Replay cached (student_hidden, teacher_hidden) pairs through projector
        to compute foresight loss without re-running generator forward.
        Used for projector-only updates on non-generator steps.
        Processes one pair at a time to avoid OOM.
        """
        if not self.cached_foresight_pairs or self.mirai_projector is None:
            return None

        device = next(self.mirai_projector.parameters()).device
        total_loss = 0.0

        # Subsample pairs if limit is set (0 = use all)
        import random
        pairs = self.cached_foresight_pairs
        if self.max_replay_pairs > 0 and len(pairs) > self.max_replay_pairs:
            pairs = random.sample(pairs, self.max_replay_pairs)
        num_pairs = len(pairs)

        for pair in pairs:
            student_h = pair['student_hidden'].to(device)
            teacher_h = pair['teacher_hidden'].to(device)
            delta = pair['delta']
            ts = pair['timestep']

            if self.block_wise_using_frame_wise_foresight:
                num_frames = min(pair['student_num_frames'], pair['teacher_num_frames'])
                frame_losses = []
                for fi in range(num_frames):
                    s_start = fi * self.frame_seq_length
                    s_end = (fi + 1) * self.frame_seq_length
                    sf = student_h[:, s_start:s_end, :]
                    tf = teacher_h[:, s_start:s_end, :]

                    if self.foresight_projector_type != "simple":
                        pf = self.mirai_projector(sf, delta=delta, timestep=ts)
                    else:
                        pf = self.mirai_projector(sf)
                    if self.foresight_dim_proj is not None:
                        pf = self.foresight_dim_proj(pf)

                    frame_losses.append(self._foresight_loss(pf.view(-1, pf.shape[-1]), tf.view(-1, tf.shape[-1])))
                pair_loss = torch.stack(frame_losses).mean()
            else:
                if self.foresight_projector_type != "simple":
                    projected = self.mirai_projector(student_h, delta=delta, timestep=ts)
                else:
                    projected = self.mirai_projector(student_h)
                if self.foresight_dim_proj is not None:
                    projected = self.foresight_dim_proj(projected)

                pair_loss = self._foresight_loss(projected.view(-1, projected.shape[-1]), teacher_h.view(-1, teacher_h.shape[-1]))

            # Accumulate loss as python float to avoid graph buildup
            total_loss += pair_loss.item()
            # Backward per pair to release activations immediately
            (pair_loss * self.foresight_weight / num_pairs).backward()
            del student_h, teacher_h, pair_loss
            torch.cuda.empty_cache()

        # Return total for logging (already backward-ed, caller should NOT call .backward())
        return torch.tensor(total_loss * self.foresight_weight / num_pairs, device=device)

    def _foresight_loss(self, proj_flat, teacher_flat):
        """Cosine foresight loss.

        Args:
            proj_flat:    [N, D] projected student features
            teacher_flat: [N, D] teacher features
        Returns:
            scalar loss
        """
        return -F.cosine_similarity(proj_flat, teacher_flat, dim=-1).mean()

    def _apply_spatial_crop_offset(self, proj, teacher, num_frames, dh, dw):
        """
        Align proj token (r, c) with teacher token (r+dh, c+dw) using crop mode.
        Only the overlapping spatial region contributes to the loss.

        Args:
            proj:    [B, num_frames * H_tok * W_tok, dim]  (projected student, requires grad)
            teacher: [B, num_frames * H_tok * W_tok, dim]  (teacher hidden, detached)
            num_frames: number of frames in the token sequence
            dh, dw:  integer spatial offsets (can be negative)
        Returns:
            proj_crop, teacher_crop: flat tensors over the overlapping region
        """
        if dh == 0 and dw == 0:
            return proj, teacher

        B, _, D = proj.shape
        H, W = self.foresight_H_tok, self.foresight_W_tok

        proj_2d = proj.reshape(B, num_frames, H, W, D)
        teacher_2d = teacher.reshape(B, num_frames, H, W, D)

        # student (r, c) aligns with teacher (r+dh, c+dw)
        if dh >= 0:
            s_h = slice(0, H - dh)
            t_h = slice(dh, H)
        else:
            s_h = slice(-dh, H)
            t_h = slice(0, H + dh)

        if dw >= 0:
            s_w = slice(0, W - dw)
            t_w = slice(dw, W)
        else:
            s_w = slice(-dw, W)
            t_w = slice(0, W + dw)

        proj_crop = proj_2d[:, :, s_h, s_w, :].reshape(B, -1, D)
        teacher_crop = teacher_2d[:, :, t_h, t_w, :].reshape(B, -1, D)
        return proj_crop, teacher_crop

    def inference_with_trajectory_and_foresight(
            self,
            noise: torch.Tensor,
            clean_image_or_video: torch.Tensor = None,
            initial_latent: Optional[torch.Tensor] = None,
            gt_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> Tuple[torch.Tensor, ...]:
        """
        Enhanced trajectory generation with Foresight-Forcing.
        Runs student (generator) on current blocks and EMA (teacher) on future blocks,
        aligning student's projected hidden states with teacher's hidden states.

        Returns:
            output, denoised_timestep_from, denoised_timestep_to, foresight_loss
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device, dtype=noise.dtype
        )

        # Step 1: Initialize KV caches for both student and EMA teacher
        self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)

        # Wan teacher (Path A) uses one bidirectional forward pass without KV cache,
        # so skip EMA cache allocation to save ~6 GB GPU memory.
        # Step 2: Cache context feature (initial latent)
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
                # Wan teacher (Path A) doesn't use EMA KV cache, skip this forward pass
                # to avoid a costly 14B model allgather (~28 GB peak) with no benefit.
            current_start_frame += 1

        # Step 3: Temporal denoising loop with foresight
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21

        # Storage for foresight distillation
        student_hidden_states = []
        foresight_losses = []
        foresight_loss = None

        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])

                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device, dtype=torch.int64) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # Final step: get output and optionally hidden state for foresight
                    if self.foresight_virtual_blocks:
                        remaining_future_blocks = len(all_num_frames) - block_index - 1
                        has_any_target = remaining_future_blocks >= 1 or self.foresight_include_current
                    elif self.foresight_include_current:
                        # With include_current, delta=0 (self-alignment) is always valid
                        has_any_target = block_index < len(all_num_frames)
                    else:
                        # All projector types now use multi-delta range, only need at least 1 future block
                        has_any_target = block_index + 1 < len(all_num_frames)
                    should_extract_hidden = (
                        self.enable_foresight and
                        current_start_frame >= start_gradient_frame_index and
                        has_any_target
                    )

                    if current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                    else:
                        if should_extract_hidden:
                            _, denoised_pred, student_hidden = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                                return_hidden_at_layer=self.student_foresight_layer
                            )
                            current_timestep_val = self.denoising_step_list[
                                exit_flags[0] if self.same_step_across_blocks else exit_flags[block_index]]
                            student_hidden_states.append(
                                (block_index, student_hidden, current_timestep_val,
                                 current_start_frame, current_num_frames))
                        else:
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                    break

            # Step 3.2: Record output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: Rerun with context noise to update cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            denoised_pred_noisy = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])

            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred_noisy,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

                # Also update EMA cache with the same output
                # (skip for Wan teacher — it uses one bidirectional forward pass, no KV cache)
            current_start_frame += current_num_frames

        # Step 3.5: EMA generates virtual future blocks beyond the training window
        num_virtual_blocks = 0
        if self.foresight_virtual_blocks and self.enable_foresight and len(student_hidden_states) > 0:
            last_student_block_idx = student_hidden_states[-1][0]
            max_needed_block = last_student_block_idx + self.foresight_delta
            num_virtual_blocks = max(0, max_needed_block - (len(all_num_frames) - 1))

            if num_virtual_blocks > 0:
                virtual_frame_start = current_start_frame

                for v_idx in range(num_virtual_blocks):
                    virtual_num_frames = self.num_frame_per_block
                    virtual_noise = torch.randn(
                        [batch_size, virtual_num_frames, noise.shape[2], noise.shape[3], noise.shape[4]],
                        device=noise.device, dtype=noise.dtype
                    )
                    virtual_timestep_val = self.denoising_step_list[
                        exit_flags[0] if self.same_step_across_blocks else exit_flags[-1]
                    ]
                    virtual_timestep = torch.ones(
                        [batch_size, virtual_num_frames],
                        device=noise.device, dtype=torch.int64
                    ) * virtual_timestep_val

                    with torch.no_grad():
                        _, virtual_denoised = self.ema_generator(
                            noisy_image_or_video=virtual_noise,
                            conditional_dict=conditional_dict,
                            timestep=virtual_timestep,
                            kv_cache=self.ema_kv_cache,
                            crossattn_cache=self.ema_crossattn_cache,
                            current_start=virtual_frame_start * self.frame_seq_length
                        )

                        ctx_ts = torch.ones_like(virtual_timestep) * self.context_noise
                        virtual_denoised_noisy = self.scheduler.add_noise(
                            virtual_denoised.flatten(0, 1),
                            torch.randn_like(virtual_denoised.flatten(0, 1)),
                            ctx_ts * torch.ones(
                                [batch_size * virtual_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, virtual_denoised.shape[:2])

                        self.ema_generator(
                            noisy_image_or_video=virtual_denoised_noisy,
                            conditional_dict=conditional_dict,
                            timestep=ctx_ts,
                            kv_cache=self.ema_kv_cache,
                            crossattn_cache=self.ema_crossattn_cache,
                            current_start=virtual_frame_start * self.frame_seq_length
                        )

                    all_num_frames.append(virtual_num_frames)
                    virtual_frame_start += virtual_num_frames

        # Step 4: Compute foresight loss AFTER all blocks are generated
        foresight_loss = None
        if self.enable_foresight and len(student_hidden_states) > 0:

            # Sample one 2D spatial offset per training step (used in all frame-pair losses)
            if self.foresight_spatial_offset:
                _max = self.foresight_spatial_offset_max
                _dh = int(torch.randint(-_max, _max + 1, (1,)).item())
                _dw = int(torch.randint(-_max, _max + 1, (1,)).item())
            else:
                _dh, _dw = 0, 0
            # One forward pass on all generated frames (no KV cache needed)
            teacher_input = output[:, :current_start_frame].detach()
            teacher_timestep = torch.zeros(
                [batch_size, current_start_frame],
                device=noise.device, dtype=torch.int64)

            with torch.no_grad():
                _, _, teacher_all_hidden = self.ema_generator(
                    noisy_image_or_video=teacher_input,
                    conditional_dict=conditional_dict,
                    timestep=teacher_timestep,
                    return_hidden_at_layer=self.teacher_foresight_layer
                )
            # teacher_all_hidden: [batch, total_frames * frame_seq_length, dim]

            # Determine deltas to use (same logic as Path B)
            future_deltas = list(range(1 + self.foresight_skip, self.foresight_delta + 1 + self.foresight_skip))
            deltas_to_use = ([0] + future_deltas) if self.foresight_include_current else future_deltas

            total_teacher_frames = current_start_frame  # total frames available in teacher hidden

            # Compute loss per student block with delta offset
            for student_block_idx, student_hidden, student_timestep_val, student_frame_start, student_num_frames in student_hidden_states:
                if self.foresight_delta_mean_pool:
                    # Collect teacher hiddens for all valid deltas, then average into one target
                    teacher_hiddens_for_avg = []
                    for delta in deltas_to_use:
                        teacher_block_idx = student_block_idx + delta
                        if teacher_block_idx >= len(all_num_frames):
                            continue
                        teacher_frame_start = sum(all_num_frames[:teacher_block_idx])
                        if initial_latent is not None:
                            teacher_frame_start += 1
                        teacher_num_frames_delta = all_num_frames[teacher_block_idx]

                        if teacher_frame_start + teacher_num_frames_delta > total_teacher_frames:
                            continue

                        t_start = teacher_frame_start * self.frame_seq_length
                        t_end = (teacher_frame_start + teacher_num_frames_delta) * self.frame_seq_length
                        teacher_hiddens_for_avg.append(teacher_all_hidden[:, t_start:t_end, :])

                    if not teacher_hiddens_for_avg:
                        continue

                    # Average teacher features across all deltas: [B, T, D]
                    avg_teacher = torch.stack(teacher_hiddens_for_avg, dim=0).mean(dim=0)

                    # Spatial mean-pool: [B, frames*seq_len, D] → [B, frames, D]
                    if self.foresight_online_mean_pool:
                        B = avg_teacher.shape[0]
                        avg_teacher = avg_teacher.view(
                            B, student_num_frames, self.frame_seq_length, avg_teacher.shape[-1]
                        ).mean(dim=2)
                        student_hidden_for_proj = student_hidden.view(
                            B, student_num_frames, self.frame_seq_length, student_hidden.shape[-1]
                        ).mean(dim=2)
                    else:
                        student_hidden_for_proj = student_hidden

                    # Cache pair for projector replay (delta=0 used for projector conditioning)
                    if self.cache_foresight_pairs:
                        self.cached_foresight_pairs.append({
                            'student_hidden': student_hidden_for_proj.detach().cpu(),
                            'teacher_hidden': avg_teacher.detach().cpu(),
                            'delta': 0,
                            'timestep': student_timestep_val,
                            'student_num_frames': student_num_frames,
                            'teacher_num_frames': student_num_frames,
                        })

                    # Single block-level loss against averaged teacher target
                    if self.foresight_projector_type != "simple":
                        projected_student = self.mirai_projector(student_hidden_for_proj, delta=0, timestep=student_timestep_val)
                    else:
                        projected_student = self.mirai_projector(student_hidden_for_proj)
                    if self.foresight_dim_proj is not None:
                        projected_student = self.foresight_dim_proj(projected_student)

                    teacher_for_loss = avg_teacher.detach()
                    if not self.foresight_online_mean_pool:
                        projected_student, teacher_for_loss = self._apply_spatial_crop_offset(
                            projected_student, teacher_for_loss, student_num_frames, _dh, _dw)

                    foresight_losses.append(self._foresight_loss(
                        projected_student.view(-1, projected_student.shape[-1]),
                        teacher_for_loss.view(-1, teacher_for_loss.shape[-1])
                    ))
                else:
                    for delta in deltas_to_use:
                        # Compute teacher frame position with delta offset (block-level)
                        teacher_block_idx = student_block_idx + delta
                        if teacher_block_idx >= len(all_num_frames):
                            continue
                        teacher_frame_start = sum(all_num_frames[:teacher_block_idx])
                        if initial_latent is not None:
                            teacher_frame_start += 1
                        teacher_num_frames = all_num_frames[teacher_block_idx]

                        # Bounds check: teacher frames must be within teacher_all_hidden
                        if teacher_frame_start + teacher_num_frames > total_teacher_frames:
                            continue

                        t_start = teacher_frame_start * self.frame_seq_length
                        t_end = (teacher_frame_start + teacher_num_frames) * self.frame_seq_length
                        teacher_hidden = teacher_all_hidden[:, t_start:t_end, :]

                        # Spatial mean-pool: [B, frames*seq_len, D] → [B, frames, D]
                        if self.foresight_online_mean_pool:
                            B = teacher_hidden.shape[0]
                            teacher_hidden = teacher_hidden.view(
                                B, teacher_num_frames, self.frame_seq_length, teacher_hidden.shape[-1]
                            ).mean(dim=2)
                            student_hidden_for_loss = student_hidden.view(
                                B, student_num_frames, self.frame_seq_length, student_hidden.shape[-1]
                            ).mean(dim=2)
                        else:
                            student_hidden_for_loss = student_hidden

                        # Cache pair for projector replay (store on CPU to save GPU memory)
                        if self.cache_foresight_pairs:
                            self.cached_foresight_pairs.append({
                                'student_hidden': student_hidden_for_loss.detach().cpu(),
                                'teacher_hidden': teacher_hidden.detach().cpu(),
                                'delta': delta,
                                'timestep': student_timestep_val,
                                'student_num_frames': student_num_frames,
                                'teacher_num_frames': teacher_num_frames,
                            })

                        # Frame-wise or block-level alignment
                        # When foresight_online_mean_pool=True, always use block-level path
                        if self.block_wise_using_frame_wise_foresight and not self.foresight_online_mean_pool:
                            num_frames_in_block = min(student_num_frames, teacher_num_frames)
                            frame_losses = []
                            for fi in range(num_frames_in_block):
                                s_start = fi * self.frame_seq_length
                                s_end = (fi + 1) * self.frame_seq_length
                                student_frame = student_hidden[:, s_start:s_end, :]
                                teacher_frame = teacher_hidden[:, s_start:s_end, :].detach()

                                if self.foresight_projector_type != "simple":
                                    proj_frame = self.mirai_projector(student_frame, delta=delta, timestep=student_timestep_val)
                                else:
                                    proj_frame = self.mirai_projector(student_frame)
                                if self.foresight_dim_proj is not None:
                                    proj_frame = self.foresight_dim_proj(proj_frame)

                                proj_frame, teacher_frame = self._apply_spatial_crop_offset(
                                    proj_frame, teacher_frame, 1, _dh, _dw)

                                frame_losses.append(self._foresight_loss(proj_frame.view(-1, proj_frame.shape[-1]), teacher_frame.view(-1, teacher_frame.shape[-1])))
                            foresight_losses.append(torch.stack(frame_losses).mean())
                        else:
                            # Block-level alignment (also used when foresight_online_mean_pool=True)
                            if self.foresight_projector_type != "simple":
                                projected_student = self.mirai_projector(student_hidden_for_loss, delta=delta, timestep=student_timestep_val)
                            else:
                                projected_student = self.mirai_projector(student_hidden_for_loss)
                            if self.foresight_dim_proj is not None:
                                projected_student = self.foresight_dim_proj(projected_student)

                            teacher_for_loss = teacher_hidden.detach()
                            if not self.foresight_online_mean_pool:
                                projected_student, teacher_for_loss = self._apply_spatial_crop_offset(
                                    projected_student, teacher_for_loss, student_num_frames, _dh, _dw)

                            foresight_losses.append(self._foresight_loss(projected_student.view(-1, projected_student.shape[-1]), teacher_for_loss.view(-1, teacher_for_loss.shape[-1])))

            # Release teacher activations to prevent OOM before backward pass
            del teacher_all_hidden, teacher_input, teacher_timestep
            torch.cuda.empty_cache()

            if foresight_losses:
                foresight_loss = torch.stack(foresight_losses).mean() * self.foresight_weight
        # Step 5: Return denoised timestep info
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1, foresight_loss

        return output, denoised_timestep_from, denoised_timestep_to, foresight_loss
