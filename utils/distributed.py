from contextlib import contextmanager
from datetime import timedelta
from functools import partial
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None, cpu_offload=False):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False  # Load ckpt on rank 0 and sync to other ranks
    )
    return module


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend,
                            init_method=init_method, timeout=timedelta(minutes=30))
    torch.cuda.set_device(local_rank)


class EMA_FSDP:
    """EMA that works on FSDP sharded parameters directly.

    update() operates on local shards only -- no allgather, no extra GPU memory.
    state_dict() temporarily swaps EMA shards into the model and calls
    fsdp_state_dict() to collect the full state dict (collective, only at save).
    """

    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self._fsdp_module = fsdp_module
        # Clone each rank's local shard to CPU float32 -- no communication
        self._sharded_shadow = [
            p.data.detach().clone().float().cpu()
            for p in fsdp_module.parameters()
        ]

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        for shadow, p in zip(self._sharded_shadow, fsdp_module.parameters()):
            shadow.mul_(d).add_(p.data.detach().float().cpu(), alpha=1. - d)

    def state_dict(self):
        """Return full (unsharded) EMA state dict for saving.

        All ranks must call this together (collective operation).
        """
        # Backup original sharded params to CPU
        original_data = [
            p.data.detach().clone().cpu()
            for p in self._fsdp_module.parameters()
        ]
        # Temporarily write EMA shards into the model
        for shadow, p in zip(self._sharded_shadow, self._fsdp_module.parameters()):
            p.data.copy_(shadow.to(dtype=p.dtype, device=p.device))
        # Allgather + unflatten via fsdp_state_dict (rank0_only)
        full_sd = fsdp_state_dict(self._fsdp_module)
        # Restore original weights
        for orig, p in zip(original_data, self._fsdp_module.parameters()):
            p.data.copy_(orig.to(device=p.device))
        return full_sd

    @torch.no_grad()
    def load_state_dict(self, sd):
        """Load a full (unsharded) EMA state_dict back into the sharded shadow.

        Counterpart to state_dict(): round-trip through the FSDP module so that
        FSDP handles the resharding, then copy the per-rank shard out of the
        module parameters into _sharded_shadow. All ranks must call together.
        """
        # Backup original module params (stay on GPU to avoid PCIe traffic)
        original_data = [
            p.data.detach().clone()
            for p in self._fsdp_module.parameters()
        ]
        # Load the full state_dict into the FSDP module; FSDP reshards it.
        with FSDP.state_dict_type(self._fsdp_module, StateDictType.FULL_STATE_DICT):
            self._fsdp_module.load_state_dict(sd)
        # Snapshot each rank's local shard into _sharded_shadow (CPU float32,
        # matching the storage format used by __init__/update).
        for i, p in enumerate(self._fsdp_module.parameters()):
            self._sharded_shadow[i] = p.data.detach().clone().float().cpu()
        # Restore original module params.
        for orig, p in zip(original_data, self._fsdp_module.parameters()):
            p.data.copy_(orig)

    def copy_to(self, fsdp_module):
        """Copy EMA shards directly into an FSDP module's parameters."""
        for shadow, p in zip(self._sharded_shadow, fsdp_module.parameters()):
            p.data.copy_(shadow.to(dtype=p.dtype, device=p.device))

    def get_ema_model(self):
        """
        Returns a callable proxy that runs the FSDP generator forward with EMA
        weights temporarily swapped in, then restores original weights. This
        lets downstream code (e.g. DMD pipeline) call `ema_generator(...)`
        without knowing about the swap mechanism.
        """
        return _EMAForwardProxy(self)

    @contextmanager
    def swap_in(self):
        """Temporarily swap EMA weights into the model for a forward pass.

        Usage:
            with ema.swap_in():
                output = model(input)   # runs with EMA weights
            # original weights restored automatically
        """
        # Backup original sharded params on GPU (GPU-GPU copy; avoids PCIe).
        original_data = [
            p.data.detach().clone()
            for p in self._fsdp_module.parameters()
        ]
        # Write EMA shards into the model (CPU -> GPU transfer).
        for shadow, p in zip(self._sharded_shadow, self._fsdp_module.parameters()):
            p.data.copy_(shadow.to(dtype=p.dtype, device=p.device))
        try:
            yield self._fsdp_module
        finally:
            # Restore original weights (GPU-GPU copy).
            for orig, p in zip(original_data, self._fsdp_module.parameters()):
                p.data.copy_(orig)


class _EMAForwardProxy:
    """Callable that forwards through the FSDP module with EMA weights active.

    Gradients are not accumulated into EMA shadow (shadow is CPU-only anyway),
    and callers can continue to use `with torch.no_grad()` around the
    invocation as usual.
    """

    def __init__(self, ema_fsdp):
        self._ema = ema_fsdp

    def __call__(self, *args, **kwargs):
        with self._ema.swap_in():
            return self._ema._fsdp_module(*args, **kwargs)

    def __getattr__(self, name):
        # Fall through to the underlying module for attribute access the
        # pipeline might do (e.g. .module, .parameters()). __getattr__ is only
        # called when the attribute isn't found on self, so `_ema` access is safe.
        return getattr(self._ema._fsdp_module, name)
