import functools
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import fully_shard, OffloadPolicy, CPUOffloadPolicy, FSDPModule
from torch.nn.parallel.distributed import DistributedDataParallel

from kronfluence.utils.state import State


def apply_ddp(
    model: nn.Module,
    local_rank: int,
    rank: int,
    world_size: int,
) -> DistributedDataParallel:
    """Applies DistributedDataParallel (DDP) to the given PyTorch model.

    Args:
        model (nn.Module):
            The PyTorch model to be parallelized.
        local_rank (int):
            The local rank of the current process within its node.
        rank (int):
            The global rank of the current process across all nodes.
        world_size (int):
            The total number of processes in the distributed setup.

    Returns:
        DistributedDataParallel:
            The input model wrapped with DDP.

    Raises:
        RuntimeError:
            If the distributed initialization fails.
    """
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    ddp_cfg = {
        "device_ids": [local_rank],
        "output_device": local_rank,
    }

    model = model.to(device=device)
    model = DistributedDataParallel(model, **ddp_cfg)

    return model


def apply_fsdp(
    model: nn.Module,
    sequential: nn.Sequential,
    cpu_offload: bool = True,
) -> FSDPModule:
    """Applies FSDP2 to the given PyTorch model.

    Args:
        model (nn.Module):
            The PyTorch model to be parallelized.
        cpu_offload (bool):
            Whether to offload parameters to CPU. Defaults to `True`.

    Returns:
        FullyShardedDataParallel:
            The input model wrapped with FSDP.

    Raises:
        ValueError:
            If an invalid sharding strategy is provided or if `layer_to_wrap` is not provided for transformer models.
        RuntimeError:
            If the distributed initialization fails.
    """
    state = State()

    if cpu_offload:
        offload_policy = OffloadPolicy()
    else:
        offload_policy = CPUOffloadPolicy()

    for i in range(len(sequential)):
        fully_shard(
            sequential[i],
            offload_policy=offload_policy,
            mesh=state.mesh,
        )

    model = fully_shard(
        model,
        offload_policy=offload_policy,
        mesh=state.mesh,
    )

    return model
