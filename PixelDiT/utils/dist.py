from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return get_rank() == 0


def barrier():
    if is_distributed():
        dist.barrier()


def cleanup_distributed():
    if is_distributed():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()


def init_distributed(backend: str = "nccl"):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not is_distributed():
        if torch.cuda.is_available() and backend == "nccl":
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if world_size > 1 else 0)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    return rank, world_size, local_rank, device


def all_reduce_mean_scalar(value: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return value.to(dtype=torch.float32)
    x = value.detach().clone().to(dtype=torch.float32)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= float(get_world_size())
    return x


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
