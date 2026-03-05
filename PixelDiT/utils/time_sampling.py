from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TimeSamplingConfig:
    logit_mean: float = 0.0
    logit_std: float = 1.0
    data_proportion: float = 0.5


def logit_normal_dist(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    z = torch.randn((batch_size, 1, 1, 1), device=device, dtype=torch.float32)
    z = mean + std * z
    t = torch.sigmoid(z)
    return t.to(dtype=dtype)


def sample_tr(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    data_proportion: float = 0.5,
):
    t = logit_normal_dist(batch_size, device=device, dtype=dtype, mean=logit_mean, std=logit_std)
    r = logit_normal_dist(batch_size, device=device, dtype=dtype, mean=logit_mean, std=logit_std)

    data_size = int(batch_size * float(data_proportion))
    fm_mask = torch.arange(batch_size, device=device) < data_size
    fm_mask = fm_mask.view(batch_size, 1, 1, 1)
    r = torch.where(fm_mask, t, r)

    t, r = torch.maximum(t, r), torch.minimum(t, r)
    return t, r, fm_mask
