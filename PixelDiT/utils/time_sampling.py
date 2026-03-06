from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TimeSamplingConfig:
    # Match outer pMF naming/behavior.
    P_mean: float = 0.0
    P_std: float = 1.0
    data_proportion: float = 0.5
    tr_uniform: bool = False
    tr_uniform_prob: float = 0.1


def logit_normal_dist(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    P_mean: float = 0.0,
    P_std: float = 1.0,
) -> torch.Tensor:
    # Keep the sampling path aligned with outer pMF (which samples directly in `dtype`).
    rnd_normal = torch.randn((batch_size, 1, 1, 1), device=device, dtype=dtype)
    return torch.sigmoid(rnd_normal * float(P_std) + float(P_mean))


def sample_tr(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    P_mean: float = 0.0,
    P_std: float = 1.0,
    data_proportion: float = 0.5,
    tr_uniform: bool = False,
    tr_uniform_prob: float = 0.1,
):
    t = logit_normal_dist(batch_size, device=device, dtype=dtype, P_mean=P_mean, P_std=P_std)
    r = logit_normal_dist(batch_size, device=device, dtype=dtype, P_mean=P_mean, P_std=P_std)

    if bool(tr_uniform):
        unif_mask = torch.rand((batch_size, 1, 1, 1), device=device, dtype=dtype) < float(tr_uniform_prob)
        t_uniform = torch.rand((batch_size, 1, 1, 1), device=device, dtype=dtype)
        r_uniform = torch.rand((batch_size, 1, 1, 1), device=device, dtype=dtype)
        t = torch.where(unif_mask, t_uniform, t)
        r = torch.where(unif_mask, r_uniform, r)

    data_size = int(batch_size * float(data_proportion))
    fm_mask = torch.arange(batch_size, device=device) < data_size
    fm_mask = fm_mask.view(batch_size, 1, 1, 1)
    r = torch.where(fm_mask, t, r)

    t, r = torch.maximum(t, r), torch.minimum(t, r)
    return t, r, fm_mask
