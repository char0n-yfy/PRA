from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class LossConfig:
    norm_p: float = 0.5
    norm_eps: float = 1e-3
    perceptual_max_t: float = 1.0
    lpips_weight: float = 1.0
    convnext_weight: float = 0.0
    enable_lpips: bool = True
    enable_convnext: bool = True


def adp_wt_fn(loss_vec: torch.Tensor, norm_eps: float, norm_p: float) -> torch.Tensor:
    adp = (loss_vec + float(norm_eps)) ** float(norm_p)
    return loss_vec / adp.detach()


def masked_perceptual_loss(
    t_flat: torch.Tensor,
    lpips_vec: Optional[torch.Tensor],
    convnext_vec: Optional[torch.Tensor],
    cfg: LossConfig,
) -> torch.Tensor:
    device = t_flat.device
    dtype = t_flat.dtype
    bsz = t_flat.shape[0]

    loss = torch.zeros((bsz,), device=device, dtype=dtype)
    mask = t_flat < float(cfg.perceptual_max_t)
    if not mask.any():
        return loss

    if cfg.enable_lpips and lpips_vec is not None:
        lp = adp_wt_fn(lpips_vec.to(device=device, dtype=dtype), cfg.norm_eps, cfg.norm_p)
        loss = loss + float(cfg.lpips_weight) * torch.where(mask, lp, torch.zeros_like(lp))

    if cfg.enable_convnext and convnext_vec is not None:
        cv = adp_wt_fn(convnext_vec.to(device=device, dtype=dtype), cfg.norm_eps, cfg.norm_p)
        loss = loss + float(cfg.convnext_weight) * torch.where(mask, cv, torch.zeros_like(cv))

    return loss


def build_metrics(
    loss: torch.Tensor,
    loss_u: torch.Tensor,
    loss_v: torch.Tensor,
    lpips_vec: Optional[torch.Tensor] = None,
    convnext_vec: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {
        "loss": loss.detach(),
        "loss_u": loss_u.detach().mean(),
        "loss_v": loss_v.detach().mean(),
    }
    if lpips_vec is not None:
        out["aux_loss_lpips"] = lpips_vec.detach().mean()
    if convnext_vec is not None:
        out["aux_loss_convnext"] = convnext_vec.detach().mean()
    return out
