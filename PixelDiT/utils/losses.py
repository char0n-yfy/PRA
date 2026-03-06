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


def _perceptual_gate_cosine(t_flat: torch.Tensor, t0: float) -> torch.Tensor:
    """
    Smooth gate for perceptual losses.

    w(t) = 0.5 * (1 + cos(pi * t / t0)) for t <= t0, else 0.
    """
    if t0 <= 0.0:
        return torch.zeros_like(t_flat)
    t = t_flat.to(torch.float32)
    t0_f = float(t0)
    w = 0.5 * (1.0 + torch.cos(torch.pi * (t / t0_f)))
    w = torch.where(t <= t0_f, w, torch.zeros_like(w))
    return w.to(dtype=t_flat.dtype)


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
    if float(cfg.perceptual_max_t) <= 0.0:
        return loss

    w = _perceptual_gate_cosine(t_flat, float(cfg.perceptual_max_t))

    if cfg.enable_lpips and lpips_vec is not None:
        lp = adp_wt_fn(lpips_vec.to(device=device, dtype=dtype), cfg.norm_eps, cfg.norm_p)
        loss = loss + float(cfg.lpips_weight) * (w * lp)

    if cfg.enable_convnext and convnext_vec is not None:
        cv = adp_wt_fn(convnext_vec.to(device=device, dtype=dtype), cfg.norm_eps, cfg.norm_p)
        loss = loss + float(cfg.convnext_weight) * (w * cv)

    return loss


def build_metrics(
    loss: torch.Tensor,
    loss_u: torch.Tensor,
    loss_v: torch.Tensor,
    loss_u_raw: Optional[torch.Tensor] = None,
    loss_v_raw: Optional[torch.Tensor] = None,
    lpips_vec: Optional[torch.Tensor] = None,
    convnext_vec: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    # DDP safety: always emit a fixed metric key set so every rank performs
    # the same collective sequence in _reduce_metrics().
    zero = loss.detach().new_zeros(())
    out: Dict[str, torch.Tensor] = {
        "loss": loss.detach(),
        "loss_u": loss_u.detach().mean(),
        "loss_v": loss_v.detach().mean(),
        "loss_u_raw": zero,
        "loss_v_raw": zero,
        "aux_loss_lpips": zero,
        "aux_loss_convnext": zero,
    }
    if loss_u_raw is not None:
        out["loss_u_raw"] = loss_u_raw.detach().mean()
    if loss_v_raw is not None:
        out["loss_v_raw"] = loss_v_raw.detach().mean()
    if lpips_vec is not None:
        out["aux_loss_lpips"] = lpips_vec.detach().mean()
    if convnext_vec is not None:
        out["aux_loss_convnext"] = convnext_vec.detach().mean()
    return out
