from __future__ import annotations

from typing import Optional, Tuple

import torch


def broadcast_null_tokens(
    null_token: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return null_token.to(device=device, dtype=dtype).expand(int(batch_size), int(seq_len), -1)


def apply_cond_dropout(
    sem_tokens: torch.Tensor,
    null_tokens: torch.Tensor,
    drop_prob: float,
    force_drop_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz = sem_tokens.shape[0]
    device = sem_tokens.device
    if force_drop_mask is not None:
        drop_mask = force_drop_mask.to(device=device, dtype=torch.bool)
    else:
        drop_mask = torch.rand((bsz,), device=device) < float(drop_prob)

    mixed = torch.where(drop_mask[:, None, None], null_tokens, sem_tokens)
    return mixed, drop_mask
