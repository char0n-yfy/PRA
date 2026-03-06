from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _resolve_hw(size_cfg, fallback: int = 224) -> Tuple[int, int]:
    if isinstance(size_cfg, dict):
        if "height" in size_cfg and "width" in size_cfg:
            return int(size_cfg["height"]), int(size_cfg["width"])
        if "shortest_edge" in size_cfg:
            s = int(size_cfg["shortest_edge"])
            return s, s
        if "longest_edge" in size_cfg:
            s = int(size_cfg["longest_edge"])
            return s, s
    if isinstance(size_cfg, (tuple, list)):
        if len(size_cfg) == 2:
            return int(size_cfg[0]), int(size_cfg[1])
        if len(size_cfg) == 1:
            s = int(size_cfg[0])
            return s, s
    if isinstance(size_cfg, (int, float)):
        s = int(size_cfg)
        return s, s
    return int(fallback), int(fallback)


def _stats_to_torch(stat, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(stat, (int, float)):
        stat = [float(stat)] * 3
    t = torch.tensor(stat, device=device, dtype=dtype).view(1, -1, 1, 1)
    return t


def preprocess_for_hf_vision_encoder(
    images_nchw: torch.Tensor,
    processor,
    out_hw: Optional[Tuple[int, int]] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Normalize NCHW images to match HF image processor stats/size.

    Args:
      images_nchw: (B,C,H,W), expected range [-1,1] or [0,1] float.
    """
    if images_nchw.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got shape={tuple(images_nchw.shape)}")
    device = images_nchw.device

    x = images_nchw.to(device=device, dtype=torch.float32)
    x_min = float(x.detach().amin().item())
    x_max = float(x.detach().amax().item())
    if x_min < -5.0 or x_max > 5.0:
        raise ValueError(
            "preprocess_for_hf_vision_encoder expects inputs in [0,1] or [-1,1]. "
            f"Observed range [{x_min:.4f}, {x_max:.4f}]"
        )
    if x_min < 0.0:
        x = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
    else:
        x = x.clamp(0.0, 1.0)

    if out_hw is None:
        out_hw = _resolve_hw(getattr(processor, "size", 224), fallback=224)

    mean = _stats_to_torch(getattr(processor, "image_mean", [0.5, 0.5, 0.5]), device=device, dtype=torch.float32)
    std = _stats_to_torch(getattr(processor, "image_std", [0.5, 0.5, 0.5]), device=device, dtype=torch.float32)

    x = F.interpolate(
        x,
        size=out_hw,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    x = (x - mean) / std
    return x.to(dtype=dtype)


@dataclass
class DinoEncoderConfig:
    model_name: str
    use_dense: bool = True
    # If None, use the processor default size. Otherwise force resize to (image_size,image_size).
    image_size: Optional[int] = None


class DinoEncoder:
    """Frozen HF DINO encoder that returns last_hidden_state tokens on device."""

    def __init__(self, cfg: DinoEncoderConfig, device: torch.device):
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Online DINO extraction requires `transformers`.") from exc

        self.cfg = cfg
        self.device = device

        self.processor = AutoImageProcessor.from_pretrained(str(cfg.model_name), use_fast=False)
        self.model = AutoModel.from_pretrained(str(cfg.model_name)).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, images_nchw: torch.Tensor, amp_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """
        Args:
          images_nchw: (B,C,H,W) in [-1,1] or [0,1] float.
        Returns:
          tokens: (B, N, 768) float/bf16 on self.device. Dense tokens if cfg.use_dense else CLS only.
        """
        if images_nchw.device != self.device:
            images_nchw = images_nchw.to(device=self.device, non_blocking=True)

        out_hw = None
        if self.cfg.image_size is not None:
            s = int(self.cfg.image_size)
            out_hw = (s, s)

        pixel_values = preprocess_for_hf_vision_encoder(
            images_nchw,
            self.processor,
            out_hw=out_hw,
            dtype=torch.float32,
        )

        autocast_ctx = nullcontext()
        if self.device.type == "cuda" and torch.cuda.is_available():
            autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
        with autocast_ctx:
            out = self.model(pixel_values=pixel_values)

        hidden = out.last_hidden_state
        if bool(self.cfg.use_dense) and hidden.shape[1] > 1:
            hidden = hidden[:, 1:, :]
        else:
            hidden = hidden[:, :1, :]
        return hidden


__all__ = ["DinoEncoder", "DinoEncoderConfig", "preprocess_for_hf_vision_encoder"]

