from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cuda_bf16_autocast(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def paired_random_resized_crop(
    x1: torch.Tensor,
    x2: torch.Tensor,
    out_size: int = 224,
    scale: Tuple[float, float] = (0.08, 1.0),
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
):
    b, _, h, w = x1.shape
    device = x1.device
    area = float(h * w)
    out1, out2 = [], []
    for i in range(b):
        target_area = area * float(torch.empty((), device=device).uniform_(*scale))
        log_ratio = torch.log(torch.tensor(ratio, device=device, dtype=torch.float32))
        aspect = torch.exp(torch.empty((), device=device).uniform_(log_ratio[0], log_ratio[1]))
        crop_w = int(torch.clamp(torch.round(torch.sqrt(target_area * aspect)), 1, w).item())
        crop_h = int(torch.clamp(torch.round(torch.sqrt(target_area / aspect)), 1, h).item())
        top = int(torch.randint(0, h - crop_h + 1, (), device=device).item())
        left = int(torch.randint(0, w - crop_w + 1, (), device=device).item())

        a = x1[i : i + 1, :, top : top + crop_h, left : left + crop_w]
        b_ = x2[i : i + 1, :, top : top + crop_h, left : left + crop_w]
        out1.append(F.interpolate(a, size=(out_size, out_size), mode="bicubic", align_corners=False, antialias=True))
        out2.append(F.interpolate(b_, size=(out_size, out_size), mode="bicubic", align_corners=False, antialias=True))
    return torch.cat(out1, dim=0), torch.cat(out2, dim=0)


def _resolve_hw(size_cfg, fallback=224):
    if isinstance(size_cfg, dict):
        if "height" in size_cfg and "width" in size_cfg:
            return int(size_cfg["height"]), int(size_cfg["width"])
        if "shortest_edge" in size_cfg:
            s = int(size_cfg["shortest_edge"])
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


class ConvNextFeatureHead(nn.Module):
    def __init__(self, model_name: str = "facebook/convnextv2-base-22k-224"):
        super().__init__()
        from transformers import AutoImageProcessor, ConvNextV2Model

        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
        self.model = ConvNextV2Model.from_pretrained(model_name)

        mean = np.asarray(getattr(self.processor, "image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.asarray(getattr(self.processor, "image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
        if mean.ndim == 0:
            mean = np.repeat(mean, 3)
        if std.ndim == 0:
            std = np.repeat(std, 3)

        self.input_hw = _resolve_hw(getattr(self.processor, "size", 224), fallback=224)
        self.register_buffer("mean", torch.from_numpy(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.from_numpy(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 3:
            raise ValueError(f"ConvNextFeatureHead expects NCHW with C=3, got {tuple(x.shape)}")
        x = x.to(torch.float32)
        if x.min() < 0:
            x = (x + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)
        x = F.interpolate(x, size=self.input_hw, mode="bicubic", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std

        with cuda_bf16_autocast(x.device):
            out = self.model(pixel_values=x)
        feat = getattr(out, "pooler_output", None)
        if feat is None:
            hidden = out.last_hidden_state
            feat = hidden.mean(dim=(2, 3)) if hidden.ndim == 4 else hidden.mean(dim=1)
        return feat.float()


@dataclass
class PerceptualConfig:
    enable_lpips: bool = True
    enable_convnext: bool = True
    convnext_model_name: str = "facebook/convnextv2-base-22k-224"


class PerceptualLoss:
    def __init__(self, cfg: PerceptualConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.lpips_model = None
        self.convnext_model = None

        if bool(cfg.enable_lpips):
            import lpips

            self.lpips_model = lpips.LPIPS(net="vgg").to(device)
            self.lpips_model.eval()
            for p in self.lpips_model.parameters():
                p.requires_grad_(False)

        if bool(cfg.enable_convnext):
            self.convnext_model = ConvNextFeatureHead(model_name=str(cfg.convnext_model_name)).to(device)
            self.convnext_model.eval()
            for p in self.convnext_model.parameters():
                p.requires_grad_(False)

    @property
    def enabled(self) -> bool:
        return self.lpips_model is not None or self.convnext_model is not None

    def __call__(self, pred_x: torch.Tensor, gt_x: torch.Tensor):
        pred_crop, gt_crop = paired_random_resized_crop(pred_x, gt_x, out_size=224)
        pred_crop = pred_crop.to(device=self.device, dtype=torch.float32)
        gt_crop = gt_crop.to(device=self.device, dtype=torch.float32)
        bsz = pred_crop.shape[0]

        lpips_vec = torch.zeros((bsz,), device=self.device, dtype=torch.float32)
        conv_vec = torch.zeros((bsz,), device=self.device, dtype=torch.float32)

        if self.lpips_model is not None:
            out = self.lpips_model(pred_crop, gt_crop, normalize=False)
            lpips_vec = out.reshape(out.shape[0], -1).mean(dim=1)

        if self.convnext_model is not None:
            with cuda_bf16_autocast(self.device):
                pred_feat = self.convnext_model(pred_crop)
            with torch.no_grad():
                with cuda_bf16_autocast(self.device):
                    gt_feat = self.convnext_model(gt_crop)
            conv_vec = torch.sum((pred_feat.float() - gt_feat.float()) ** 2, dim=-1)

        return lpips_vec, conv_vec
