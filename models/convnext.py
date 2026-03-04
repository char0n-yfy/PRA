"""
PyTorch ConvNeXt-V2 feature extractor aligned with pMF-main.
"""

import numpy as np
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_hw(size_cfg, fallback=224):
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


class ConvNextFeatureHead(nn.Module):
    def __init__(self, model_name: str = "facebook/convnextv2-base-22k-224"):
        super().__init__()
        try:
            from transformers import AutoImageProcessor, ConvNextV2Model
        except Exception as exc:
            raise ImportError(
                "ConvNeXt-V2 perceptual head requires `transformers`."
            ) from exc

        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=False)
        self.model = ConvNextV2Model.from_pretrained(self.model_name)

        mean = np.asarray(
            getattr(self.processor, "image_mean", [0.485, 0.456, 0.406]),
            dtype=np.float32,
        )
        std = np.asarray(
            getattr(self.processor, "image_std", [0.229, 0.224, 0.225]),
            dtype=np.float32,
        )
        if mean.ndim == 0:
            mean = np.repeat(mean, 3)
        if std.ndim == 0:
            std = np.repeat(std, 3)
        self.input_hw = _resolve_hw(getattr(self.processor, "size", 224), fallback=224)
        self.register_buffer(
            "mean",
            torch.from_numpy(mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.from_numpy(std).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"Expected rank-4 input, got shape {tuple(x.shape)}")
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3-channel input, got shape {tuple(x.shape)}")
        x = x.to(torch.float32)
        if x.min() < 0:
            x = (x + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)
        x = F.interpolate(
            x,
            size=self.input_hw,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        x = (x - self.mean) / self.std

        if x.device.type == "cuda" and torch.cuda.is_available():
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            autocast_ctx = nullcontext()
        with autocast_ctx:
            out = self.model(pixel_values=x)
        feat = getattr(out, "pooler_output", None)
        if feat is None:
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                raise ValueError("ConvNeXt-V2 output does not contain usable features.")
            if hidden.ndim == 4:
                feat = hidden.mean(dim=(2, 3))
            elif hidden.ndim == 3:
                feat = hidden.mean(dim=1)
            else:
                raise ValueError(
                    f"Unexpected ConvNeXt-V2 hidden state rank: {hidden.ndim}"
                )
        return feat.float()


def load_convnext_jax_model(device=None, model_name: str = "facebook/convnextv2-base-22k-224"):
    """
    Backward-compatible name from the old JAX implementation.
    Returns a PyTorch module and ``None`` params.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNextFeatureHead(model_name=model_name).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, None
