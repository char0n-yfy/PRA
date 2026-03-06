from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _gaussian_kernel2d(
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
    kernel_size: Optional[int] = None,
) -> torch.Tensor:
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0 for gaussian kernel")
    if kernel_size is None:
        # 3-sigma rule.
        radius = int(math.ceil(3.0 * float(sigma)))
        kernel_size = 2 * radius + 1
    k = int(kernel_size)
    if k <= 0 or (k % 2) == 0:
        raise ValueError(f"kernel_size must be positive odd, got {kernel_size}")

    x = torch.arange(k, device=device, dtype=torch.float32) - (k - 1) / 2.0
    g = torch.exp(-(x**2) / (2.0 * float(sigma) ** 2))
    g = g / torch.clamp(g.sum(), min=1e-12)
    k2d = (g[:, None] * g[None, :]).to(dtype=dtype)
    return k2d


def gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Args:
      x: (B,C,H,W)
    """
    if sigma <= 0.0:
        return x
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW, got {tuple(x.shape)}")
    b, c, _h, _w = x.shape
    device = x.device
    dtype = x.dtype

    k2d = _gaussian_kernel2d(float(sigma), device=device, dtype=dtype)
    k = int(k2d.shape[-1])
    k2d = k2d.view(1, 1, k, k)
    weight = k2d.repeat(c, 1, 1, 1)
    return F.conv2d(x, weight=weight, bias=None, stride=1, padding=k // 2, groups=c)


def _to_01(images_nchw: torch.Tensor) -> torch.Tensor:
    x = images_nchw.to(torch.float32)
    if float(x.detach().amin().item()) < 0.0:
        x = (x + 1.0) * 0.5
    return x.clamp(0.0, 1.0)


def rgb_to_gray(images_nchw: torch.Tensor) -> torch.Tensor:
    if images_nchw.ndim != 4:
        raise ValueError(f"Expected NCHW, got {tuple(images_nchw.shape)}")
    if images_nchw.shape[1] == 1:
        return images_nchw
    if images_nchw.shape[1] != 3:
        raise ValueError(f"Expected C=1 or C=3, got {images_nchw.shape[1]}")
    r, g, b = images_nchw[:, 0:1], images_nchw[:, 1:2], images_nchw[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def sobel_edge_map(
    images_nchw: torch.Tensor,
    blur_sigma: float = 1.0,
    threshold: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute a 1-channel edge magnitude map from images.

    Args:
      images_nchw: (B,C,H,W) in [-1,1] or [0,1].
      blur_sigma: gaussian blur sigma before Sobel (reduces high-freq watermark artifacts).
      threshold: if >0, binarize edges with `mag > threshold` after normalization to [0,1].
    Returns:
      edge_map: (B,1,H,W) in [0,1].
    """
    x01 = _to_01(images_nchw)
    gray = rgb_to_gray(x01)
    gray = gaussian_blur_2d(gray, float(blur_sigma))

    device = gray.device
    dtype = gray.dtype

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + float(eps))

    # Normalize per-image to [0,1] for stable downstream conditioning.
    denom = torch.amax(mag, dim=(2, 3), keepdim=True)
    mag = mag / torch.clamp(denom, min=float(eps))
    mag = mag.clamp(0.0, 1.0)

    if float(threshold) > 0.0:
        mag = (mag > float(threshold)).to(dtype=mag.dtype)
    return mag


__all__ = ["sobel_edge_map", "gaussian_blur_2d", "rgb_to_gray"]
