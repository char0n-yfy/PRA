from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F

from .edge import gaussian_blur_2d, rgb_to_gray, sobel_edge_map


def _to_01(images_nchw: torch.Tensor) -> torch.Tensor:
    x = images_nchw.to(torch.float32)
    if float(x.detach().amin().item()) < 0.0:
        x = (x + 1.0) * 0.5
    return x.clamp(0.0, 1.0)


def _conv2d_same(
    x: torch.Tensor,
    kernel: torch.Tensor,
    dilation: int = 1,
    mode: str = "reflect",
) -> torch.Tensor:
    kh, kw = int(kernel.shape[-2]), int(kernel.shape[-1])
    eff_h = dilation * (kh - 1) + 1
    eff_w = dilation * (kw - 1) + 1
    pad_h = eff_h - 1
    pad_w = eff_w - 1
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode=mode)
    return F.conv2d(x_pad, kernel, stride=1, padding=0, dilation=dilation)


def _haar_kernels(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, ...]:
    low = torch.tensor([1.0, 1.0], device=device, dtype=dtype) / (2.0**0.5)
    high = torch.tensor([-1.0, 1.0], device=device, dtype=dtype) / (2.0**0.5)
    ll = torch.outer(low, low).view(1, 1, 2, 2)
    lh = torch.outer(low, high).view(1, 1, 2, 2)
    hl = torch.outer(high, low).view(1, 1, 2, 2)
    hh = torch.outer(high, high).view(1, 1, 2, 2)
    return ll, lh, hl, hh


def _stationary_haar_high_bands(gray: torch.Tensor, num_levels: int) -> Sequence[torch.Tensor]:
    if gray.ndim != 4 or gray.shape[1] != 1:
        raise ValueError(f"gray must be (B,1,H,W), got {tuple(gray.shape)}")
    ll, lh, hl, hh = _haar_kernels(device=gray.device, dtype=gray.dtype)
    cur = gray
    out = []
    for level in range(int(num_levels)):
        dilation = 2**level
        out.append(_conv2d_same(cur, lh, dilation=dilation))
        out.append(_conv2d_same(cur, hl, dilation=dilation))
        out.append(_conv2d_same(cur, hh, dilation=dilation))
        cur = _conv2d_same(cur, ll, dilation=dilation)
    return out


def build_texture_map(
    images_nchw: torch.Tensor,
    wavelet_levels: int = 3,
    dog_sigmas: Iterable[Tuple[float, float]] = ((0.8, 1.6), (1.6, 3.2)),
    norm_quantile: float = 0.95,
    norm_clip: float = 3.0,
    norm_stats_stride: int = 4,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Build a 12-channel texture tensor:
      3-level stationary Haar high bands (9)
      + DoG bands (2)
      + edge magnitude (1)

    Returns:
      tex_map: (B,C,H,W) float32, roughly normalized to [-1,1].
    """
    x01 = _to_01(images_nchw)
    gray = rgb_to_gray(x01).to(torch.float32)

    bands = list(_stationary_haar_high_bands(gray, num_levels=int(wavelet_levels)))
    for sigma_small, sigma_large in dog_sigmas:
        lo = gaussian_blur_2d(gray, float(sigma_small))
        hi = gaussian_blur_2d(gray, float(sigma_large))
        bands.append(lo - hi)
    bands.append(sobel_edge_map(images_nchw, blur_sigma=0.0, threshold=0.0).to(torch.float32))

    tex = torch.cat(bands, dim=1)

    stats = tex.abs()
    stride = max(1, int(norm_stats_stride))
    if stride > 1:
        stats = F.avg_pool2d(stats, kernel_size=stride, stride=stride)
    flat = stats.flatten(2)
    q = float(norm_quantile)
    if 0.0 < q < 1.0:
        scale = torch.quantile(flat, q=q, dim=2, keepdim=True)
    else:
        scale = torch.amax(flat, dim=2, keepdim=True)
    scale = scale.view(tex.shape[0], tex.shape[1], 1, 1)

    clip = max(float(norm_clip), float(eps))
    tex = tex / torch.clamp(scale, min=float(eps))
    tex = tex.clamp(-clip, clip) / clip
    return tex


__all__ = ["build_texture_map"]
