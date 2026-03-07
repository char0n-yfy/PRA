from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import shutil
import sys
import gc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image

if __package__ is None or __package__ == "":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from PixelDiT.models import PixelDiTT2IPMF
from PixelDiT.train import build_imagefolder_dataset
from PixelDiT.utils.dino_encoder import DinoEncoder, DinoEncoderConfig, preprocess_for_hf_vision_encoder
from PixelDiT.utils.edge import sobel_edge_map
from PixelDiT.utils.semantic_cache import DinoSemanticCache, IndexedDatasetWrapper


def _as_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _as_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_as_namespace(v) for v in obj]
    return obj


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _sanitize_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", x)


def _to_01(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)


def _normed_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1, eps=1e-8)
    b = F.normalize(b, dim=-1, eps=1e-8)
    return torch.sum(a * b, dim=-1)


def _pairwise_indices(n: int) -> Iterable[Tuple[int, int]]:
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j


def _default_workdir_from_ckpt(ckpt_path: str) -> str:
    p = Path(ckpt_path).resolve()
    # .../<run>/checkpoints/checkpoint_step_xxx.pt -> .../<run>
    if p.parent.name == "checkpoints":
        return str(p.parent.parent)
    return str(p.parent)


def _build_model(cfg, device: torch.device) -> PixelDiTT2IPMF:
    m = cfg.model
    spatial = getattr(cfg, "spatial", None)
    enable_edge = bool(getattr(spatial, "enable_edge", False)) if spatial is not None else False
    edge_channels = int(getattr(spatial, "edge_channels", 64)) if spatial is not None else 64
    model = PixelDiTT2IPMF(
        input_size=int(m.input_size),
        patch_size=int(m.patch_size),
        in_channels=int(m.in_channels),
        hidden_size=int(m.hidden_size),
        pixel_dim=int(m.pixel_dim),
        patch_depth=int(m.patch_depth),
        pixel_depth=int(m.pixel_depth),
        pixel_head_depth=int(m.pixel_head_depth),
        num_heads=int(m.num_heads),
        pixel_num_heads=int(m.pixel_num_heads),
        mlp_ratio=float(m.mlp_ratio),
        sem_in_dim=int(m.sem_in_dim),
        sem_num_tokens=int(getattr(m, "sem_num_tokens", 64)),
        sem_pool_num_heads=int(getattr(m, "sem_pool_num_heads", 12)),
        enable_edge_cond=enable_edge,
        edge_channels=edge_channels,
        use_qknorm=bool(m.use_qknorm),
        use_swiglu=bool(m.use_swiglu),
        use_rope=bool(m.use_rope),
        use_rmsnorm=bool(m.use_rmsnorm),
        use_checkpoint=False,
        null_token_learnable=bool(getattr(m, "null_token_learnable", True)),
    ).to(device)
    model.eval()
    return model


def _load_checkpoint(path: str, model: torch.nn.Module) -> Tuple[int, int]:
    # Epoch-end eval loads the full training checkpoint, which includes optimizer/scaler/RNG
    # states. PyTorch 2.6 defaults torch.load(..., weights_only=True), which rejects these
    # non-tensor objects. We trust locally produced checkpoints here and opt into full load.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        step = int(ckpt.get("step", 0))
        epoch = int(ckpt.get("epoch", 0))
    else:
        state = ckpt
        step = 0
        epoch = 0
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing model keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected model keys: {len(unexpected)}")
    return step, epoch


class LPIPSMetric:
    def __init__(self, device: torch.device):
        import lpips

        self.model = lpips.LPIPS(net="vgg").to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Expects [-1,1] input.
        out = self.model(x, y, normalize=False)
        return out.reshape(out.shape[0], -1).mean(dim=1)


class CLIPImageEncoder:
    def __init__(self, model_name: str, device: torch.device):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device

    @torch.inference_mode()
    def encode(self, x_nchw: torch.Tensor) -> torch.Tensor:
        if x_nchw.device != self.device:
            x_nchw = x_nchw.to(self.device, non_blocking=True)
        pixel_values = preprocess_for_hf_vision_encoder(
            x_nchw,
            self.processor,
            out_hw=None,
            dtype=torch.float32,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(self.device.type == "cuda")):
            out = self.model(pixel_values=pixel_values)
        if hasattr(out, "image_embeds") and out.image_embeds is not None:
            feat = out.image_embeds
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
            feat = hidden.mean(dim=(2, 3)) if hidden.ndim == 4 else hidden.mean(dim=1)
        else:
            raise RuntimeError("Unsupported CLIP model outputs: cannot derive image feature.")
        return feat.float()


class SSIMMetric:
    def __init__(self, channels: int = 3, window_size: int = 11, sigma: float = 1.5, device: torch.device = torch.device("cpu")):
        self.channels = int(channels)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.device = device
        self.window = self._create_window().to(device=device, dtype=torch.float32)

    def _create_window(self) -> torch.Tensor:
        k = self.window_size
        x = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
        g = torch.exp(-(x**2) / (2.0 * self.sigma**2))
        g = g / torch.clamp(g.sum(), min=1e-12)
        k2d = g[:, None] * g[None, :]
        w = k2d.view(1, 1, k, k).repeat(self.channels, 1, 1, 1)
        return w

    @torch.inference_mode()
    def __call__(self, x_01: torch.Tensor, y_01: torch.Tensor) -> torch.Tensor:
        c1 = 0.01**2
        c2 = 0.03**2
        pad = self.window_size // 2
        w = self.window.to(device=x_01.device, dtype=x_01.dtype)

        mu_x = F.conv2d(x_01, w, padding=pad, groups=self.channels)
        mu_y = F.conv2d(y_01, w, padding=pad, groups=self.channels)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x_01 * x_01, w, padding=pad, groups=self.channels) - mu_x2
        sigma_y2 = F.conv2d(y_01 * y_01, w, padding=pad, groups=self.channels) - mu_y2
        sigma_xy = F.conv2d(x_01 * y_01, w, padding=pad, groups=self.channels) - mu_xy

        num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        ssim_map = num / torch.clamp(den, min=1e-12)
        return ssim_map.mean(dim=(1, 2, 3))


@dataclass
class SweepItem:
    num_steps: int
    omega: float
    t_min: float
    t_max: float

    @property
    def tag(self) -> str:
        return f"omega{self.omega:g}_steps{self.num_steps}_int{self.t_min:g}-{self.t_max:g}"

    @property
    def safe_tag(self) -> str:
        return _sanitize_name(self.tag)


def _parse_sweep(eval_cfg, task_name: str) -> List[SweepItem]:
    sweep = getattr(eval_cfg.sweep, task_name)
    steps = [int(x) for x in list(sweep.num_steps)]
    omegas = [float(x) for x in list(sweep.omegas)]
    intervals_raw = list(sweep.intervals)
    intervals = []
    for item in intervals_raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"evaluation.sweep.{task_name}.intervals must be [t_min,t_max] pairs.")
        intervals.append((float(item[0]), float(item[1])))
    out = []
    for n, o, (t0, t1) in itertools.product(steps, omegas, intervals):
        out.append(SweepItem(num_steps=n, omega=o, t_min=t0, t_max=t1))
    return out


def _make_vis_grid(rows: List[torch.Tensor], nrow: int) -> Optional[torch.Tensor]:
    if not rows:
        return None
    grid = make_grid(
        torch.stack(rows, dim=0),
        nrow=nrow,
        normalize=True,
        value_range=(-1, 1),
    )
    return grid


def _save_grid(grid: Optional[torch.Tensor], out_path: str):
    if grid is None:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(grid, out_path)


def _psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    # Inputs are in [0,1], so max value is 1.
    return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-12))


@torch.inference_mode()
def _run_guided(
    model: PixelDiTT2IPMF,
    z_start: torch.Tensor,
    t_start: float,
    r_end: float,
    num_steps: int,
    sem_cond: torch.Tensor,
    edge_map: Optional[torch.Tensor],
    omega: float,
    t_min: float,
    t_max: float,
    t_eps: float,
) -> torch.Tensor:
    z = z_start
    b = z.shape[0]
    dtype = z.dtype
    device = z.device
    ts = torch.linspace(float(t_start), float(r_end), int(num_steps) + 1, device=device, dtype=dtype)
    for i in range(int(num_steps)):
        t_val = float(ts[i].item())
        r_val = float(ts[i + 1].item())
        t = torch.full((b,), t_val, device=device, dtype=dtype)
        h = torch.full((b,), t_val - r_val, device=device, dtype=dtype)

        out_c = model(x=z, t=t, h=h, sem_tokens=sem_cond, sem_mask=None, sem_drop_mask=None, edge_map=edge_map)
        x_hat_c = out_c["x_hat_u"]
        u_c = (z - x_hat_c) / torch.clamp(t.view(-1, 1, 1, 1), min=float(t_eps))

        drop_mask = torch.ones((b,), device=device, dtype=torch.bool)
        out_u = model(
            x=z,
            t=t,
            h=h,
            sem_tokens=sem_cond,
            sem_mask=None,
            sem_drop_mask=drop_mask,
            edge_map=edge_map,
        )
        x_hat_u = out_u["x_hat_u"]
        u_u = (z - x_hat_u) / torch.clamp(t.view(-1, 1, 1, 1), min=float(t_eps))

        omega_eff = float(omega) if (float(t_min) <= t_val <= float(t_max)) else 1.0
        u_g = u_u + omega_eff * (u_c - u_u)
        z = z - (t.view(-1, 1, 1, 1) - r_val) * u_g
    return z


def _make_seed(base_seed: int, sample_idx: int, t_val: float, variant_idx: int) -> int:
    return int(base_seed + sample_idx * 100003 + int(round(t_val * 1000.0)) * 101 + variant_idx * 1009)


def _build_eval_loader(cfg, max_samples: int, num_workers_override: Optional[int] = None) -> DataLoader:
    eval_bs = int(getattr(cfg.evaluation, "batch_size", 1))
    if eval_bs < 1:
        raise ValueError(f"evaluation.batch_size must be >=1, got {eval_bs}")
    val_ds = build_imagefolder_dataset(cfg, split=str(cfg.dataset.val_split))
    val_ds = IndexedDatasetWrapper(val_ds)
    if int(max_samples) > 0:
        n = min(int(max_samples), len(val_ds))
        val_ds = Subset(val_ds, list(range(n)))
    num_workers = int(num_workers_override) if num_workers_override is not None else int(getattr(cfg.evaluation, "num_workers", 2))
    return DataLoader(
        val_ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=bool(getattr(cfg.dataset, "pin_memory", True)),
        drop_last=False,
        persistent_workers=False,
    )


def _build_semantic_sources(cfg, device: torch.device):
    use_cache = bool(cfg.semantic.use_offline_cache)
    cache = None
    cond_encoder = None
    if use_cache:
        cache = DinoSemanticCache(
            cache_root=str(cfg.semantic.cache_root),
            dino_feature_dim=int(cfg.semantic.dino_feature_dim),
            num_dino_tokens=int(cfg.semantic.num_dino_tokens),
            dataset_root=str(cfg.dataset.root),
            image_size=int(cfg.dataset.image_size),
            strict=bool(cfg.semantic.strict),
        )
    else:
        cond_encoder = DinoEncoder(
            cfg=DinoEncoderConfig(
                model_name=str(cfg.semantic.dino_model_name),
                use_dense=bool(getattr(cfg.semantic, "dino_use_dense", True)),
                image_size=int(getattr(cfg.semantic, "dino_image_size", cfg.dataset.image_size)),
            ),
            device=device,
        )
    # Metric encoder always needs online features for generated outputs.
    metric_encoder = DinoEncoder(
        cfg=DinoEncoderConfig(
            model_name=str(cfg.semantic.dino_model_name),
            use_dense=bool(getattr(cfg.semantic, "dino_use_dense", True)),
            image_size=int(getattr(cfg.semantic, "dino_image_size", cfg.dataset.image_size)),
        ),
        device=device,
    )
    return cache, cond_encoder, metric_encoder


def _mean_feature_from_tokens(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.mean(dim=1)


def _write_scalars(writer, step: int, pairs: Dict[str, float]):
    for k, v in pairs.items():
        writer.add_scalar(k, float(v), int(step))
    writer.flush()


def _to_float_or_nan(x) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _score_posthoc(mean_metrics: Dict[str, float], lpips_weight: float) -> float:
    psnr = _to_float_or_nan(mean_metrics.get("psnr", float("nan")))
    lpips = _to_float_or_nan(mean_metrics.get("lpips", float("nan")))
    if math.isnan(psnr) or math.isnan(lpips):
        return float("-inf")
    return float(psnr - float(lpips_weight) * lpips)


def _score_regen(mean_metrics: Dict[str, float], clip_weight: float, div_weight: float) -> float:
    dino = _to_float_or_nan(mean_metrics.get("dino_sim", float("nan")))
    clip = _to_float_or_nan(mean_metrics.get("clip_sim", float("nan")))
    div_lp = _to_float_or_nan(mean_metrics.get("div_lpips", float("nan")))
    if math.isnan(dino) or math.isnan(div_lp):
        return float("-inf")
    clip_term = 0.0 if math.isnan(clip) else float(clip_weight) * clip
    return float(dino + clip_term + float(div_weight) * div_lp)


def _load_json_or_none(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_best_if_improved(
    name: str,
    score: float,
    payload: dict,
    source_checkpoint: str,
    ckpt_dir: str,
    min_improvement: float,
) -> Tuple[bool, dict]:
    os.makedirs(ckpt_dir, exist_ok=True)
    json_path = os.path.join(ckpt_dir, f"best_{name}.json")
    pt_path = os.path.join(ckpt_dir, f"best_{name}.pt")

    prev = _load_json_or_none(json_path)
    prev_score = float(prev.get("score", float("-inf"))) if isinstance(prev, dict) else float("-inf")
    improved = float(score) > (prev_score + float(min_improvement))
    if not improved:
        return False, (prev if isinstance(prev, dict) else {"score": prev_score})

    src_abs = os.path.abspath(source_checkpoint)
    dst_abs = os.path.abspath(pt_path)
    if src_abs != dst_abs:
        shutil.copy2(source_checkpoint, pt_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return True, payload


def run_eval(args):
    from torch.utils.tensorboard import SummaryWriter

    cfg_dict = _load_yaml(args.config)
    cfg = _as_namespace(cfg_dict)

    if not hasattr(cfg, "evaluation"):
        raise ValueError("Config missing `evaluation` section.")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        raise RuntimeError("Evaluation expects CUDA for practical runtime.")

    torch.manual_seed(int(cfg.evaluation.seed))
    torch.cuda.manual_seed_all(int(cfg.evaluation.seed))

    writer = None
    loader = None
    model = None
    cache = cond_encoder = dino_metric = None
    lpips_metric = ssim_metric = clip_encoder = None
    try:
        model = _build_model(cfg, device)
        step_from_ckpt, epoch_from_ckpt = _load_checkpoint(args.checkpoint, model)
        global_step = int(args.global_step) if args.global_step is not None else int(step_from_ckpt)

        base_workdir = args.workdir.strip() if args.workdir.strip() else _default_workdir_from_ckpt(args.checkpoint)
        eval_root = os.path.join(base_workdir, "eval", _now_stamp())
        os.makedirs(eval_root, exist_ok=True)
        tb_dir = os.path.join(eval_root, "tensorboard")
        writer = SummaryWriter(log_dir=tb_dir)
        checkpoints_dir = os.path.join(base_workdir, "checkpoints")

        max_samples = int(args.max_samples) if args.max_samples is not None else int(cfg.evaluation.max_samples)
        num_workers_override = getattr(args, "num_workers_override", None)
        loader = _build_eval_loader(cfg, max_samples=max_samples, num_workers_override=num_workers_override)

        cache, cond_encoder, dino_metric = _build_semantic_sources(cfg, device)
        lpips_metric = LPIPSMetric(device=device)
        ssim_metric = SSIMMetric(channels=int(cfg.model.in_channels), device=device)
        use_clip = bool(getattr(cfg.evaluation.metrics, "enable_clip", False))
        if use_clip:
            clip_encoder = CLIPImageEncoder(
                model_name=str(cfg.evaluation.metrics.clip_model_name),
                device=device,
            )

        posthoc_sweeps = _parse_sweep(cfg.evaluation, "posthoc")
        regen_sweeps = _parse_sweep(cfg.evaluation, "regen")

        posthoc_t_values = [float(x) for x in list(cfg.evaluation.posthoc_t_values)]
        regen_t_values = [float(x) for x in list(cfg.evaluation.regen_t_values)]
        regen_num_variants = int(cfg.evaluation.regen_num_variants)
        t_eps = float(getattr(cfg.evaluation, "t_eps", cfg.training.t_eps))
        noise_scale = float(cfg.training.noise_scale)

        enable_edge = bool(getattr(cfg.spatial, "enable_edge", False))
        edge_blur_sigma = float(getattr(cfg.spatial, "edge_blur_sigma", 1.0))
        edge_threshold = float(getattr(cfg.spatial, "edge_threshold", 0.0))

        save_vis = bool(getattr(cfg.evaluation.visualization, "save", True))
        vis_to_tb = bool(getattr(cfg.evaluation.visualization, "to_tensorboard", True))
        vis_num_images = int(getattr(cfg.evaluation.visualization, "num_images", 8))
        vis_regen_variants = int(getattr(cfg.evaluation.visualization, "regen_show_variants", 4))

        results = {
            "config": os.path.abspath(args.config),
            "checkpoint": os.path.abspath(args.checkpoint),
            "global_step": global_step,
            "epoch": int(epoch_from_ckpt),
            "posthoc": {},
            "regen": {},
        }

        best_cfg = getattr(cfg.evaluation, "best", SimpleNamespace())
        best_enabled = bool(getattr(best_cfg, "enabled", True))
        best_save_overall = bool(getattr(best_cfg, "save_overall", False))
        best_min_improvement = float(getattr(best_cfg, "min_improvement", 1e-6))
        best_posthoc_lpips_w = float(getattr(best_cfg, "posthoc_lpips_weight", 10.0))
        best_regen_clip_w = float(getattr(best_cfg, "regen_clip_weight", 0.5))
        best_regen_div_w = float(getattr(best_cfg, "regen_div_weight", 0.3))
        best_overall_w_posthoc = float(getattr(best_cfg, "overall_posthoc_weight", 0.5))
        best_overall_w_regen = float(getattr(best_cfg, "overall_regen_weight", 0.5))

        posthoc_best_candidate = None
        regen_best_candidate = None

        def get_sem_cond_batch(global_indices: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            if cache is not None:
                idx = global_indices.to(device="cpu", dtype=torch.int64)
                sem = cache.get_batch(split=str(cfg.dataset.val_split), indices=idx, device=device)
                return sem.to(dtype=torch.float32)
            assert cond_encoder is not None
            return cond_encoder.encode(x, amp_dtype=torch.bfloat16).to(dtype=torch.float32)

        for sweep in posthoc_sweeps:
            tag = sweep.tag
            safe_tag = sweep.safe_tag
            acc = {t: {"psnr": [], "ssim": [], "lpips": []} for t in posthoc_t_values}
            vis_rows: Dict[float, List[torch.Tensor]] = {t: [] for t in posthoc_t_values}
            vis_seen: Dict[float, int] = {t: 0 for t in posthoc_t_values}

            for _batch_i, batch in enumerate(loader):
                global_idx, images, _labels = batch
                x = images.to(device=device, non_blocking=True)
                bsz = int(x.shape[0])
                sem_cond_batch = get_sem_cond_batch(global_idx, x)
                edge_map_batch = (
                    sobel_edge_map(x, blur_sigma=edge_blur_sigma, threshold=edge_threshold) if enable_edge else None
                )

                for t_val in posthoc_t_values:
                    for bi in range(bsz):
                        x_b = x[bi : bi + 1]
                        sem_b = sem_cond_batch[bi : bi + 1]
                        edge_b = edge_map_batch[bi : bi + 1] if edge_map_batch is not None else None
                        sample_key = int(global_idx[bi].item())

                        g = torch.Generator(device=device)
                        g.manual_seed(_make_seed(int(cfg.evaluation.seed), sample_key, t_val, 0))
                        eps = torch.randn(x_b.shape, device=device, dtype=x_b.dtype, generator=g)
                        z_t = (1.0 - t_val) * x_b + t_val * (noise_scale * eps)
                        x_hat = _run_guided(
                            model=model,
                            z_start=z_t,
                            t_start=t_val,
                            r_end=0.0,
                            num_steps=sweep.num_steps,
                            sem_cond=sem_b,
                            edge_map=edge_b,
                            omega=sweep.omega,
                            t_min=sweep.t_min,
                            t_max=sweep.t_max,
                            t_eps=t_eps,
                        ).clamp(-1.0, 1.0)

                        x01 = _to_01(x_b)
                        y01 = _to_01(x_hat)
                        mse = torch.mean((x01 - y01) ** 2, dim=(1, 2, 3))
                        psnr = float(_psnr_from_mse(mse).mean().item())
                        ssim = float(ssim_metric(x01, y01).mean().item())
                        lp = float(lpips_metric(x_hat, x_b).mean().item())

                        acc[t_val]["psnr"].append(psnr)
                        acc[t_val]["ssim"].append(ssim)
                        acc[t_val]["lpips"].append(lp)

                        if (save_vis or vis_to_tb) and vis_seen[t_val] < vis_num_images:
                            edge_vis = torch.zeros_like(x_b)
                            if edge_b is not None:
                                edge_vis = edge_b.repeat(1, 3, 1, 1) * 2.0 - 1.0
                            z_t_vis = z_t.clamp(-1.0, 1.0)
                            vis_rows[t_val].extend(
                                [
                                    x_b[0].detach().cpu(),
                                    z_t_vis[0].detach().cpu(),
                                    edge_vis[0].detach().cpu(),
                                    x_hat[0].detach().cpu(),
                                ]
                            )
                            vis_seen[t_val] += 1

            metrics_to_tb = {}
            posthoc_out = {}
            all_psnr, all_ssim, all_lpips = [], [], []
            for t_val in posthoc_t_values:
                t_str = f"{t_val:.2f}"
                mean_psnr = float(np.mean(acc[t_val]["psnr"])) if acc[t_val]["psnr"] else float("nan")
                mean_ssim = float(np.mean(acc[t_val]["ssim"])) if acc[t_val]["ssim"] else float("nan")
                mean_lp = float(np.mean(acc[t_val]["lpips"])) if acc[t_val]["lpips"] else float("nan")
                posthoc_out[t_str] = {"psnr": mean_psnr, "ssim": mean_ssim, "lpips": mean_lp}
                metrics_to_tb[f"eval/posthoc/psnr_t{t_str}/mean/{tag}"] = mean_psnr
                metrics_to_tb[f"eval/posthoc/ssim_t{t_str}/mean/{tag}"] = mean_ssim
                metrics_to_tb[f"eval/posthoc/lpips_t{t_str}/mean/{tag}"] = mean_lp
                all_psnr.append(mean_psnr)
                all_ssim.append(mean_ssim)
                all_lpips.append(mean_lp)

                if save_vis or vis_to_tb:
                    out_path = os.path.join(eval_root, "vis", "posthoc", f"{safe_tag}_t{t_str}.png")
                    grid = _make_vis_grid(vis_rows[t_val], nrow=4)
                    if save_vis:
                        _save_grid(grid, out_path=out_path)
                    if vis_to_tb and grid is not None:
                        writer.add_image(f"eval/vis/posthoc/t{t_str}/{safe_tag}", grid, global_step)

            metrics_to_tb[f"eval/posthoc/psnr/mean_over_t/{tag}"] = float(np.mean(all_psnr))
            metrics_to_tb[f"eval/posthoc/ssim/mean_over_t/{tag}"] = float(np.mean(all_ssim))
            metrics_to_tb[f"eval/posthoc/lpips/mean_over_t/{tag}"] = float(np.mean(all_lpips))
            _write_scalars(writer, global_step, metrics_to_tb)
            score_posthoc = _score_posthoc(
                {
                    "psnr": metrics_to_tb[f"eval/posthoc/psnr/mean_over_t/{tag}"],
                    "lpips": metrics_to_tb[f"eval/posthoc/lpips/mean_over_t/{tag}"],
                },
                lpips_weight=best_posthoc_lpips_w,
            )
            results["posthoc"][tag] = {
                "sweep": {"num_steps": sweep.num_steps, "omega": sweep.omega, "t_min": sweep.t_min, "t_max": sweep.t_max},
                "per_t": posthoc_out,
                "mean_over_t": {
                    "psnr": metrics_to_tb[f"eval/posthoc/psnr/mean_over_t/{tag}"],
                    "ssim": metrics_to_tb[f"eval/posthoc/ssim/mean_over_t/{tag}"],
                    "lpips": metrics_to_tb[f"eval/posthoc/lpips/mean_over_t/{tag}"],
                },
                "score_posthoc": float(score_posthoc),
            }
            cand = {
                "tag": tag,
                "score": float(score_posthoc),
                "infer_cfg": {
                    "omega": float(sweep.omega),
                    "t_min": float(sweep.t_min),
                    "t_max": float(sweep.t_max),
                    "num_steps": int(sweep.num_steps),
                    "t_eps": float(t_eps),
                },
                "metrics": results["posthoc"][tag]["mean_over_t"],
            }
            if (posthoc_best_candidate is None) or (cand["score"] > posthoc_best_candidate["score"]):
                posthoc_best_candidate = cand
            print(f"[posthoc] done {tag}")

        for sweep in regen_sweeps:
            tag = sweep.tag
            safe_tag = sweep.safe_tag
            acc = {
                t: {
                    "dino_sim": [],
                    "clip_sim": [],
                    "div_lpips": [],
                    "div_dino": [],
                }
                for t in regen_t_values
            }
            vis_rows: Dict[float, List[torch.Tensor]] = {t: [] for t in regen_t_values}
            vis_seen: Dict[float, int] = {t: 0 for t in regen_t_values}

            for _batch_i, batch in enumerate(loader):
                global_idx, images, _labels = batch
                x = images.to(device=device, non_blocking=True)
                bsz = int(x.shape[0])
                sem_cond_batch = get_sem_cond_batch(global_idx, x)
                edge_map_batch = (
                    sobel_edge_map(x, blur_sigma=edge_blur_sigma, threshold=edge_threshold) if enable_edge else None
                )

                with torch.no_grad():
                    dino_x_batch = _mean_feature_from_tokens(dino_metric.encode(x, amp_dtype=torch.bfloat16))
                    clip_x_batch = clip_encoder.encode(x) if clip_encoder is not None else None

                for t_val in regen_t_values:
                    for bi in range(bsz):
                        x_b = x[bi : bi + 1]
                        sem_b = sem_cond_batch[bi : bi + 1]
                        edge_b = edge_map_batch[bi : bi + 1] if edge_map_batch is not None else None
                        dino_x = dino_x_batch[bi : bi + 1]
                        clip_x = clip_x_batch[bi : bi + 1] if clip_x_batch is not None else None
                        sample_key = int(global_idx[bi].item())

                        outs = []
                        dino_feats = []
                        z_t_vis_ref: Optional[torch.Tensor] = None
                        for m in range(regen_num_variants):
                            g = torch.Generator(device=device)
                            g.manual_seed(_make_seed(int(cfg.evaluation.seed), sample_key, t_val, m))
                            eps = torch.randn(x_b.shape, device=device, dtype=x_b.dtype, generator=g)
                            z_t = (1.0 - t_val) * x_b + t_val * (noise_scale * eps)
                            if z_t_vis_ref is None:
                                z_t_vis_ref = z_t.clamp(-1.0, 1.0)
                            x_hat = _run_guided(
                                model=model,
                                z_start=z_t,
                                t_start=t_val,
                                r_end=0.0,
                                num_steps=sweep.num_steps,
                                sem_cond=sem_b,
                                edge_map=edge_b,
                                omega=sweep.omega,
                                t_min=sweep.t_min,
                                t_max=sweep.t_max,
                                t_eps=t_eps,
                            ).clamp(-1.0, 1.0)
                            outs.append(x_hat)

                            with torch.no_grad():
                                d_feat = _mean_feature_from_tokens(dino_metric.encode(x_hat, amp_dtype=torch.bfloat16))
                                dino_feats.append(d_feat)
                                dino_sim = float(_normed_cosine(d_feat, dino_x).mean().item())
                                acc[t_val]["dino_sim"].append(dino_sim)

                                if clip_encoder is not None and clip_x is not None:
                                    c_feat = clip_encoder.encode(x_hat)
                                    clip_sim = float(_normed_cosine(c_feat, clip_x).mean().item())
                                    acc[t_val]["clip_sim"].append(clip_sim)

                        div_lpips = []
                        div_dino = []
                        for i, j in _pairwise_indices(len(outs)):
                            lp = float(lpips_metric(outs[i], outs[j]).mean().item())
                            div_lpips.append(lp)
                            di = 1.0 - float(_normed_cosine(dino_feats[i], dino_feats[j]).mean().item())
                            div_dino.append(di)
                        if div_lpips:
                            acc[t_val]["div_lpips"].append(float(np.mean(div_lpips)))
                        if div_dino:
                            acc[t_val]["div_dino"].append(float(np.mean(div_dino)))

                        if (save_vis or vis_to_tb) and vis_seen[t_val] < vis_num_images:
                            edge_vis = torch.zeros_like(x_b)
                            if edge_b is not None:
                                edge_vis = edge_b.repeat(1, 3, 1, 1) * 2.0 - 1.0
                            vis_rows[t_val].append(x_b[0].detach().cpu())
                            if z_t_vis_ref is not None:
                                vis_rows[t_val].append(z_t_vis_ref[0].detach().cpu())
                            vis_rows[t_val].append(edge_vis[0].detach().cpu())
                            for out_img in outs[: max(1, vis_regen_variants)]:
                                vis_rows[t_val].append(out_img[0].detach().cpu())
                            vis_seen[t_val] += 1

            metrics_to_tb = {}
            regen_out = {}
            all_dino, all_clip, all_divlp = [], [], []
            for t_val in regen_t_values:
                t_str = f"{t_val:.2f}"
                mean_dino = float(np.mean(acc[t_val]["dino_sim"])) if acc[t_val]["dino_sim"] else float("nan")
                mean_clip = float(np.mean(acc[t_val]["clip_sim"])) if acc[t_val]["clip_sim"] else float("nan")
                mean_div_lpips = float(np.mean(acc[t_val]["div_lpips"])) if acc[t_val]["div_lpips"] else float("nan")
                mean_div_dino = float(np.mean(acc[t_val]["div_dino"])) if acc[t_val]["div_dino"] else float("nan")

                regen_out[t_str] = {
                    "dino_sim": mean_dino,
                    "clip_sim": mean_clip,
                    "div_lpips": mean_div_lpips,
                    "div_dino": mean_div_dino,
                }
                metrics_to_tb[f"eval/regen/dino_sim_t{t_str}/mean/{tag}"] = mean_dino
                metrics_to_tb[f"eval/regen/div_lpips_t{t_str}/mean/{tag}"] = mean_div_lpips
                metrics_to_tb[f"eval/regen/div_dino_t{t_str}/mean/{tag}"] = mean_div_dino
                if not math.isnan(mean_clip):
                    metrics_to_tb[f"eval/regen/clip_sim_t{t_str}/mean/{tag}"] = mean_clip

                all_dino.append(mean_dino)
                all_divlp.append(mean_div_lpips)
                if not math.isnan(mean_clip):
                    all_clip.append(mean_clip)

                if save_vis or vis_to_tb:
                    nrow = 3 + max(1, vis_regen_variants)
                    out_path = os.path.join(eval_root, "vis", "regen", f"{safe_tag}_t{t_str}.png")
                    grid = _make_vis_grid(vis_rows[t_val], nrow=nrow)
                    if save_vis:
                        _save_grid(grid, out_path=out_path)
                    if vis_to_tb and grid is not None:
                        writer.add_image(f"eval/vis/regen/t{t_str}/{safe_tag}", grid, global_step)

            mean_dino_all = float(np.mean(all_dino))
            mean_divlp_all = float(np.mean(all_divlp))
            metrics_to_tb[f"eval/regen/dino_sim/mean_over_t/{tag}"] = mean_dino_all
            metrics_to_tb[f"eval/regen/div_lpips/mean_over_t/{tag}"] = mean_divlp_all
            if all_clip:
                metrics_to_tb[f"eval/regen/clip_sim/mean_over_t/{tag}"] = float(np.mean(all_clip))
            pareto = mean_dino_all - (1.0 / max(mean_divlp_all, 1e-6))
            metrics_to_tb[f"eval/regen/pareto/{tag}"] = float(pareto)
            _write_scalars(writer, global_step, metrics_to_tb)
            score_regen = _score_regen(
                {
                    "dino_sim": mean_dino_all,
                    "clip_sim": float(np.mean(all_clip)) if all_clip else float("nan"),
                    "div_lpips": mean_divlp_all,
                },
                clip_weight=best_regen_clip_w,
                div_weight=best_regen_div_w,
            )

            results["regen"][tag] = {
                "sweep": {"num_steps": sweep.num_steps, "omega": sweep.omega, "t_min": sweep.t_min, "t_max": sweep.t_max},
                "per_t": regen_out,
                "mean_over_t": {
                    "dino_sim": mean_dino_all,
                    "clip_sim": float(np.mean(all_clip)) if all_clip else float("nan"),
                    "div_lpips": mean_divlp_all,
                    "pareto": float(pareto),
                },
                "score_regen": float(score_regen),
            }
            cand = {
                "tag": tag,
                "score": float(score_regen),
                "infer_cfg": {
                    "omega": float(sweep.omega),
                    "t_min": float(sweep.t_min),
                    "t_max": float(sweep.t_max),
                    "num_steps": int(sweep.num_steps),
                    "t_eps": float(t_eps),
                },
                "metrics": results["regen"][tag]["mean_over_t"],
            }
            if (regen_best_candidate is None) or (cand["score"] > regen_best_candidate["score"]):
                regen_best_candidate = cand
            print(f"[regen] done {tag}")

        if best_enabled:
            run_meta = {
                "config": os.path.abspath(args.config),
                "source_checkpoint": os.path.abspath(args.checkpoint),
                "ckpt_step": int(global_step),
                "ckpt_epoch": int(epoch_from_ckpt),
                "evaluated_at": datetime.now().isoformat(timespec="seconds"),
                "eval": {
                    "split": str(cfg.dataset.val_split),
                    "n": int(max_samples),
                    "seed": int(cfg.evaluation.seed),
                    "posthoc_t_values": posthoc_t_values,
                    "regen_t_values": regen_t_values,
                    "regen_num_variants": int(regen_num_variants),
                },
            }

            if posthoc_best_candidate is not None:
                payload = {
                    "best_type": "posthoc",
                    "score_name": "S_posthoc = PSNR_mean - lambda * LPIPS_mean",
                    "score_weights": {"posthoc_lpips_weight": best_posthoc_lpips_w},
                    "score": float(posthoc_best_candidate["score"]),
                    "metrics": posthoc_best_candidate["metrics"],
                    "infer_cfg": posthoc_best_candidate["infer_cfg"],
                    "sweep_tag": posthoc_best_candidate["tag"],
                    **run_meta,
                }
                improved, best_record = _save_best_if_improved(
                    name="posthoc",
                    score=float(posthoc_best_candidate["score"]),
                    payload=payload,
                    source_checkpoint=args.checkpoint,
                    ckpt_dir=checkpoints_dir,
                    min_improvement=best_min_improvement,
                )
                writer.add_scalar("best/posthoc/score", float(best_record.get("score", float("nan"))), int(global_step))
                writer.add_scalar("best/posthoc/step", int(best_record.get("ckpt_step", global_step)), int(global_step))
                if improved:
                    writer.add_text("best/posthoc/infer_cfg", json.dumps(posthoc_best_candidate["infer_cfg"], ensure_ascii=False), int(global_step))
                    print(f"[best] updated best_posthoc.pt score={posthoc_best_candidate['score']:.6f}")

            if regen_best_candidate is not None:
                payload = {
                    "best_type": "regen",
                    "score_name": "S_regen = DINO_sim + alpha * CLIP_sim + beta * DIV_LPIPS",
                    "score_weights": {
                        "regen_clip_weight": best_regen_clip_w,
                        "regen_div_weight": best_regen_div_w,
                    },
                    "score": float(regen_best_candidate["score"]),
                    "metrics": regen_best_candidate["metrics"],
                    "infer_cfg": regen_best_candidate["infer_cfg"],
                    "sweep_tag": regen_best_candidate["tag"],
                    **run_meta,
                }
                improved, best_record = _save_best_if_improved(
                    name="regen",
                    score=float(regen_best_candidate["score"]),
                    payload=payload,
                    source_checkpoint=args.checkpoint,
                    ckpt_dir=checkpoints_dir,
                    min_improvement=best_min_improvement,
                )
                writer.add_scalar("best/regen/score", float(best_record.get("score", float("nan"))), int(global_step))
                writer.add_scalar("best/regen/step", int(best_record.get("ckpt_step", global_step)), int(global_step))
                if improved:
                    writer.add_text("best/regen/infer_cfg", json.dumps(regen_best_candidate["infer_cfg"], ensure_ascii=False), int(global_step))
                    print(f"[best] updated best_regen.pt score={regen_best_candidate['score']:.6f}")

            if best_save_overall and (posthoc_best_candidate is not None) and (regen_best_candidate is not None):
                overall_score = (
                    float(best_overall_w_posthoc) * float(posthoc_best_candidate["score"])
                    + float(best_overall_w_regen) * float(regen_best_candidate["score"])
                )
                payload = {
                    "best_type": "overall",
                    "score_name": "S_overall = w1 * S_posthoc + w2 * S_regen",
                    "score_weights": {
                        "overall_posthoc_weight": best_overall_w_posthoc,
                        "overall_regen_weight": best_overall_w_regen,
                    },
                    "score": float(overall_score),
                    "metrics": {
                        "posthoc": posthoc_best_candidate["metrics"],
                        "regen": regen_best_candidate["metrics"],
                    },
                    "infer_cfg": {
                        "posthoc": posthoc_best_candidate["infer_cfg"],
                        "regen": regen_best_candidate["infer_cfg"],
                    },
                    "sweep_tag": {
                        "posthoc": posthoc_best_candidate["tag"],
                        "regen": regen_best_candidate["tag"],
                    },
                    **run_meta,
                }
                improved, best_record = _save_best_if_improved(
                    name="overall",
                    score=float(overall_score),
                    payload=payload,
                    source_checkpoint=args.checkpoint,
                    ckpt_dir=checkpoints_dir,
                    min_improvement=best_min_improvement,
                )
                writer.add_scalar("best/overall/score", float(best_record.get("score", float("nan"))), int(global_step))
                writer.add_scalar("best/overall/step", int(best_record.get("ckpt_step", global_step)), int(global_step))
                if improved:
                    writer.add_text("best/overall/infer_cfg", json.dumps(payload["infer_cfg"], ensure_ascii=False), int(global_step))
                    print(f"[best] updated best_overall.pt score={overall_score:.6f}")

        summary_path = os.path.join(eval_root, "metrics_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[done] Evaluation finished. Summary: {summary_path}")
        print(f"[done] TensorBoard: {tb_dir}")
    finally:
        if writer is not None:
            writer.close()
        if loader is not None:
            loader_iter = getattr(loader, "_iterator", None)
            if loader_iter is not None and hasattr(loader_iter, "_shutdown_workers"):
                loader_iter._shutdown_workers()
        del loader
        del writer
        del model
        del cache
        del cond_encoder
        del dino_metric
        del lpips_metric
        del ssim_metric
        del clip_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--workdir", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--global-step", type=int, default=None)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    main()
