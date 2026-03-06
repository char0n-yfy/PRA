from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

if __package__ is None or __package__ == "":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from PixelDiT.models import PixelDiTT2IPMF
from PixelDiT.utils.semantic_cache import DinoSemanticCache
from PixelDiT.utils.edge import sobel_edge_map


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model(cfg: dict, device: torch.device) -> PixelDiTT2IPMF:
    m = cfg["model"]
    spatial = cfg.get("spatial", {}) or {}
    model = PixelDiTT2IPMF(
        input_size=int(m["input_size"]),
        patch_size=int(m["patch_size"]),
        in_channels=int(m["in_channels"]),
        hidden_size=int(m["hidden_size"]),
        pixel_dim=int(m["pixel_dim"]),
        patch_depth=int(m["patch_depth"]),
        pixel_depth=int(m["pixel_depth"]),
        pixel_head_depth=int(m.get("pixel_head_depth", 1)),
        num_heads=int(m["num_heads"]),
        pixel_num_heads=int(m["pixel_num_heads"]),
        mlp_ratio=float(m["mlp_ratio"]),
        sem_in_dim=int(m["sem_in_dim"]),
        sem_num_tokens=int(m.get("sem_num_tokens", 64)),
        sem_pool_num_heads=int(m.get("sem_pool_num_heads", 12)),
        enable_edge_cond=bool(spatial.get("enable_edge", False)),
        edge_channels=int(spatial.get("edge_channels", 64)),
        use_qknorm=bool(m["use_qknorm"]),
        use_swiglu=bool(m["use_swiglu"]),
        use_rope=bool(m["use_rope"]),
        use_rmsnorm=bool(m["use_rmsnorm"]),
        use_checkpoint=False,
        null_token_learnable=bool(m.get("null_token_learnable", True)),
    ).to(device)
    model.eval()
    return model


def _load_checkpoint(path: str, model: torch.nn.Module):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")


def _load_sem_tokens(args, cfg, device: torch.device):
    if args.sem_tokens_file:
        arr = np.load(args.sem_tokens_file)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(f"sem_tokens_file must be [L,D] or [B,L,D], got {arr.shape}")
        sem = torch.from_numpy(arr[:1]).to(device=device, dtype=torch.float32)
        return sem

    if not args.cache_root:
        raise ValueError("Either --sem_tokens_file or --cache-root must be provided")

    cache = DinoSemanticCache(
        cache_root=args.cache_root,
        dino_feature_dim=int(cfg["semantic"]["dino_feature_dim"]),
        num_dino_tokens=int(cfg["semantic"]["num_dino_tokens"]),
        dataset_root=str(cfg["dataset"]["root"]),
        image_size=int(cfg["dataset"]["image_size"]),
        strict=bool(cfg["semantic"].get("strict", True)),
    )
    idx = torch.tensor([int(args.sample_index)], dtype=torch.int64)
    sem = cache.get_batch(split=args.split, indices=idx, device=device).to(dtype=torch.float32)
    return sem


def _to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float().clamp(-1, 1)
    x = (x + 1.0) * 0.5
    x = (x * 255.0).round().clamp(0, 255).to(torch.uint8)
    x = x[0].permute(1, 2, 0).numpy()
    return x


def _load_edge_image_tensor(path: str, image_size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((int(image_size), int(image_size)), resample=Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    x = x * 2.0 - 1.0
    return x


def _build_edge_map(args, cfg: dict, device: torch.device) -> torch.Tensor | None:
    spatial = cfg.get("spatial", {}) or {}
    enable_edge = bool(spatial.get("enable_edge", False))
    if not enable_edge:
        return None

    image_size = int(cfg["dataset"]["image_size"])
    sigma_cfg = float(spatial.get("edge_blur_sigma", 1.0))
    thresh_cfg = float(spatial.get("edge_threshold", 0.0))
    sigma = float(args.edge_blur_sigma) if args.edge_blur_sigma is not None else sigma_cfg
    threshold = float(args.edge_threshold) if args.edge_threshold is not None else thresh_cfg

    if args.edge_image:
        edge_src = _load_edge_image_tensor(args.edge_image, image_size=image_size, device=device)
        edge_map = sobel_edge_map(edge_src, blur_sigma=sigma, threshold=threshold)
        print(f"[info] spatial edge enabled with --edge-image: {args.edge_image}")
        return edge_map

    # Keep inference runnable when spatial is enabled but no explicit edge source is provided.
    edge_map = torch.zeros((1, 1, image_size, image_size), device=device, dtype=torch.float32)
    print("[warn] spatial.enable_edge=true but --edge-image is empty; using zero edge map.")
    return edge_map


def run_infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_yaml(args.config)

    model = _build_model(cfg, device)
    _load_checkpoint(args.checkpoint, model)
    edge_map = _build_edge_map(args, cfg, device)

    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    sem_cond = _load_sem_tokens(args, cfg, device)
    b, _l_sem, _d_sem = sem_cond.shape
    assert b == 1, "inference currently supports batch_size=1"

    h_img = int(cfg["dataset"]["image_size"])
    w_img = int(cfg["dataset"]["image_size"])
    c_img = int(cfg["model"]["in_channels"])

    z = torch.randn((1, c_img, h_img, w_img), device=device, dtype=torch.float32)
    t_steps = torch.linspace(1.0, 0.0, int(args.num_steps) + 1, device=device, dtype=torch.float32)

    with torch.no_grad():
        for i in range(int(args.num_steps)):
            t_val = t_steps[i]
            r_val = t_steps[i + 1]
            h_val = t_val - r_val

            t = torch.full((1,), float(t_val.item()), device=device, dtype=torch.float32)
            h = torch.full((1,), float(h_val.item()), device=device, dtype=torch.float32)

            out_cond = model(x=z, t=t, h=h, sem_tokens=sem_cond, edge_map=edge_map)
            x_hat_u_cond = out_cond["x_hat_u"]
            u_cond = (z - x_hat_u_cond) / torch.clamp(t.view(-1, 1, 1, 1), min=float(args.t_eps))

            drop_mask = torch.ones((1,), device=device, dtype=torch.bool)
            out_uncond = model(
                x=z,
                t=t,
                h=h,
                sem_tokens=sem_cond,
                sem_drop_mask=drop_mask,
                edge_map=edge_map,
            )
            x_hat_u_uncond = out_uncond["x_hat_u"]
            u_uncond = (z - x_hat_u_uncond) / torch.clamp(t.view(-1, 1, 1, 1), min=float(args.t_eps))

            omega_eff = float(args.omega) if float(args.t_min) <= float(t_val.item()) <= float(args.t_max) else 1.0
            u_guided = u_uncond + omega_eff * (u_cond - u_uncond)

            z = z - (t.view(-1, 1, 1, 1) - r_val) * u_guided

    img = _to_uint8(z)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    Image.fromarray(img).save(args.output)
    print(f"Saved: {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--omega", type=float, default=7.5)
    parser.add_argument("--t-min", type=float, default=0.0)
    parser.add_argument("--t-max", type=float, default=1.0)
    parser.add_argument("--t-eps", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--cache-root", type=str, default="")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sem-tokens-file", type=str, default="")
    parser.add_argument("--edge-image", type=str, default="")
    parser.add_argument("--edge-blur-sigma", type=float, default=None)
    parser.add_argument("--edge-threshold", type=float, default=None)

    args = parser.parse_args()
    run_infer(args)


if __name__ == "__main__":
    main()
