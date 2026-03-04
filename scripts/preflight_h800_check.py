#!/usr/bin/env python3
"""
H800 training preflight checks for pMF-main (pure PyTorch).

Checks:
- CUDA visibility / device info
- NCCL availability and (optional) distributed all-reduce smoke test
- bf16 autocast + backward smoke test
- Config-driven DDP / AMP / Muon enablement
- Whether train.py would use official torch.optim.Muon or local fallback

Usage (single process):
    python scripts/preflight_h800_check.py --config configs/base/pMF_L_32_config.yml --expect-gpus 2

Usage (DDP/NCCL smoke, recommended for 2 GPUs):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      scripts/preflight_h800_check.py --config configs/base/pMF_L_32_config.yml --expect-gpus 2
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import socket
import sys
import traceback
from pathlib import Path
from typing import List, Tuple

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_config(config_arg: str):
    from configs.load_config import get_config, get_config_from_file

    cfg_path = Path(config_arg)
    if cfg_path.is_file():
        return get_config_from_file(str(cfg_path))
    return get_config(config_arg)


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _print(msg: str, rank0_only: bool = False):
    if rank0_only and _rank() != 0:
        return
    print(msg, flush=True)


def _env_summary():
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "NCCL_DEBUG",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_DISABLE",
        "NCCL_P2P_DISABLE",
        "NCCL_ASYNC_ERROR_HANDLING",
        "OMP_NUM_THREADS",
    ]
    _print("== Environment ==")
    for k in keys:
        _print(f"{k}={os.environ.get(k, '<unset>')}", rank0_only=True)


def _check_cuda(expect_gpus: int | None, strict_h800: bool) -> Tuple[List[str], List[str]]:
    errs, warns = [], []
    _print("== CUDA / Device Visibility ==")
    _print(f"torch={torch.__version__}, cuda_runtime={torch.version.cuda}, python={sys.version.split()[0]}", rank0_only=True)
    _print(f"cuda_available={torch.cuda.is_available()}", rank0_only=True)
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    _print(f"visible_gpu_count={device_count}", rank0_only=True)

    if not torch.cuda.is_available():
        errs.append("CUDA is not available.")
        return errs, warns

    if expect_gpus is not None and device_count < expect_gpus:
        errs.append(f"Visible GPUs ({device_count}) < expected ({expect_gpus}).")

    for i in range(device_count):
        prop = torch.cuda.get_device_properties(i)
        total_gb = prop.total_memory / (1024 ** 3)
        cc = f"{prop.major}.{prop.minor}"
        name = prop.name
        _print(
            f"gpu[{i}]: name={name}, cc={cc}, mem={total_gb:.1f} GiB, mp={prop.multi_processor_count}",
            rank0_only=True,
        )
        if strict_h800 and "H800" not in name.upper():
            warns.append(f"GPU {i} is not reported as H800 (name={name}).")

    try:
        bf16_supported = torch.cuda.is_bf16_supported()
        _print(f"cuda_bf16_supported={bf16_supported}", rank0_only=True)
        if not bf16_supported:
            errs.append("torch.cuda.is_bf16_supported() returned False.")
    except Exception as exc:
        warns.append(f"Could not query bf16 support: {exc}")

    try:
        nccl_ver = torch.cuda.nccl.version()
        _print(f"torch.cuda.nccl.version={nccl_ver}", rank0_only=True)
    except Exception as exc:
        warns.append(f"Could not query torch.cuda.nccl.version(): {exc}")

    return errs, warns


def _bf16_smoke_all_visible_gpus(matmul_size: int) -> Tuple[List[str], List[str]]:
    errs, warns = [], []
    if not torch.cuda.is_available():
        return ["bf16 smoke skipped because CUDA is unavailable."], warns

    _print("== bf16 Autocast Smoke ==")
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.set_device(i)
            a = torch.randn(matmul_size, matmul_size, device=f"cuda:{i}", requires_grad=True)
            b = torch.randn(matmul_size, matmul_size, device=f"cuda:{i}", requires_grad=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                out = (a @ b).square().mean()
            out.backward()
            finite = bool(torch.isfinite(out.detach()).item())
            grad_ok = bool(torch.isfinite(a.grad).all().item() and torch.isfinite(b.grad).all().item())
            _print(
                f"gpu[{i}] bf16_autocast_backward: ok={finite and grad_ok}, loss={float(out.detach().float()):.6f}",
                rank0_only=True,
            )
            if not (finite and grad_ok):
                errs.append(f"bf16 autocast smoke failed numerical finite check on gpu[{i}].")
            del a, b, out
            torch.cuda.synchronize(i)
        except Exception as exc:
            errs.append(f"bf16 autocast smoke failed on gpu[{i}]: {type(exc).__name__}: {exc}")
    return errs, warns


def _maybe_init_dist(timeout_sec: int) -> Tuple[str, torch.device]:
    ws = _world_size()
    rank = _rank()
    local_rank = _local_rank()
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank if ws > 1 else 0)
        device = torch.device("cuda", local_rank if ws > 1 else 0)
    else:
        device = torch.device("cpu")

    if ws > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=dt.timedelta(seconds=timeout_sec),
        )
    _print(
        f"dist: initialized={dist.is_initialized()}, backend={backend}, rank={rank}, world_size={ws}, local_rank={local_rank}, host={socket.gethostname()}",
        rank0_only=False,
    )
    return backend, device


def _ddp_nccl_smoke(device: torch.device) -> Tuple[List[str], List[str]]:
    errs, warns = [], []
    ws = _world_size()
    _print("== Distributed / NCCL Smoke ==")
    _print(f"dist_available={dist.is_available()}, dist_initialized={dist.is_initialized()}", rank0_only=True)
    _print(f"nccl_available={dist.is_nccl_available() if dist.is_available() else False}", rank0_only=True)

    if ws <= 1:
        warns.append("WORLD_SIZE=1; DDP/NCCL all-reduce smoke not executed. Run via torchrun for full check.")
        return errs, warns

    if not dist.is_available():
        errs.append("torch.distributed is not available.")
        return errs, warns
    if not dist.is_initialized():
        errs.append("Process group not initialized under WORLD_SIZE>1.")
        return errs, warns

    try:
        val = torch.tensor([float(_rank() + 1)], device=device)
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
        expected = float(ws * (ws + 1) // 2)
        ok = abs(float(val.item()) - expected) < 1e-5
        _print(f"all_reduce_sum={float(val.item())} expected={expected} ok={ok}", rank0_only=False)
        if not ok:
            errs.append(f"All-reduce returned {float(val.item())}, expected {expected}.")
        dist.barrier()
    except Exception as exc:
        errs.append(f"DDP/NCCL smoke failed: {type(exc).__name__}: {exc}")
    return errs, warns


def _check_config_and_muon(config_path_or_mode: str) -> Tuple[List[str], List[str]]:
    errs, warns = [], []
    _print("== Config / Optimizer / AMP ==")
    try:
        cfg = _load_config(config_path_or_mode)
    except Exception as exc:
        return [f"Failed to load config `{config_path_or_mode}`: {exc}"], warns

    optimizer_name = str(getattr(cfg.training, "optimizer", "adamw")).lower()
    half_precision = bool(getattr(cfg.training, "half_precision", False))
    enable_distributed = bool(getattr(cfg.training, "enable_distributed", True))
    cond_device = str(getattr(getattr(cfg, "condition", object()), "device", "unknown"))
    _print(f"config.training.optimizer={optimizer_name}", rank0_only=True)
    _print(f"config.training.half_precision={half_precision}", rank0_only=True)
    _print(f"config.training.enable_distributed={enable_distributed}", rank0_only=True)
    _print(f"config.condition.device={cond_device}", rank0_only=True)

    if half_precision and not torch.cuda.is_available():
        errs.append("Config enables half_precision, but CUDA is unavailable.")
    if half_precision and torch.cuda.is_available():
        _print("train.py uses torch.autocast(..., dtype=torch.bfloat16) when half_precision=True on CUDA.", rank0_only=True)

    has_official_muon = hasattr(torch.optim, "Muon")
    _print(f"torch.optim.Muon available={has_official_muon}", rank0_only=True)

    if optimizer_name == "muon":
        if has_official_muon:
            try:
                from utils.muon import MuonWithAuxAdamW

                p2 = torch.nn.Parameter(torch.randn(8, 8))
                p1 = torch.nn.Parameter(torch.randn(8))
                opt = MuonWithAuxAdamW(
                    muon_params=[p2],
                    adamw_params=[p1],
                    muon_kwargs=dict(
                        lr=float(getattr(cfg.training, "learning_rate", 1e-3)),
                        weight_decay=float(getattr(cfg.training, "weight_decay", 0.0)),
                        momentum=float(getattr(cfg.training, "muon_momentum", 0.95)),
                        nesterov=bool(getattr(cfg.training, "muon_nesterov", True)),
                        eps=float(getattr(cfg.training, "muon_eps", 1e-7)),
                        ns_steps=int(getattr(cfg.training, "muon_ns_steps", 5)),
                    ),
                    adamw_kwargs=dict(
                        lr=float(getattr(cfg.training, "adam_learning_rate", getattr(cfg.training, "learning_rate", 1e-3)) or getattr(cfg.training, "learning_rate", 1e-3)),
                        betas=(
                            float(getattr(cfg.training, "adam_b1", 0.9)),
                            float(getattr(cfg.training, "adam_b2", 0.95)),
                        ),
                        eps=float(getattr(cfg.training, "adam_eps", 1e-8)),
                        weight_decay=float(getattr(cfg.training, "adam_weight_decay", 0.0)),
                    ),
                )
                _print(
                    "Muon path: ENABLED (train.py will use MuonWithAuxAdamW: official torch.optim.Muon for 2D params + AdamW for non-2D params)",
                    rank0_only=True,
                )
                _print(
                    f"Muon smoke instantiate ok: muon_opt={type(opt.muon_opt).__name__ if opt.muon_opt is not None else None}, adamw_opt={type(opt.adamw_opt).__name__ if opt.adamw_opt is not None else None}",
                    rank0_only=True,
                )
            except Exception as exc:
                errs.append(f"Config requests Muon and torch.optim.Muon exists, but composite optimizer smoke instantiate failed: {exc}")
        else:
            try:
                from utils.muon import Muon as LocalMuon

                p = torch.nn.Parameter(torch.randn(8, 8))
                _ = LocalMuon([p], lr=float(getattr(cfg.training, "learning_rate", 1e-3)))
                warns.append(
                    "Config requests Muon, but official torch.optim.Muon is unavailable. train.py will fall back to local utils.muon.Muon."
                )
                _print("Muon path: FALLBACK local utils.muon.Muon", rank0_only=True)
            except Exception as exc:
                errs.append(f"Config requests Muon, but local utils.muon.Muon instantiate failed: {exc}")
    elif optimizer_name == "adamw":
        _print("Optimizer path: AdamW (Muon disabled by config).", rank0_only=True)
    else:
        errs.append(f"Unsupported config.training.optimizer={optimizer_name}")

    return errs, warns


def _finalize_dist():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="H800 preflight checks for pMF-main")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base/pMF_L_32_config.yml",
        help="Config file path or mode name (e.g., pMF_L_32).",
    )
    parser.add_argument(
        "--expect-gpus",
        type=int,
        default=None,
        help="Minimum visible GPU count expected (e.g., 2).",
    )
    parser.add_argument(
        "--matmul-size",
        type=int,
        default=1024,
        help="Square matmul size for bf16 autocast smoke test.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=120,
        help="Distributed init timeout in seconds when launched with torchrun.",
    )
    parser.add_argument(
        "--skip-bf16-smoke",
        action="store_true",
        help="Skip bf16 autocast smoke test.",
    )
    parser.add_argument(
        "--skip-dist-smoke",
        action="store_true",
        help="Skip distributed/NCCL all-reduce smoke test even if WORLD_SIZE>1.",
    )
    parser.add_argument(
        "--strict-h800-name",
        action="store_true",
        help="Warn if visible GPUs are not reported with 'H800' in name.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (non-zero exit).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    failures: List[str] = []
    warnings: List[str] = []

    _print("===== pMF H800 Preflight Check =====", rank0_only=True)
    _env_summary()

    errs, warns = _check_cuda(args.expect_gpus, args.strict_h800_name)
    failures.extend(errs)
    warnings.extend(warns)

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    try:
        _, device = _maybe_init_dist(args.timeout_sec)
    except Exception as exc:
        failures.append(f"Distributed init failed: {type(exc).__name__}: {exc}")
        _print(traceback.format_exc(), rank0_only=True)

    if not args.skip_dist_smoke:
        errs, warns = _ddp_nccl_smoke(device)
        failures.extend(errs)
        warnings.extend(warns)

    if not args.skip_bf16_smoke:
        errs, warns = _bf16_smoke_all_visible_gpus(args.matmul_size)
        failures.extend(errs)
        warnings.extend(warns)

    errs, warns = _check_config_and_muon(args.config)
    failures.extend(errs)
    warnings.extend(warns)

    # Rank 0 aggregates only local process observations; distributed smoke itself checks connectivity.
    if _rank() == 0:
        _print("== Summary ==", rank0_only=True)
        if warnings:
            for w in warnings:
                _print(f"[WARN] {w}", rank0_only=True)
        if failures:
            for e in failures:
                _print(f"[FAIL] {e}", rank0_only=True)
            _print(f"RESULT: FAIL ({len(failures)} failure(s), {len(warnings)} warning(s))", rank0_only=True)
        else:
            _print(f"RESULT: PASS ({len(warnings)} warning(s))", rank0_only=True)

    _finalize_dist()

    if failures:
        sys.exit(1)
    if args.strict and warnings:
        sys.exit(2)


if __name__ == "__main__":
    main()

