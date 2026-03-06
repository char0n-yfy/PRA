#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def _find_latest_checkpoint(workdir: str) -> str:
    ckpt_dir = Path(workdir) / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")
    ckpts = sorted(ckpt_dir.glob("checkpoint_step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint_step_*.pt under: {ckpt_dir}")
    return str(ckpts[-1].resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test PixelDiT eval pipeline.")
    parser.add_argument("--config", type=str, default="PixelDiT/configs/base_t2i_pmf.yml")
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--global-step", type=int, default=None)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from PixelDiT.eval import run_eval

    workdir = os.path.abspath(args.workdir)
    if args.checkpoint.strip():
        checkpoint = os.path.abspath(args.checkpoint)
    else:
        checkpoint = _find_latest_checkpoint(workdir)

    cfg_path = os.path.abspath(args.config)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config not found: {cfg_path}")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    ns = SimpleNamespace(
        config=cfg_path,
        checkpoint=checkpoint,
        workdir=workdir,
        max_samples=int(args.max_samples) if args.max_samples is not None else None,
        global_step=args.global_step,
        device=args.device,
    )

    print("[smoke-eval] config:", cfg_path)
    print("[smoke-eval] checkpoint:", checkpoint)
    print("[smoke-eval] workdir:", workdir)
    print("[smoke-eval] max_samples:", ns.max_samples)
    run_eval(ns)
    print("[smoke-eval] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
