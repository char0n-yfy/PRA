#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def _default_workdir() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.abspath(os.path.join("output", "PixelDiT_eval_preflight", stamp))


def _make_fast_eval_config(cfg: dict, max_samples: int) -> dict:
    out = deepcopy(cfg)
    ev = out.setdefault("evaluation", {})
    ev["enabled"] = True
    ev["fail_on_error"] = True
    ev["max_samples"] = int(max_samples)
    ev["batch_size"] = 1
    ev["num_workers"] = min(int(ev.get("num_workers", 2)), 2)
    ev["posthoc_t_values"] = [0.10]
    ev["regen_t_values"] = [0.90]
    ev["regen_num_variants"] = 2
    ev["sweep"] = {
        "posthoc": {
            "num_steps": [1],
            "omegas": [1.0],
            "intervals": [[0.0, 1.0]],
        },
        "regen": {
            "num_steps": [1],
            "omegas": [1.0],
            "intervals": [[0.0, 1.0]],
        },
    }
    ev.setdefault("metrics", {})
    ev["metrics"]["enable_clip"] = False
    ev.setdefault("visualization", {})
    ev["visualization"]["save"] = False
    ev["visualization"]["to_tensorboard"] = False
    ev.setdefault("best", {})
    ev["best"]["enabled"] = False
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight eval before training.")
    parser.add_argument("--config", type=str, default="PixelDiT/configs/base_t2i_pmf.yml")
    parser.add_argument("--workdir", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--full-config",
        action="store_true",
        help="Use full evaluation config/sweeps from yaml (only override max_samples).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from PixelDiT.eval import run_eval
    from PixelDiT.train import _as_namespace, _build_model

    cfg_path = os.path.abspath(args.config)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config not found: {cfg_path}")

    workdir = os.path.abspath(args.workdir) if args.workdir.strip() else _default_workdir()
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, "checkpoints"), exist_ok=True)

    raw_cfg = _load_yaml(cfg_path)
    if args.full_config:
        fast_cfg = deepcopy(raw_cfg)
        fast_cfg.setdefault("evaluation", {})
        fast_cfg["evaluation"]["enabled"] = True
        fast_cfg["evaluation"]["fail_on_error"] = True
        fast_cfg["evaluation"]["max_samples"] = int(args.max_samples)
    else:
        fast_cfg = _make_fast_eval_config(raw_cfg, max_samples=int(args.max_samples))
    fast_cfg_path = os.path.join(workdir, "preflight_eval_config.yml")
    _dump_yaml(fast_cfg_path, fast_cfg)

    # Create bootstrap checkpoint from randomly initialized model.
    model = _build_model(_as_namespace(fast_cfg))
    ckpt_path = os.path.join(workdir, "checkpoints", "checkpoint_step_00000000.pt")
    torch.save(
        {
            "epoch": 0,
            "step": 0,
            "model": model.state_dict(),
        },
        ckpt_path,
    )

    ns = SimpleNamespace(
        config=fast_cfg_path,
        checkpoint=ckpt_path,
        workdir=workdir,
        max_samples=int(args.max_samples),
        global_step=0,
        device=args.device,
    )

    print("[preflight-eval] config:", fast_cfg_path)
    print("[preflight-eval] checkpoint:", ckpt_path)
    print("[preflight-eval] workdir:", workdir)
    print("[preflight-eval] max_samples:", int(args.max_samples))
    run_eval(ns)
    print("[preflight-eval] OK: eval pipeline is runnable before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
