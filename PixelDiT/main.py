from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

if __package__ is None or __package__ == "":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from PixelDiT.train import run_train, smoke_test_forward
from PixelDiT.utils.logging import rank0_info, setup_logger


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_workdir(config_path: str) -> str:
    cfg_name = Path(config_path).stem.replace("_config", "")
    return os.path.join("output", "PixelDiT", cfg_name)


def _timestamped_dir(base: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = os.path.join(base, stamp)
    idx = 1
    while os.path.exists(d):
        d = os.path.join(base, f"{stamp}_{idx:02d}")
        idx += 1
    return d


def _save_used_config(cfg: dict, src_path: str, workdir: str):
    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, "used_config.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    with open(os.path.join(workdir, "source_config_path.txt"), "w", encoding="utf-8") as f:
        f.write(os.path.abspath(src_path) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--workdir", default="", type=str)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    setup_logger()
    cfg = _load_yaml(args.config)

    if args.smoke_test:
        out = smoke_test_forward(cfg)
        rank0_info(f"Smoke forward passed: {out}")
        return

    base_workdir = args.workdir.strip() if args.workdir.strip() else _default_workdir(args.config)
    workdir = _timestamped_dir(base_workdir)

    _save_used_config(cfg, args.config, workdir)
    rank0_info(f"Config loaded from: {args.config}")
    rank0_info(f"Workdir: {workdir}")

    run_train(cfg, workdir)


if __name__ == "__main__":
    main()
