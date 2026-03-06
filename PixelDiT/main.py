from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import time
import traceback
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
    # Match the outer pMF layout: output/<run_name>/<timestamp>.
    # Keep a PixelDiT prefix at the same level to avoid mixing runs.
    run_name = cfg_name
    if not run_name.lower().startswith("pixeldit"):
        run_name = f"PixelDiT_{run_name}"
    return os.path.join("output", run_name)


def _timestamped_dir(base: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = os.path.join(base, stamp)
    idx = 1
    while os.path.exists(d):
        d = os.path.join(base, f"{stamp}_{idx:02d}")
        idx += 1
    return d


def _env_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _env_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _resolve_workdir_for_launch(base_workdir: str) -> str:
    """
    torchrun starts multiple processes before torch.distributed is initialized.
    To avoid each rank racing and choosing different timestamped output dirs,
    rank0 picks the workdir and writes it to a small sync file under base_workdir.
    """
    rank = _env_rank()
    world = _env_world_size()
    if world <= 1:
        return _timestamped_dir(base_workdir)

    os.makedirs(base_workdir, exist_ok=True)
    sync_key = "_".join(
        [
            os.environ.get("TORCHELASTIC_RUN_ID", "none"),
            os.environ.get("MASTER_ADDR", "localhost"),
            os.environ.get("MASTER_PORT", "0"),
            str(world),
        ]
    ).replace("/", "_")
    sync_path = os.path.join(base_workdir, f".workdir_sync_{sync_key}.txt")

    if rank == 0:
        workdir = _timestamped_dir(base_workdir)
        with open(sync_path, "w", encoding="utf-8") as f:
            f.write(workdir + "\n")
        return workdir

    deadline = time.time() + 120.0
    while time.time() < deadline:
        if os.path.exists(sync_path):
            try:
                with open(sync_path, "r", encoding="utf-8") as f:
                    line = f.readline().strip()
                if line:
                    return line
            except OSError:
                pass
        time.sleep(0.05)
    raise RuntimeError(f"Timed out waiting for rank0 workdir sync file: {sync_path}")


def _save_used_config(cfg: dict, src_path: str, workdir: str):
    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, "used_config.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    with open(os.path.join(workdir, "source_config_path.txt"), "w", encoding="utf-8") as f:
        f.write(os.path.abspath(src_path) + "\n")


def _save_traceback_file(trace_root: str, exc: BaseException):
    rank = _env_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    pid = os.getpid()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(trace_root, exist_ok=True)
    path = os.path.join(trace_root, f"fatal_rank{rank}_local{local_rank}_pid{pid}_{stamp}.log")
    tb_str = traceback.format_exc()
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"timestamp: {stamp}\n")
        f.write(f"rank: {rank}\n")
        f.write(f"local_rank: {local_rank}\n")
        f.write(f"pid: {pid}\n")
        f.write(f"exception: {type(exc).__name__}: {exc}\n\n")
        f.write(tb_str)
    return path


def main():
    faulthandler.enable()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--workdir", default="", type=str)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    setup_logger()
    cfg = _load_yaml(args.config)

    workdir = None
    base_workdir = None
    try:
        if args.smoke_test:
            out = smoke_test_forward(cfg)
            rank0_info(f"Smoke forward passed: {out}")
            return

        base_workdir = args.workdir.strip() if args.workdir.strip() else _default_workdir(args.config)
        base_workdir = os.path.abspath(base_workdir)
        workdir = _resolve_workdir_for_launch(base_workdir)

        if _env_rank() == 0:
            _save_used_config(cfg, args.config, workdir)
        rank0_info(f"Config loaded from: {args.config}")
        rank0_info(f"Workdir: {workdir}")

        run_train(cfg, workdir)
    except Exception as exc:
        rank = _env_rank()
        print(
            f"[rank{rank}] FATAL: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()

        trace_root = None
        if workdir:
            trace_root = os.path.join(workdir, "tracebacks")
        elif base_workdir:
            trace_root = os.path.join(base_workdir, "tracebacks")
        else:
            trace_root = os.path.abspath(os.path.join("output", "PixelDiT_tracebacks"))
        try:
            trace_path = _save_traceback_file(trace_root, exc)
            print(f"[rank{rank}] Traceback saved to: {trace_path}", file=sys.stderr, flush=True)
        except Exception as save_exc:
            print(
                f"[rank{rank}] Failed to save traceback file: {type(save_exc).__name__}: {save_exc}",
                file=sys.stderr,
                flush=True,
            )
        raise


if __name__ == "__main__":
    main()
