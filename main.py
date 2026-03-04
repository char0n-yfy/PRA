"""Main entry for pure PyTorch pMF training/evaluation."""

import argparse
import logging
import os
import shutil
import time
from pathlib import Path
from datetime import datetime
import yaml

from configs.load_config import get_config, get_config_from_file
import train
from utils.logging_util import log_for_0


def _parse_config_arg(config_arg: str):
    # Legacy style: configs/load_config.py:pMF_B_16
    if ":" in config_arg and config_arg.endswith(
        tuple(
            [
                "pMF_B_16",
                "pMF_B_32",
                "pMF_L_16",
                "pMF_L_32",
                "pMF_H_16",
                "pMF_H_32",
                "eval",
            ]
        )
    ):
        mode = config_arg.split(":")[-1]
        return get_config(mode)
    # Generic legacy style with :
    if ":" in config_arg and config_arg.split(":")[0].endswith("load_config.py"):
        mode = config_arg.split(":")[-1]
        return get_config(mode)
    # Direct yml file path.
    if os.path.isfile(config_arg):
        return get_config_from_file(config_arg)
    # Mode shorthand.
    return get_config(config_arg)


def _config_name_from_arg(config_arg: str) -> str:
    if ":" in config_arg:
        name = config_arg.split(":")[-1]
    elif os.path.isfile(config_arg):
        name = Path(config_arg).stem
    else:
        name = config_arg

    if name.endswith("_config"):
        name = name[: -len("_config")]
    name = name.replace(os.sep, "_").replace(":", "_").strip()
    return name or "run"


def _default_workdir(config_arg: str) -> str:
    run_name = _config_name_from_arg(config_arg)
    return os.path.join("output", run_name)


def _build_timestamped_workdir(base_workdir: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = os.path.join(base_workdir, stamp)
    idx = 1
    while os.path.exists(candidate):
        candidate = os.path.join(base_workdir, f"{stamp}_{idx:02d}")
        idx += 1
    return candidate


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
        return _build_timestamped_workdir(base_workdir)

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
        workdir = _build_timestamped_workdir(base_workdir)
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


def _save_run_config(cfg, config_arg: str, workdir: str):
    os.makedirs(workdir, exist_ok=True)

    # Save merged/effective configuration used for this run.
    merged_cfg_path = os.path.join(workdir, "used_config.yml")
    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    with open(merged_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False, allow_unicode=True)

    # Save original input config for traceability.
    if os.path.isfile(config_arg):
        src_path = os.path.abspath(config_arg)
        dst_path = os.path.join(workdir, "source_config.yml")
        shutil.copy2(src_path, dst_path)
    else:
        with open(os.path.join(workdir, "source_config.txt"), "w", encoding="utf-8") as f:
            f.write(str(config_arg).strip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir",
        default="",
        type=str,
        help="Base directory for outputs. Default: output/<config_name>. "
        "A timestamped subdirectory is created for each run.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Config spec. e.g. configs/load_config.py:pMF_L_32 or configs/base/pMF_L_32_config.yml",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    cfg = _parse_config_arg(args.config)
    base_workdir = args.workdir.strip() if args.workdir else ""
    if not base_workdir:
        base_workdir = _default_workdir(args.config)
    base_workdir = os.path.abspath(base_workdir)
    workdir = _resolve_workdir_for_launch(base_workdir)

    if _env_rank() == 0:
        _save_run_config(cfg, args.config, workdir)

    log_for_0(f"Config loaded from: {args.config}")
    log_for_0(f"Workdir: {workdir}")

    if bool(getattr(cfg, "eval_only", False)):
        train.just_evaluate(cfg, workdir)
    else:
        train.train_and_evaluate(cfg, workdir)


if __name__ == "__main__":
    main()
