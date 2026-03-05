from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .dist import is_main_process


@dataclass
class LoggerConfig:
    use_tensorboard: bool = True


def setup_logger(level: int = logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def rank0_info(msg: str):
    if is_main_process():
        logging.info(msg)


class ScalarWriter:
    def __init__(self, workdir: str, use_tensorboard: bool = True):
        self.tb_writer = None
        self.workdir = workdir
        if not is_main_process():
            return
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except Exception:
                SummaryWriter = None
            if SummaryWriter is not None:
                tb_dir = os.path.join(workdir, "tensorboard")
                os.makedirs(tb_dir, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=tb_dir)
                rank0_info(f"TensorBoard enabled at {tb_dir}")

    def write_scalars(self, step: int, scalar_dict: Dict[str, float]):
        if not is_main_process():
            return
        line = f"[{step}] " + ", ".join([f"{k}={float(v):.6g}" for k, v in scalar_dict.items()])
        logging.info(line)
        if self.tb_writer is not None:
            for k, v in scalar_dict.items():
                self.tb_writer.add_scalar(self._tag(k), float(v), int(step))
            self.tb_writer.flush()

    @staticmethod
    def _tag(k: str) -> str:
        if "/" in k:
            return k
        if k in {"loss", "loss_u", "loss_v"}:
            return f"train/loss/{k}"
        if k.startswith("aux_loss_"):
            return f"train/aux/{k}"
        if k in {"lr", "steps_per_second"}:
            return f"train/system/{k}"
        return f"train/misc/{k}"

    def close(self):
        if self.tb_writer is not None:
            self.tb_writer.close()


class AverageMeter:
    def __init__(self):
        self.sums: Dict[str, float] = {}
        self.count = 0

    def update(self, metrics: Dict[str, torch.Tensor | float]):
        for k, v in metrics.items():
            if torch.is_tensor(v):
                v = float(v.detach().mean().item())
            else:
                v = float(v)
            self.sums[k] = self.sums.get(k, 0.0) + v
        self.count += 1

    def pop(self) -> Dict[str, float]:
        if self.count == 0:
            return {}
        out = {k: v / float(self.count) for k, v in self.sums.items()}
        self.sums = {}
        self.count = 0
        return out
