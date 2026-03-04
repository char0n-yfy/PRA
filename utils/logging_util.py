import logging
import os
import shutil
import time
from typing import Dict

import numpy as np
from PIL import Image
import torch

try:
    import wandb
except Exception:
    wandb = None


def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank():
    if is_distributed():
        return torch.distributed.get_rank()
    # torchrun sets RANK before torch.distributed.init_process_group(); use it to
    # avoid duplicate "rank0-only" logs during early startup.
    return int(os.environ.get("RANK", "0"))


def get_world_size():
    return torch.distributed.get_world_size() if is_distributed() else 1


def barrier():
    if is_distributed():
        torch.distributed.barrier()


def log_for_0(msg, *args):
    if get_rank() == 0:
        if args:
            logging.info(msg, *args)
        else:
            logging.info(msg)


def supress_checkpt_info():
    pass


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def elapse_without_reset(self):
        return time.time() - self.start_time

    def elapse_with_reset(self):
        a = time.time() - self.start_time
        self.reset()
        return a

    def reset(self):
        self.start_time = time.time()

    def __str__(self):
        return f"{self.elapse_with_reset():.2f} s"


class MetricsTracker:
    def __init__(self):
        self._sum = {}
        self._n = 0

    def update(self, metrics: Dict[str, torch.Tensor]):
        for k, v in metrics.items():
            if torch.is_tensor(v):
                val = float(v.detach().mean().item())
            else:
                val = float(v)
            self._sum[k] = self._sum.get(k, 0.0) + val
        self._n += 1

    def finalize(self):
        if self._n == 0:
            return {}
        out = {k: v / self._n for k, v in self._sum.items()}
        self._sum, self._n = {}, 0
        return out


class Writer:
    def __init__(self, config, workdir):
        self.workdir = workdir
        self.use_wandb = bool(getattr(config.logging, "use_wandb", False))
        self.use_tensorboard = bool(getattr(config.logging, "use_tensorboard", True))
        self.tb_writer = None
        if get_rank() != 0:
            return
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except Exception:
                SummaryWriter = None
            if SummaryWriter is not None:
                tb_dir = os.path.join(workdir, "tensorboard")
                os.makedirs(tb_dir, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=tb_dir)
                log_for_0(f"TensorBoard logging enabled at: {tb_dir}")
            else:
                log_for_0("TensorBoard is unavailable; install `tensorboard` to enable it.")
        else:
            log_for_0("TensorBoard logging is disabled by config.logging.use_tensorboard=False.")
        if self.use_wandb and wandb is not None:
            wandb.init(
                project=config.logging.wandb_project,
                entity=config.logging.wandb_entity if config.logging.wandb_entity else None,
                notes=config.logging.wandb_notes if config.logging.wandb_notes else None,
                tags=config.logging.wandb_tags if config.logging.wandb_tags else None,
                dir="/tmp",
            )
            if hasattr(config, "to_dict"):
                wandb.config.update(config.to_dict(), allow_val_change=True)
        else:
            self.use_wandb = False
            log_for_0("Wandb logging is disabled.")

    def write_scalars(self, step, scalar_dict):
        if get_rank() != 0:
            return
        log_str = f"[{step}] " + ", ".join(
            [f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in scalar_dict.items()]
        )
        logging.info(log_str)
        if self.tb_writer is not None:
            for k, v in scalar_dict.items():
                if torch.is_tensor(v):
                    v = float(v.detach().mean().item())
                else:
                    v = float(v)
                self.tb_writer.add_scalar(self._tensorboard_tag(k), v, step)
            self.tb_writer.flush()
        if self.use_wandb:
            wandb.log(scalar_dict, step=step)

    @staticmethod
    def _tensorboard_tag(key: str) -> str:
        # Keep pre-grouped tags (e.g. eval/...).
        if "/" in key:
            return key

        if key in {
            "loss",
            "loss_u",
            "loss_v",
            "loss_perc",
            "loss_sem",
            "loss_sem_raw",
        }:
            return f"train/loss/{key}"
        if key.startswith("aux_loss_"):
            return f"train/aux/{key}"
        if key.startswith("lambda_"):
            return f"train/weights/{key}"
        if key in {"high_noise_ratio", "cond_drop_ratio", "h_over_t"}:
            return f"train/sampling/{key}"
        if key == "lr":
            return "train/optim/lr"
        if key in {"steps_per_second", "epoch"}:
            return f"train/system/{key}"
        return f"train/misc/{key}"

    def write_images(self, step, image_dict):
        if get_rank() != 0:
            return

        def to_numpy_hwc(v):
            if isinstance(v, Image.Image):
                return np.asarray(v)
            if torch.is_tensor(v):
                v = v.detach().cpu().numpy()
            assert isinstance(v, np.ndarray)
            if v.ndim == 3 and v.shape[0] in (1, 3):
                v = np.transpose(v, (1, 2, 0))
            if v.dtype != np.uint8:
                v = np.clip(v, 0, 255).astype(np.uint8)
            return v

        np_images = {k: to_numpy_hwc(v) for k, v in image_dict.items()}
        if self.tb_writer is not None:
            for k, v in np_images.items():
                self.tb_writer.add_image(k, v, step, dataformats="HWC")
            self.tb_writer.flush()
        if self.use_wandb:
            wandb.log({k: wandb.Image(Image.fromarray(v)) for k, v in np_images.items()}, step=step)
        else:
            out_dir = os.path.join(self.workdir, "images")
            os.makedirs(out_dir, exist_ok=True)
            for k, v in np_images.items():
                Image.fromarray(v).save(os.path.join(out_dir, f"{step}_{k}.png"))

    def __del__(self):
        # Interpreter shutdown can tear down imported modules (e.g., torch)
        # before object finalizers run, so make cleanup best-effort only.
        try:
            if get_rank() != 0:
                return
            if self.tb_writer is not None:
                self.tb_writer.close()
            if self.use_wandb and wandb is not None:
                wandb.finish()
                shutil.rmtree("/tmp/wandb", ignore_errors=True)
        except Exception:
            return
