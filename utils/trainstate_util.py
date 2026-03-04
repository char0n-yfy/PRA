from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from utils.logging_util import log_for_0
from utils.state_util import print_params


@dataclass
class TrainState:
    """
    Lightweight PyTorch train state kept for compatibility with older utility imports.
    """

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    ema_params: Dict[float, Dict[str, torch.Tensor]]
    step: int = 0

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema_params": {
                str(k): {n: t.detach().cpu() for n, t in v.items()}
                for k, v in self.ema_params.items()
            },
            "step": int(self.step),
        }

    def load_state_dict(self, ckpt: dict):
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = int(ckpt.get("step", 0))
        if "ema_params" in ckpt:
            parsed = {}
            for k, v in ckpt["ema_params"].items():
                key = float(k)
                parsed[key] = {n: t.to(next(self.model.parameters()).device) for n, t in v.items()}
            self.ema_params = parsed


def create_train_state(
    config,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    """
    Create a compatibility train state for pure PyTorch runs.
    """
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.training.learning_rate),
            betas=(0.9, float(getattr(config.training, "adam_b2", 0.95))),
            weight_decay=float(getattr(config.training, "weight_decay", 0.0)),
        )

    param_count = sum(p.numel() for p in model.parameters())
    log_for_0(f"Total trainable parameters: {param_count}")
    print_params(model.state_dict())

    ema_vals = config.training.ema_val
    if isinstance(ema_vals, (float, int)):
        ema_vals = [ema_vals]
    ema_params = {float(v): deepcopy(model.state_dict()) for v in ema_vals}

    return TrainState(
        model=model,
        optimizer=optimizer,
        ema_params=ema_params,
        step=0,
    )
