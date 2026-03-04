import math

import torch


def lr_schedules(config, steps_per_epoch):
    base_lr = float(config.training.learning_rate)
    warmup_epochs = int(getattr(config.training, "warmup_epochs", 0))
    schedule_kind = getattr(config.training, "lr_schedule", "warmup_const")
    total_steps = int(config.training.num_epochs * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    lr_min_factor = float(getattr(config.training, "lr_min_factor", 0.0))

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if schedule_kind == "warmup_const":
            return 1.0
        if schedule_kind == "warmup_cosine":
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min_factor + (1.0 - lr_min_factor) * cosine
        raise ValueError(
            f"Unknown lr_schedule '{schedule_kind}'. "
            "Supported: 'warmup_const', 'warmup_cosine'."
        )

    return torch.optim.lr_scheduler.LambdaLR, lr_lambda, base_lr
