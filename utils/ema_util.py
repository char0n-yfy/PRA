from copy import deepcopy

import torch


def const_schedule(step, ema_value):
    return float(ema_value)


def edm_schedule(step, ema_halflife_kimg):
    ema_halflife_nimg = float(ema_halflife_kimg) * 1000.0
    ema_rampup_ratio = 0.05
    ema_halflife_nimg = min(ema_halflife_nimg, float(step) * 1024.0 * ema_rampup_ratio)
    ema_beta = 0.5 ** (1024.0 / max(ema_halflife_nimg, 1e-8))
    return float(ema_beta)


def ema_schedules(config):
    ema_type = getattr(config.training, "ema_type", "const")
    if ema_type == "const":
        return const_schedule
    if ema_type == "edm":
        return edm_schedule
    raise ValueError("Unknown EMA type.")


@torch.no_grad()
def update_ema(ema_model, model, alpha):
    alpha = float(alpha)
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(alpha).add_(p.data, alpha=(1.0 - alpha))
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.data.copy_(b.data)


def clone_model(model):
    ema_model = deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model
