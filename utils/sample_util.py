import math

import numpy as np
import torch

from utils import fid_util
from utils.logging_util import get_rank, get_world_size, log_for_0


def run_p_sample_step(p_sample_step, state, sample_idx, ema: float = None, **kwargs):
    """
    Run one sampling step and return uint8 BHWC numpy images.
    """
    samples = p_sample_step(state=state, sample_idx=sample_idx, ema=ema, **kwargs)
    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()
    samples = np.asarray(samples)

    if samples.ndim != 4:
        raise ValueError(f"Expected rank-4 samples, got shape {samples.shape}")
    if samples.shape[1] in (1, 3):  # BCHW
        samples = np.transpose(samples, (0, 2, 3, 1))

    if samples.dtype != np.uint8:
        samples = np.clip(127.5 * samples + 128.0, 0, 255).astype(np.uint8)
    return samples


def generate_fid_samples(state, config, p_sample_step, run_p_sample_step_fn, ema: float = None, **kwargs):
    """
    Generate samples for FID evaluation.
    """
    world_size = max(1, get_world_size())
    per_rank = int(math.ceil(config.fid.num_samples / world_size))
    num_steps = int(math.ceil(per_rank / config.fid.device_batch_size))

    samples_all = []
    log_for_0("Note: the first sample may be slower due to model warmup.")
    for step in range(num_steps):
        begin = step * config.fid.device_batch_size
        end = min(per_rank, begin + config.fid.device_batch_size)
        sample_idx = np.arange(begin, end, dtype=np.int64) + get_rank() * per_rank
        log_for_0(f"Sampling step {step + 1}/{num_steps}...")
        samples = run_p_sample_step_fn(
            p_sample_step,
            state,
            sample_idx=sample_idx,
            ema=ema,
            **kwargs,
        )
        samples_all.append(samples)

    samples_all = np.concatenate(samples_all, axis=0)
    return samples_all[:per_rank]


def get_fid_evaluator(config, writer):
    """
    Create FID evaluator function.
    """
    inception_net = fid_util.build_jax_inception(batch_size=config.fid.device_batch_size)
    stats_ref = fid_util.get_reference(config.fid.cache_ref)

    def _evaluate_one_mode(state, p_sample_step, ema: float = None, **kwargs):
        samples_all = generate_fid_samples(
            state,
            config,
            p_sample_step,
            run_p_sample_step,
            ema=ema,
            **kwargs,
        )
        stats = fid_util.compute_stats(samples_all, inception_net)
        metric = {}

        mode_str = f"ema_{ema}" if ema is not None else "online"
        omega = float(np.asarray(kwargs.get("omega", 0.0)).reshape(-1)[0])
        t_min = float(np.asarray(kwargs.get("t_min", 0.0)).reshape(-1)[0])
        t_max = float(np.asarray(kwargs.get("t_max", 1.0)).reshape(-1)[0])
        descriptor = f"omega_{omega:.2f}_tmin_{t_min:.2f}_tmax_{t_max:.2f}_{mode_str}"

        fid = fid_util.compute_fid(stats_ref["mu"], stats["mu"], stats_ref["sigma"], stats["sigma"])
        is_score, _ = fid_util.compute_inception_score(stats["logits"])

        metric[f"FID_{descriptor}"] = fid
        metric[f"IS_{descriptor}"] = is_score
        log_for_0(f"FID ({descriptor}): {fid:.4f}, IS ({descriptor}): {is_score:.4f}")
        return metric, fid, is_score

    def evaluator(state, p_sample_step, step, ema_only=False, **kwargs):
        metric_dict = {}
        fid, is_score = None, None
        ema = kwargs.pop("ema", None)
        ema_list = [ema] if ema is not None else list(getattr(state, "ema_params", {}).keys())

        for ema_item in ema_list:
            metric, fid, is_score = _evaluate_one_mode(state, p_sample_step, ema=ema_item, **kwargs)
            metric_dict.update(metric)

        if not ema_only:
            metric, fid, is_score = _evaluate_one_mode(state, p_sample_step, ema=None, **kwargs)
            metric_dict.update(metric)

        writer.write_scalars(step + 1, metric_dict)
        return fid, is_score

    return evaluator
