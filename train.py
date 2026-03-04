"""
Training and evaluation for pixel MeanFlow (pure PyTorch).
"""

import itertools
import gc
import math
import os
import shutil
import traceback
import warnings
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.datasets.folder import pil_loader

from pmf import pixelMeanFlow
import utils.input_pipeline as input_pipeline
from utils.auxloss_util import init_auxloss
from utils.data_util import build_semantic_encoder
from utils.ema_util import clone_model, ema_schedules, update_ema
from utils.fid_util import (
    build_jax_inception,
    compute_fid,
    compute_inception_score,
    compute_stats,
    get_reference,
)
from utils.logging_util import (
    MetricsTracker,
    Timer,
    Writer,
    barrier,
    get_rank,
    get_world_size,
    log_for_0,
)
from utils.lr_utils import lr_schedules
from utils.muon import MuonWithAuxAdamW
from utils.vis_util import make_grid_visualization


def _build_lpips_vgg(device):
    import lpips

    # lpips internally triggers torchvision deprecated weight/pretrained warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The parameter 'pretrained' is deprecated since 0.13",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Arguments other than a weight enum or `None` for 'weights' are deprecated",
            category=UserWarning,
        )
        model = lpips.LPIPS(net="vgg").to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _cuda_reset_peak_memory(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)


def _cuda_peak_memory_mb(device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize(device)
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)


def _build_muon_optimizer(model, config, lr: float, weight_decay: float, adam_b2: float):
    """
    Build Muon optimizer recipe used in this project:
    - 2D parameters -> official torch.optim.Muon
    - non-2D parameters -> torch.optim.AdamW
    Requires torch>=2.9 with torch.optim.Muon available.
    """
    official_muon = getattr(torch.optim, "Muon", None)
    if official_muon is None:
        raise RuntimeError(
            "training.optimizer='muon' requires torch.optim.Muon (PyTorch >= 2.9). "
            "Upgrade PyTorch or switch config.training.optimizer='adamw'."
        )

    def _cfg_float(name: str, default: float) -> float:
        v = getattr(config.training, name, None)
        if v is None:
            return float(default)
        if isinstance(v, str) and v.strip() == "":
            return float(default)
        return float(v)

    trainable = [p for p in model.parameters() if p.requires_grad]
    muon_params = [p for p in trainable if p.ndim == 2]
    adamw_params = [p for p in trainable if p.ndim != 2]

    momentum = float(
        getattr(
            config.training,
            "muon_momentum",
            getattr(config.training, "muon_beta1", 0.95),
        )
    )
    muon_kwargs = dict(
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=bool(getattr(config.training, "muon_nesterov", True)),
        eps=float(getattr(config.training, "muon_eps", 1e-7)),
        ns_steps=int(getattr(config.training, "muon_ns_steps", 5)),
    )
    ns_coefficients = getattr(config.training, "muon_ns_coefficients", None)
    if ns_coefficients is not None:
        muon_kwargs["ns_coefficients"] = tuple(float(x) for x in ns_coefficients)
    adjust_lr_fn = getattr(config.training, "muon_adjust_lr_fn", None)
    if adjust_lr_fn is not None and str(adjust_lr_fn).strip() != "":
        muon_kwargs["adjust_lr_fn"] = str(adjust_lr_fn)

    adamw_kwargs = dict(
        lr=_cfg_float("adam_learning_rate", lr),
        betas=(
            _cfg_float("adam_b1", 0.9),
            adam_b2,
        ),
        eps=_cfg_float("adam_eps", 1e-8),
        weight_decay=_cfg_float("adam_weight_decay", 0.0),
    )

    optimizer = MuonWithAuxAdamW(
        muon_params=muon_params,
        adamw_params=adamw_params,
        muon_kwargs=muon_kwargs,
        adamw_kwargs=adamw_kwargs,
    )

    log_for_0(
        "Optimizer=MuonWithAuxAdamW"
        f"(muon_lr={lr}, muon_momentum={momentum}, muon_wd={weight_decay}, "
        f"muon_nesterov={muon_kwargs['nesterov']}, muon_ns_steps={muon_kwargs['ns_steps']}, "
        f"adam_lr={adamw_kwargs['lr']}, adam_betas={adamw_kwargs['betas']}, "
        f"adam_wd={adamw_kwargs['weight_decay']}, params_2d={len(muon_params)}, "
        f"params_non2d={len(adamw_params)})"
    )
    return optimizer


def _is_distributed():
    return dist.is_available() and dist.is_initialized()


def init_distributed(config=None):
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    enable_distributed = True
    if config is not None and hasattr(config, "training"):
        enable_distributed = bool(getattr(config.training, "enable_distributed", True))

    if not enable_distributed and env_world_size > 1:
        raise ValueError(
            "training.enable_distributed=False, but WORLD_SIZE>1 detected. "
            "Launch with a single process (no torchrun) or set training.enable_distributed=True."
        )

    if enable_distributed:
        world_size = env_world_size
        rank = env_rank
        local_rank = env_local_rank
        if world_size > 1 and not _is_distributed():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")
    else:
        world_size = 1
        rank = 0
        local_rank = 0

    if torch.cuda.is_available():
        device_index = local_rank if world_size > 1 else 0
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    return rank, world_size, device


def _to_device_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _reduce_mean_scalar(x: torch.Tensor):
    if not _is_distributed():
        return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= float(get_world_size())
    return y


def _reduce_metrics(metrics):
    out = {}
    for k, v in metrics.items():
        if torch.is_tensor(v):
            out[k] = _reduce_mean_scalar(v)
        else:
            t = torch.tensor(float(v), device="cuda" if torch.cuda.is_available() else "cpu")
            out[k] = _reduce_mean_scalar(t)
    return out


def _broadcast_bool_from_rank0(flag: bool, device: torch.device) -> bool:
    if not _is_distributed():
        return bool(flag)
    x = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
    dist.broadcast(x, src=0)
    return bool(int(x.item()))


def _save_checkpoint(
    workdir,
    epoch,
    step,
    model,
    optimizer,
    scheduler,
    ema_models,
):
    if get_rank() != 0:
        return
    os.makedirs(workdir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "ema_models": {str(k): v.state_dict() for k, v in ema_models.items()},
    }
    path_last = os.path.join(workdir, "checkpoint_last.pt")
    path_epoch = os.path.join(workdir, f"checkpoint_epoch_{epoch:04d}.pt")
    torch.save(ckpt, path_last)
    torch.save(ckpt, path_epoch)
    log_for_0(f"Saved checkpoint: {path_last}")


def _load_checkpoint(path, model, optimizer=None, scheduler=None, ema_models=None):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
        if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ema_models is not None and "ema_models" in ckpt:
            for k, v in ema_models.items():
                key = str(k)
                if key in ckpt["ema_models"]:
                    v.load_state_dict(ckpt["ema_models"][key], strict=False)
        epoch = int(ckpt.get("epoch", 0))
        step = int(ckpt.get("step", 0))
    else:
        # Support raw model state_dict checkpoints.
        model.load_state_dict(ckpt, strict=False)
        epoch = 0
        step = 0
    return epoch, step


def _generate_visual(
    model_core,
    config,
    device,
    n_sample=16,
    omega=None,
    t_min=None,
    t_max=None,
):
    if omega is None:
        omega = float(config.sampling.omega)
    if t_min is None:
        t_min = float(config.sampling.t_min)
    if t_max is None:
        t_max = float(config.sampling.t_max)
    with torch.no_grad():
        images = model_core.generate(
            n_sample=n_sample,
            num_steps=int(config.sampling.num_steps),
            omega=float(omega),
            t_min=float(t_min),
            t_max=float(t_max),
            cond_embeddings=None,
            device=device,
            dtype=torch.float32,
            image_size=int(config.dataset.image_size),
            image_channels=int(config.dataset.image_channels),
        )
        images = ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        vis = make_grid_visualization(images, grid=4, max_bz=1)[0]
    return vis


def _save_best_checkpoint(workdir, name, epoch, step, model, metric_name, metric_value):
    if get_rank() != 0:
        return
    os.makedirs(workdir, exist_ok=True)
    ckpt = {
        "epoch": int(epoch),
        "step": int(step),
        "model": model.state_dict(),
        "metric_name": str(metric_name),
        "metric_value": float(metric_value),
    }
    path = os.path.join(workdir, f"checkpoint_best_{name}.pt")
    torch.save(ckpt, path)
    log_for_0(f"Saved best checkpoint `{name}`: {path} ({metric_name}={metric_value:.6f})")


def _update_topk_checkpoints(
    workdir,
    name,
    model,
    metric_name,
    metric_value,
    epoch,
    step,
    topk_entries,
    top_k=3,
    higher_is_better=True,
):
    """
    Keep top-k checkpoints for a metric with fixed rank filenames:
      checkpoint_best_{name}_rank1.pt ... rankK.pt
    New better models overwrite rank files.
    """
    if get_rank() != 0:
        return topk_entries
    if not np.isfinite(metric_value):
        return topk_entries
    top_k = max(int(top_k), 1)

    # Entries are always kept sorted by metric_value according to optimization direction.
    entries = list(topk_entries)
    if len(entries) >= top_k:
        worst_val = entries[-1]["metric_value"]
        if higher_is_better:
            if metric_value <= worst_val:
                return entries
        else:
            if metric_value >= worst_val:
                return entries

    os.makedirs(workdir, exist_ok=True)
    candidate_path = os.path.join(workdir, f"checkpoint_best_{name}_candidate.pt")
    ckpt = {
        "epoch": int(epoch),
        "step": int(step),
        "model": model.state_dict(),
        "metric_name": str(metric_name),
        "metric_value": float(metric_value),
    }
    torch.save(ckpt, candidate_path)

    entries.append(
        {
            "metric_value": float(metric_value),
            "path": candidate_path,
        }
    )
    entries = sorted(entries, key=lambda x: x["metric_value"], reverse=bool(higher_is_better))[:top_k]

    # Copy all retained sources to temporary files to avoid overwrite conflicts.
    tmp_src = {}
    for ent in entries:
        src = ent["path"]
        if src == candidate_path or not os.path.exists(src):
            continue
        tmp_path = f"{src}.tmp_rankcopy"
        shutil.copy2(src, tmp_path)
        tmp_src[src] = tmp_path

    new_entries = []
    for i, ent in enumerate(entries, start=1):
        dst = os.path.join(workdir, f"checkpoint_best_{name}_rank{i}.pt")
        src = ent["path"]
        src_to_copy = src if src == candidate_path else tmp_src.get(src, src)
        if src_to_copy != dst:
            shutil.copy2(src_to_copy, dst)
        new_entries.append({"metric_value": ent["metric_value"], "path": dst})

    # Remove stale rank files.
    for i in range(len(new_entries) + 1, top_k + 1):
        stale = os.path.join(workdir, f"checkpoint_best_{name}_rank{i}.pt")
        if os.path.exists(stale):
            os.remove(stale)

    # Cleanup temp files.
    if os.path.exists(candidate_path):
        os.remove(candidate_path)
    for p in tmp_src.values():
        if os.path.exists(p):
            os.remove(p)

    log_for_0(
        f"Updated top-{top_k} `{name}` with {metric_name}={metric_value:.6f}. "
        f"Current best={new_entries[0]['metric_value']:.6f}"
    )
    return new_entries


def _cfg_to_dict(cfg_obj):
    if cfg_obj is None:
        return {}
    if hasattr(cfg_obj, "to_dict"):
        return cfg_obj.to_dict()
    return dict(cfg_obj)


def _get_eval_cfg(config):
    cfg = getattr(config, "evaluation", None)

    def _get(key, default):
        return getattr(cfg, key, default) if cfg is not None else default

    out = {
        "enable": bool(_get("enable", False)),
        "every_n_epochs": int(_get("every_n_epochs", 10)),
        "max_samples": int(_get("max_samples", 1000)),
        "batch_size": int(_get("batch_size", 32)),
        "seed": int(_get("seed", 2026)),
        "low_t": float(_get("low_t", 0.2)),
        "low_rho": float(_get("low_rho", 0.25)),
        "high_t": float(_get("high_t", 0.9)),
        "high_rho": float(_get("high_rho", 0.9)),
        "probe_t_values": list(_get("probe_t_values", [0.25, 0.65, 0.9])),
        "probe_rho_values": list(_get("probe_rho_values", [0.0, 0.25, 0.65, 0.9])),
        "probe_pairs_per_t": int(_get("probe_pairs_per_t", 8)),
        "probe_rho_zero_ratio": float(_get("probe_rho_zero_ratio", 0.25)),
        "probe_nonzero_rho_values": list(_get("probe_nonzero_rho_values", [0.25, 0.65, 0.9])),
        "cfg_w_mid": float(_get("cfg_w_mid", 4.0)),
        "cfg_w_high": float(_get("cfg_w_high", 8.0)),
        "interval_on": list(_get("interval_on", [0.5, 1.0])),
        "interval_off": list(_get("interval_off", [0.0, 0.5])),
        "num_vis": int(_get("num_vis", 4)),
        "compute_fid": bool(_get("compute_fid", True)),
        "compute_is": bool(_get("compute_is", True)),
        "preflight_check": bool(_get("preflight_check", True)),
        "preflight_max_samples": int(_get("preflight_max_samples", 32)),
        "preflight_include_fid_is": bool(_get("preflight_include_fid_is", False)),
        "preflight_fid_num_images": int(_get("preflight_fid_num_images", 0)),
        "preflight_fid_device_batch_size": int(_get("preflight_fid_device_batch_size", 0)),
        "lpips_resize": int(_get("lpips_resize", 224)),
        "best_lpips_psnr_tradeoff": float(_get("best_lpips_psnr_tradeoff", 10.0)),
        "best_top_k": int(_get("best_top_k", 3)),
        "freq_hf_cutoff": float(_get("freq_hf_cutoff", 0.5)),
        "freq_angle_bins": int(_get("freq_angle_bins", 16)),
        "freq_radial_bins": int(_get("freq_radial_bins", 24)),
    }
    return SimpleNamespace(**out)


def _cleanup_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _init_eval_runtime_models(config, condition_cfg, eval_cfg, device):
    """
    Lazily initialize heavy eval-only models on rank0 and release after use.
    """
    semantic_encoder_metric = None
    lpips_eval_model = None
    inception_eval = None
    fid_ref = None

    cond_metric_dict = _cfg_to_dict(condition_cfg)
    if cond_metric_dict:
        metric_device = cond_metric_dict.get("device", "cpu")
        if metric_device == "cuda" and not torch.cuda.is_available():
            metric_device = "cpu"
        cond_metric_dict["device"] = metric_device
        cond_metric_cfg = SimpleNamespace(**cond_metric_dict)
        try:
            semantic_encoder_metric = build_semantic_encoder(
                cond_metric_cfg,
                num_dino_tokens=int(getattr(config.model, "num_dino_tokens", 4)),
                use_clip=True,
                use_dino=True,
            )
        except Exception as exc:
            semantic_encoder_metric = None
            log_for_0(f"Semantic eval encoder init failed: {exc}")

    try:
        lpips_eval_model = _build_lpips_vgg(device)
    except Exception as exc:
        lpips_eval_model = None
        log_for_0(f"LPIPS eval model unavailable: {exc}")

    if bool(getattr(eval_cfg, "compute_fid", False)):
        cache_ref = getattr(config.fid, "cache_ref", "")
        if cache_ref and os.path.exists(cache_ref):
            try:
                inception_eval = build_jax_inception(
                    batch_size=int(getattr(config.fid, "device_batch_size", 40))
                )
                fid_ref = get_reference(cache_ref)
            except Exception as exc:
                inception_eval = None
                fid_ref = None
                log_for_0(f"FID evaluator init failed: {exc}")
        else:
            log_for_0("FID cache_ref missing; skip FID during validation.")

    return semantic_encoder_metric, lpips_eval_model, inception_eval, fid_ref


def _preflight_fid_is_memory_smoke(config, eval_cfg, device):
    """
    Minimal GPU-memory smoke test for FID/IS stage.
    Does not run the full validation loop; it only instantiates Inception and runs
    a tiny synthetic batch through the FID/IS statistics path to estimate peak
    memory and catch OOMs early.
    """
    if not (bool(getattr(eval_cfg, "compute_fid", False)) or bool(getattr(eval_cfg, "compute_is", False))):
        return {
            "enabled": False,
            "peak_mem_mb": 0.0,
            "oom": False,
            "reason": "compute_fid/compute_is both disabled",
        }
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "enabled": False,
            "peak_mem_mb": 0.0,
            "oom": False,
            "reason": "cuda unavailable",
        }

    fid_bs_cfg = int(getattr(config.fid, "device_batch_size", 40))
    fid_bs = int(getattr(eval_cfg, "preflight_fid_device_batch_size", 0)) or fid_bs_cfg
    num_images = int(getattr(eval_cfg, "preflight_fid_num_images", 0)) or fid_bs
    num_images = max(1, int(num_images))
    fid_bs = max(1, int(fid_bs))
    image_size = int(getattr(config.dataset, "image_size", 256))

    inception_eval = None
    peak_mem_mb = 0.0
    try:
        _cleanup_cuda_memory()
        _cuda_reset_peak_memory(device)

        inception_eval = build_jax_inception(batch_size=fid_bs)
        # Random uint8 images are sufficient to measure Inception/FID/IS stage peak
        # memory; this stage is independent from semantic metric models and does not
        # require realistic samples for an OOM smoke test.
        images = np.random.randint(
            0,
            256,
            size=(num_images, image_size, image_size, 3),
            dtype=np.uint8,
        )
        stats_gen = compute_stats(images, inception_eval)
        if bool(getattr(eval_cfg, "compute_is", False)):
            compute_inception_score(stats_gen["logits"])

        peak_mem_mb = _cuda_peak_memory_mb(device)
        return {
            "enabled": True,
            "peak_mem_mb": peak_mem_mb,
            "oom": False,
            "fid_device_batch_size": fid_bs,
            "num_images": num_images,
        }
    except torch.cuda.OutOfMemoryError:
        peak_mem_mb = _cuda_peak_memory_mb(device)
        return {
            "enabled": True,
            "peak_mem_mb": peak_mem_mb,
            "oom": True,
            "fid_device_batch_size": fid_bs,
            "num_images": num_images,
        }
    finally:
        inception_eval = None
        _cleanup_cuda_memory()


def _create_eval_subset_loader(dataset_cfg, batch_size: int, max_samples: int, semantic_cache=None):
    dataset = input_pipeline.build_imagenet_dataset(dataset_cfg, "val")
    if semantic_cache is not None:
        semantic_cache.validate_against_imagefolder(split="val", imagefolder_dataset=dataset)
    if max_samples > 0:
        n = min(max_samples, len(dataset))
        dataset = Subset(dataset, list(range(n)))
    if semantic_cache is not None:
        dataset = input_pipeline.IndexedDatasetWrapper(dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=int(getattr(dataset_cfg, "num_workers", 0)),
        pin_memory=bool(getattr(dataset_cfg, "pin_memory", True)),
        prefetch_factor=(
            int(getattr(dataset_cfg, "prefetch_factor", 2))
            if int(getattr(dataset_cfg, "num_workers", 0)) > 0
            else None
        ),
        persistent_workers=True if int(getattr(dataset_cfg, "num_workers", 0)) > 0 else False,
    )
    return loader, len(dataset)


def _to_uint8_bhwc(x_nchw: torch.Tensor):
    x = ((x_nchw.detach().float().cpu() + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return x.permute(0, 2, 3, 1).contiguous().numpy()


def _make_comparison_grid(tensors_nchw, num_vis=4):
    arrays = [_to_uint8_bhwc(t) for t in tensors_nchw]
    n = min(num_vis, arrays[0].shape[0])
    rows = []
    for i in range(n):
        rows.append(np.concatenate([arr[i] for arr in arrays], axis=1))
    return np.concatenate(rows, axis=0)


def _cosine_np(a, b, eps=1e-8):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), eps, None)
    b = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), eps, None)
    return np.sum(a * b, axis=-1)


def _stats_from_list(values):
    if not values:
        return np.nan, np.nan, np.nan
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(arr)), float(np.quantile(arr, 0.5)), float(np.quantile(arr, 0.9))


def _build_bin_key(t_val: float, rho_val: float):
    if t_val < 0.5:
        t_bin = "t_0_0p5"
    elif t_val < 0.8:
        t_bin = "t_0p5_0p8"
    else:
        t_bin = "t_0p8_1p0"

    if abs(rho_val) <= 1e-6:
        rho_bin = "rho_0"
    elif rho_val <= 0.5:
        rho_bin = "rho_0_0p5"
    elif rho_val <= 0.8:
        rho_bin = "rho_0p5_0p8"
    else:
        rho_bin = "rho_0p8_1p0"
    return t_bin, rho_bin


def _build_probe_rho_schedule(eval_cfg):
    total = max(1, int(eval_cfg.probe_pairs_per_t))
    target_ratio = float(max(0.0, min(1.0, eval_cfg.probe_rho_zero_ratio)))
    zero_count = int(round(total * target_ratio))
    zero_count = max(0, min(total, zero_count))
    nonzero_count = total - zero_count

    nonzero_vals = [float(v) for v in list(eval_cfg.probe_nonzero_rho_values) if float(v) > 0.0]
    if not nonzero_vals:
        nonzero_vals = [float(v) for v in list(eval_cfg.probe_rho_values) if float(v) > 0.0]

    schedule = [0.0] * zero_count
    if nonzero_count > 0:
        if not nonzero_vals:
            schedule.extend([0.0] * nonzero_count)
        else:
            schedule.extend([nonzero_vals[i % len(nonzero_vals)] for i in range(nonzero_count)])

    rng = np.random.default_rng(int(eval_cfg.seed))
    schedule = np.asarray(schedule, dtype=np.float32)
    rng.shuffle(schedule)
    return schedule.tolist()


def _compute_psnr_mse(pred_x, gt_x):
    pred01 = ((pred_x + 1.0) * 0.5).clamp(0.0, 1.0)
    gt01 = ((gt_x + 1.0) * 0.5).clamp(0.0, 1.0)
    mse = torch.mean((pred01 - gt01) ** 2, dim=(1, 2, 3))
    psnr = -10.0 * torch.log10(torch.clamp(mse, min=1e-8))
    return mse.detach().cpu().numpy(), psnr.detach().cpu().numpy()


def _compute_ssim(pred_x, gt_x, window_size: int = 11, sigma: float = 1.5):
    pred = ((pred_x + 1.0) * 0.5).clamp(0.0, 1.0).float()
    gt = ((gt_x + 1.0) * 0.5).clamp(0.0, 1.0).float()
    c = pred.shape[1]
    ws = int(window_size)
    half = ws // 2

    coords = torch.arange(ws, device=pred.device, dtype=pred.dtype) - half
    gauss_1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    gauss_1d = gauss_1d / torch.clamp(gauss_1d.sum(), min=1e-8)
    kernel_2d = torch.outer(gauss_1d, gauss_1d)
    kernel = kernel_2d.view(1, 1, ws, ws).repeat(c, 1, 1, 1)

    mu_pred = F.conv2d(pred, kernel, padding=half, groups=c)
    mu_gt = F.conv2d(gt, kernel, padding=half, groups=c)
    mu_pred_sq = mu_pred * mu_pred
    mu_gt_sq = mu_gt * mu_gt
    mu_pred_gt = mu_pred * mu_gt

    sigma_pred_sq = F.conv2d(pred * pred, kernel, padding=half, groups=c) - mu_pred_sq
    sigma_gt_sq = F.conv2d(gt * gt, kernel, padding=half, groups=c) - mu_gt_sq
    sigma_pred_gt = F.conv2d(pred * gt, kernel, padding=half, groups=c) - mu_pred_gt

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2.0 * mu_pred_gt + c1) * (2.0 * sigma_pred_gt + c2)
    denominator = (mu_pred_sq + mu_gt_sq + c1) * (sigma_pred_sq + sigma_gt_sq + c2)
    ssim_map = numerator / torch.clamp(denominator, min=1e-8)
    ssim = torch.mean(ssim_map, dim=(1, 2, 3))
    return ssim.detach().cpu().numpy().astype(np.float32)


def _compute_frequency_metrics(
    pred_x,
    gt_x,
    hf_cutoff: float = 0.5,
    angle_bins: int = 16,
    radial_bins: int = 24,
    eps: float = 1e-8,
):
    pred = ((pred_x + 1.0) * 0.5).clamp(0.0, 1.0).float().mean(dim=1)  # [B,H,W]
    gt = ((gt_x + 1.0) * 0.5).clamp(0.0, 1.0).float().mean(dim=1)
    bsz, h, w = pred.shape
    device = pred.device

    f_pred = torch.fft.fft2(pred, norm="ortho")
    f_gt = torch.fft.fft2(gt, norm="ortho")
    mag_pred = torch.abs(f_pred)
    mag_gt = torch.abs(f_gt)
    power_pred = mag_pred ** 2
    power_gt = mag_gt ** 2
    phase_pred = torch.angle(f_pred)
    phase_gt = torch.angle(f_gt)

    fy = torch.fft.fftfreq(h, d=1.0, device=device)
    fx = torch.fft.fftfreq(w, d=1.0, device=device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    rr = torch.sqrt(xx ** 2 + yy ** 2)
    rr_norm = rr / torch.clamp(torch.max(rr), min=eps)
    theta = torch.atan2(yy, xx)

    hf_mask = rr_norm >= float(hf_cutoff)
    non_dc_mask = rr > 0
    hf_mask = hf_mask & non_dc_mask

    # High-frequency magnitude coherence.
    pred_hf = mag_pred[:, hf_mask]
    gt_hf = mag_gt[:, hf_mask]
    coh_num = torch.sum(pred_hf * gt_hf, dim=1)
    coh_den = torch.sqrt(torch.sum(pred_hf * pred_hf, dim=1) * torch.sum(gt_hf * gt_hf, dim=1) + eps)
    coh_hf = coh_num / torch.clamp(coh_den, min=eps)

    # High-frequency phase coherence, weighted by GT magnitude.
    delta_hf = phase_pred[:, hf_mask] - phase_gt[:, hf_mask]
    w_hf = gt_hf
    phc_hf = torch.sum(torch.cos(delta_hf) * w_hf, dim=1) / torch.clamp(torch.sum(w_hf, dim=1), min=eps)

    # Directional anisotropy ratio in high frequency.
    a_bins = max(4, int(angle_bins))
    angle_edges = torch.linspace(-math.pi, math.pi, a_bins + 1, device=device)
    angle_masks = []
    for i in range(a_bins):
        mask_i = (theta >= angle_edges[i]) & (theta < angle_edges[i + 1]) & hf_mask
        angle_masks.append(mask_i)
    angle_masks = torch.stack(angle_masks, dim=0).float()  # [A,H,W]
    angle_counts = torch.clamp(torch.sum(angle_masks, dim=(1, 2)), min=1.0)  # [A]

    pred_dir = torch.sum(power_pred.unsqueeze(1) * angle_masks.unsqueeze(0), dim=(2, 3)) / angle_counts.unsqueeze(0)
    gt_dir = torch.sum(power_gt.unsqueeze(1) * angle_masks.unsqueeze(0), dim=(2, 3)) / angle_counts.unsqueeze(0)
    anis_pred = (torch.max(pred_dir, dim=1).values + eps) / (torch.min(pred_dir, dim=1).values + eps)
    anis_gt = (torch.max(gt_dir, dim=1).values + eps) / (torch.min(gt_dir, dim=1).values + eps)
    anis_delta = anis_pred - anis_gt

    # Radial PSD slope (log-power vs log-frequency).
    r_bins = max(8, int(radial_bins))
    radial_edges = torch.linspace(0.0, 1.0, r_bins + 1, device=device)
    radial_masks = []
    radial_centers = []
    for i in range(r_bins):
        r0, r1 = radial_edges[i], radial_edges[i + 1]
        mask_i = (rr_norm >= r0) & (rr_norm < r1) & non_dc_mask
        radial_masks.append(mask_i)
        radial_centers.append(0.5 * (r0 + r1))
    radial_masks = torch.stack(radial_masks, dim=0).float()  # [R,H,W]
    radial_counts = torch.sum(radial_masks, dim=(1, 2))
    valid = radial_counts > 0
    radial_masks = radial_masks[valid]
    radial_counts = torch.clamp(radial_counts[valid], min=1.0)
    x_log = torch.log(torch.clamp(torch.stack(radial_centers)[valid], min=eps))

    pred_rad = torch.sum(power_pred.unsqueeze(1) * radial_masks.unsqueeze(0), dim=(2, 3)) / radial_counts.unsqueeze(0)
    gt_rad = torch.sum(power_gt.unsqueeze(1) * radial_masks.unsqueeze(0), dim=(2, 3)) / radial_counts.unsqueeze(0)
    y_pred = torch.log(torch.clamp(pred_rad, min=eps))
    y_gt = torch.log(torch.clamp(gt_rad, min=eps))

    x_center = x_log - torch.mean(x_log)
    denom = torch.sum(x_center * x_center) + eps
    slope_pred = torch.sum((y_pred - torch.mean(y_pred, dim=1, keepdim=True)) * x_center.view(1, -1), dim=1) / denom
    slope_gt = torch.sum((y_gt - torch.mean(y_gt, dim=1, keepdim=True)) * x_center.view(1, -1), dim=1) / denom
    slope_delta = slope_pred - slope_gt

    return {
        "coh_hf": coh_hf.detach().cpu().numpy().astype(np.float32),
        "phc_hf": phc_hf.detach().cpu().numpy().astype(np.float32),
        "anisotropy_ratio_pred": anis_pred.detach().cpu().numpy().astype(np.float32),
        "anisotropy_ratio_gt": anis_gt.detach().cpu().numpy().astype(np.float32),
        "anisotropy_ratio_delta": anis_delta.detach().cpu().numpy().astype(np.float32),
        "psd_slope_pred": slope_pred.detach().cpu().numpy().astype(np.float32),
        "psd_slope_gt": slope_gt.detach().cpu().numpy().astype(np.float32),
        "psd_slope_delta": slope_delta.detach().cpu().numpy().astype(np.float32),
    }


def _compute_lpips(lpips_model, pred_x, gt_x, resize=224):
    if lpips_model is None:
        return np.full((pred_x.shape[0],), np.nan, dtype=np.float32)
    pred = F.interpolate(pred_x.float(), size=(resize, resize), mode="bicubic", align_corners=False, antialias=True)
    gt = F.interpolate(gt_x.float(), size=(resize, resize), mode="bicubic", align_corners=False, antialias=True)
    with torch.no_grad():
        out = lpips_model(pred, gt, normalize=False).reshape(pred.shape[0], -1).mean(dim=1)
    return out.detach().cpu().numpy().astype(np.float32)


def _compute_semantic_metrics(semantic_encoder, pred_x, gt_x, gt_sem_targets=None):
    if semantic_encoder is None:
        nan_vec = np.full((pred_x.shape[0],), np.nan, dtype=np.float32)
        return nan_vec, nan_vec, None, None

    pred_u8 = _to_uint8_bhwc(pred_x)
    clip_pred, dino_pred = semantic_encoder.encode(pred_u8)

    clip_gt = None
    dino_gt = None
    if isinstance(gt_sem_targets, dict):
        clip_cached = gt_sem_targets.get("clip", None)
        dino_cached = gt_sem_targets.get("dino", None)
        if clip_cached is not None:
            if torch.is_tensor(clip_cached):
                clip_cached = clip_cached.detach().cpu().numpy()
            clip_gt = np.asarray(clip_cached, dtype=np.float32)
            if clip_gt.ndim == 1:
                clip_gt = clip_gt[None, :]
        if dino_cached is not None:
            if torch.is_tensor(dino_cached):
                dino_cached = dino_cached.detach().cpu().numpy()
            dino_gt = np.asarray(dino_cached, dtype=np.float32)
            if dino_gt.ndim == 2:
                dino_gt = dino_gt[:, None, :]

    need_clip_gt = clip_pred is not None and clip_gt is None
    need_dino_gt = dino_pred is not None and dino_gt is None
    if need_clip_gt or need_dino_gt:
        gt_u8 = _to_uint8_bhwc(gt_x)
        clip_gt_online, dino_gt_online = semantic_encoder.encode(gt_u8)
        if need_clip_gt:
            clip_gt = clip_gt_online
        if need_dino_gt:
            dino_gt = dino_gt_online

    if clip_pred is not None and clip_gt is not None:
        clip_cos = _cosine_np(clip_pred, clip_gt).astype(np.float32)
    else:
        clip_cos = np.full((pred_x.shape[0],), np.nan, dtype=np.float32)

    if dino_pred is not None and dino_gt is not None:
        dino_pred_global = dino_pred.mean(axis=1) if dino_pred.ndim == 3 else dino_pred
        dino_gt_global = dino_gt.mean(axis=1) if dino_gt.ndim == 3 else dino_gt
        dino_cos = _cosine_np(dino_pred_global, dino_gt_global).astype(np.float32)
    else:
        dino_pred_global = None
        dino_cos = np.full((pred_x.shape[0],), np.nan, dtype=np.float32)

    return clip_cos, dino_cos, clip_pred, dino_pred_global


def _run_probe(
    model_core,
    x,
    cond_embeddings,
    t_value,
    rho_value,
    omega_value,
    t_min_value,
    t_max_value,
    noise,
    force_null_condition=False,
):
    bsz = x.shape[0]
    device = x.device
    dtype = x.dtype

    t_scalar = float(max(min(t_value, 1.0), 0.0))
    rho_scalar = float(max(min(rho_value, 1.0), 0.0))
    h_scalar = t_scalar * rho_scalar
    r_scalar = t_scalar - h_scalar

    t = torch.full((bsz, 1, 1, 1), t_scalar, device=device, dtype=dtype)
    r = torch.full((bsz, 1, 1, 1), r_scalar, device=device, dtype=dtype)
    h = t - r
    z_t = (1.0 - t) * x + t * noise
    v_t = (z_t - x) / torch.clamp(t, min=0.05)

    cond_embeddings = model_core.normalize_condition(cond_embeddings, bsz, device=device, dtype=dtype)
    null_cond = model_core.null_condition(bsz, device=device, dtype=dtype)
    if force_null_condition:
        cond_embeddings = null_cond

    high_noise_mask = t.reshape(-1) >= model_core.schedule_t0
    clip_cond = cond_embeddings["clip"] if model_core.use_clip_condition else null_cond["clip"]
    dino_cond = cond_embeddings["dino"] if model_core.use_dino_condition else null_cond["dino"]
    cond_embeddings = {
        "clip": torch.where(high_noise_mask[:, None], clip_cond, null_cond["clip"]),
        "dino": torch.where(high_noise_mask[:, None, None], dino_cond, null_cond["dino"]),
    }

    omega = torch.full((bsz,), float(omega_value), device=device, dtype=dtype)
    t_min = torch.full((bsz,), float(t_min_value), device=device, dtype=dtype)
    t_max = torch.full((bsz,), float(t_max_value), device=device, dtype=dtype)
    fm_mask = (h.reshape(-1) <= 1e-8).view(-1, 1, 1, 1)

    with torch.enable_grad():
        v_g, v_c = model_core.guidance_fn(
            v_t,
            z_t,
            t.reshape(-1),
            r.reshape(-1),
            cond_embeddings,
            fm_mask,
            omega,
            t_min,
            t_max,
        )

        def u_only(z_in, t_in, r_in):
            u_out = model_core.u_fn(
                z_in,
                t_in.reshape(-1),
                (t_in - r_in).reshape(-1),
                omega,
                t_min,
                t_max,
                y=cond_embeddings,
            )[0]
            return u_out

        dtdt = torch.ones_like(t)
        dtdr = torch.zeros_like(t)
        u, v = model_core.u_fn(
            z_t,
            t.reshape(-1),
            (t - r).reshape(-1),
            omega,
            t_min,
            t_max,
            y=cond_embeddings,
        )
        _, du_dt = torch.autograd.functional.jvp(
            u_only,
            (z_t, t, r),
            (v_c, dtdt, dtdr),
            create_graph=False,
            strict=False,
        )

    V = u + h * du_dt.detach()
    pred_x = z_t - t * u
    v_g = v_g.detach()
    rho = torch.clamp(h.reshape(-1) / torch.clamp(t.reshape(-1), min=1e-6), 0.0, 1.0)

    out = {
        "pred_x": pred_x.detach(),
        "z_t": z_t.detach(),
        "t": t.reshape(-1).detach(),
        "rho": rho.detach(),
        "mse_V_vg": torch.mean((V - v_g) ** 2, dim=(1, 2, 3)).detach(),
        "mse_v_vg": torch.mean((v - v_g) ** 2, dim=(1, 2, 3)).detach(),
        "norm_u2": torch.mean(u ** 2, dim=(1, 2, 3)).detach(),
        "norm_V2": torch.mean(V ** 2, dim=(1, 2, 3)).detach(),
        "norm_du_dt2": torch.mean(du_dt ** 2, dim=(1, 2, 3)).detach(),
        "norm_h_du_dt2": torch.mean((h * du_dt) ** 2, dim=(1, 2, 3)).detach(),
    }
    return out


def _run_validation_cycle(
    model_core,
    config,
    eval_cfg,
    eval_loader,
    device,
    semantic_encoder_cond,
    semantic_cache_cond,
    semantic_encoder_metric,
    lpips_model,
    inception_net=None,
    fid_ref=None,
    use_clip_condition=True,
    use_dino_condition=True,
):
    was_training = bool(model_core.training)
    model_core.eval()

    generator = torch.Generator(device=("cuda" if device.type == "cuda" else "cpu"))
    generator.manual_seed(int(eval_cfg.seed))

    metric_lists = defaultdict(list)
    binned_lists = defaultdict(list)
    images_for_quality = []
    vis_images = {}
    probe_rho_schedule = _build_probe_rho_schedule(eval_cfg)
    rho0_ratio_actual = float(np.mean(np.isclose(np.asarray(probe_rho_schedule), 0.0)))
    metric_lists["probe/rho0_ratio_actual"].append(rho0_ratio_actual)
    metric_lists["probe/rho0_ratio_target"].append(float(eval_cfg.probe_rho_zero_ratio))

    for batch_idx, batch in enumerate(eval_loader):
        batch_data = input_pipeline.prepare_batch_data(
            batch,
            semantic_encoder=semantic_encoder_cond,
            semantic_cache=semantic_cache_cond,
            semantic_cache_split="val",
            clip_feature_dim=int(config.model.clip_feature_dim),
            dino_feature_dim=int(config.model.dino_feature_dim),
            num_dino_tokens=int(getattr(config.model, "num_dino_tokens", 4)),
            use_clip_condition=use_clip_condition,
            use_dino_condition=use_dino_condition,
            use_flip=False,
        )
        batch_data = _to_device_batch(batch_data, device=device)
        x = batch_data["image"]
        gt_sem_metric_targets = None
        if semantic_encoder_metric is not None:
            gt_sem_metric_targets = {}
            if use_clip_condition and batch_data.get("clip_emb", None) is not None:
                gt_sem_metric_targets["clip"] = batch_data["clip_emb"]
            if use_dino_condition and batch_data.get("dino_emb", None) is not None:
                gt_sem_metric_targets["dino"] = batch_data["dino_emb"]
            if not gt_sem_metric_targets:
                gt_sem_metric_targets = None
        if use_clip_condition or use_dino_condition:
            cond_embeddings = {"clip": batch_data["clip_emb"], "dino": batch_data["dino_emb"]}
        else:
            cond_embeddings = None

        # Probe sweep for t/rho bins and training-target diagnostics.
        for t_probe in eval_cfg.probe_t_values:
            for rho_probe in probe_rho_schedule:
                noise_probe = torch.randn(
                    x.shape, generator=generator, device=device, dtype=x.dtype
                ) * model_core.noise_scale
                probe = _run_probe(
                    model_core=model_core,
                    x=x,
                    cond_embeddings=cond_embeddings,
                    t_value=float(t_probe),
                    rho_value=float(rho_probe),
                    omega_value=float(eval_cfg.cfg_w_mid),
                    t_min_value=float(eval_cfg.interval_on[0]),
                    t_max_value=float(eval_cfg.interval_on[1]),
                    noise=noise_probe,
                    force_null_condition=False,
                )
                t_bin, rho_bin = _build_bin_key(float(t_probe), float(rho_probe))
                for key in (
                    "mse_V_vg",
                    "mse_v_vg",
                    "norm_u2",
                    "norm_V2",
                    "norm_du_dt2",
                    "norm_h_du_dt2",
                ):
                    vals = probe[key].detach().cpu().numpy()
                    metric_lists[f"probe/{key}"].extend(vals.tolist())
                    binned_lists[f"probe/{key}/{t_bin}/{rho_bin}"].extend(vals.tolist())

        # Low-noise purification probe.
        noise_low = torch.randn(x.shape, generator=generator, device=device, dtype=x.dtype) * model_core.noise_scale
        probe_low = _run_probe(
            model_core=model_core,
            x=x,
            cond_embeddings=cond_embeddings,
            t_value=float(eval_cfg.low_t),
            rho_value=float(eval_cfg.low_rho),
            omega_value=float(eval_cfg.cfg_w_mid),
            t_min_value=float(eval_cfg.interval_on[0]),
            t_max_value=float(eval_cfg.interval_on[1]),
            noise=noise_low,
            force_null_condition=False,
        )
        low_mse, low_psnr = _compute_psnr_mse(probe_low["pred_x"], x)
        low_lpips = _compute_lpips(lpips_model, probe_low["pred_x"], x, resize=int(eval_cfg.lpips_resize))
        low_ssim = _compute_ssim(probe_low["pred_x"], x)
        low_freq = _compute_frequency_metrics(
            probe_low["pred_x"],
            x,
            hf_cutoff=float(eval_cfg.freq_hf_cutoff),
            angle_bins=int(eval_cfg.freq_angle_bins),
            radial_bins=int(eval_cfg.freq_radial_bins),
        )
        metric_lists["low/mse"].extend(low_mse.tolist())
        metric_lists["low/psnr"].extend(low_psnr.tolist())
        metric_lists["low/lpips"].extend(low_lpips.tolist())
        metric_lists["low/ssim"].extend(low_ssim.tolist())
        for key, values in low_freq.items():
            metric_lists[f"low/freq/{key}"].extend(values.tolist())

        # High-noise, same noise for controlled comparisons.
        noise_high = torch.randn(x.shape, generator=generator, device=device, dtype=x.dtype) * model_core.noise_scale

        probe_w1 = _run_probe(
            model_core, x, cond_embeddings,
            t_value=float(eval_cfg.high_t),
            rho_value=float(eval_cfg.high_rho),
            omega_value=1.0,
            t_min_value=float(eval_cfg.interval_on[0]),
            t_max_value=float(eval_cfg.interval_on[1]),
            noise=noise_high,
            force_null_condition=False,
        )
        probe_wmid = _run_probe(
            model_core, x, cond_embeddings,
            t_value=float(eval_cfg.high_t),
            rho_value=float(eval_cfg.high_rho),
            omega_value=float(eval_cfg.cfg_w_mid),
            t_min_value=float(eval_cfg.interval_on[0]),
            t_max_value=float(eval_cfg.interval_on[1]),
            noise=noise_high,
            force_null_condition=False,
        )
        probe_whigh = _run_probe(
            model_core, x, cond_embeddings,
            t_value=float(eval_cfg.high_t),
            rho_value=float(eval_cfg.high_rho),
            omega_value=float(eval_cfg.cfg_w_high),
            t_min_value=float(eval_cfg.interval_on[0]),
            t_max_value=float(eval_cfg.interval_on[1]),
            noise=noise_high,
            force_null_condition=False,
        )

        for name, probe in (
            ("w1", probe_w1),
            ("wmid", probe_wmid),
            ("whigh", probe_whigh),
        ):
            clip_cos, dino_cos, clip_feat, _ = _compute_semantic_metrics(
                semantic_encoder_metric, probe["pred_x"], x, gt_sem_targets=gt_sem_metric_targets
            )
            mse, psnr = _compute_psnr_mse(probe["pred_x"], x)
            lpips_arr = _compute_lpips(lpips_model, probe["pred_x"], x, resize=int(eval_cfg.lpips_resize))
            ssim_arr = _compute_ssim(probe["pred_x"], x)
            metric_lists[f"high/{name}/clip_cos"].extend(clip_cos.tolist())
            metric_lists[f"high/{name}/dino_cos"].extend(dino_cos.tolist())
            metric_lists[f"high/{name}/mse"].extend(mse.tolist())
            metric_lists[f"high/{name}/psnr"].extend(psnr.tolist())
            metric_lists[f"high/{name}/lpips"].extend(lpips_arr.tolist())
            metric_lists[f"high/{name}/ssim"].extend(ssim_arr.tolist())
            if clip_feat is not None:
                metric_lists[f"high/{name}/clip_var"].append(float(np.var(clip_feat, axis=0).mean()))
            if name == "whigh":
                high_freq = _compute_frequency_metrics(
                    probe["pred_x"],
                    x,
                    hf_cutoff=float(eval_cfg.freq_hf_cutoff),
                    angle_bins=int(eval_cfg.freq_angle_bins),
                    radial_bins=int(eval_cfg.freq_radial_bins),
                )
                for key, values in high_freq.items():
                    metric_lists[f"high/{name}/freq/{key}"].extend(values.tolist())

        clip_w1 = np.asarray(metric_lists["high/w1/clip_cos"][-x.shape[0]:], dtype=np.float32)
        clip_wmid = np.asarray(metric_lists["high/wmid/clip_cos"][-x.shape[0]:], dtype=np.float32)
        clip_whigh = np.asarray(metric_lists["high/whigh/clip_cos"][-x.shape[0]:], dtype=np.float32)
        monotonic = (clip_w1 <= clip_wmid) & (clip_wmid <= clip_whigh)
        metric_lists["high/clip_monotonic_ratio"].append(float(np.mean(monotonic.astype(np.float32))))

        # Interval on/off test.
        probe_interval_on = probe_whigh
        probe_interval_off = _run_probe(
            model_core, x, cond_embeddings,
            t_value=float(eval_cfg.high_t),
            rho_value=float(eval_cfg.high_rho),
            omega_value=float(eval_cfg.cfg_w_high),
            t_min_value=float(eval_cfg.interval_off[0]),
            t_max_value=float(eval_cfg.interval_off[1]),
            noise=noise_high,
            force_null_condition=False,
        )
        clip_on, dino_on, _, _ = _compute_semantic_metrics(
            semantic_encoder_metric, probe_interval_on["pred_x"], x, gt_sem_targets=gt_sem_metric_targets
        )
        clip_off, dino_off, _, _ = _compute_semantic_metrics(
            semantic_encoder_metric, probe_interval_off["pred_x"], x, gt_sem_targets=gt_sem_metric_targets
        )
        metric_lists["interval/delta_clip_sem"].extend((clip_on - clip_off).tolist())
        metric_lists["interval/delta_dino_sem"].extend((dino_on - dino_off).tolist())

        # Condition vs null test in high-noise.
        probe_cond = probe_whigh
        probe_null = _run_probe(
            model_core, x, cond_embeddings,
            t_value=float(eval_cfg.high_t),
            rho_value=float(eval_cfg.high_rho),
            omega_value=float(eval_cfg.cfg_w_high),
            t_min_value=float(eval_cfg.interval_on[0]),
            t_max_value=float(eval_cfg.interval_on[1]),
            noise=noise_high,
            force_null_condition=True,
        )
        clip_cond, dino_cond, _, _ = _compute_semantic_metrics(
            semantic_encoder_metric, probe_cond["pred_x"], x, gt_sem_targets=gt_sem_metric_targets
        )
        clip_null, dino_null, _, _ = _compute_semantic_metrics(
            semantic_encoder_metric, probe_null["pred_x"], x, gt_sem_targets=gt_sem_metric_targets
        )
        metric_lists["cond/delta_clip_sem"].extend((clip_cond - clip_null).tolist())
        metric_lists["cond/delta_dino_sem"].extend((dino_cond - dino_null).tolist())

        images_for_quality.append(_to_uint8_bhwc(probe_whigh["pred_x"]))

        if batch_idx == 0:
            vis_images["eval_compare_low"] = _make_comparison_grid(
                [x, probe_low["z_t"], probe_low["pred_x"]],
                num_vis=int(eval_cfg.num_vis),
            )
            vis_images["eval_compare_high_w"] = _make_comparison_grid(
                [x, probe_w1["z_t"], probe_w1["pred_x"], probe_wmid["pred_x"], probe_whigh["pred_x"]],
                num_vis=int(eval_cfg.num_vis),
            )
            vis_images["eval_compare_interval"] = _make_comparison_grid(
                [x, probe_interval_on["pred_x"], probe_interval_off["pred_x"]],
                num_vis=int(eval_cfg.num_vis),
            )
            vis_images["eval_compare_condnull"] = _make_comparison_grid(
                [x, probe_cond["pred_x"], probe_null["pred_x"]],
                num_vis=int(eval_cfg.num_vis),
            )

    out = {}
    for k, vals in metric_lists.items():
        mean, p50, p90 = _stats_from_list(vals)
        out[f"eval/{k}/mean"] = mean
        out[f"eval/{k}/p50"] = p50
        out[f"eval/{k}/p90"] = p90
    for k, vals in binned_lists.items():
        mean, p50, p90 = _stats_from_list(vals)
        out[f"eval_bins/{k}/mean"] = mean
        out[f"eval_bins/{k}/p50"] = p50
        out[f"eval_bins/{k}/p90"] = p90

    # FID/IS on high-noise w_high outputs.
    if images_for_quality and eval_cfg.compute_fid and inception_net is not None and fid_ref is not None:
        generated = np.concatenate(images_for_quality, axis=0)
        stats_gen = compute_stats(generated, inception_net)
        out["eval/high/quality/fid"] = compute_fid(
            fid_ref["mu"], stats_gen["mu"], fid_ref["sigma"], stats_gen["sigma"]
        )
        if eval_cfg.compute_is:
            is_mean, is_std = compute_inception_score(stats_gen["logits"])
            out["eval/high/quality/is_mean"] = float(is_mean)
            out["eval/high/quality/is_std"] = float(is_std)

    # Composite score for best-checkpoint selection.
    low_psnr = out.get("eval/low/psnr/mean", np.nan)
    low_lpips = out.get("eval/low/lpips/mean", np.nan)
    if np.isfinite(low_psnr) and np.isfinite(low_lpips):
        out["eval/score/lpips_psnr"] = float(
            low_psnr - float(eval_cfg.best_lpips_psnr_tradeoff) * low_lpips
        )

    if was_training:
        model_core.train()
    else:
        model_core.eval()
    return out, vis_images


def train_and_evaluate(config, workdir: str):
    rank, world_size, device = init_distributed(config=config)
    if rank == 0:
        os.makedirs(workdir, exist_ok=True)

    seed = int(config.training.seed) + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    writer = Writer(config, workdir)

    global_batch_size = int(config.training.batch_size)
    if global_batch_size % world_size != 0:
        raise ValueError(f"batch_size={global_batch_size} is not divisible by world_size={world_size}.")
    local_batch_size = global_batch_size // world_size
    log_for_0(f"World size={world_size}, local_batch_size={local_batch_size}, device={device}")

    condition_cfg = config.condition if hasattr(config, "condition") else None
    condition_mode, use_clip_condition, use_dino_condition = input_pipeline.resolve_condition_mode(
        condition_cfg
    )
    semantic_cache_train = None
    if use_clip_condition or use_dino_condition:
        try:
            semantic_cache_train = input_pipeline.create_semantic_cache(
                condition_cfg=condition_cfg,
                model_cfg=config.model,
                dataset_cfg=config.dataset,
            )
            if semantic_cache_train is not None:
                log_for_0(
                    "Semantic condition cache enabled "
                    f"(root={semantic_cache_train.cache_root}, mode={condition_mode})"
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize semantic mmap cache: {exc}") from exc

    train_loader, steps_per_epoch = input_pipeline.create_imagenet_split(
        config.dataset,
        local_batch_size,
        split="train",
        semantic_cache=semantic_cache_train,
    )
    sem_loss_enabled = (
        bool(getattr(config.model, "enable_semantic_loss", True))
        and getattr(config.model, "lambda_sem_max", 0.0) > 0.0
        and (use_clip_condition or use_dino_condition)
    )
    semantic_encoder = input_pipeline.create_semantic_encoder(
        condition_cfg,
        num_dino_tokens=int(getattr(config.model, "num_dino_tokens", 4)),
        use_clip=use_clip_condition,
        use_dino=use_dino_condition,
    )
    eval_cfg = _get_eval_cfg(config)

    model_cfg = deepcopy(config.model.to_dict() if hasattr(config.model, "to_dict") else dict(config.model))
    model_cfg["condition_mode"] = condition_mode
    model_cfg["enable_semantic_loss"] = sem_loss_enabled
    model = pixelMeanFlow(**model_cfg).to(device)
    model_core = model

    if world_size > 1:
        ddp_device_ids = [device.index] if device.type == "cuda" else None
        model = DDP(model, device_ids=ddp_device_ids, find_unused_parameters=False)
        model_core = model.module

    optimizer_name = str(getattr(config.training, "optimizer", "adamw")).lower()
    lr = float(config.training.learning_rate)
    adam_b2 = float(getattr(config.training, "adam_b2", 0.95))
    weight_decay = float(getattr(config.training, "weight_decay", 0.0))

    if optimizer_name == "muon":
        optimizer = _build_muon_optimizer(
            model=model,
            config=config,
            lr=lr,
            weight_decay=weight_decay,
            adam_b2=adam_b2,
        )
    elif optimizer_name == "adamw":
        optim_kwargs = dict(
            lr=lr,
            betas=(0.9, adam_b2),
            weight_decay=weight_decay,
        )
        optimizer = torch.optim.AdamW(model.parameters(), **optim_kwargs)
        log_for_0(f"Optimizer=AdamW(lr={lr}, betas=(0.9, {adam_b2}), weight_decay={weight_decay})")
    else:
        raise ValueError(f"Unsupported training.optimizer={optimizer_name}. Choose from ['muon', 'adamw'].")
    sched_cls, lr_lambda, _ = lr_schedules(config, steps_per_epoch)
    scheduler = sched_cls(optimizer, lr_lambda=lr_lambda)

    ema_vals = config.training.ema_val
    if isinstance(ema_vals, (int, float)):
        ema_vals = [ema_vals]
    ema_models = {float(v): clone_model(model_core).to(device) for v in ema_vals}
    ema_fn = ema_schedules(config)

    start_epoch = 0
    global_step = 0
    load_from = getattr(config, "load_from", "")
    if load_from:
        start_epoch, global_step = _load_checkpoint(
            load_from, model_core, optimizer, scheduler, ema_models
        )
        log_for_0(f"Restored checkpoint from {load_from} at epoch={start_epoch}, step={global_step}")

    use_semantic_aux = sem_loss_enabled
    if config.model.convnext or config.model.lpips or use_semantic_aux:
        aux_fn = init_auxloss(config, semantic_encoder=semantic_encoder, device=device)
    else:
        aux_fn = None
        log_for_0("Aux losses are disabled.")

    eval_loader = None
    topk_clip_high = []
    topk_lpips_psnr = []
    topk_train_loss = []
    eval_enabled_runtime = bool(eval_cfg.enable)
    if eval_enabled_runtime and rank == 0:
        try:
            eval_loader, eval_samples = _create_eval_subset_loader(
                config.dataset,
                batch_size=int(eval_cfg.batch_size),
                max_samples=int(eval_cfg.max_samples),
                semantic_cache=semantic_cache_train,
            )
            log_for_0(f"Validation loader ready with {eval_samples} samples.")
        except Exception as exc:
            eval_enabled_runtime = False
            log_for_0(
                f"Evaluation init failed; disabling runtime evaluation. Error: {exc}"
            )

    # Run one preflight evaluation before training starts.
    # This is a smoke-check only (no tensorboard/images/checkpoint writes).
    if eval_enabled_runtime and rank == 0 and bool(eval_cfg.preflight_check):
        preflight_samples = max(
            1,
            min(
                int(eval_cfg.preflight_max_samples),
                int(eval_cfg.max_samples),
            ),
        )
        preflight_semantic_encoder_metric = None
        preflight_lpips_eval_model = None
        preflight_loader = None
        preflight_cfg = None
        fid_mem = None
        preflight_core_peak_mem_mb = np.nan
        preflight_core_oom = False
        preflight_fid_peak_mem_mb = np.nan
        preflight_fid_oom = False
        try:
            preflight_loader, preflight_n = _create_eval_subset_loader(
                config.dataset,
                batch_size=int(eval_cfg.batch_size),
                max_samples=preflight_samples,
                semantic_cache=semantic_cache_train,
            )
            preflight_cfg = SimpleNamespace(**vars(eval_cfg))
            preflight_cfg.max_samples = int(preflight_n)
            preflight_cfg.compute_fid = False
            preflight_cfg.compute_is = False
            preflight_cfg.num_vis = max(1, int(eval_cfg.num_vis))
            (
                preflight_semantic_encoder_metric,
                preflight_lpips_eval_model,
                _,
                _,
            ) = _init_eval_runtime_models(
                config=config,
                condition_cfg=condition_cfg,
                eval_cfg=preflight_cfg,
                device=device,
            )
            _cuda_reset_peak_memory(device)
            _run_validation_cycle(
                model_core=model_core,
                config=config,
                eval_cfg=preflight_cfg,
                eval_loader=preflight_loader,
                device=device,
                semantic_encoder_cond=semantic_encoder,
                semantic_cache_cond=semantic_cache_train,
                semantic_encoder_metric=preflight_semantic_encoder_metric,
                lpips_model=preflight_lpips_eval_model,
                inception_net=None,
                fid_ref=None,
                use_clip_condition=use_clip_condition,
                use_dino_condition=use_dino_condition,
            )
            preflight_core_peak_mem_mb = _cuda_peak_memory_mb(device)

            if bool(getattr(eval_cfg, "preflight_include_fid_is", False)):
                fid_preflight_cfg = SimpleNamespace(**vars(eval_cfg))
                fid_mem = _preflight_fid_is_memory_smoke(
                    config=config,
                    eval_cfg=fid_preflight_cfg,
                    device=device,
                )
                if fid_mem.get("enabled", False):
                    preflight_fid_peak_mem_mb = float(fid_mem.get("peak_mem_mb", np.nan))
                    preflight_fid_oom = bool(fid_mem.get("oom", False))
                    log_for_0(
                        "Evaluation preflight FID/IS memory smoke: "
                        f"peak_mem_mb={preflight_fid_peak_mem_mb:.1f}, "
                        f"oom={preflight_fid_oom}, "
                        f"fid_device_batch_size={fid_mem.get('fid_device_batch_size')}, "
                        f"num_images={fid_mem.get('num_images')}"
                    )
                    if preflight_fid_oom:
                        raise RuntimeError(
                            "Preflight FID/IS memory smoke OOM; "
                            "reduce config.fid.device_batch_size or disable FID/IS during training eval."
                        )
                else:
                    log_for_0(
                        "Evaluation preflight FID/IS memory smoke skipped: "
                        f"{fid_mem.get('reason', 'unknown')}"
                    )
            log_for_0(
                f"Evaluation preflight passed ({preflight_n} samples). "
                f"core_peak_mem_mb={preflight_core_peak_mem_mb:.1f}, "
                f"fid_peak_mem_mb={preflight_fid_peak_mem_mb:.1f}. "
                "No eval artifacts were written."
            )
        except torch.cuda.OutOfMemoryError as exc:
            eval_enabled_runtime = False
            preflight_core_oom = True
            preflight_core_peak_mem_mb = _cuda_peak_memory_mb(device)
            try:
                if exc.__traceback__ is not None:
                    traceback.clear_frames(exc.__traceback__)
            except Exception:
                pass
            log_for_0(
                "Evaluation preflight failed; disabling runtime evaluation. "
                f"core_peak_mem_mb={preflight_core_peak_mem_mb:.1f}, "
                f"core_oom={preflight_core_oom}, fid_oom={preflight_fid_oom}. "
                "Error: CUDA out of memory"
            )
        except Exception as exc:
            eval_enabled_runtime = False
            try:
                if exc.__traceback__ is not None:
                    traceback.clear_frames(exc.__traceback__)
            except Exception:
                pass
            log_for_0(
                "Evaluation preflight failed; disabling runtime evaluation. "
                f"core_peak_mem_mb={preflight_core_peak_mem_mb:.1f}, "
                f"core_oom={preflight_core_oom}, "
                f"fid_peak_mem_mb={preflight_fid_peak_mem_mb:.1f}, "
                f"fid_oom={preflight_fid_oom}. "
                f"Error: {exc}"
            )
        finally:
            preflight_loader = None
            preflight_cfg = None
            fid_mem = None
            preflight_semantic_encoder_metric = None
            preflight_lpips_eval_model = None
            _cleanup_cuda_memory()

    eval_enabled_runtime = _broadcast_bool_from_rank0(eval_enabled_runtime, device=device)
    if eval_cfg.enable and not eval_enabled_runtime:
        log_for_0("Evaluation is disabled at runtime; training will continue without eval.")

    use_amp = bool(getattr(config.training, "half_precision", False)) and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    metrics_tracker = MetricsTracker()
    timer = Timer()
    model.train()

    for epoch in range(start_epoch, int(config.training.num_epochs)):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        log_for_0(f"epoch {epoch}...")
        timer.reset()
        epoch_metrics_tracker = MetricsTracker()

        for n_batch, batch in enumerate(train_loader):
            batch = input_pipeline.prepare_batch_data(
                batch,
                semantic_encoder=semantic_encoder,
                semantic_cache=semantic_cache_train,
                semantic_cache_split="train",
                clip_feature_dim=int(config.model.clip_feature_dim),
                dino_feature_dim=int(config.model.dino_feature_dim),
                num_dino_tokens=int(getattr(config.model, "num_dino_tokens", 4)),
                use_clip_condition=use_clip_condition,
                use_dino_condition=use_dino_condition,
                use_flip=bool(config.dataset.use_flip),
            )
            batch = _to_device_batch(batch, device=device)
            images = batch["image"]
            if use_clip_condition or use_dino_condition:
                cond_embeddings = {"clip": batch["clip_emb"], "dino": batch["dino_emb"]}
            else:
                cond_embeddings = None

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss, metrics = model(images=images, cond_embeddings=cond_embeddings, aux_fn=aux_fn)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for ema_val, ema_model in ema_models.items():
                    alpha = ema_fn(global_step + 1, ema_val)
                    update_ema(ema_model, model_core, alpha)

            reduced_metrics = _reduce_metrics(metrics)
            reduced_metrics["lr"] = torch.tensor(optimizer.param_groups[0]["lr"], device=device)
            metrics_tracker.update(reduced_metrics)
            epoch_metrics_tracker.update(reduced_metrics)

            if (global_step + 1) % int(config.training.log_per_step) == 0:
                summary = metrics_tracker.finalize()
                summary["steps_per_second"] = float(config.training.log_per_step) / max(timer.elapse_with_reset(), 1e-6)
                summary["epoch"] = float(epoch)
                writer.write_scalars(global_step + 1, summary)

            global_step += 1

        epoch_train_summary = epoch_metrics_tracker.finalize()
        if (not eval_enabled_runtime) and rank == 0 and epoch_train_summary:
            train_loss_key = "train/loss/epoch_mean"
            train_loss_val = float(epoch_train_summary.get("loss", np.nan))
            topk_train_loss = _update_topk_checkpoints(
                workdir=workdir,
                name="train_loss",
                model=model_core,
                metric_name=train_loss_key,
                metric_value=train_loss_val,
                epoch=epoch + 1,
                step=global_step,
                topk_entries=topk_train_loss,
                top_k=int(eval_cfg.best_top_k),
                higher_is_better=False,
            )

        if (epoch + 1) % int(config.training.sample_per_epoch) == 0:
            vis = _generate_visual(model_core, config, device=device, n_sample=16)
            writer.write_images(global_step, {"vis_sample": vis})

        do_eval = eval_enabled_runtime and (
            (epoch + 1) % int(eval_cfg.every_n_epochs) == 0
            or (epoch + 1) == int(config.training.num_epochs)
        )
        if do_eval:
            barrier()
            eval_failed = False
            if rank == 0:
                eval_semantic_encoder_metric = None
                eval_lpips_eval_model = None
                eval_inception = None
                eval_fid_ref = None
                try:
                    (
                        eval_semantic_encoder_metric,
                        eval_lpips_eval_model,
                        eval_inception,
                        eval_fid_ref,
                    ) = _init_eval_runtime_models(
                        config=config,
                        condition_cfg=condition_cfg,
                        eval_cfg=eval_cfg,
                        device=device,
                    )
                    eval_metrics, eval_images = _run_validation_cycle(
                        model_core=model_core,
                        config=config,
                        eval_cfg=eval_cfg,
                        eval_loader=eval_loader,
                        device=device,
                        semantic_encoder_cond=semantic_encoder,
                        semantic_cache_cond=semantic_cache_train,
                        semantic_encoder_metric=eval_semantic_encoder_metric,
                        lpips_model=eval_lpips_eval_model,
                        inception_net=eval_inception,
                        fid_ref=eval_fid_ref,
                        use_clip_condition=use_clip_condition,
                        use_dino_condition=use_dino_condition,
                    )
                    eval_metrics["eval/epoch"] = float(epoch + 1)
                    writer.write_scalars(global_step, eval_metrics)
                    if eval_images:
                        writer.write_images(global_step, eval_images)

                    clip_key = "eval/high/whigh/clip_cos/mean"
                    clip_val = float(eval_metrics.get(clip_key, np.nan))
                    topk_clip_high = _update_topk_checkpoints(
                        workdir=workdir,
                        name="clip_high",
                        model=model_core,
                        metric_name=clip_key,
                        metric_value=clip_val,
                        epoch=epoch + 1,
                        step=global_step,
                        topk_entries=topk_clip_high,
                        top_k=int(eval_cfg.best_top_k),
                    )

                    score_key = "eval/score/lpips_psnr"
                    score_val = float(eval_metrics.get(score_key, np.nan))
                    topk_lpips_psnr = _update_topk_checkpoints(
                        workdir=workdir,
                        name="lpips_psnr",
                        model=model_core,
                        metric_name=score_key,
                        metric_value=score_val,
                        epoch=epoch + 1,
                        step=global_step,
                        topk_entries=topk_lpips_psnr,
                        top_k=int(eval_cfg.best_top_k),
                    )
                except Exception as exc:
                    eval_failed = True
                    log_for_0(
                        f"Evaluation failed at epoch={epoch + 1}; disabling runtime evaluation. Error: {exc}"
                    )
                finally:
                    eval_semantic_encoder_metric = None
                    eval_lpips_eval_model = None
                    eval_inception = None
                    eval_fid_ref = None
                    _cleanup_cuda_memory()
            eval_failed = _broadcast_bool_from_rank0(eval_failed, device=device)
            if eval_failed:
                eval_enabled_runtime = False
            barrier()

        if (epoch + 1) % int(config.training.checkpoint_per_epoch) == 0 or (
            epoch + 1
        ) == int(config.training.num_epochs):
            _save_checkpoint(
                workdir=workdir,
                epoch=epoch + 1,
                step=global_step,
                model=model_core,
                optimizer=optimizer,
                scheduler=scheduler,
                ema_models=ema_models,
            )

    barrier()
    return model_core


@torch.no_grad()
def just_evaluate(config, workdir: str):
    rank, _, device = init_distributed(config=config)
    os.makedirs(workdir, exist_ok=True) if rank == 0 else None

    model_cfg = deepcopy(config.model.to_dict() if hasattr(config.model, "to_dict") else dict(config.model))
    condition_cfg = config.condition if hasattr(config, "condition") else None
    condition_mode, use_clip_condition, use_dino_condition = input_pipeline.resolve_condition_mode(condition_cfg)
    semantic_cache_eval = None
    if use_clip_condition or use_dino_condition:
        semantic_cache_eval = input_pipeline.create_semantic_cache(
            condition_cfg=condition_cfg,
            model_cfg=config.model,
            dataset_cfg=config.dataset,
        )
        if semantic_cache_eval is not None:
            log_for_0(
                "Semantic condition cache enabled for eval "
                f"(root={semantic_cache_eval.cache_root}, mode={condition_mode})"
            )
    model_cfg["condition_mode"] = condition_mode
    model_cfg["enable_semantic_loss"] = bool(getattr(config.model, "enable_semantic_loss", True))
    # Keep full u/v heads in eval-only mode so validation diagnostics
    # (e.g. v-related probe metrics) stay consistent with training-time definitions.
    model_cfg["eval"] = False
    model = pixelMeanFlow(**model_cfg).to(device)
    eval_cfg = _get_eval_cfg(config)

    if not getattr(config, "load_from", ""):
        raise ValueError("config.load_from must be specified for evaluation.")
    _load_checkpoint(config.load_from, model)
    model.eval()

    intervals = getattr(config.sampling, "interval", [[config.sampling.t_min, config.sampling.t_max]])
    omegas = getattr(config.sampling, "omegas", [config.sampling.omega])

    writer = Writer(config, workdir)
    combo_iter = itertools.product(omegas, intervals)
    for idx, (omega, interval) in enumerate(combo_iter):
        t_min, t_max = float(interval[0]), float(interval[1])
        vis = _generate_visual(
            model,
            config,
            device=device,
            n_sample=16,
            omega=omega,
            t_min=t_min,
            t_max=t_max,
        )
        writer.write_images(
            idx,
            {f"eval_omega_{omega}_t_{t_min}_{t_max}": vis},
        )
        log_for_0(f"Generated eval sample for omega={omega}, interval=[{t_min}, {t_max}]")

    if eval_cfg.enable and rank == 0:
        semantic_encoder_cond = input_pipeline.create_semantic_encoder(
            condition_cfg,
            num_dino_tokens=int(getattr(config.model, "num_dino_tokens", 4)),
            use_clip=use_clip_condition,
            use_dino=use_dino_condition,
        )
        eval_loader, eval_samples = _create_eval_subset_loader(
            config.dataset,
            batch_size=int(eval_cfg.batch_size),
            max_samples=int(eval_cfg.max_samples),
            semantic_cache=semantic_cache_eval,
        )
        log_for_0(f"Running full validation on {eval_samples} samples...")

        cond_metric_dict = _cfg_to_dict(condition_cfg)
        semantic_encoder_metric = None
        if cond_metric_dict:
            metric_device = cond_metric_dict.get("device", "cpu")
            if metric_device == "cuda" and not torch.cuda.is_available():
                metric_device = "cpu"
            cond_metric_dict["device"] = metric_device
            cond_metric_cfg = SimpleNamespace(**cond_metric_dict)
            try:
                semantic_encoder_metric = build_semantic_encoder(
                    cond_metric_cfg,
                    num_dino_tokens=int(getattr(config.model, "num_dino_tokens", 4)),
                    use_clip=True,
                    use_dino=True,
                )
            except Exception as exc:
                semantic_encoder_metric = None
                log_for_0(f"Semantic eval encoder init failed: {exc}")

        try:
            lpips_eval_model = _build_lpips_vgg(device)
        except Exception as exc:
            lpips_eval_model = None
            log_for_0(f"LPIPS eval model unavailable: {exc}")

        inception_eval = None
        fid_ref = None
        if eval_cfg.compute_fid:
            cache_ref = getattr(config.fid, "cache_ref", "")
            if cache_ref and os.path.exists(cache_ref):
                try:
                    inception_eval = build_jax_inception(
                        batch_size=int(getattr(config.fid, "device_batch_size", 40))
                    )
                    fid_ref = get_reference(cache_ref)
                except Exception as exc:
                    inception_eval = None
                    fid_ref = None
                    log_for_0(f"FID evaluator init failed: {exc}")

        eval_metrics, eval_images = _run_validation_cycle(
            model_core=model,
            config=config,
            eval_cfg=eval_cfg,
            eval_loader=eval_loader,
            device=device,
            semantic_encoder_cond=semantic_encoder_cond,
            semantic_cache_cond=semantic_cache_eval,
            semantic_encoder_metric=semantic_encoder_metric,
            lpips_model=lpips_eval_model,
            inception_net=inception_eval,
            fid_ref=fid_ref,
            use_clip_condition=use_clip_condition,
            use_dino_condition=use_dino_condition,
        )
        writer.write_scalars(0, eval_metrics)
        if eval_images:
            writer.write_images(0, eval_images)

    barrier()
    return model
