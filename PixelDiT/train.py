from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from PixelDiT.models import PixelDiTT2IPMF
from PixelDiT.utils.dist import (
    all_reduce_mean_scalar,
    barrier,
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
    set_seed,
)
from PixelDiT.utils.logging import AverageMeter, ScalarWriter, rank0_info
from PixelDiT.utils.losses import LossConfig, adp_wt_fn, build_metrics, masked_perceptual_loss
from PixelDiT.utils.perceptual import PerceptualConfig, PerceptualLoss
from PixelDiT.utils.semantic_cache import DinoSemanticCache, IndexedDatasetWrapper
from PixelDiT.utils.text_cond import apply_cond_dropout
from PixelDiT.utils.time_sampling import sample_tr


@dataclass
class TrainState:
    epoch: int
    step: int


def _as_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _as_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_as_namespace(v) for v in obj]
    return obj


def _to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: _to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, list):
        return [_to_dict(x) for x in ns]
    return ns


def center_crop_arr(pil_image, image_size: int):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


class DeterministicCenterCrop:
    def __init__(self, image_size: int):
        self.image_size = int(image_size)

    def __call__(self, img):
        return center_crop_arr(img, self.image_size)


def build_imagefolder_dataset(cfg, split: str):
    root = os.path.join(str(cfg.dataset.root), split)

    image_size = int(cfg.dataset.image_size)
    augment_train = bool(getattr(cfg.dataset, "augment_train", False)) and split == str(cfg.dataset.train_split)
    use_flip = bool(getattr(cfg.dataset, "use_flip", False)) and split == str(cfg.dataset.train_split)

    if augment_train:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5 if use_flip else 0.0),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                DeterministicCenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    base = datasets.ImageFolder(root=root, transform=transform)
    return base


def _reduce_metrics(metrics: Dict[str, torch.Tensor | float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        if torch.is_tensor(v):
            t = v.detach()
        else:
            t = torch.tensor(float(v), device="cuda" if torch.cuda.is_available() else "cpu")
        t = all_reduce_mean_scalar(t)
        out[k] = float(t.item())
    return out


def _save_checkpoint(workdir: str, state: TrainState, model, optimizer, scaler, keep_last_k: int = 3):
    if not is_main_process():
        return
    ckpt_dir = os.path.join(workdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"checkpoint_step_{state.step:08d}.pt")
    torch.save(
        {
            "epoch": state.epoch,
            "step": state.step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
        },
        path,
    )

    all_ckpts = sorted(Path(ckpt_dir).glob("checkpoint_step_*.pt"))
    if len(all_ckpts) > int(keep_last_k):
        stale = all_ckpts[: len(all_ckpts) - int(keep_last_k)]
        for p in stale:
            p.unlink(missing_ok=True)


def _save_topk_train_loss(
    workdir: str,
    model,
    epoch: int,
    step: int,
    metric_value: float,
    entries: List[dict],
    top_k: int = 3,
):
    if not is_main_process():
        return entries

    candidate = {
        "metric_value": float(metric_value),
        "epoch": int(epoch),
        "step": int(step),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    entries = list(entries) + [candidate]
    entries = sorted(entries, key=lambda x: x["metric_value"])[: int(top_k)]

    out_dir = os.path.join(workdir, "best_checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    for i, e in enumerate(entries, start=1):
        path = os.path.join(out_dir, f"checkpoint_best_train_loss_rank{i}.pt")
        torch.save(
            {
                "model": e["state_dict"],
                "epoch": e["epoch"],
                "step": e["step"],
                "metric_name": "train/loss",
                "metric_value": e["metric_value"],
            },
            path,
        )

    for i in range(len(entries) + 1, int(top_k) + 1):
        stale = os.path.join(out_dir, f"checkpoint_best_train_loss_rank{i}.pt")
        if os.path.isfile(stale):
            os.remove(stale)

    return entries


def _build_model(cfg) -> PixelDiTT2IPMF:
    m = cfg.model
    return PixelDiTT2IPMF(
        input_size=int(m.input_size),
        patch_size=int(m.patch_size),
        in_channels=int(m.in_channels),
        hidden_size=int(m.hidden_size),
        pixel_dim=int(m.pixel_dim),
        patch_depth=int(m.patch_depth),
        pixel_depth=int(m.pixel_depth),
        pixel_head_depth=int(m.pixel_head_depth),
        num_heads=int(m.num_heads),
        pixel_num_heads=int(m.pixel_num_heads),
        mlp_ratio=float(m.mlp_ratio),
        sem_in_dim=int(m.sem_in_dim),
        use_qknorm=bool(m.use_qknorm),
        use_swiglu=bool(m.use_swiglu),
        use_rope=bool(m.use_rope),
        use_rmsnorm=bool(m.use_rmsnorm),
        use_checkpoint=bool(m.use_checkpoint),
        null_token_learnable=bool(m.null_token_learnable),
    )


def train(cfg, workdir: str):
    rank, world_size, local_rank, device = init_distributed(backend=str(cfg.ddp.backend))

    if is_main_process():
        os.makedirs(workdir, exist_ok=True)

    set_seed(int(cfg.training.seed) + rank)

    writer = ScalarWriter(workdir=workdir, use_tensorboard=bool(cfg.logging.tb))

    train_ds = build_imagefolder_dataset(cfg, split=str(cfg.dataset.train_split))
    indexed_train_ds = IndexedDatasetWrapper(train_ds)

    semantic_cache = None
    if bool(cfg.semantic.use_offline_cache):
        semantic_cache = DinoSemanticCache(
            cache_root=str(cfg.semantic.cache_root),
            dino_feature_dim=int(cfg.semantic.dino_feature_dim),
            num_dino_tokens=int(cfg.semantic.num_dino_tokens),
            dataset_root=str(cfg.dataset.root),
            image_size=int(cfg.dataset.image_size),
            strict=bool(cfg.semantic.strict),
        )
        semantic_cache.validate_against_imagefolder(split=str(cfg.dataset.train_split), imagefolder_dataset=train_ds)
        rank0_info(
            "Semantic DINO cache enabled "
            f"(root={semantic_cache.cache_root}, split={cfg.dataset.train_split}, samples={len(train_ds)})"
        )

    sampler = DistributedSampler(indexed_train_ds, shuffle=True, drop_last=True) if world_size > 1 else None
    train_loader = DataLoader(
        indexed_train_ds,
        batch_size=int(cfg.training.batch_size) // world_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg.dataset.num_workers),
        pin_memory=bool(cfg.dataset.pin_memory),
        drop_last=True,
        persistent_workers=bool(cfg.dataset.persistent_workers),
        prefetch_factor=int(cfg.dataset.prefetch_factor) if int(cfg.dataset.num_workers) > 0 else None,
    )

    model = _build_model(cfg).to(device)
    model_core = model

    if world_size > 1:
        ddp_device_ids = [device.index] if device.type == "cuda" else None
        model = DDP(
            model,
            device_ids=ddp_device_ids,
            find_unused_parameters=bool(cfg.ddp.find_unused_parameters),
        )
        model_core = model.module

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
        betas=(0.9, 0.95),
    )

    use_amp = bool(cfg.training.use_amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(cfg.training.amp_dtype).lower() == "bf16" else torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    loss_cfg = LossConfig(
        norm_p=float(cfg.loss.norm_p),
        norm_eps=float(cfg.loss.norm_eps),
        perceptual_max_t=float(cfg.loss.perceptual_max_t),
        lpips_weight=float(cfg.loss.lpips_weight),
        convnext_weight=float(cfg.loss.convnext_weight),
        enable_lpips=bool(cfg.loss.enable_lpips),
        enable_convnext=bool(cfg.loss.enable_convnext),
    )

    perceptual = PerceptualLoss(
        cfg=PerceptualConfig(
            enable_lpips=bool(cfg.loss.enable_lpips),
            enable_convnext=bool(cfg.loss.enable_convnext),
            convnext_model_name=str(cfg.loss.convnext_model_name),
        ),
        device=device,
    )

    rank0_info(
        f"Train setup: world_size={world_size}, local_rank={local_rank}, "
        f"global_batch={int(cfg.training.batch_size)}, local_batch={int(cfg.training.batch_size)//world_size}, "
        f"amp={use_amp}, dtype={cfg.training.amp_dtype}"
    )

    state = TrainState(epoch=0, step=0)
    topk_entries: List[dict] = []
    log_meter = AverageMeter()
    epoch_meter = AverageMeter()

    t_eps = float(cfg.training.t_eps)
    start_time = time.time()

    model.train()
    for epoch in range(int(cfg.training.num_epochs)):
        state.epoch = epoch
        if sampler is not None:
            sampler.set_epoch(epoch)

        rank0_info(f"epoch {epoch}...")
        epoch_meter = AverageMeter()

        for batch_idx, batch in enumerate(train_loader):
            indices, images, _labels = batch
            images = images.to(device=device, non_blocking=True)
            bsz = images.shape[0]

            if semantic_cache is None:
                raise RuntimeError("semantic.use_offline_cache must be true for this training path.")
            sem_tokens = semantic_cache.get_batch(
                split=str(cfg.dataset.train_split),
                indices=indices,
                device=device,
            ).to(dtype=torch.float32)

            t, r, _fm_mask = sample_tr(
                batch_size=bsz,
                device=device,
                dtype=images.dtype,
                logit_mean=float(cfg.time_sampling.logit_mean),
                logit_std=float(cfg.time_sampling.logit_std),
                data_proportion=float(cfg.time_sampling.data_proportion),
            )
            h = t - r

            noise = torch.randn_like(images) * float(cfg.training.noise_scale)
            z_t = (1.0 - t) * images + t * noise
            t_clamp = torch.clamp(t, min=t_eps)
            v_t = (z_t - images) / t_clamp

            null_sem = model_core.get_null_sem_tokens(
                batch_size=bsz,
                seq_len=sem_tokens.shape[1],
                device=device,
                dtype=sem_tokens.dtype,
            )
            sem_used, drop_mask = apply_cond_dropout(
                sem_tokens=sem_tokens,
                null_tokens=null_sem,
                drop_prob=float(cfg.training.class_dropout_prob),
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(
                    x=z_t,
                    t=t.reshape(-1),
                    h=h.reshape(-1),
                    sem_tokens=sem_used,
                    sem_mask=None,
                )
                x_hat_u = out["x_hat_u"]
                x_hat_v = out["x_hat_v"]

                u_pred = (z_t - x_hat_u) / t_clamp
                v_pred = (z_t - x_hat_v) / t_clamp

                def u_only(z_in, t_in, r_in):
                    out_u = model(
                        x=z_in,
                        t=t_in.reshape(-1),
                        h=(t_in - r_in).reshape(-1),
                        sem_tokens=sem_used,
                        sem_mask=None,
                    )["x_hat_u"]
                    return (z_in - out_u) / torch.clamp(t_in, min=t_eps)

                dtdt = torch.ones_like(t)
                dtdr = torch.zeros_like(t)

                # Training-time create_graph=True avoids DDP reducer issues observed
                # in some PyTorch builds when jvp is evaluated inside forward.
                _, du_dt = torch.autograd.functional.jvp(
                    u_only,
                    (z_t, t, r),
                    (v_pred.detach(), dtdt, dtdr),
                    create_graph=model.training,
                    strict=False,
                )
                V = u_pred + (t - r) * du_dt.detach()

                loss_u_vec = torch.sum((V - v_t) ** 2, dim=(1, 2, 3))
                loss_u_vec = adp_wt_fn(loss_u_vec, norm_eps=loss_cfg.norm_eps, norm_p=loss_cfg.norm_p)

                loss_v_vec = torch.sum((v_pred - v_t) ** 2, dim=(1, 2, 3))
                loss_v_vec = adp_wt_fn(loss_v_vec, norm_eps=loss_cfg.norm_eps, norm_p=loss_cfg.norm_p)

                lpips_vec, convnext_vec = (None, None)
                perc_vec = torch.zeros_like(loss_u_vec)
                if perceptual.enabled:
                    lpips_vec, convnext_vec = perceptual(x_hat_u, images)
                    perc_vec = masked_perceptual_loss(
                        t_flat=t.reshape(-1),
                        lpips_vec=lpips_vec,
                        convnext_vec=convnext_vec,
                        cfg=loss_cfg,
                    )

                total_vec = loss_u_vec + loss_v_vec + perc_vec
                loss = total_vec.mean()

            if use_amp:
                scaler.scale(loss).backward()
                if float(cfg.training.grad_clip) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.grad_clip))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if float(cfg.training.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.grad_clip))
                optimizer.step()

            state.step += 1

            metrics = build_metrics(
                loss=loss,
                loss_u=loss_u_vec,
                loss_v=loss_v_vec,
                lpips_vec=lpips_vec,
                convnext_vec=convnext_vec,
            )
            metrics["cond_drop_ratio"] = drop_mask.float().mean().detach()
            metrics["lr"] = torch.tensor(optimizer.param_groups[0]["lr"], device=device)

            reduced = _reduce_metrics(metrics)
            log_meter.update(reduced)
            epoch_meter.update(reduced)

            if state.step % int(cfg.logging.log_interval) == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                summary = log_meter.pop()
                summary["steps_per_second"] = float(cfg.logging.log_interval) / elapsed
                writer.write_scalars(step=state.step, scalar_dict=summary)
                start_time = time.time()

            if state.step % int(cfg.checkpoint.save_interval) == 0:
                _save_checkpoint(
                    workdir=workdir,
                    state=state,
                    model=model_core,
                    optimizer=optimizer,
                    scaler=scaler,
                    keep_last_k=int(cfg.checkpoint.keep_last_k),
                )

        epoch_summary = epoch_meter.pop()
        if is_main_process() and epoch_summary:
            train_loss_epoch = float(epoch_summary.get("loss", np.nan))
            topk_entries = _save_topk_train_loss(
                workdir=workdir,
                model=model_core,
                epoch=epoch + 1,
                step=state.step,
                metric_value=train_loss_epoch,
                entries=topk_entries,
                top_k=3,
            )

        if bool(cfg.checkpoint.save_per_epoch):
            _save_checkpoint(
                workdir=workdir,
                state=state,
                model=model_core,
                optimizer=optimizer,
                scaler=scaler,
                keep_last_k=int(cfg.checkpoint.keep_last_k),
            )

        barrier()

    writer.close()
    cleanup_distributed()


def run_train(config_dict: dict, workdir: str):
    cfg = _as_namespace(config_dict)
    train(cfg, workdir)


def smoke_test_forward(config_dict: dict):
    cfg = _as_namespace(config_dict)
    model = _build_model(cfg)
    model.eval()

    b = 2
    h = int(cfg.dataset.image_size)
    w = int(cfg.dataset.image_size)
    sem_toks = int(cfg.semantic.num_dino_tokens)
    sem_dim = int(cfg.semantic.dino_feature_dim)

    x = torch.randn(b, int(cfg.model.in_channels), h, w)
    t = torch.rand(b)
    h_span = torch.rand(b)
    sem = torch.randn(b, sem_toks, sem_dim)

    with torch.no_grad():
        out = model(x=x, t=t, h=h_span, sem_tokens=sem)
    assert out["x_hat_u"].shape == x.shape
    assert out["x_hat_v"].shape == x.shape
    return {"ok": True, "shape": tuple(out["x_hat_u"].shape)}
