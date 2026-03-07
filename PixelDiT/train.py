from __future__ import annotations

import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional
import traceback

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from PixelDiT.models import PixelDiTT2IPMF
from PixelDiT.utils.dist import (
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
from PixelDiT.utils.time_sampling import sample_tr
from PixelDiT.utils.dino_encoder import DinoEncoder, DinoEncoderConfig
from PixelDiT.utils.edge import sobel_edge_map

try:
    from torch.func import jvp as func_jvp
except Exception:  # pragma: no cover
    func_jvp = None


@dataclass
class TrainState:
    epoch: int
    step: int


def _capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict):
    if not isinstance(state, dict):
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.random.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


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


def _jvp_sdpa_context():
    """Use a conservative SDP backend for JVP to avoid unsupported flash-SDP derivatives."""
    if not torch.cuda.is_available():
        return nullcontext()
    if hasattr(torch, "nn") and hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
        from torch.nn.attention import sdpa_kernel, SDPBackend

        return sdpa_kernel([SDPBackend.MATH])
    if hasattr(torch.backends.cuda, "sdp_kernel"):
        # Backward-compat for older torch releases.
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
            enable_cudnn=False,
        )
    return nullcontext()


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


class FlatImageDataset(Dataset):
    """Flat image directory dataset with ImageFolder-like attributes.

    Used as a fallback for val directories shaped like `val/*.jpg` without class subfolders.
    Labels are dummy zeros because current PixelDiT eval/training pipeline does not consume labels.
    """

    def __init__(self, root: str, transform):
        self.root = str(root)
        self.transform = transform
        self.loader = default_loader
        self.classes = ["dummy"]
        self.class_to_idx = {"dummy": 0}

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = []
        for name in sorted(os.listdir(self.root)):
            p = os.path.join(self.root, name)
            if os.path.isfile(p) and os.path.splitext(name)[1].lower() in exts:
                files.append(p)

        if not files:
            raise FileNotFoundError(f"No images found under flat dataset directory: {self.root}")

        self.samples = [(p, 0) for p in files]
        self.targets = [0 for _ in files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, target = self.samples[int(idx)]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


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

    try:
        base = datasets.ImageFolder(root=root, transform=transform)
        return base
    except FileNotFoundError as exc:
        # Fallback for flat validation layout: val/*.jpg without class folders.
        is_val = split == str(cfg.dataset.val_split)
        if (not is_val) or ("Couldn't find any class folder" not in str(exc)):
            raise
        rank0_info(
            "Using flat val dataset fallback without class folders; assigning dummy labels=0 "
            "(labels are unused by current PixelDiT train/eval paths)."
        )
        return FlatImageDataset(root=root, transform=transform)


def _reduce_metrics(metrics: Dict[str, torch.Tensor | float]) -> Dict[str, float]:
    if not metrics:
        return {}

    keys = sorted(metrics.keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vals: List[torch.Tensor] = []
    for k in keys:
        v = metrics[k]
        if torch.is_tensor(v):
            t = v.detach()
            if t.numel() != 1:
                t = t.mean()
        else:
            t = torch.tensor(float(v), device=device)
        if t.device != device:
            t = t.to(device=device)
        vals.append(t.to(dtype=torch.float32).reshape(()))

    vec = torch.stack(vals, dim=0)
    if is_distributed():
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        vec = vec / float(get_world_size())

    return {k: float(vec[i].item()) for i, k in enumerate(keys)}


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
            "rng_state": _capture_rng_state(),
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
    top_k: int = 1,
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


def _load_topk_train_loss_entries(workdir: str, top_k: int = 1) -> List[dict]:
    out_dir = os.path.join(workdir, "best_checkpoints")
    if not os.path.isdir(out_dir):
        return []

    entries: List[dict] = []
    for i in range(1, int(top_k) + 1):
        path = os.path.join(out_dir, f"checkpoint_best_train_loss_rank{i}.pt")
        if not os.path.isfile(path):
            continue
        ckpt = torch.load(path, map_location="cpu")
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            continue
        entries.append(
            {
                "metric_value": float(ckpt.get("metric_value", float("inf"))),
                "epoch": int(ckpt.get("epoch", 0)),
                "step": int(ckpt.get("step", 0)),
                "state_dict": ckpt["model"],
            }
        )
    return sorted(entries, key=lambda x: x["metric_value"])[: int(top_k)]


def _load_training_checkpoint(
    checkpoint_path: str,
    model,
    optimizer,
    scaler,
    strict_model: bool = True,
    resume_optimizer: bool = True,
    resume_scaler: bool = True,
    resume_rng_state: bool = True,
) -> dict:
    # Full training checkpoints include optimizer/scaler/RNG states. Under PyTorch 2.6,
    # torch.load defaults to weights_only=True, which rejects these non-tensor objects.
    # Resume uses locally produced checkpoints and must restore the full training state.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Invalid training checkpoint: {checkpoint_path}")

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=bool(strict_model))
    if missing:
        rank0_info(f"[resume] missing model keys: {len(missing)}")
    if unexpected:
        rank0_info(f"[resume] unexpected model keys: {len(unexpected)}")

    if bool(resume_optimizer) and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if bool(resume_scaler) and scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    rng_restored = False
    if bool(resume_rng_state) and ckpt.get("rng_state") is not None and get_world_size() <= 1:
        _restore_rng_state(ckpt["rng_state"])
        rng_restored = True

    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "step": int(ckpt.get("step", 0)),
        "has_rng_state": ckpt.get("rng_state") is not None,
        "rng_restored": rng_restored,
    }


def _build_model(cfg) -> PixelDiTT2IPMF:
    m = cfg.model
    spatial = getattr(cfg, "spatial", None)
    enable_edge = bool(getattr(spatial, "enable_edge", False)) if spatial is not None else False
    edge_channels = int(getattr(spatial, "edge_channels", 64)) if spatial is not None else 64
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
        sem_num_tokens=int(getattr(m, "sem_num_tokens", 64)),
        sem_pool_num_heads=int(getattr(m, "sem_pool_num_heads", 12)),
        enable_edge_cond=enable_edge,
        edge_channels=edge_channels,
        use_qknorm=bool(m.use_qknorm),
        use_swiglu=bool(m.use_swiglu),
        use_rope=bool(m.use_rope),
        use_rmsnorm=bool(m.use_rmsnorm),
        use_checkpoint=bool(m.use_checkpoint),
        null_token_learnable=bool(m.null_token_learnable),
    )


def _maybe_run_epoch_eval(
    cfg,
    workdir: str,
    checkpoint_path: str,
    step: int,
    epoch: int,
    device: torch.device,
):
    eval_cfg = getattr(cfg, "evaluation", None)
    if eval_cfg is None:
        return
    if not bool(getattr(eval_cfg, "enabled", False)):
        return
    every = max(1, int(getattr(eval_cfg, "run_every_n_epochs", 1)))
    if ((int(epoch) + 1) % every) != 0:
        return
    if not is_main_process():
        return

    try:
        from PixelDiT.eval import run_eval
    except Exception as exc:
        rank0_info(f"[eval] import failed, skip epoch eval: {type(exc).__name__}: {exc}")
        if bool(getattr(eval_cfg, "fail_on_error", False)):
            raise
        return

    epoch_max_samples = getattr(eval_cfg, "epoch_max_samples", None)
    if epoch_max_samples is not None:
        epoch_max_samples = int(epoch_max_samples)
        if epoch_max_samples <= 0:
            epoch_max_samples = None

    args = SimpleNamespace(
        config=os.path.join(workdir, "used_config.yml"),
        checkpoint=checkpoint_path,
        workdir=workdir,
        max_samples=epoch_max_samples,
        global_step=int(step),
        device=f"cuda:{device.index}" if device.type == "cuda" and device.index is not None else device.type,
        num_workers_override=int(getattr(eval_cfg, "epoch_end_num_workers", 0)),
    )
    rank0_info(
        f"[eval] epoch-end evaluation starts: epoch={int(epoch)+1}, step={int(step)}, checkpoint={checkpoint_path}"
    )
    try:
        run_eval(args)
    except Exception as exc:
        rank0_info(f"[eval] epoch-end evaluation failed: {type(exc).__name__}: {exc}")
        rank0_info(traceback.format_exc())
        if bool(getattr(eval_cfg, "fail_on_error", False)):
            raise
    else:
        rank0_info(f"[eval] epoch-end evaluation done: epoch={int(epoch)+1}, step={int(step)}")


def train(cfg, workdir: str):
    rank, world_size, local_rank, device = init_distributed(backend=str(cfg.ddp.backend))
    def _stage(msg: str):
        print(f"[rank{rank}] {msg}", flush=True)
    def _nccl_probe(tag: str):
        if world_size <= 1:
            return
        _stage(f"stage: NCCL probe ({tag}) start")
        probe = torch.tensor([float(rank + 1)], device=device)
        dist.all_reduce(probe, op=dist.ReduceOp.SUM)
        expected = float(world_size * (world_size + 1) // 2)
        if abs(float(probe.item()) - expected) > 1e-4:
            raise RuntimeError(
                f"NCCL probe mismatch on rank={rank}, tag={tag}: got={float(probe.item())}, expected={expected}"
            )
        _stage(f"stage: NCCL probe ({tag}) done (sum={float(probe.item()):.1f})")

    # This training loop relies on `autograd.functional.jvp` through attention blocks.
    # On CPU builds, higher-order derivatives for some SDP kernels are not available.
    if device.type != "cuda":
        raise RuntimeError(
            "PixelDiT training requires CUDA (got device=%s). "
            "Use `--smoke-test` for CPU-only shape checks." % (device,)
        )

    if is_main_process():
        os.makedirs(workdir, exist_ok=True)

    global_batch = int(cfg.training.batch_size)
    if global_batch % max(world_size, 1) != 0:
        raise ValueError(f"training.batch_size={global_batch} must be divisible by world_size={world_size}")
    local_batch = global_batch // max(world_size, 1)

    set_seed(int(cfg.training.seed) + rank)
    _nccl_probe("early")

    writer = ScalarWriter(
        workdir=workdir,
        use_tensorboard=bool(cfg.logging.tb),
        flush_every=int(getattr(cfg.logging, "tb_flush_every", 20)),
    )

    _stage("stage: build train dataset start")
    train_ds = build_imagefolder_dataset(cfg, split=str(cfg.dataset.train_split))
    _stage(f"stage: build train dataset done (samples={len(train_ds)})")
    indexed_train_ds = IndexedDatasetWrapper(train_ds)

    spatial_cfg = getattr(cfg, "spatial", None)
    use_edge_cond = bool(getattr(spatial_cfg, "enable_edge", False)) if spatial_cfg is not None else False
    edge_blur_sigma = float(getattr(spatial_cfg, "edge_blur_sigma", 1.0)) if spatial_cfg is not None else 1.0
    edge_threshold = float(getattr(spatial_cfg, "edge_threshold", 0.0)) if spatial_cfg is not None else 0.0

    semantic_cache = None
    dino_encoder = None
    dino_model_name = ""
    dino_image_size = int(getattr(cfg.semantic, "dino_image_size", cfg.dataset.image_size))
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
    else:
        dino_model_name = str(getattr(cfg.semantic, "dino_model_name", "")).strip()
        if dino_model_name == "":
            raise ValueError("semantic.use_offline_cache=false requires semantic.dino_model_name for online extraction.")
        rank0_info(
            "Semantic DINO online configured "
            f"(model={dino_model_name}, image_size={dino_image_size}); init deferred until after DDP."
        )

    sampler = DistributedSampler(indexed_train_ds, shuffle=True, drop_last=True) if world_size > 1 else None
    num_workers = int(cfg.dataset.num_workers)
    mp_ctx = "spawn" if num_workers > 0 else None
    _stage(
        "stage: dataloader create "
        f"(num_workers={num_workers}, local_batch={local_batch}, world_size={world_size}, mp_ctx={mp_ctx})"
    )
    train_loader = DataLoader(
        indexed_train_ds,
        batch_size=local_batch,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(cfg.dataset.pin_memory),
        drop_last=True,
        persistent_workers=bool(cfg.dataset.persistent_workers) if num_workers > 0 else False,
        prefetch_factor=int(cfg.dataset.prefetch_factor) if num_workers > 0 else None,
        multiprocessing_context=mp_ctx,
    )
    _stage("stage: dataloader create done")

    _nccl_probe("pre_ddp")

    _stage("stage: model build start")
    model = _build_model(cfg).to(device)
    _stage("stage: model build done")
    param_tensors = sum(1 for _ in model.parameters())
    trainable_tensors = sum(1 for p in model.parameters() if p.requires_grad)
    print(
        f"[rank{rank}] model param tensors={param_tensors}, trainable_tensors={trainable_tensors}, device={device}",
        flush=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_core = model

    if world_size > 1:
        ddp_device_ids = [device.index] if device.type == "cuda" else None
        _stage("stage: DDP wrap start")
        model = DDP(
            model,
            device_ids=ddp_device_ids,
            find_unused_parameters=bool(cfg.ddp.find_unused_parameters),
            broadcast_buffers=False,
        )
        model_core = model.module
        _stage("stage: DDP wrap done")

    if (not bool(cfg.semantic.use_offline_cache)) and (dino_encoder is None):
        dino_encoder = DinoEncoder(
            cfg=DinoEncoderConfig(
                model_name=dino_model_name,
                use_dense=bool(getattr(cfg.semantic, "dino_use_dense", True)),
                image_size=int(dino_image_size),
            ),
            device=device,
        )
        rank0_info(
            "Semantic DINO online enabled "
            f"(model={dino_model_name}, image_size={int(dino_image_size)})"
        )

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

    resume_cfg = getattr(cfg, "resume", None)
    resume_enabled = bool(getattr(resume_cfg, "enabled", False)) if resume_cfg is not None else False
    resume_path = str(getattr(resume_cfg, "checkpoint", "")).strip() if resume_cfg is not None else ""
    resume_strict_model = bool(getattr(resume_cfg, "strict_model", True)) if resume_cfg is not None else True
    resume_optimizer = bool(getattr(resume_cfg, "optimizer", True)) if resume_cfg is not None else True
    resume_scaler = bool(getattr(resume_cfg, "scaler", True)) if resume_cfg is not None else True
    resume_rng_state = bool(getattr(resume_cfg, "rng_state", True)) if resume_cfg is not None else True

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
        f"global_batch={global_batch}, local_batch={local_batch}, "
        f"amp={use_amp}, dtype={cfg.training.amp_dtype}"
    )

    state = TrainState(epoch=0, step=0)
    topk_entries: List[dict] = []
    log_meter = AverageMeter()
    epoch_meter = AverageMeter()

    grad_accum_steps = int(getattr(cfg.training, "gradient_accumulation_steps", 1))
    if grad_accum_steps != 1:
        raise ValueError(
            "Gradient accumulation is disabled in this training pipeline. "
            "Set training.gradient_accumulation_steps=1."
        )

    log_interval = int(getattr(cfg.training, "log_per_step", getattr(cfg.logging, "log_interval", 100)))
    log_interval = max(int(log_interval), 1)

    diag_cfg = getattr(cfg, "diagnostics", None)
    cfg_probe_interval = int(getattr(diag_cfg, "cfg_probe_interval", 500)) if diag_cfg is not None else 500
    cfg_probe_batch_size = int(getattr(diag_cfg, "cfg_probe_batch_size", 4)) if diag_cfg is not None else 4
    cfg_probe_enabled = cfg_probe_interval > 0 and cfg_probe_batch_size > 0

    t_eps = float(cfg.training.t_eps)
    start_time = time.time()
    jvp_fallback_emitted = False
    start_epoch = 0
    resume_skip_batches = 0
    # torch.func.jvp can fail on some operator stacks (e.g., captured in-place copy_)
    # and may leave the current step graph in a bad state. Keep it opt-in.
    use_func_jvp = bool(getattr(cfg.training, "use_func_jvp", False))
    if is_main_process() and (not use_func_jvp):
        rank0_info("JVP backend: finite-difference (torch.func.jvp disabled; set training.use_func_jvp=true to opt in).")

    if resume_enabled:
        if resume_path == "":
            raise ValueError("resume.enabled=true requires resume.checkpoint")
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_info = _load_training_checkpoint(
            checkpoint_path=resume_path,
            model=model_core,
            optimizer=optimizer,
            scaler=scaler,
            strict_model=resume_strict_model,
            resume_optimizer=resume_optimizer,
            resume_scaler=resume_scaler,
            resume_rng_state=resume_rng_state,
        )
        state.epoch = int(resume_info["epoch"])
        state.step = int(resume_info["step"])
        steps_per_epoch = max(1, len(train_loader))
        resume_skip_batches = int(state.step % steps_per_epoch)
        start_epoch = int(state.epoch) + (1 if resume_skip_batches == 0 else 0)
        topk_entries = _load_topk_train_loss_entries(workdir=workdir, top_k=1)
        rank0_info(
            f"[resume] loaded checkpoint={resume_path}, epoch={state.epoch}, step={state.step}, "
            f"start_epoch={start_epoch}, skip_batches={resume_skip_batches}"
        )
        if bool(resume_info.get("has_rng_state", False)) and not bool(resume_info.get("rng_restored", False)):
            rank0_info(
                "[resume] RNG state is not restored under multi-rank training because the checkpoint stores only "
                "rank0 RNG. Model/optimizer/scaler resume is exact; randomness resume is approximate."
            )
        elif not bool(resume_info.get("has_rng_state", False)):
            rank0_info(
                "[resume] checkpoint has no RNG state; exact randomness cannot be restored. "
                "This epoch-end checkpoint can still resume effectively seamlessly."
            )

    model.train()
    for epoch in range(int(start_epoch), int(cfg.training.num_epochs)):
        state.epoch = epoch
        if sampler is not None:
            sampler.set_epoch(epoch)

        rank0_info(f"epoch {epoch}...")
        epoch_meter = AverageMeter()

        for batch_idx, batch in enumerate(train_loader):
            if epoch == int(start_epoch) and resume_skip_batches > 0 and batch_idx < resume_skip_batches:
                continue
            indices, images, _labels = batch
            images = images.to(device=device, non_blocking=True)
            bsz = images.shape[0]

            edge_map = None
            if use_edge_cond:
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        edge_map = sobel_edge_map(
                            images,
                            blur_sigma=edge_blur_sigma,
                            threshold=edge_threshold,
                        )

            if semantic_cache is not None:
                sem_tokens = semantic_cache.get_batch(
                    split=str(cfg.dataset.train_split),
                    indices=indices,
                    device=device,
                )
            else:
                assert dino_encoder is not None
                sem_tokens = dino_encoder.encode(images, amp_dtype=amp_dtype)

            t, r, _fm_mask = sample_tr(
                batch_size=bsz,
                device=device,
                dtype=images.dtype,
                P_mean=float(getattr(cfg.time_sampling, "P_mean", 0.0)),
                P_std=float(getattr(cfg.time_sampling, "P_std", 1.0)),
                data_proportion=float(cfg.time_sampling.data_proportion),
                tr_uniform=bool(getattr(cfg.time_sampling, "tr_uniform", False)),
                tr_uniform_prob=float(getattr(cfg.time_sampling, "tr_uniform_prob", 0.1)),
            )
            h = t - r

            noise = torch.randn_like(images) * float(cfg.training.noise_scale)
            z_t = (1.0 - t) * images + t * noise
            t_clamp = torch.clamp(t, min=t_eps)
            v_t = (z_t - images) / t_clamp

            drop_mask = torch.rand((bsz,), device=device) < float(cfg.training.class_dropout_prob)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(
                    x=z_t,
                    t=t.reshape(-1),
                    h=h.reshape(-1),
                    sem_tokens=sem_tokens,
                    sem_mask=None,
                    sem_drop_mask=drop_mask,
                    edge_map=edge_map,
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
                        sem_tokens=sem_tokens,
                        sem_mask=None,
                        sem_drop_mask=drop_mask,
                        edge_map=edge_map,
                    )["x_hat_u"]
                    return (z_in - out_u) / torch.clamp(t_in, min=t_eps)

                dtdt = torch.ones_like(t)
                dtdr = torch.zeros_like(t)

                jvp_exc = None
                if use_func_jvp and (func_jvp is not None):
                    try:
                        with _jvp_sdpa_context():
                            _, du_dt = func_jvp(
                                u_only,
                                (z_t, t, r),
                                (v_t.detach(), dtdt, dtdr),
                            )
                    except Exception as exc:
                        jvp_exc = exc
                else:
                    jvp_exc = RuntimeError("skip_func_jvp")

                if jvp_exc is not None:
                    # Robust fallback: finite-difference directional derivative in no-grad mode.
                    # This avoids higher-order graph/autograd path issues while preserving
                    # the V = u + (t-r) * du_dt structure.
                    fd_eps = float(getattr(cfg.training, "jvp_fd_eps", 1.0e-3))
                    with torch.no_grad():
                        z_eps = z_t + fd_eps * v_t.detach()
                        t_eps_fd = torch.clamp(t + fd_eps * dtdt, min=t_eps)
                        r_eps = r + fd_eps * dtdr
                        u_eps = u_only(z_eps, t_eps_fd, r_eps)
                        du_dt = (u_eps - u_pred.detach()) / fd_eps
                    if use_func_jvp and (not jvp_fallback_emitted) and is_main_process():
                        print(
                            "\n"
                            + "=" * 100
                            + "\n[JVP-FALLBACK] torch.func.jvp failed, switched to finite-difference JVP.\n"
                            + f"Reason: {type(jvp_exc).__name__}: {jvp_exc}\n"
                            + f"fd_eps={fd_eps}\n"
                            + "This warning is shown once per run."
                            + "\n"
                            + "=" * 100
                            + "\n"
                        )
                        jvp_fallback_emitted = True
                V = u_pred + (t - r) * du_dt.detach()

                loss_u_raw_vec = torch.mean((V - v_t) ** 2, dim=(1, 2, 3))
                loss_u_vec = torch.sum((V - v_t) ** 2, dim=(1, 2, 3))
                loss_u_vec = adp_wt_fn(loss_u_vec, norm_eps=loss_cfg.norm_eps, norm_p=loss_cfg.norm_p)

                loss_v_raw_vec = torch.mean((v_pred - v_t) ** 2, dim=(1, 2, 3))
                loss_v_vec = torch.sum((v_pred - v_t) ** 2, dim=(1, 2, 3))
                loss_v_vec = adp_wt_fn(loss_v_vec, norm_eps=loss_cfg.norm_eps, norm_p=loss_cfg.norm_p)

                lpips_vec, convnext_vec = (None, None)
                perc_vec = torch.zeros_like(loss_u_vec)
                if perceptual.enabled and float(loss_cfg.perceptual_max_t) > 0.0:
                    # Only run aux forward on the subset where the perceptual gate is non-zero.
                    t_flat = t.reshape(-1)
                    perc_mask = t_flat <= float(loss_cfg.perceptual_max_t)

                    need_lpips = bool(loss_cfg.enable_lpips) and float(loss_cfg.lpips_weight) > 0.0
                    need_convnext = bool(loss_cfg.enable_convnext) and float(loss_cfg.convnext_weight) > 0.0

                    if (need_lpips or need_convnext) and bool(perc_mask.any()):
                        idx = torch.nonzero(perc_mask, as_tuple=False).squeeze(1)
                        lp_sub, cv_sub = perceptual(x_hat_u[idx], images[idx])

                        if need_lpips:
                            lpips_vec = torch.zeros((bsz,), device=device, dtype=lp_sub.dtype)
                            lpips_vec.scatter_(0, idx, lp_sub.to(device=device))
                        if need_convnext:
                            convnext_vec = torch.zeros((bsz,), device=device, dtype=cv_sub.dtype)
                            convnext_vec.scatter_(0, idx, cv_sub.to(device=device))

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

            # Low-frequency probe: verify cond/uncond branch separation and edge usage.
            if cfg_probe_enabled and (state.step % int(cfg_probe_interval) == 0):
                probe_n = min(int(cfg_probe_batch_size), bsz)
                if probe_n > 0:
                    z_probe = z_t[:probe_n].detach()
                    t_probe = t[:probe_n].detach().reshape(-1)
                    h_probe = h[:probe_n].detach().reshape(-1)
                    sem_probe = sem_tokens[:probe_n].detach()
                    edge_probe = edge_map[:probe_n].detach() if edge_map is not None else None
                    t_probe_clamp = torch.clamp(t_probe.view(-1, 1, 1, 1), min=t_eps)
                    cond_mask = torch.zeros((probe_n,), device=device, dtype=torch.bool)
                    uncond_mask = torch.ones((probe_n,), device=device, dtype=torch.bool)

                    with torch.no_grad():
                        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                            out_cond = model(
                                x=z_probe,
                                t=t_probe,
                                h=h_probe,
                                sem_tokens=sem_probe,
                                sem_mask=None,
                                sem_drop_mask=cond_mask,
                                edge_map=edge_probe,
                            )
                            out_uncond = model(
                                x=z_probe,
                                t=t_probe,
                                h=h_probe,
                                sem_tokens=sem_probe,
                                sem_mask=None,
                                sem_drop_mask=uncond_mask,
                                edge_map=edge_probe,
                            )

                            u_cond = (z_probe - out_cond["x_hat_u"]) / t_probe_clamp
                            u_uncond = (z_probe - out_uncond["x_hat_u"]) / t_probe_clamp
                            v_cond = (z_probe - out_cond["x_hat_v"]) / t_probe_clamp
                            v_uncond = (z_probe - out_uncond["x_hat_v"]) / t_probe_clamp

                            u_delta = torch.sqrt(torch.mean((u_cond - u_uncond) ** 2, dim=(1, 2, 3)) + 1e-12).mean()
                            v_delta = torch.sqrt(torch.mean((v_cond - v_uncond) ** 2, dim=(1, 2, 3)) + 1e-12).mean()

                            probe_metrics = {
                                "train/cfg/u_cond_minus_uncond_norm_mean": u_delta,
                                "train/cfg/v_cond_minus_uncond_norm_mean": v_delta,
                            }

                            if use_edge_cond and edge_probe is not None:
                                # torch.quantile requires float32/float64; probe tensors can be bf16 under AMP.
                                edge_probe_f32 = edge_probe.to(dtype=torch.float32)
                                edge_flat = edge_probe_f32.reshape(probe_n, -1)
                                edge_p95 = torch.quantile(edge_flat, q=0.95, dim=1).mean()
                                probe_metrics["train/edge/edge_map_mean"] = edge_probe_f32.mean()
                                probe_metrics["train/edge/edge_map_p95"] = edge_p95

                                if getattr(model_core, "enable_edge_cond", False):
                                    patch_size = int(getattr(model_core, "patch_size"))
                                    seq_len = (int(z_probe.shape[-2]) // patch_size) * (int(z_probe.shape[-1]) // patch_size)
                                    edge_res = model_core._encode_edge(
                                        edge_probe.to(device=device),
                                        expected_seq_len=seq_len,
                                        dtype=z_probe.dtype,
                                    )
                                    edge_res_norm = torch.sqrt(torch.mean(edge_res ** 2, dim=(1, 2)) + 1e-12).mean()
                                    probe_metrics["train/edge/edge_res_norm_mean"] = edge_res_norm
                                    if getattr(model_core, "edge_gate", None) is not None:
                                        gate = model_core.edge_gate.detach()
                                        probe_metrics["train/edge/edge_gate_mean"] = gate.mean()
                                        probe_metrics["train/edge/edge_gate_max_layer"] = torch.argmax(
                                            torch.abs(gate)
                                        ).to(dtype=torch.float32)

                    reduced_probe = _reduce_metrics(probe_metrics)
                    writer.write_scalars(step=state.step, scalar_dict=reduced_probe)

            metrics = build_metrics(
                loss=loss,
                loss_u=loss_u_vec,
                loss_v=loss_v_vec,
                loss_u_raw=loss_u_raw_vec,
                loss_v_raw=loss_v_raw_vec,
                lpips_vec=lpips_vec,
                convnext_vec=convnext_vec,
            )
            metrics["cond_drop_ratio"] = drop_mask.float().mean().detach()
            metrics["lr"] = torch.tensor(optimizer.param_groups[0]["lr"], device=device)

            # Keep per-step metrics local; synchronize only on logging boundaries.
            log_meter.update(metrics)
            epoch_meter.update(metrics)

            if state.step % log_interval == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                summary = _reduce_metrics(log_meter.pop())
                summary["steps_per_second"] = float(log_interval) / elapsed
                summary["epoch"] = float(epoch)
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

        epoch_summary = _reduce_metrics(epoch_meter.pop())
        if is_main_process() and epoch_summary:
            train_loss_epoch = float(epoch_summary.get("loss", np.nan))
            topk_entries = _save_topk_train_loss(
                workdir=workdir,
                model=model_core,
                epoch=epoch + 1,
                step=state.step,
                metric_value=train_loss_epoch,
                entries=topk_entries,
                top_k=1,
            )

        eval_cfg = getattr(cfg, "evaluation", None)
        eval_enabled = bool(getattr(eval_cfg, "enabled", False)) if eval_cfg is not None else False
        eval_every = max(1, int(getattr(eval_cfg, "run_every_n_epochs", 1))) if eval_cfg is not None else 1
        need_epoch_eval = eval_enabled and (((epoch + 1) % eval_every) == 0)
        need_epoch_ckpt = bool(cfg.checkpoint.save_per_epoch) or need_epoch_eval

        checkpoint_path = os.path.join(workdir, "checkpoints", f"checkpoint_step_{state.step:08d}.pt")
        if need_epoch_ckpt:
            _save_checkpoint(
                workdir=workdir,
                state=state,
                model=model_core,
                optimizer=optimizer,
                scaler=scaler,
                keep_last_k=int(cfg.checkpoint.keep_last_k),
            )
        if need_epoch_eval:
            _maybe_run_epoch_eval(
                cfg=cfg,
                workdir=workdir,
                checkpoint_path=checkpoint_path,
                step=state.step,
                epoch=epoch,
                device=device,
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

    b = 1
    h = int(cfg.dataset.image_size)
    w = int(cfg.dataset.image_size)
    sem_toks = int(cfg.semantic.num_dino_tokens)
    sem_dim = int(cfg.semantic.dino_feature_dim)

    x = torch.randn(b, int(cfg.model.in_channels), h, w)
    t = torch.rand(b)
    h_span = torch.rand(b)
    sem = torch.randn(b, sem_toks, sem_dim)
    edge_map = None
    if bool(getattr(getattr(cfg, "spatial", None), "enable_edge", False)):
        edge_map = torch.rand(b, 1, h, w)

    with torch.no_grad():
        out = model(x=x, t=t, h=h_span, sem_tokens=sem, edge_map=edge_map)
    assert out["x_hat_u"].shape == x.shape
    assert out["x_hat_v"].shape == x.shape
    return {"ok": True, "shape": tuple(out["x_hat_u"].shape)}
