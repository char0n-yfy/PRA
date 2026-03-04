#!/usr/bin/env python3
"""
Build offline semantic feature cache (CLIP/DINO) for ImageNet in fp32.

Design goals (for training-time throughput):
- Single packed contiguous feature array per split: one memmap row read per sample.
- ImageFolder sequential index order is preserved exactly (shuffle=False).
- Optional multi-GPU extraction via torchrun; each rank writes disjoint contiguous rows.
- File format is .npy memmap (NumPy open_memmap), easy to mmap in training workers.

Output layout:
  <output_root>/
    meta.json
    train/
      cond_packed_fp32.npy    # [N, total_cond_dim], float32
      labels_int64.npy        # [N], int64
      relpaths.txt            # N lines, path relative to dataset root
    val/
      ...

`cond_packed_fp32.npy` layout is recorded in meta.json:
- CLIP segment: [offset, offset + clip_dim)
- DINO segment: [offset, offset + num_dino_tokens * dino_dim)

Example (2 GPUs):
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    scripts/build_semantic_cache.py \
    --dataset-root /root/autodl-tmp/imagenet \
    --config configs/base/pMF_B_16_config.yml \
    --output-root /root/autodl-tmp/imagenet_sem_cache_fp32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.datasets.folder import pil_loader
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.load_config import get_config, get_config_from_file  # noqa: E402
from utils.data_util import build_semantic_encoder, parse_condition_mode  # noqa: E402
from utils.input_pipeline import build_imagenet_dataset, center_crop_arr  # noqa: E402


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _log(msg: str, rank0_only: bool = False):
    if rank0_only and _rank() != 0:
        return
    print(msg, flush=True)


def _barrier():
    if _is_dist():
        dist.barrier()


def _dist_device():
    if torch.cuda.is_available():
        return torch.device("cuda", _local_rank())
    return torch.device("cpu")


def _reduce_sum_int(x: int) -> int:
    if not _is_dist():
        return int(x)
    t = torch.tensor([int(x)], device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def _reduce_max_float(x: float) -> float:
    if not _is_dist():
        return float(x)
    t = torch.tensor([float(x)], device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _broadcast_bool_from_rank0(flag: bool) -> bool:
    if not _is_dist():
        return bool(flag)
    t = torch.tensor([1 if flag else 0], device=_dist_device(), dtype=torch.int32)
    dist.broadcast(t, src=0)
    return bool(int(t.item()))


def _init_dist_if_needed(timeout_sec: int):
    ws = _world_size()
    if ws <= 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(_local_rank())
        backend = "nccl"
    else:
        backend = "gloo"

    if not _is_dist():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=torch.distributed.constants.default_pg_timeout
            if timeout_sec is None
            else __import__("datetime").timedelta(seconds=int(timeout_sec)),
        )
    _log(
        f"[rank {_rank()}] dist initialized: backend={backend}, world_size={ws}, local_rank={_local_rank()}",
        rank0_only=False,
    )


def _destroy_dist():
    if _is_dist():
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _load_cfg(config_arg: str):
    p = Path(config_arg)
    if p.is_file():
        return get_config_from_file(str(p))
    return get_config(config_arg)


def _worker_init_fn(worker_id: int):
    base = 12345 + 1000 * _rank()
    seed = base + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)


def _crop_loader(path: str, image_size: int):
    img = pil_loader(path)
    img = center_crop_arr(img, image_size)
    return np.asarray(img, dtype=np.uint8)


class IndexedSubsetDataset(Dataset):
    def __init__(self, base: Dataset, start: int, end: int):
        self.base = base
        self.start = int(start)
        self.end = int(end)
        if self.end < self.start:
            raise ValueError(f"Invalid subset range [{self.start}, {self.end})")

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int):
        global_idx = self.start + int(idx)
        image, label = self.base[global_idx]
        return global_idx, image, label


def _partition_contiguous(n: int, rank: int, world_size: int) -> Tuple[int, int]:
    start = (n * rank) // world_size
    end = (n * (rank + 1)) // world_size
    return start, end


def _sha256_relpaths(relpaths: List[str]) -> str:
    h = hashlib.sha256()
    for p in relpaths:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _build_imagefolder(dataset_root: str, split: str, image_size: int):
    # Deprecated helper kept for backward compatibility; use input_pipeline builder.
    class _DatasetCfgShim:
        def __init__(self, root, image_size):
            self.root = root
            self.image_size = image_size
            self.val_labels_file = ""

    return build_imagenet_dataset(_DatasetCfgShim(dataset_root, image_size), split)


def _ensure_split_files(
    split_dir: Path,
    n: int,
    total_dim: int,
    labels: np.ndarray,
    relpaths: List[str],
    overwrite: bool,
):
    split_dir.mkdir(parents=True, exist_ok=True)
    feat_path = split_dir / "cond_packed_fp32.npy"
    label_path = split_dir / "labels_int64.npy"
    relpaths_path = split_dir / "relpaths.txt"

    if overwrite:
        for p in [feat_path, label_path, relpaths_path]:
            if p.exists():
                p.unlink()

    if not feat_path.exists():
        mm = np.lib.format.open_memmap(
            feat_path,
            mode="w+",
            dtype=np.float32,
            shape=(n, total_dim),
        )
        mm.flush()
        del mm

    if not label_path.exists():
        mm = np.lib.format.open_memmap(
            label_path,
            mode="w+",
            dtype=np.int64,
            shape=(n,),
        )
        mm[:] = labels.astype(np.int64, copy=False)
        mm.flush()
        del mm

    if not relpaths_path.exists():
        relpaths_path.write_text("\n".join(relpaths) + ("\n" if relpaths else ""), encoding="utf-8")

    return feat_path, label_path, relpaths_path


def _open_feature_memmap(path: Path):
    mm = np.load(path, mmap_mode="r+")
    if not isinstance(mm, np.memmap):
        raise RuntimeError(f"Expected memmap when opening {path}, got {type(mm)}")
    return mm


def _sample_row_indices(n: int, k: int, seed: int = 20260222) -> np.ndarray:
    n = int(n)
    k = int(max(0, min(k, n)))
    if n <= 0 or k <= 0:
        return np.zeros((0,), dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)

    anchor = np.array([0, n - 1], dtype=np.int64) if n > 1 else np.array([0], dtype=np.int64)
    lin_k = min(max(2, k // 4), k)
    lin = np.linspace(0, n - 1, num=lin_k, dtype=np.int64)
    idx = np.unique(np.concatenate([anchor, lin], axis=0))

    if idx.size < k:
        rng = np.random.default_rng(seed + n + k)
        extra = rng.integers(0, n, size=(k - idx.size) * 2 + 8, dtype=np.int64)
        idx = np.unique(np.concatenate([idx, extra], axis=0))
    if idx.size < k:
        # Deterministic backfill.
        full = np.arange(n, dtype=np.int64)
        idx = np.unique(np.concatenate([idx, full[: (k - idx.size)]], axis=0))
    return np.sort(idx[:k].astype(np.int64, copy=False))


def _validate_split_cache_integrity(
    output_root: Path,
    split: str,
    split_meta: Dict[str, object],
    sample_rows: int = 1024,
):
    feat_path = output_root / str(split_meta["packed_feature_file"])
    label_path = output_root / str(split_meta["labels_file"])
    relpaths_path = output_root / str(split_meta["relpaths_file"])

    for p in (feat_path, label_path, relpaths_path):
        if not p.exists():
            raise FileNotFoundError(f"[{split}] missing cache artifact: {p}")

    feat = np.load(feat_path, mmap_mode="r")
    if not isinstance(feat, np.memmap):
        raise RuntimeError(f"[{split}] feature file is not memmap-backed .npy: {feat_path}")
    exp_feat_shape = tuple(int(v) for v in split_meta["feature_shape"])
    if tuple(feat.shape) != exp_feat_shape:
        raise ValueError(f"[{split}] feature shape mismatch: got {tuple(feat.shape)}, expected {exp_feat_shape}")
    if feat.dtype != np.float32:
        raise ValueError(f"[{split}] feature dtype mismatch: got {feat.dtype}, expected float32")

    labels = np.load(label_path, mmap_mode="r")
    if tuple(labels.shape) != (int(split_meta["num_samples"]),):
        raise ValueError(
            f"[{split}] labels shape mismatch: got {tuple(labels.shape)}, expected {(int(split_meta['num_samples']),)}"
        )
    if labels.dtype != np.int64:
        raise ValueError(f"[{split}] labels dtype mismatch: got {labels.dtype}, expected int64")

    relpaths = relpaths_path.read_text(encoding="utf-8").splitlines()
    if len(relpaths) != int(split_meta["num_samples"]):
        raise ValueError(
            f"[{split}] relpaths count mismatch: got {len(relpaths)}, expected {int(split_meta['num_samples'])}"
        )
    expected_hash = str(split_meta.get("relpaths_sha256", ""))
    if expected_hash:
        got_hash = _sha256_relpaths(relpaths)
        if got_hash != expected_hash:
            raise ValueError(
                f"[{split}] relpaths hash mismatch: got {got_hash}, expected {expected_hash}"
            )

    n, d = exp_feat_shape
    idx = _sample_row_indices(n, sample_rows)
    if idx.size > 0:
        rows = np.asarray(feat[idx], dtype=np.float32)
        if rows.shape != (idx.size, d):
            raise ValueError(
                f"[{split}] sampled feature block shape mismatch: got {rows.shape}, expected {(idx.size, d)}"
            )
        if not np.isfinite(rows).all():
            bad = np.argwhere(~np.isfinite(rows))
            first_bad = tuple(int(v) for v in bad[0]) if bad.size > 0 else ("?", "?")
            raise ValueError(f"[{split}] sampled features contain non-finite values at sample offset {first_bad}")
        row_l2 = np.linalg.norm(rows, axis=1)
        zero_rows = int(np.sum(row_l2 <= 1e-12))
        if zero_rows > 0:
            raise ValueError(
                f"[{split}] sampled features contain {zero_rows}/{idx.size} near-zero rows; cache may be partially unwritten."
            )

    del feat
    del labels


def _build_split_cache(
    cfg,
    dataset_root: str,
    output_root: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    overwrite: bool,
):
    rank = _rank()
    world_size = _world_size()

    model_cfg = cfg.model
    cond_cfg = cfg.condition
    mode, use_clip, use_dino = parse_condition_mode(cond_cfg)
    if not (use_clip or use_dino):
        raise ValueError(
            "condition.mode resolves to `uncond`; no semantic branch is enabled. "
            "Offline semantic cache is unnecessary."
        )

    clip_dim = int(getattr(model_cfg, "clip_feature_dim", 1024)) if use_clip else 0
    dino_dim = int(getattr(model_cfg, "dino_feature_dim", 768)) if use_dino else 0
    num_dino_tokens = int(getattr(model_cfg, "num_dino_tokens", 4)) if use_dino else 0
    total_dim = clip_dim + (num_dino_tokens * dino_dim)
    if total_dim <= 0:
        raise ValueError("Computed total condition dim <= 0.")

    image_size = int(getattr(cfg.dataset, "image_size", 256))
    ds_cfg = SimpleNamespace(
        root=dataset_root,
        image_size=image_size,
        val_labels_file=getattr(cfg.dataset, "val_labels_file", ""),
    )
    ds = build_imagenet_dataset(ds_cfg, split)
    n = len(ds)
    split_dir = output_root / split

    relpaths = [os.path.relpath(path, dataset_root) for path, _ in ds.samples]
    labels = np.asarray([target for _, target in ds.samples], dtype=np.int64)

    if rank == 0:
        _log(
            f"[{split}] size={n}, classes={len(ds.classes)}, image_size={image_size}, mode={mode}, total_dim={total_dim}",
            rank0_only=True,
        )
        _ensure_split_files(
            split_dir=split_dir,
            n=n,
            total_dim=total_dim,
            labels=labels,
            relpaths=relpaths,
            overwrite=overwrite,
        )
    _barrier()

    feat_mm = _open_feature_memmap(split_dir / "cond_packed_fp32.npy")

    local_start, local_end = _partition_contiguous(n, rank, world_size)
    local_ds = IndexedSubsetDataset(ds, local_start, local_end)

    # Build one encoder per rank so each rank uses its own GPU when launched with torchrun.
    semantic_encoder = build_semantic_encoder(
        cond_cfg,
        num_dino_tokens=num_dino_tokens if use_dino else 1,
        use_clip=use_clip,
        use_dino=use_dino,
    )
    if semantic_encoder is None:
        raise RuntimeError("Failed to build semantic encoder while semantic mode is enabled.")

    loader = DataLoader(
        local_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        worker_init_fn=_worker_init_fn,
    )

    local_count = 0
    tic = time.time()
    last_log = tic

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=len(local_ds),
            desc=f"{split}[r{rank}]",
            position=rank,
            leave=True,
            dynamic_ncols=True,
            mininterval=1.0,
            smoothing=0.1,
            unit="img",
        )

    clip_offset = 0
    dino_offset = clip_dim
    try:
        for step, batch in enumerate(loader):
            global_idx, images, _labels = batch
            if not torch.is_tensor(global_idx):
                global_idx = torch.as_tensor(global_idx, dtype=torch.int64)
            idx_np = global_idx.numpy().astype(np.int64, copy=False)

            # images: torch uint8 tensor [B,H,W,C] due to default collate on np arrays
            clip_np, dino_np = semantic_encoder.encode(images)
            bsz = int(idx_np.shape[0])

            # Validate shapes on first batch of each split/rank.
            if step == 0:
                if use_clip:
                    if clip_np is None or clip_np.shape != (bsz, clip_dim):
                        raise ValueError(
                            f"[{split}] CLIP cache shape mismatch on rank {rank}: "
                            f"got {None if clip_np is None else clip_np.shape}, expected {(bsz, clip_dim)}"
                        )
                if use_dino:
                    exp_shape = (bsz, num_dino_tokens, dino_dim)
                    if dino_np is None or dino_np.shape != exp_shape:
                        raise ValueError(
                            f"[{split}] DINO cache shape mismatch on rank {rank}: "
                            f"got {None if dino_np is None else dino_np.shape}, expected {exp_shape}"
                        )

            contiguous = (
                bsz > 0
                and idx_np[0] >= local_start
                and idx_np[-1] < local_end
                and np.all(idx_np == np.arange(idx_np[0], idx_np[0] + bsz, dtype=np.int64))
            )

            if contiguous:
                dst = feat_mm[idx_np[0] : idx_np[0] + bsz]
                if use_clip:
                    dst[:, clip_offset : clip_offset + clip_dim] = clip_np.astype(np.float32, copy=False)
                if use_dino:
                    flat_dino = dino_np.reshape(bsz, -1)
                    dst[:, dino_offset : dino_offset + flat_dino.shape[1]] = flat_dino.astype(
                        np.float32, copy=False
                    )
            else:
                if use_clip:
                    feat_mm[idx_np, clip_offset : clip_offset + clip_dim] = clip_np.astype(
                        np.float32, copy=False
                    )
                if use_dino:
                    flat_dino = dino_np.reshape(bsz, -1)
                    feat_mm[idx_np, dino_offset : dino_offset + flat_dino.shape[1]] = flat_dino.astype(
                        np.float32, copy=False
                    )

            local_count += bsz
            if pbar is not None:
                pbar.update(bsz)
                elapsed = max(time.time() - tic, 1e-6)
                pbar.set_postfix({"img/s": f"{local_count / elapsed:.1f}"}, refresh=False)

            now = time.time()
            if (now - last_log) >= 10.0:
                elapsed = now - tic
                ips = local_count / max(elapsed, 1e-6)
                _log(
                    f"[rank {rank}] [{split}] step={step+1}/{len(loader)} local_done={local_count}/{len(local_ds)} ({ips:.1f} img/s)",
                    rank0_only=False,
                )
                last_log = now
    finally:
        if pbar is not None:
            pbar.close()

    feat_mm.flush()
    del feat_mm

    elapsed_local = time.time() - tic
    total_done = _reduce_sum_int(local_count)
    max_elapsed = _reduce_max_float(elapsed_local)

    if rank == 0:
        global_ips = total_done / max(max_elapsed, 1e-6)
        feat_bytes = n * total_dim * 4
        _log(
            f"[{split}] done: samples={total_done}, feat_shape=({n}, {total_dim}), "
            f"feat_size={feat_bytes / (1024 ** 3):.2f} GiB, throughput~{global_ips:.1f} img/s (world_size={world_size})",
            rank0_only=True,
        )

    split_meta = {
        "num_samples": int(n),
        "num_classes": int(len(ds.classes)),
        "feature_dtype": "float32",
        "packed_feature_file": str((split_dir / "cond_packed_fp32.npy").relative_to(output_root)),
        "labels_file": str((split_dir / "labels_int64.npy").relative_to(output_root)),
        "relpaths_file": str((split_dir / "relpaths.txt").relative_to(output_root)),
        "feature_shape": [int(n), int(total_dim)],
        "feature_size_bytes": int(n * total_dim * 4),
        "relpaths_sha256": _sha256_relpaths(relpaths),
    }
    return split_meta


def _write_meta(
    output_root: Path,
    cfg,
    dataset_root: str,
    split_metas: Dict[str, dict],
):
    mode, use_clip, use_dino = parse_condition_mode(cfg.condition)
    clip_dim = int(getattr(cfg.model, "clip_feature_dim", 1024)) if use_clip else 0
    dino_dim = int(getattr(cfg.model, "dino_feature_dim", 768)) if use_dino else 0
    num_dino_tokens = int(getattr(cfg.model, "num_dino_tokens", 4)) if use_dino else 0

    offset = 0
    layout = {}
    if use_clip:
        layout["clip"] = {
            "enabled": True,
            "offset": offset,
            "length": clip_dim,
            "shape": [clip_dim],
        }
        offset += clip_dim
    else:
        layout["clip"] = {"enabled": False}

    if use_dino:
        dino_flat = num_dino_tokens * dino_dim
        layout["dino"] = {
            "enabled": True,
            "offset": offset,
            "length": dino_flat,
            "shape": [num_dino_tokens, dino_dim],
        }
        offset += dino_flat
    else:
        layout["dino"] = {"enabled": False}

    meta = {
        "format_version": 1,
        "dataset_root": dataset_root,
        "build_time_unix": int(time.time()),
        "image_size": int(getattr(cfg.dataset, "image_size", 256)),
        "condition_mode": mode,
        "condition": {
            "clip_model_name": str(getattr(cfg.condition, "clip_model_name", "")),
            "dino_model_name": str(getattr(cfg.condition, "dino_model_name", "")),
            "dino_use_dense": bool(getattr(cfg.condition, "dino_use_dense", True)),
        },
        "packed_layout": layout,
        "total_condition_dim": int(offset),
        "storage": {
            "feature_dtype": "float32",
            "format": "numpy_npy_memmap_packed",
            "ordering": "torchvision.datasets.ImageFolder sequential order (shuffle=False) over each split",
        },
        "splits": split_metas,
    }

    (output_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _load_existing_meta_if_any(output_root: Path):
    meta_path = output_root / "meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read existing metadata at {meta_path}: {exc}") from exc


def _expected_meta_signature_from_cfg(cfg):
    mode, use_clip, use_dino = parse_condition_mode(cfg.condition)
    clip_dim = int(getattr(cfg.model, "clip_feature_dim", 1024)) if use_clip else 0
    dino_dim = int(getattr(cfg.model, "dino_feature_dim", 768)) if use_dino else 0
    num_dino_tokens = int(getattr(cfg.model, "num_dino_tokens", 4)) if use_dino else 0
    total_dim = clip_dim + (num_dino_tokens * dino_dim)
    return {
        "format_version": 1,
        "image_size": int(getattr(cfg.dataset, "image_size", 256)),
        "condition_mode": mode,
        "clip_model_name": str(getattr(cfg.condition, "clip_model_name", "")),
        "dino_model_name": str(getattr(cfg.condition, "dino_model_name", "")),
        "dino_use_dense": bool(getattr(cfg.condition, "dino_use_dense", True)),
        "total_condition_dim": int(total_dim),
    }


def _assert_existing_meta_compatible(existing_meta, cfg, dataset_root: str):
    sig = _expected_meta_signature_from_cfg(cfg)
    if int(existing_meta.get("format_version", -1)) != int(sig["format_version"]):
        raise ValueError(
            f"Existing meta format_version mismatch: {existing_meta.get('format_version')} vs {sig['format_version']}"
        )
    if int(existing_meta.get("image_size", -1)) != int(sig["image_size"]):
        raise ValueError(
            f"Existing meta image_size mismatch: {existing_meta.get('image_size')} vs {sig['image_size']}"
        )
    if str(existing_meta.get("condition_mode", "")).lower() != str(sig["condition_mode"]).lower():
        raise ValueError(
            f"Existing meta condition_mode mismatch: {existing_meta.get('condition_mode')} vs {sig['condition_mode']}"
        )
    cond_meta = existing_meta.get("condition", {})
    if str(cond_meta.get("clip_model_name", "")) != sig["clip_model_name"]:
        raise ValueError("Existing meta clip_model_name mismatch.")
    if str(cond_meta.get("dino_model_name", "")) != sig["dino_model_name"]:
        raise ValueError("Existing meta dino_model_name mismatch.")
    if bool(cond_meta.get("dino_use_dense", True)) != bool(sig["dino_use_dense"]):
        raise ValueError("Existing meta dino_use_dense mismatch.")
    if int(existing_meta.get("total_condition_dim", -1)) != int(sig["total_condition_dim"]):
        raise ValueError(
            f"Existing meta total_condition_dim mismatch: {existing_meta.get('total_condition_dim')} vs {sig['total_condition_dim']}"
        )
    old_root = os.path.abspath(str(existing_meta.get("dataset_root", "")))
    if old_root and old_root != os.path.abspath(dataset_root):
        raise ValueError(f"Existing meta dataset_root mismatch: {old_root} vs {os.path.abspath(dataset_root)}")


def parse_args():
    p = argparse.ArgumentParser(description="Build offline CLIP/DINO semantic cache for ImageNet.")
    p.add_argument("--dataset-root", type=str, default="/root/autodl-tmp/imagenet")
    p.add_argument("--config", type=str, default="configs/base/pMF_B_16_config.yml")
    p.add_argument("--output-root", type=str, required=True)
    p.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Delete and rebuild existing split files.")
    p.add_argument(
        "--validate-sample-rows",
        type=int,
        default=1024,
        help="Number of rows to sample per split for post-write integrity validation.",
    )
    p.add_argument("--timeout-sec", type=int, default=1800)
    return p.parse_args()


def main():
    args = parse_args()
    dataset_root = os.path.abspath(args.dataset_root)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    _init_dist_if_needed(args.timeout_sec)

    if _rank() == 0:
        _log("=== Offline Semantic Cache Builder ===", rank0_only=True)
        _log(f"dataset_root={dataset_root}", rank0_only=True)
        _log(f"output_root={str(output_root)}", rank0_only=True)
        _log(f"config={args.config}", rank0_only=True)
        _log(f"world_size={_world_size()}, torch={torch.__version__}", rank0_only=True)
        if torch.cuda.is_available():
            _log(
                f"cuda_runtime={torch.version.cuda}, gpu_count={torch.cuda.device_count()}",
                rank0_only=True,
            )

    cfg = _load_cfg(args.config)
    split_metas = {}
    existing_meta = None
    if _rank() == 0:
        existing_meta = _load_existing_meta_if_any(output_root)
        if existing_meta is not None:
            _assert_existing_meta_compatible(existing_meta, cfg, dataset_root=dataset_root)
            existing_splits = sorted(list(existing_meta.get("splits", {}).keys()))
            _log(
                f"Found existing meta.json with splits={existing_splits}; unprocessed splits will be preserved.",
                rank0_only=True,
            )

    try:
        for split in args.splits:
            _barrier()
            split_meta = _build_split_cache(
                cfg=cfg,
                dataset_root=dataset_root,
                output_root=output_root,
                split=split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                pin_memory=(not args.no_pin_memory),
                overwrite=args.overwrite,
            )
            _barrier()
            validation_failed = False
            if _rank() == 0:
                try:
                    _log(
                        f"[{split}] validating cache integrity (sample_rows={int(args.validate_sample_rows)})...",
                        rank0_only=True,
                    )
                    _validate_split_cache_integrity(
                        output_root=output_root,
                        split=split,
                        split_meta=split_meta,
                        sample_rows=int(args.validate_sample_rows),
                    )
                    _log(f"[{split}] integrity validation passed.", rank0_only=True)
                except Exception as exc:
                    validation_failed = True
                    _log(f"[{split}] integrity validation failed: {exc}", rank0_only=True)
            validation_failed = _broadcast_bool_from_rank0(validation_failed)
            if validation_failed:
                raise RuntimeError(
                    f"Split `{split}` cache integrity validation failed. "
                    "Aborting before processing subsequent splits."
                )
            if _rank() == 0:
                split_metas[split] = split_meta
            _barrier()
        _barrier()

        if _rank() == 0:
            if existing_meta is not None:
                old_splits = dict(existing_meta.get("splits", {}))
                for k, v in old_splits.items():
                    if k not in split_metas:
                        split_metas[k] = v
            _write_meta(output_root=output_root, cfg=cfg, dataset_root=dataset_root, split_metas=split_metas)
            total_bytes = sum(m["feature_size_bytes"] for m in split_metas.values())
            _log(
                f"Finished. Total feature bytes={total_bytes} ({total_bytes / (1024 ** 3):.2f} GiB).",
                rank0_only=True,
            )
            _log(f"Metadata: {output_root / 'meta.json'}", rank0_only=True)
    finally:
        _destroy_dist()


if __name__ == "__main__":
    main()
