"""ImageNet input pipeline for pure PyTorch training."""

import hashlib
import json
import os
import random
import re
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision.datasets.folder import find_classes, pil_loader

from utils.data_util import build_semantic_encoder, parse_condition_mode
from utils.logging_util import get_rank, get_world_size, log_for_0


def loader(path: str):
    return pil_loader(path)


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def _sha256_relpaths(relpaths):
    h = hashlib.sha256()
    for p in relpaths:
        h.update(str(p).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _normalize_label_path(path_str: str, dataset_root: str):
    if path_str is None:
        return ""
    p = str(path_str).strip()
    if p == "":
        return ""
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(dataset_root, p))


def _resolve_flat_val_labels_file(dataset_cfg):
    dataset_root = os.path.abspath(str(getattr(dataset_cfg, "root", "")))
    explicit = _normalize_label_path(getattr(dataset_cfg, "val_labels_file", ""), dataset_root)
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(
        [
            os.path.join(dataset_root, "val_labels.txt"),
            os.path.join(dataset_root, "val.txt"),
            os.path.join(dataset_root, "ILSVRC2012_validation_ground_truth.txt"),
            os.path.join(dataset_root, "devkit", "data", "ILSVRC2012_validation_ground_truth.txt"),
            os.path.join(dataset_root, "imagenet_val_labels.txt"),
        ]
    )
    seen = set()
    for c in candidates:
        c_abs = os.path.abspath(c)
        if c_abs in seen:
            continue
        seen.add(c_abs)
        if os.path.isfile(c_abs):
            return c_abs
    return ""


def _list_flat_val_images(val_root: str):
    files = []
    for name in sorted(os.listdir(val_root)):
        path = os.path.join(val_root, name)
        if not os.path.isfile(path):
            continue
        if not name.lower().endswith(_IMG_EXTS):
            continue
        files.append(path)
    return files


def _load_train_class_index(dataset_root: str):
    train_root = os.path.join(dataset_root, "train")
    if not os.path.isdir(train_root):
        return [], {}
    try:
        classes, class_to_idx = find_classes(train_root)
        return classes, class_to_idx
    except Exception:
        return [], {}


def _parse_flat_val_labels(label_file: str, image_paths, dataset_root: str):
    image_names = [os.path.basename(p) for p in image_paths]
    n = len(image_names)
    lines = Path(label_file).read_text(encoding="utf-8").splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines:
        raise ValueError(f"Validation labels file is empty: {label_file}")

    pair_map = {}
    seq_labels = []
    pair_count = 0
    seq_count = 0
    for ln in lines:
        raw = ln.replace(",", " ").replace("\t", " ")
        parts = [p for p in raw.split(" ") if p]
        if len(parts) >= 2 and parts[0].lower().endswith(_IMG_EXTS):
            pair_map[parts[0]] = parts[-1]
            pair_count += 1
        elif len(parts) >= 2 and parts[0].startswith("ILSVRC2012_val_"):
            pair_map[parts[0]] = parts[-1]
            pair_count += 1
        elif len(parts) == 1:
            seq_labels.append(parts[0])
            seq_count += 1
        else:
            raise ValueError(f"Unsupported val label line format: {ln}")

    if pair_count > 0 and seq_count > 0:
        raise ValueError(
            f"Mixed val label formats detected in {label_file}. Use either `filename label` lines or one label per line."
        )

    if pair_count > 0:
        missing = [nm for nm in image_names if nm not in pair_map]
        if missing:
            raise ValueError(
                f"Validation labels file missing {len(missing)} filenames (e.g. {missing[:3]})"
            )
        label_tokens = [pair_map[nm] for nm in image_names]
    else:
        if len(seq_labels) != n:
            raise ValueError(
                f"Sequential val labels count mismatch: labels={len(seq_labels)}, images={n}. "
                "Expected one label per image in sorted filename order."
            )
        label_tokens = seq_labels

    classes, class_to_idx = _load_train_class_index(dataset_root)

    # Try integer labels first. If values are 1..K, convert to 0-based.
    int_vals = []
    ints_ok = True
    for tok in label_tokens:
        try:
            int_vals.append(int(tok))
        except Exception:
            ints_ok = False
            break

    if ints_ok:
        labels = np.asarray(int_vals, dtype=np.int64)
        if labels.size > 0 and labels.min() >= 1 and (labels.max() <= 1000) and not np.any(labels == 0):
            labels = labels - 1
        if labels.min(initial=0) < 0:
            raise ValueError("Validation integer labels contain negative values after normalization.")
        if not classes:
            max_cls = int(labels.max(initial=-1)) + 1
            classes = [str(i) for i in range(max_cls)]
            class_to_idx = {c: i for i, c in enumerate(classes)}
        return labels.tolist(), classes, class_to_idx

    if not class_to_idx:
        raise ValueError(
            "Validation labels are non-integer (e.g. synset names), but train class folders were not found to build class_to_idx."
        )

    labels = []
    missing_tokens = []
    for tok in label_tokens:
        if tok not in class_to_idx:
            missing_tokens.append(tok)
        else:
            labels.append(int(class_to_idx[tok]))
    if missing_tokens:
        uniq = sorted(set(missing_tokens))
        raise ValueError(
            f"Validation labels contain unknown class tokens not present in train/: {uniq[:5]}"
        )
    return labels, classes, class_to_idx


class FlatLabeledImageDataset(Dataset):
    """Image dataset for flat val directory + label file, with ImageFolder-like attributes."""

    def __init__(self, root, image_paths, labels, loader_fn, classes, class_to_idx):
        self.root = str(root)
        self.loader = loader_fn
        self.classes = list(classes)
        self.class_to_idx = dict(class_to_idx)
        self.samples = [(str(p), int(y)) for p, y in zip(image_paths, labels)]
        self.targets = [int(y) for _, y in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[int(idx)]
        sample = self.loader(path)
        return sample, target


def build_imagenet_dataset(dataset_cfg, split):
    """Build train/val dataset. Supports ImageFolder val or flat val + label file."""
    image_size = int(getattr(dataset_cfg, "image_size", 256))
    dataset_root = os.path.abspath(str(getattr(dataset_cfg, "root", "")))

    def loader_with_crop(path: str):
        img = pil_loader(path)
        img_cropped = center_crop_arr(img, image_size)
        return np.array(img_cropped)

    root = os.path.join(dataset_root, split)
    if split != "val":
        return datasets.ImageFolder(root, transform=None, loader=loader_with_crop)

    try:
        return datasets.ImageFolder(root, transform=None, loader=loader_with_crop)
    except FileNotFoundError as exc:
        # Fallback for flat ImageNet val layout: val/*.JPEG + labels file.
        if "Couldn't find any class folder" not in str(exc):
            raise

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Validation split not found: {root}")

    label_file = _resolve_flat_val_labels_file(dataset_cfg)
    image_paths = _list_flat_val_images(root)
    if not image_paths:
        raise FileNotFoundError(f"No images found under flat val directory: {root}")

    if label_file == "":
        # Labels are not used by the pMF semantic cache / eval logic; keep an ImageFolder-like
        # interface with dummy labels so flat val layouts can be consumed without devkit metadata.
        classes, class_to_idx = _load_train_class_index(dataset_root)
        if not classes:
            classes = ["dummy"]
            class_to_idx = {"dummy": 0}
        labels = [0] * len(image_paths)
        log_for_0(
            "Using flat val dataset fallback without label file; assigning dummy labels=0. "
            "This is safe for current cache/eval pipelines (labels are unused)."
        )
    else:
        labels, classes, class_to_idx = _parse_flat_val_labels(
            label_file, image_paths, dataset_root=dataset_root
        )
        log_for_0(
            f"Using flat val dataset fallback with label file: {label_file} "
            f"(images={len(image_paths)}, classes={len(classes)})"
        )

    ds = FlatLabeledImageDataset(
        root=root,
        image_paths=image_paths,
        labels=labels,
        loader_fn=loader_with_crop,
        classes=classes,
        class_to_idx=class_to_idx,
    )
    return ds


class IndexedDatasetWrapper(Dataset):
    """Wraps a dataset to emit (global_index, image, label)."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[int(idx)]
        return int(idx), image, label


class SemanticFeatureCache:
    """
    Read-only mmap cache for offline CLIP/DINO condition features.

    Expects layout produced by scripts/build_semantic_cache.py.
    """

    def __init__(self, cache_root, condition_cfg, model_cfg, dataset_cfg=None, strict=True):
        self.cache_root = os.path.abspath(str(cache_root))
        self.strict = bool(strict)
        meta_path = os.path.join(self.cache_root, "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Semantic cache metadata not found: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.mode, self.use_clip, self.use_dino = parse_condition_mode(condition_cfg)
        self.clip_feature_dim = int(getattr(model_cfg, "clip_feature_dim", 1024))
        self.dino_feature_dim = int(getattr(model_cfg, "dino_feature_dim", 768))
        self.num_dino_tokens = int(getattr(model_cfg, "num_dino_tokens", 4))
        self.image_size = int(getattr(dataset_cfg, "image_size", 0)) if dataset_cfg is not None else None
        self.dataset_root = (
            os.path.abspath(str(getattr(dataset_cfg, "root", ""))) if dataset_cfg is not None else None
        )
        self.clip_model_name = str(getattr(condition_cfg, "clip_model_name", ""))
        self.dino_model_name = str(getattr(condition_cfg, "dino_model_name", ""))
        self.dino_use_dense = bool(getattr(condition_cfg, "dino_use_dense", True))
        self._memmaps = {}
        self._validated_splits = set()

        self._validate_meta_compatibility()

    def _validate_meta_compatibility(self):
        meta = self.meta
        if int(meta.get("format_version", 0)) != 1:
            raise ValueError(f"Unsupported semantic cache format_version={meta.get('format_version')}")

        if self.image_size is not None:
            meta_image_size = int(meta.get("image_size", -1))
            if meta_image_size != self.image_size:
                raise ValueError(
                    f"Semantic cache image_size mismatch: cache={meta_image_size}, config={self.image_size}"
                )

        cache_mode = str(meta.get("condition_mode", "")).lower()
        if cache_mode != self.mode:
            # Allow using a superset cache (clip_dino) with a subset mode (clip/dino/uncond) if layouts match.
            allowed_superset = cache_mode == "clip_dino" and self.mode in {"clip", "dino", "clip_dino"}
            if not allowed_superset:
                raise ValueError(
                    f"Semantic cache condition_mode mismatch: cache={cache_mode}, config={self.mode}"
                )

        cache_cond = meta.get("condition", {})
        if self.use_clip and str(cache_cond.get("clip_model_name", "")) != self.clip_model_name:
            raise ValueError(
                "Semantic cache CLIP model mismatch: "
                f"cache={cache_cond.get('clip_model_name')}, config={self.clip_model_name}"
            )
        if self.use_dino and str(cache_cond.get("dino_model_name", "")) != self.dino_model_name:
            raise ValueError(
                "Semantic cache DINO model mismatch: "
                f"cache={cache_cond.get('dino_model_name')}, config={self.dino_model_name}"
            )
        if self.use_dino and bool(cache_cond.get("dino_use_dense", True)) != self.dino_use_dense:
            raise ValueError(
                "Semantic cache dino_use_dense mismatch: "
                f"cache={cache_cond.get('dino_use_dense')}, config={self.dino_use_dense}"
            )

        layout = meta.get("packed_layout", {})
        clip_layout = layout.get("clip", {"enabled": False})
        dino_layout = layout.get("dino", {"enabled": False})
        if self.use_clip:
            if not bool(clip_layout.get("enabled", False)):
                raise ValueError("Semantic cache does not contain CLIP features, but CLIP condition is enabled.")
            if int(clip_layout.get("length", -1)) != self.clip_feature_dim:
                raise ValueError(
                    f"Semantic cache CLIP dim mismatch: cache={clip_layout.get('length')}, config={self.clip_feature_dim}"
                )
        if self.use_dino:
            if not bool(dino_layout.get("enabled", False)):
                raise ValueError("Semantic cache does not contain DINO features, but DINO condition is enabled.")
            dino_shape = dino_layout.get("shape", None)
            if dino_shape != [self.num_dino_tokens, self.dino_feature_dim]:
                raise ValueError(
                    f"Semantic cache DINO shape mismatch: cache={dino_shape}, "
                    f"config={[self.num_dino_tokens, self.dino_feature_dim]}"
                )

    def validate_against_imagefolder(self, split, imagefolder_dataset):
        split = str(split)
        if split in self._validated_splits:
            return
        if split not in self.meta.get("splits", {}):
            raise KeyError(f"Semantic cache missing split `{split}`.")
        split_meta = self.meta["splits"][split]

        if len(imagefolder_dataset) != int(split_meta.get("num_samples", -1)):
            raise ValueError(
                f"Semantic cache split `{split}` size mismatch: "
                f"cache={split_meta.get('num_samples')}, dataset={len(imagefolder_dataset)}"
            )

        if self.dataset_root:
            cache_root_meta = os.path.abspath(str(self.meta.get("dataset_root", "")))
            if cache_root_meta and cache_root_meta != self.dataset_root:
                raise ValueError(
                    f"Semantic cache dataset_root mismatch: cache={cache_root_meta}, config={self.dataset_root}"
                )

        relpaths_expected = [os.path.relpath(path, self.dataset_root) for path, _ in imagefolder_dataset.samples]
        relpaths_hash = _sha256_relpaths(relpaths_expected)
        cache_hash = str(split_meta.get("relpaths_sha256", ""))
        if cache_hash and relpaths_hash != cache_hash:
            raise ValueError(
                f"Semantic cache relpaths hash mismatch for split `{split}`. "
                "ImageFolder ordering/path set differs from cache."
            )
        self._validated_splits.add(split)

    def _open_split_memmap(self, split):
        split = str(split)
        if split in self._memmaps:
            return self._memmaps[split]
        split_meta = self.meta["splits"][split]
        feat_path = os.path.join(self.cache_root, split_meta["packed_feature_file"])
        arr = np.load(feat_path, mmap_mode="r")
        if not isinstance(arr, np.memmap):
            raise RuntimeError(f"Expected memmap array for {feat_path}, got {type(arr)}")
        self._memmaps[split] = arr
        return arr

    def get_batch(self, split, indices, device=None):
        idx = indices
        if torch.is_tensor(idx):
            idx_np = idx.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            idx_np = np.asarray(idx, dtype=np.int64)
        if idx_np.ndim != 1:
            idx_np = idx_np.reshape(-1)

        arr = self._open_split_memmap(split)
        layout = self.meta["packed_layout"]
        out = {}

        if self.use_clip:
            clip_layout = layout["clip"]
            start = int(clip_layout["offset"])
            end = start + int(clip_layout["length"])
            clip_np = arr[idx_np, start:end]
            out["clip"] = torch.from_numpy(np.asarray(clip_np, dtype=np.float32))
        else:
            out["clip"] = None

        if self.use_dino:
            dino_layout = layout["dino"]
            start = int(dino_layout["offset"])
            end = start + int(dino_layout["length"])
            dino_np = arr[idx_np, start:end]
            dino_np = np.asarray(dino_np, dtype=np.float32).reshape(
                idx_np.shape[0], self.num_dino_tokens, self.dino_feature_dim
            )
            out["dino"] = torch.from_numpy(dino_np)
        else:
            out["dino"] = None

        if device is not None:
            for k, v in out.items():
                if torch.is_tensor(v):
                    out[k] = v.to(device=device, non_blocking=True)
        return out


def create_semantic_cache(condition_cfg, model_cfg, dataset_cfg):
    if condition_cfg is None:
        return None
    cache_root = str(getattr(condition_cfg, "semantic_cache_root", "")).strip()
    if cache_root == "":
        return None
    strict = bool(getattr(condition_cfg, "semantic_cache_strict", True))
    return SemanticFeatureCache(
        cache_root=cache_root,
        condition_cfg=condition_cfg,
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
        strict=strict,
    )


def prepare_batch_data(
    batch,
    semantic_encoder=None,
    semantic_cache=None,
    semantic_cache_split=None,
    clip_feature_dim=1024,
    dino_feature_dim=768,
    num_dino_tokens=4,
    use_clip_condition=True,
    use_dino_condition=True,
    use_flip=False,
):
    """
    Args:
      batch = (image, label) or (index, image, label)
        image: torch uint8 tensor, shape (B, H, W, C)
        label: torch int tensor, shape (B,)
    Returns:
      dict with:
        image: float tensor (B, C, H, W), normalized to [-1, 1]
        clip_emb: float tensor (B, clip_feature_dim)
        dino_emb: float tensor (B, num_dino_tokens, dino_feature_dim)
        label: int tensor
    """
    if not isinstance(batch, (tuple, list)):
        raise TypeError(f"Expected batch tuple/list, got {type(batch)}")
    if len(batch) == 3:
        sample_idx, image, label = batch
    elif len(batch) == 2:
        sample_idx = None
        image, label = batch
    else:
        raise ValueError(f"Unsupported batch length={len(batch)}. Expected 2 or 3 elements.")

    if sample_idx is not None and not torch.is_tensor(sample_idx):
        sample_idx = torch.as_tensor(sample_idx, dtype=torch.int64)
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    if not torch.is_tensor(label):
        label = torch.as_tensor(label)

    bsz = image.shape[0]
    clip_emb = torch.zeros((bsz, clip_feature_dim), dtype=torch.float32)
    dino_emb = torch.zeros((bsz, num_dino_tokens, dino_feature_dim), dtype=torch.float32)

    clip_emb_np = None
    dino_emb_np = None
    used_cache = False
    if semantic_cache is not None and sample_idx is not None and semantic_cache_split is not None:
        cached = semantic_cache.get_batch(split=semantic_cache_split, indices=sample_idx)
        used_cache = True
        if use_clip_condition and cached["clip"] is not None:
            clip_emb = cached["clip"]
        if use_dino_condition and cached["dino"] is not None:
            dino_emb = cached["dino"]

    if semantic_encoder is not None and not used_cache:
        clip_emb_np, dino_emb_np = semantic_encoder.encode(image)

        if use_clip_condition:
            if clip_emb_np is None:
                raise ValueError("CLIP condition is enabled but CLIP embeddings are unavailable.")
            if clip_emb_np.shape[1] != clip_feature_dim:
                raise ValueError(
                    f"CLIP embedding dim mismatch: got {clip_emb_np.shape[1]}, expected {clip_feature_dim}"
                )
            clip_emb = torch.from_numpy(clip_emb_np)

        if use_dino_condition:
            if dino_emb_np is None:
                raise ValueError("DINO condition is enabled but DINO embeddings are unavailable.")
            if dino_emb_np.ndim != 3:
                raise ValueError(
                    f"DINO dense embedding rank mismatch: got rank={dino_emb_np.ndim}, expected 3."
                )
            if dino_emb_np.shape[2] != dino_feature_dim:
                raise ValueError(
                    f"DINO embedding dim mismatch: got {dino_emb_np.shape[2]}, expected {dino_feature_dim}"
                )
            if dino_emb_np.shape[1] != int(num_dino_tokens):
                raise ValueError(
                    f"DINO token count mismatch: got {dino_emb_np.shape[1]}, expected {num_dino_tokens}"
                )
            dino_emb = torch.from_numpy(dino_emb_np)

    image = image.float() / 255.0  # NHWC
    if use_flip:
        flip_mask = torch.rand((image.shape[0], 1, 1, 1), dtype=image.dtype) < 0.5
        image_flipped = torch.flip(image, dims=[2])  # horizontal flip in NHWC (W axis)
        image = torch.where(flip_mask, image_flipped, image)
    image = (image - 0.5) / 0.5
    image = image.permute(0, 3, 1, 2).contiguous()  # NCHW

    return {
        "index": sample_idx if sample_idx is not None else torch.full((bsz,), -1, dtype=torch.int64),
        "image": image,
        "label": label,
        "clip_emb": clip_emb.float(),
        "dino_emb": dino_emb.float(),
    }


def create_semantic_encoder(condition_cfg, num_dino_tokens=4, use_clip=None, use_dino=None):
    return build_semantic_encoder(
        condition_cfg,
        num_dino_tokens=num_dino_tokens,
        use_clip=use_clip,
        use_dino=use_dino,
    )


def resolve_condition_mode(condition_cfg):
    return parse_condition_mode(condition_cfg)


def worker_init_fn(worker_id, rank):
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def create_imagenet_split(dataset_cfg, batch_size, split, semantic_cache=None):
    rank = get_rank()
    world_size = get_world_size()
    ds = build_imagenet_dataset(dataset_cfg, split)
    if semantic_cache is not None:
        semantic_cache.validate_against_imagefolder(split=split, imagefolder_dataset=ds)
        ds_for_loader = IndexedDatasetWrapper(ds)
    else:
        ds_for_loader = ds

    sampler = DistributedSampler(
        ds_for_loader,
        num_replicas=world_size,
        rank=rank,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
    )
    it = DataLoader(
        ds_for_loader,
        batch_size=batch_size,
        drop_last=(split == "train"),
        worker_init_fn=partial(worker_init_fn, rank=rank),
        sampler=sampler,
        num_workers=dataset_cfg.num_workers,
        prefetch_factor=(dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None),
        pin_memory=dataset_cfg.pin_memory,
        persistent_workers=True if dataset_cfg.num_workers > 0 else False,
    )
    steps_per_epoch = len(it)
    log_for_0(f"Dataset {split}: {len(ds)} images, steps_per_epoch={steps_per_epoch}")
    return it, steps_per_epoch
