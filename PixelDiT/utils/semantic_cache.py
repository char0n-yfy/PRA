from __future__ import annotations

import hashlib
import json
import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class IndexedDatasetWrapper(Dataset):
    """Wrap any dataset to return (global_idx, image, label)."""

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[int(idx)]
        return int(idx), image, label


def _sha256_relpaths(relpaths: List[str]) -> str:
    h = hashlib.sha256()
    for p in relpaths:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


class DinoSemanticCache:
    """Read-only mmap cache for DINO features from build_semantic_cache.py outputs."""

    def __init__(
        self,
        cache_root: str,
        dino_feature_dim: int,
        num_dino_tokens: int,
        dataset_root: Optional[str] = None,
        image_size: Optional[int] = None,
        strict: bool = True,
    ):
        self.cache_root = os.path.abspath(str(cache_root))
        self.strict = bool(strict)
        self.dataset_root = os.path.abspath(str(dataset_root)) if dataset_root else None
        self.image_size = int(image_size) if image_size is not None else None
        self.dino_feature_dim = int(dino_feature_dim)
        self.num_dino_tokens = int(num_dino_tokens)
        self._memmaps = {}
        self._validated_splits = set()

        meta_path = os.path.join(self.cache_root, "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Semantic cache metadata not found: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self._validate_meta_compatibility()

    def _validate_meta_compatibility(self):
        if int(self.meta.get("format_version", 0)) != 1:
            raise ValueError(f"Unsupported cache format_version={self.meta.get('format_version')}")

        if self.image_size is not None:
            meta_img = int(self.meta.get("image_size", -1))
            if meta_img != self.image_size:
                raise ValueError(f"Cache image_size mismatch: cache={meta_img}, config={self.image_size}")

        layout = self.meta.get("packed_layout", {})
        dino_layout = layout.get("dino", {"enabled": False})
        if not bool(dino_layout.get("enabled", False)):
            raise ValueError("Cache does not contain DINO features.")

        dino_shape = dino_layout.get("shape", None)
        exp_shape = [self.num_dino_tokens, self.dino_feature_dim]
        if dino_shape != exp_shape:
            raise ValueError(f"Cache DINO shape mismatch: cache={dino_shape}, config={exp_shape}")

        if self.dataset_root:
            cache_root_meta = os.path.abspath(str(self.meta.get("dataset_root", "")))
            if cache_root_meta and cache_root_meta != self.dataset_root:
                raise ValueError(
                    f"Cache dataset_root mismatch: cache={cache_root_meta}, config={self.dataset_root}"
                )

    def _open_split_memmap(self, split: str):
        split = str(split)
        if split in self._memmaps:
            return self._memmaps[split]
        split_meta = self.meta.get("splits", {}).get(split)
        if split_meta is None:
            raise KeyError(f"Cache missing split `{split}`")
        feat_path = os.path.join(self.cache_root, split_meta["packed_feature_file"])
        arr = np.load(feat_path, mmap_mode="r")
        if not isinstance(arr, np.memmap):
            raise RuntimeError(f"Expected memmap, got {type(arr)}")
        self._memmaps[split] = arr
        return arr

    def validate_against_imagefolder(self, split: str, imagefolder_dataset):
        split = str(split)
        if split in self._validated_splits:
            return
        split_meta = self.meta.get("splits", {}).get(split)
        if split_meta is None:
            raise KeyError(f"Cache missing split `{split}`")

        if len(imagefolder_dataset) != int(split_meta.get("num_samples", -1)):
            raise ValueError(
                f"Cache split size mismatch: cache={split_meta.get('num_samples')}, dataset={len(imagefolder_dataset)}"
            )

        if self.strict:
            if not hasattr(imagefolder_dataset, "samples"):
                raise ValueError("ImageFolder dataset with `.samples` is required for strict cache validation.")
            relpaths_expected = [
                os.path.relpath(path, self.dataset_root)
                for path, _ in imagefolder_dataset.samples
            ]
            relpaths_hash = _sha256_relpaths(relpaths_expected)
            cache_hash = str(split_meta.get("relpaths_sha256", ""))
            if cache_hash and relpaths_hash != cache_hash:
                raise ValueError(
                    f"Cache relpaths hash mismatch for split `{split}`. "
                    "Image ordering/path set differs from cache."
                )

        self._validated_splits.add(split)

    def get_batch(self, split: str, indices, device: Optional[torch.device] = None) -> torch.Tensor:
        if torch.is_tensor(indices):
            idx_np = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            idx_np = np.asarray(indices, dtype=np.int64)
        if idx_np.ndim != 1:
            idx_np = idx_np.reshape(-1)

        arr = self._open_split_memmap(split)
        dino_layout = self.meta["packed_layout"]["dino"]
        start = int(dino_layout["offset"])
        end = start + int(dino_layout["length"])

        dino_np = arr[idx_np, start:end]
        dino_np = np.asarray(dino_np, dtype=np.float32).reshape(
            idx_np.shape[0], self.num_dino_tokens, self.dino_feature_dim
        )
        dino = torch.from_numpy(dino_np)
        if device is not None:
            dino = dino.to(device=device, non_blocking=True)
        return dino
