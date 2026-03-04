import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3

from utils.data_util import create_imagenet_dataloader
from utils.logging_util import log_for_0


def _softmax(logits: np.ndarray, axis: int = -1):
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _to_uint8_bhwc(x):
    if torch.is_tensor(x):
        t = x.detach().cpu()
        if t.ndim != 4:
            raise ValueError(f"Expected rank-4 image tensor, got shape {tuple(t.shape)}")
        if t.shape[-1] == 3:
            if t.dtype != torch.uint8:
                t = torch.clamp(t, 0, 255).to(torch.uint8)
            return t.numpy()
        if t.shape[1] == 3:
            if t.dtype == torch.uint8:
                t = t.permute(0, 2, 3, 1)
            else:
                if t.min() < 0:
                    t = ((t + 1.0) * 127.5).clamp(0, 255)
                else:
                    t = (t * 255.0).clamp(0, 255)
                t = t.to(torch.uint8).permute(0, 2, 3, 1)
            return t.numpy()
        raise ValueError(f"Cannot infer image layout from tensor shape {tuple(t.shape)}")

    arr = np.asarray(x)
    if arr.ndim != 4:
        raise ValueError(f"Expected rank-4 image array, got shape {arr.shape}")
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected BHWC images, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


@dataclass
class TorchInception:
    batch_size: int = 200
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        self.model = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=False,
            transform_input=False,
        ).to(self.device)
        self.model.eval()
        self._pool = None

        def _hook(_, __, output):
            self._pool = output

        self._handle = self.model.avgpool.register_forward_hook(_hook)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, images_bhwc_uint8: np.ndarray):
        x = torch.from_numpy(images_bhwc_uint8).to(device=self.device, dtype=torch.float32)
        x = x.permute(0, 3, 1, 2) / 255.0
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        self._pool = None
        logits = self.model(x)
        if self._pool is None:
            raise RuntimeError("Inception avgpool hook did not run.")
        feats = self._pool.flatten(1)
        return feats.cpu().numpy(), logits.cpu().numpy()


def compute_fid(mu1, mu2, sigma1, sigma2, eps=1e-6):
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    diff = mu1 - mu2
    cov_prod = sigma1 @ sigma2
    cov_prod = (cov_prod + cov_prod.T) * 0.5

    try:
        from scipy import linalg

        covmean = linalg.sqrtm(cov_prod)
        if not np.isfinite(covmean).all():
            covmean = linalg.sqrtm((sigma1 + eps * np.eye(sigma1.shape[0])) @ (sigma2 + eps * np.eye(sigma2.shape[0])))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    except Exception:
        vals, vecs = np.linalg.eigh(cov_prod)
        vals = np.clip(vals, a_min=0.0, a_max=None)
        covmean = (vecs * np.sqrt(vals)[None, :]) @ vecs.T

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(np.real(fid))


def build_jax_inception(batch_size=200):
    # Kept for backward compatibility with existing call sites.
    return TorchInception(batch_size=batch_size)


def get_reference(cache_path):
    if not cache_path or not os.path.exists(cache_path):
        raise FileNotFoundError(f"FID reference stats not found: {cache_path}")
    return dict(np.load(cache_path))


def compute_stats(samples_all, inception_net):
    images = _to_uint8_bhwc(samples_all)
    feats_all = []
    logits_all = []
    bsz = int(getattr(inception_net, "batch_size", 200))

    for i in range(0, images.shape[0], bsz):
        feats, logits = inception_net(images[i : i + bsz])
        feats_all.append(feats)
        logits_all.append(logits)

    feats_all = np.concatenate(feats_all, axis=0)
    logits_all = np.concatenate(logits_all, axis=0)
    mu = np.mean(feats_all, axis=0)
    sigma = np.cov(feats_all, rowvar=False)
    return {"mu": mu, "sigma": sigma, "logits": logits_all}


def compute_inception_score(logits, splits=10):
    logits = np.asarray(logits)
    probs = _softmax(logits, axis=-1)
    n = probs.shape[0]
    splits = max(1, min(int(splits), n))
    chunk = n // splits
    scores = []
    for i in range(splits):
        part = probs[i * chunk : (i + 1) * chunk] if i < splits - 1 else probs[i * chunk :]
        if part.shape[0] == 0:
            continue
        py = np.mean(part, axis=0, keepdims=True)
        kl = np.sum(part * (np.log(np.clip(part, 1e-12, None)) - np.log(np.clip(py, 1e-12, None))), axis=1)
        scores.append(np.exp(np.mean(kl)))
    scores = np.asarray(scores, dtype=np.float64)
    return float(np.mean(scores)), float(np.std(scores))


def compute_fid_stats(
    imagenet_root,
    split,
    image_size,
    batch_size=200,
    num_workers=4,
    cache_path=None,
):
    dataloader, _, _ = create_imagenet_dataloader(
        imagenet_root=imagenet_root,
        split=split,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        for_fid=True,
    )
    inception_net = build_jax_inception(batch_size=batch_size)

    feats_all = []
    logits_all = []
    for batch_idx, batch in enumerate(dataloader):
        images, _ = batch
        images = _to_uint8_bhwc(images)
        feats, logits = inception_net(images)
        feats_all.append(feats)
        logits_all.append(logits)
        if (batch_idx + 1) % 100 == 0:
            log_for_0(f"FID stats: processed {batch_idx + 1}/{len(dataloader)} batches")

    feats_all = np.concatenate(feats_all, axis=0)
    logits_all = np.concatenate(logits_all, axis=0)
    mu = np.mean(feats_all, axis=0)
    sigma = np.cov(feats_all, rowvar=False)
    stats = {"mu": mu, "sigma": sigma, "logits": logits_all}

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True) if os.path.dirname(cache_path) else None
        np.savez(cache_path, **stats)
        log_for_0(f"Saved FID stats to {cache_path}")
        return cache_path
    return stats
