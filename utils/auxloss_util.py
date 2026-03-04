from typing import Tuple
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from models.convnext import load_convnext_jax_model
from utils.data_util import (
    build_semantic_encoder,
    cuda_bf16_autocast,
    parse_condition_mode,
    pool_dino_hidden_to_tokens,
    preprocess_for_vision_encoder,
)
from utils.logging_util import log_for_0


def paired_random_resized_crop(
    x1: torch.Tensor,
    x2: torch.Tensor,
    out_size: int = 224,
    scale: Tuple[float, float] = (0.08, 1.0),
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
):
    """
    Torch version of paired random resized crop.
    Inputs are NCHW tensors in [-1, 1].
    """
    b, c, h, w = x1.shape
    assert x2.shape == x1.shape
    device = x1.device

    area = float(h * w)
    out1 = []
    out2 = []
    for i in range(b):
        target_area = area * float(torch.empty((), device=device).uniform_(*scale))
        log_ratio = torch.log(torch.tensor(ratio, device=device, dtype=torch.float32))
        aspect = torch.exp(torch.empty((), device=device).uniform_(log_ratio[0], log_ratio[1]))
        crop_w = int(torch.clamp(torch.round(torch.sqrt(target_area * aspect)), 1, w).item())
        crop_h = int(torch.clamp(torch.round(torch.sqrt(target_area / aspect)), 1, h).item())
        top = int(torch.randint(0, h - crop_h + 1, (), device=device).item())
        left = int(torch.randint(0, w - crop_w + 1, (), device=device).item())

        x1_crop = x1[i : i + 1, :, top : top + crop_h, left : left + crop_w]
        x2_crop = x2[i : i + 1, :, top : top + crop_h, left : left + crop_w]
        out1.append(
            F.interpolate(
                x1_crop, size=(out_size, out_size), mode="bicubic", align_corners=False, antialias=True
            )
        )
        out2.append(
            F.interpolate(
                x2_crop, size=(out_size, out_size), mode="bicubic", align_corners=False, antialias=True
            )
        )
    return torch.cat(out1, dim=0), torch.cat(out2, dim=0)


def init_auxloss(config, semantic_encoder=None, device=None):
    """
    Returns a callable auxloss_fn(pred_x, gt_x) -> (lpips_dist, convnext_dist, sem_raw)
    where each output is shape (B,).

    Notes:
      - lpips_dist / convnext_dist are true perceptual distances from frozen pretrained models.
      - semantic loss is computed with frozen CLIP + DINO in pure PyTorch.
    """
    aux_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_lpips = bool(config.model.lpips)
    use_convnext = bool(config.model.convnext)
    use_perc = bool(use_lpips or use_convnext)
    condition_cfg = config.condition if hasattr(config, "condition") else None
    _, use_clip_cond, use_dino_cond = parse_condition_mode(condition_cfg)
    sem_loss_enabled = bool(getattr(config.model, "enable_semantic_loss", True))
    use_sem = (
        sem_loss_enabled
        and getattr(config.model, "lambda_sem_max", 0.0) > 0.0
        and (use_clip_cond or use_dino_cond)
    )
    use_sem_clip = bool(use_sem and use_clip_cond)
    use_sem_dino = bool(use_sem and use_dino_cond)
    dino_use_dense = bool(
        hasattr(config, "condition") and getattr(config.condition, "dino_use_dense", True)
    )

    lpips_model = None
    convnext_model = None
    if use_lpips:
        try:
            import lpips
        except Exception as exc:
            raise ImportError(
                "LPIPS loss requested, but `lpips` is not installed. "
                "Install it with: pip install lpips"
            ) from exc
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
            lpips_model = lpips.LPIPS(net="vgg").to(aux_device)
        lpips_model.eval()
        for p in lpips_model.parameters():
            p.requires_grad_(False)

    if use_convnext:
        convnext_model_name = str(
            getattr(config.model, "convnext_model_name", "facebook/convnextv2-base-22k-224")
        )
        convnext_model, _ = load_convnext_jax_model(
            device=aux_device,
            model_name=convnext_model_name,
        )
        convnext_model.eval()
        for p in convnext_model.parameters():
            p.requires_grad_(False)

    if use_sem:
        if semantic_encoder is None:
            semantic_encoder = build_semantic_encoder(
                condition_cfg,
                num_dino_tokens=int(getattr(config.model, "num_dino_tokens", 4)),
                use_clip=use_sem_clip,
                use_dino=use_sem_dino,
            )
        if semantic_encoder is None:
            raise ValueError("Semantic loss enabled but semantic encoder is unavailable.")
        clip_model = semantic_encoder.clip_model if use_sem_clip else None
        dino_model = semantic_encoder.dino_model if use_sem_dino else None
        clip_processor = semantic_encoder.clip_processor if use_sem_clip else None
        dino_processor = semantic_encoder.dino_processor if use_sem_dino else None
        sem_device = aux_device
        if use_sem_clip and next(clip_model.parameters()).device != sem_device:
            clip_model = clip_model.to(sem_device)
        if use_sem_dino and next(dino_model.parameters()).device != sem_device:
            dino_model = dino_model.to(sem_device)
        semantic_encoder.device = sem_device

        def _cosine_distance(a, b, eps=1e-6):
            a = a / torch.clamp(torch.norm(a, dim=-1, keepdim=True), min=eps)
            b = b / torch.clamp(torch.norm(b, dim=-1, keepdim=True), min=eps)
            return 1.0 - torch.sum(a * b, dim=-1)

    log_for_0(
        "Aux loss initialized "
        f"(lpips={use_lpips}, convnext={use_convnext}, semantic={use_sem}, "
        f"sem_clip={use_sem_clip}, sem_dino={use_sem_dino}, device={aux_device})"
    )

    def auxloss_fn(
        model_images,
        gt_images,
        compute_perc=True,
        compute_sem=True,
        gt_sem_targets=None,
    ):
        bsz = model_images.shape[0]
        dev = model_images.device
        dt = model_images.dtype
        compute_perc = bool(compute_perc)
        compute_sem = bool(compute_sem)

        if use_perc and compute_perc:
            model_crop, gt_crop = paired_random_resized_crop(model_images, gt_images, out_size=224)
            model_crop = model_crop.to(device=aux_device, dtype=torch.float32)
            gt_crop = gt_crop.to(device=aux_device, dtype=torch.float32)

            if use_lpips:
                lpips_out = lpips_model(model_crop, gt_crop, normalize=False)
                lpips_dist = lpips_out.reshape(lpips_out.shape[0], -1).mean(dim=1)
            else:
                lpips_dist = torch.zeros((bsz,), device=aux_device, dtype=torch.float32)

            if use_convnext:
                with cuda_bf16_autocast(aux_device):
                    convnext_pred = convnext_model(model_crop)
                with torch.no_grad():
                    with cuda_bf16_autocast(aux_device):
                        convnext_gt = convnext_model(gt_crop)
                convnext_pred = convnext_pred.float()
                convnext_gt = convnext_gt.float()
                convnext_dist = torch.sum((convnext_pred - convnext_gt) ** 2, dim=-1)
            else:
                convnext_dist = torch.zeros((bsz,), device=aux_device, dtype=torch.float32)

            lpips_dist = lpips_dist.to(device=dev, dtype=dt)
            convnext_dist = convnext_dist.to(device=dev, dtype=dt)
        else:
            lpips_dist = torch.zeros((bsz,), device=dev, dtype=dt)
            convnext_dist = torch.zeros((bsz,), device=dev, dtype=dt)

        if use_sem and compute_sem:
            pred = model_images.to(device=sem_device, dtype=torch.float32)
            gt = gt_images.to(device=sem_device, dtype=torch.float32)
            gt_sem_targets = gt_sem_targets if isinstance(gt_sem_targets, dict) else None
            gt_sem_clip_cached = None
            gt_sem_dino_cached = None
            if gt_sem_targets is not None:
                if "clip" in gt_sem_targets and gt_sem_targets["clip"] is not None:
                    gt_sem_clip_cached = torch.as_tensor(
                        gt_sem_targets["clip"], device=sem_device, dtype=torch.float32
                    )
                    if gt_sem_clip_cached.ndim == 1:
                        gt_sem_clip_cached = gt_sem_clip_cached.unsqueeze(0)
                    if gt_sem_clip_cached.shape[0] != bsz:
                        raise ValueError(
                            f"gt_sem_targets['clip'] batch mismatch: "
                            f"got {gt_sem_clip_cached.shape[0]}, expected {bsz}"
                        )
                if "dino" in gt_sem_targets and gt_sem_targets["dino"] is not None:
                    gt_sem_dino_cached = torch.as_tensor(
                        gt_sem_targets["dino"], device=sem_device, dtype=torch.float32
                    )
                    if gt_sem_dino_cached.ndim == 2:
                        gt_sem_dino_cached = gt_sem_dino_cached.unsqueeze(1)
                    if gt_sem_dino_cached.ndim != 3:
                        raise ValueError(
                            "gt_sem_targets['dino'] rank mismatch: "
                            f"got rank={gt_sem_dino_cached.ndim}, expected 2 or 3"
                        )
                    if gt_sem_dino_cached.shape[0] != bsz:
                        raise ValueError(
                            f"gt_sem_targets['dino'] batch mismatch: "
                            f"got {gt_sem_dino_cached.shape[0]}, expected {bsz}"
                        )

            if use_sem_clip:
                clip_pred_pixels = preprocess_for_vision_encoder(
                    pred, clip_processor, device=sem_device, dtype=torch.float32
                )
                with cuda_bf16_autocast(sem_device):
                    clip_pred = clip_model(
                        pixel_values=clip_pred_pixels
                    ).last_hidden_state[:, 0, :]
                clip_pred = clip_pred.float()
            else:
                clip_pred = None

            if use_sem_dino:
                dino_pred_pixels = preprocess_for_vision_encoder(
                    pred, dino_processor, device=sem_device, dtype=torch.float32
                )
                with cuda_bf16_autocast(sem_device):
                    dino_pred_hidden = dino_model(
                        pixel_values=dino_pred_pixels
                    ).last_hidden_state
                dino_pred_hidden = dino_pred_hidden.float()
                dino_pred = pool_dino_hidden_to_tokens(
                    dino_hidden=dino_pred_hidden,
                    num_dino_tokens=int(getattr(config.model, "num_dino_tokens", 4)),
                    dino_use_dense=dino_use_dense,
                )
            else:
                dino_pred = None

            with torch.no_grad():
                if use_sem_clip:
                    if gt_sem_clip_cached is not None:
                        clip_gt = gt_sem_clip_cached
                    else:
                        clip_gt_pixels = preprocess_for_vision_encoder(
                            gt, clip_processor, device=sem_device, dtype=torch.float32
                        )
                        with cuda_bf16_autocast(sem_device):
                            clip_gt = clip_model(
                                pixel_values=clip_gt_pixels
                            ).last_hidden_state[:, 0, :]
                        clip_gt = clip_gt.float()
                else:
                    clip_gt = None

                if use_sem_dino:
                    if gt_sem_dino_cached is not None:
                        dino_gt = gt_sem_dino_cached
                    else:
                        dino_gt_pixels = preprocess_for_vision_encoder(
                            gt, dino_processor, device=sem_device, dtype=torch.float32
                        )
                        with cuda_bf16_autocast(sem_device):
                            dino_gt_hidden = dino_model(
                                pixel_values=dino_gt_pixels
                            ).last_hidden_state
                        dino_gt_hidden = dino_gt_hidden.float()
                        dino_gt = pool_dino_hidden_to_tokens(
                            dino_hidden=dino_gt_hidden,
                            num_dino_tokens=int(getattr(config.model, "num_dino_tokens", 4)),
                            dino_use_dense=dino_use_dense,
                        )
                else:
                    dino_gt = None

            sem_terms = []
            if use_sem_clip:
                sem_terms.append(_cosine_distance(clip_pred, clip_gt))
            if use_sem_dino:
                dino_token_dist = _cosine_distance(dino_pred, dino_gt)
                if dino_token_dist.ndim == 2:
                    dino_token_dist = dino_token_dist.mean(dim=1)
                sem_terms.append(dino_token_dist)
            sem_raw = torch.stack(sem_terms, dim=0).mean(dim=0)
            sem_raw = sem_raw.to(device=dev, dtype=dt)
        else:
            sem_raw = torch.zeros((bsz,), device=dev, dtype=dt)

        return lpips_dist, convnext_dist, sem_raw

    return auxloss_fn
