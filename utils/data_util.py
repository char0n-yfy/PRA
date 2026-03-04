import os
from functools import partial
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F


NUM_CLASSES = 1000


def cuda_bf16_autocast(device):
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def resolve_processor_hw(size_cfg, fallback=224):
    if isinstance(size_cfg, dict):
        if "height" in size_cfg and "width" in size_cfg:
            return int(size_cfg["height"]), int(size_cfg["width"])
        if "shortest_edge" in size_cfg:
            s = int(size_cfg["shortest_edge"])
            return s, s
        if "longest_edge" in size_cfg:
            s = int(size_cfg["longest_edge"])
            return s, s
    if isinstance(size_cfg, (tuple, list)):
        if len(size_cfg) == 2:
            return int(size_cfg[0]), int(size_cfg[1])
        if len(size_cfg) == 1:
            s = int(size_cfg[0])
            return s, s
    if isinstance(size_cfg, (int, float)):
        s = int(size_cfg)
        return s, s
    return int(fallback), int(fallback)


def _stats_to_torch(stat, device, dtype=torch.float32):
    arr = np.asarray(stat, dtype=np.float32)
    if arr.ndim == 0:
        arr = np.repeat(arr, 3)
    return torch.from_numpy(arr).to(device=device, dtype=dtype).view(1, -1, 1, 1)


def preprocess_for_vision_encoder(images_nchw, processor, device=None, dtype=torch.float32):
    """
    Normalize NCHW images to match HF image processor stats/size.

    Args:
      images_nchw: torch.Tensor in shape (B, C, H, W), value range either [0, 1] or [-1, 1].
    """
    if images_nchw.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got shape={tuple(images_nchw.shape)}")

    out_device = images_nchw.device if device is None else torch.device(device)
    x = images_nchw.to(device=out_device, dtype=torch.float32)
    # Accept both training-space [-1,1] and image-space [0,1].
    # The previous check incorrectly rejected any negative value, which broke
    # valid tensors in [-1, 1] (e.g. near-zero predictions like [-0.005, 0.006]).
    x_min = x.detach().amin().item()
    x_max = x.detach().amax().item()
    # Predicted images can mildly overshoot [-1, 1] early in training
    # (e.g. [-1.03, 0.8]). That is not a pipeline bug and should be clamped
    # instead of crashing semantic/perceptual aux losses. Keep a guard only for
    # clearly wrong scales such as uint8-like [0,255].
    if x_min < -5.0 or x_max > 5.0:
        raise ValueError(
            "preprocess_for_vision_encoder expects inputs in [0,1] or [-1,1]. "
            f"Observed range [{x_min:.4f}, {x_max:.4f}]"
        )
    if x_min < 0.0:
        x = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
    else:
        x = x.clamp(0.0, 1.0)

    out_hw = resolve_processor_hw(getattr(processor, "size", 224), fallback=224)
    mean = _stats_to_torch(
        getattr(processor, "image_mean", [0.5, 0.5, 0.5]),
        device=out_device,
        dtype=torch.float32,
    )
    std = _stats_to_torch(
        getattr(processor, "image_std", [0.5, 0.5, 0.5]),
        device=out_device,
        dtype=torch.float32,
    )
    x = F.interpolate(
        x,
        size=out_hw,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    x = (x - mean) / std
    return x.to(dtype=dtype)


def pool_dense_tokens_to_fixed_count(dense_tokens: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """
    Pool patch tokens to a fixed token count.
    Input: (B, N, D), Output: (B, T, D)
    """
    if dense_tokens.ndim != 3:
        raise ValueError(f"Expected dense_tokens rank=3, got shape={tuple(dense_tokens.shape)}")
    num_tokens = max(1, int(num_tokens))
    bsz, n_tokens, d = dense_tokens.shape
    if n_tokens == num_tokens:
        return dense_tokens

    target_side = int(round(num_tokens ** 0.5))
    source_side = int(round(n_tokens ** 0.5))
    if (
        target_side * target_side == num_tokens
        and source_side * source_side == n_tokens
    ):
        dense_tokens = dense_tokens.transpose(1, 2).reshape(
            bsz, d, source_side, source_side
        )
        dense_tokens = F.adaptive_avg_pool2d(
            dense_tokens, output_size=(target_side, target_side)
        )
        return dense_tokens.flatten(2).transpose(1, 2)

    dense_tokens = dense_tokens.transpose(1, 2)
    dense_tokens = F.adaptive_avg_pool1d(dense_tokens, output_size=num_tokens)
    return dense_tokens.transpose(1, 2)


def pool_dino_hidden_to_tokens(
    dino_hidden: torch.Tensor,
    num_dino_tokens: int,
    dino_use_dense: bool = True,
) -> torch.Tensor:
    """
    Convert DINO last_hidden_state (B, N, D) into fixed tokens (B, T, D).
    """
    if dino_hidden.ndim != 3:
        raise ValueError(f"Expected dino_hidden rank=3, got shape={tuple(dino_hidden.shape)}")
    if bool(dino_use_dense) and dino_hidden.shape[1] > 1:
        dino_tokens = dino_hidden[:, 1:, :]
    else:
        dino_tokens = dino_hidden[:, :1, :]
    return pool_dense_tokens_to_fixed_count(dino_tokens, num_tokens=num_dino_tokens)


def parse_condition_mode(condition_cfg):
    if condition_cfg is None:
        return "uncond", False, False

    mode = getattr(condition_cfg, "mode", None)
    if mode is None:
        mode = "clip_dino" if bool(getattr(condition_cfg, "enable_semantic", False)) else "uncond"
    mode = str(mode).lower()
    aliases = {
        "none": "uncond",
        "unconditional": "uncond",
        "null": "uncond",
        "clip+dino": "clip_dino",
    }
    mode = aliases.get(mode, mode)
    valid_modes = {"uncond", "clip", "dino", "clip_dino"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported condition.mode={mode}. Expected one of {sorted(valid_modes)}.")

    use_clip = mode in {"clip", "clip_dino"}
    use_dino = mode in {"dino", "clip_dino"}
    return mode, use_clip, use_dino


class ClipDinoSemanticEncoder:
    """Frozen semantic encoder supporting CLIP and/or DINO branches."""

    def __init__(
        self,
        condition_cfg,
        num_dino_tokens: int = 4,
        use_clip: bool = True,
        use_dino: bool = True,
    ):
        try:
            from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel
        except ImportError as exc:
            raise ImportError(
                "Semantic condition pipeline requires `transformers`."
            ) from exc

        self.use_clip = bool(use_clip)
        self.use_dino = bool(use_dino)
        if not (self.use_clip or self.use_dino):
            raise ValueError("ClipDinoSemanticEncoder requires at least one active branch.")

        self.num_dino_tokens = max(1, int(num_dino_tokens))
        self.dino_use_dense = bool(getattr(condition_cfg, "dino_use_dense", True))

        device = getattr(condition_cfg, "device", "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.clip_processor = None
        self.clip_model = None
        if self.use_clip:
            self.clip_processor = AutoImageProcessor.from_pretrained(
                condition_cfg.clip_model_name,
                use_fast=False,
            )
            self.clip_model = CLIPVisionModel.from_pretrained(
                condition_cfg.clip_model_name
            ).to(self.device)
            self.clip_model.eval()
            for p in self.clip_model.parameters():
                p.requires_grad_(False)

        self.dino_processor = None
        self.dino_model = None
        if self.use_dino:
            self.dino_processor = AutoImageProcessor.from_pretrained(
                condition_cfg.dino_model_name,
                use_fast=False,
            )
            self.dino_model = AutoModel.from_pretrained(condition_cfg.dino_model_name).to(
                self.device
            )
            self.dino_model.eval()
            for p in self.dino_model.parameters():
                p.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, images):
        """
        Args:
            images: uint8 tensor/array of shape (B, H, W, C).
        Returns:
            clip_cls, dino_dense:
              - clip_cls: float32 array (B, clip_feature_dim) or None
              - dino_dense: float32 array (B, num_dino_tokens, dino_feature_dim) or None
        """
        if torch.is_tensor(images):
            images_t = images.detach()
            if images_t.ndim != 4:
                raise ValueError(f"Expected NHWC uint8 images, got shape={tuple(images_t.shape)}")
            if images_t.device.type != "cpu":
                images_t = images_t.cpu()
            if images_t.dtype != torch.uint8:
                images_t = images_t.to(torch.uint8)
            images_t = images_t.permute(0, 3, 1, 2).contiguous().to(self.device, dtype=torch.float32) / 255.0
        else:
            images_np = np.asarray(images)
            if images_np.ndim != 4:
                raise ValueError(f"Expected NHWC images array, got shape={tuple(images_np.shape)}")
            images_t = (
                torch.from_numpy(images_np)
                .to(device=self.device, dtype=torch.uint8)
                .permute(0, 3, 1, 2)
                .contiguous()
                .to(dtype=torch.float32)
                / 255.0
            )

        clip_cls = None
        if self.use_clip:
            clip_pixels = preprocess_for_vision_encoder(
                images_t, self.clip_processor, device=self.device, dtype=torch.float32
            )
            with cuda_bf16_autocast(self.device):
                clip_out = self.clip_model(pixel_values=clip_pixels)
            clip_cls = clip_out.last_hidden_state[:, 0, :].float().cpu().numpy().astype(np.float32)

        dino_dense = None
        if self.use_dino:
            dino_pixels = preprocess_for_vision_encoder(
                images_t, self.dino_processor, device=self.device, dtype=torch.float32
            )
            with cuda_bf16_autocast(self.device):
                dino_out = self.dino_model(pixel_values=dino_pixels)
            dino_hidden = dino_out.last_hidden_state.float()
            dino_tokens = pool_dino_hidden_to_tokens(
                dino_hidden=dino_hidden,
                num_dino_tokens=self.num_dino_tokens,
                dino_use_dense=self.dino_use_dense,
            )
            dino_dense = dino_tokens.cpu().numpy().astype(np.float32)

        return clip_cls, dino_dense


def build_semantic_encoder(
    condition_cfg,
    num_dino_tokens: int = 4,
    use_clip: bool = None,
    use_dino: bool = None,
):
    if condition_cfg is None:
        return None
    _, cfg_use_clip, cfg_use_dino = parse_condition_mode(condition_cfg)
    use_clip = cfg_use_clip if use_clip is None else bool(use_clip)
    use_dino = cfg_use_dino if use_dino is None else bool(use_dino)
    if not (use_clip or use_dino):
        return None
    return ClipDinoSemanticEncoder(
        condition_cfg,
        num_dino_tokens=num_dino_tokens,
        use_clip=use_clip,
        use_dino=use_dino,
    )


def create_imagenet_dataloader(
    imagenet_root, split, batch_size, image_size, num_workers=4, for_fid=False
):
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torchvision import datasets, transforms

    from utils.input_pipeline import center_crop_arr, loader, worker_init_fn
    from utils.logging_util import get_rank, get_world_size, log_for_0

    if for_fid:
        def fid_transform(pil_image):
            cropped = center_crop_arr(pil_image, image_size)
            return np.array(cropped)

        transform = fid_transform
    else:
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3, inplace=True),
            ]
        )

    dataset = datasets.ImageFolder(
        os.path.join(imagenet_root, split),
        transform=transform,
        loader=loader,
    )

    rank = get_rank()
    world_size = get_world_size()
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    log_for_0(
        f"Dataset {split} size={len(dataset)}, world_size={world_size}, rank={rank}"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        worker_init_fn=partial(worker_init_fn, rank=rank),
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    return dataloader, len(sampler), len(dataset)
