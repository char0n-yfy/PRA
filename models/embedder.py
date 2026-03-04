import math

import torch
import torch.nn as nn

from models.torch_models import TorchEmbedding, TorchLinear


class TimestepEmbedder(nn.Module):
    """Embed scalar timestep/conditioning values into hidden vectors."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        weight_init: str = "scaled_variance",
        init_constant: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.weight_init = weight_init
        self.init_constant = init_constant

        init_kwargs = dict(
            out_features=self.hidden_size,
            bias=True,
            weight_init=self.weight_init,
            init_constant=self.init_constant,
            bias_init="zeros",
        )
        self.mlp = nn.Sequential(
            TorchLinear(self.frequency_embedding_size, **init_kwargs),
            nn.SiLU(),
            TorchLinear(self.hidden_size, **init_kwargs),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].to(torch.float32) * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    """Class label embedding (kept for compatibility with class-conditioned code)."""

    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        weight_init: str = "scaled_variance",
        init_constant: float = 1.0,
    ):
        super().__init__()
        self.embedding_table = TorchEmbedding(
            num_classes + 1,
            hidden_size,
            weight_init=weight_init,
            init_constant=init_constant,
        )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


class BottleneckPatchEmbedder(nn.Module):
    """Image to patch embedding with a bottleneck conv projection."""

    def __init__(
        self,
        input_size: int,
        initial_patch_size: int,
        pca_channels: int,
        in_channels: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.initial_patch_size = initial_patch_size
        self.in_channels = in_channels
        self.pca_channels = pca_channels
        self.hidden_size = hidden_size
        self.bias = bias

        self.patch_size = (self.initial_patch_size, self.initial_patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(
            self.input_size
        )

        self.proj1 = nn.Conv2d(
            self.in_channels,
            self.pca_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=self.bias,
        )
        self.proj2 = nn.Conv2d(
            self.pca_channels,
            self.hidden_size,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=self.bias,
        )

        # Init convs like linears to match prior behavior.
        kh = kw = self.patch_size[0]
        fan_in = kh * kw * self.in_channels
        fan_out = self.pca_channels
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(self.proj1.weight, -limit, limit)

        fan_in = self.pca_channels
        fan_out = self.hidden_size
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(self.proj2.weight, -limit, limit)

        if self.bias:
            nn.init.zeros_(self.proj1.bias)
            nn.init.zeros_(self.proj2.bias)

    def _init_img_size(self, img_size: int):
        img_size = (img_size, img_size)
        grid_size = tuple(s // p for s, p in zip(img_size, self.patch_size))
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, D)
        x = self.proj2(self.proj1(x))
        b, d, h, w = x.shape
        return x.permute(0, 2, 3, 1).reshape(b, h * w, d)
