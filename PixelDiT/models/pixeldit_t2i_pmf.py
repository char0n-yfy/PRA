from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from .pixeldit_blocks import (
    MMDiTBlock,
    PixelFinalLayer,
    PixelTransformerBlock,
    TimestepEmbedder,
    VisionRotaryEmbeddingFast,
    get_2d_sincos_pos_embed,
)


@dataclass
class PixelDiTOutput:
    x_hat_u: torch.Tensor
    x_hat_v: torch.Tensor


class PixelDiTT2IPMF(nn.Module):
    """PixelDiT-T2I backbone adapted for pMF strategy-CFG training.

    Model conditioning is restricted to (t, h). Guidance parameters (omega/interval)
    stay outside the model and are used only in inference.
    """

    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 1024,
        pixel_dim: int = 16,
        patch_depth: int = 14,
        pixel_depth: int = 2,
        pixel_head_depth: int = 1,
        num_heads: int = 16,
        pixel_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        sem_in_dim: int = 768,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        use_checkpoint: bool = False,
        null_token_learnable: bool = True,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.pixel_dim = int(pixel_dim)
        self.use_checkpoint = bool(use_checkpoint)
        self.null_token_learnable = bool(null_token_learnable)

        self.num_patches = (self.input_size // self.patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            self.in_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_size), requires_grad=False)

        self.sem_proj = nn.Linear(int(sem_in_dim), self.hidden_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.h_embedder = TimestepEmbedder(self.hidden_size)

        if use_rope:
            half_head_dim = self.hidden_size // int(num_heads) // 2
            hw_seq_len = self.input_size // self.patch_size
            self.patch_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len)
        else:
            self.patch_rope = None

        self.pixel_embed = nn.Conv2d(self.in_channels, self.pixel_dim, kernel_size=1, stride=1)
        self.pixel_pos_embed = nn.Parameter(
            torch.zeros(1, self.input_size * self.input_size, self.pixel_dim),
            requires_grad=False,
        )

        self.patch_blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    hidden_size=self.hidden_size,
                    num_heads=int(num_heads),
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_swiglu=use_swiglu,
                    use_rmsnorm=use_rmsnorm,
                )
                for _ in range(int(patch_depth))
            ]
        )

        self.shared_pixel_blocks = nn.ModuleList(
            [
                PixelTransformerBlock(
                    pixel_dim=self.pixel_dim,
                    semantic_dim=self.hidden_size,
                    num_heads=int(pixel_num_heads),
                    patch_size=self.patch_size,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_rmsnorm=use_rmsnorm,
                )
                for _ in range(int(pixel_depth))
            ]
        )

        self.u_head_blocks = nn.ModuleList(
            [
                PixelTransformerBlock(
                    pixel_dim=self.pixel_dim,
                    semantic_dim=self.hidden_size,
                    num_heads=int(pixel_num_heads),
                    patch_size=self.patch_size,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_rmsnorm=use_rmsnorm,
                )
                for _ in range(int(pixel_head_depth))
            ]
        )
        self.v_head_blocks = nn.ModuleList(
            [
                PixelTransformerBlock(
                    pixel_dim=self.pixel_dim,
                    semantic_dim=self.hidden_size,
                    num_heads=int(pixel_num_heads),
                    patch_size=self.patch_size,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_rmsnorm=use_rmsnorm,
                )
                for _ in range(int(pixel_head_depth))
            ]
        )

        self.u_final_layer = PixelFinalLayer(
            semantic_dim=self.hidden_size,
            pixel_dim=self.pixel_dim,
            out_channels=self.in_channels,
            use_rmsnorm=use_rmsnorm,
        )
        self.v_final_layer = PixelFinalLayer(
            semantic_dim=self.hidden_size,
            pixel_dim=self.pixel_dim,
            out_channels=self.in_channels,
            use_rmsnorm=use_rmsnorm,
        )

        if self.null_token_learnable:
            self.null_sem_token = nn.Parameter(torch.zeros(1, 1, int(sem_in_dim)))
        else:
            self.register_buffer("null_sem_token", torch.zeros(1, 1, int(sem_in_dim)), persistent=False)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pixel_pos_embed = get_2d_sincos_pos_embed(self.pixel_dim, self.input_size)
        self.pixel_pos_embed.data.copy_(torch.from_numpy(pixel_pos_embed).float().unsqueeze(0))

        for embed in [self.patch_embed, self.pixel_embed]:
            w = embed.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(embed.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.h_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.h_embedder.mlp[2].weight, std=0.02)

        for block in self.patch_blocks:
            nn.init.constant_(block.adaLN_modulation_img[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation_img[-1].bias, 0)
            nn.init.constant_(block.adaLN_modulation_sem[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation_sem[-1].bias, 0)

        for block in list(self.shared_pixel_blocks) + list(self.u_head_blocks) + list(self.v_head_blocks):
            nn.init.constant_(block.pixelwise_adaln.mlp[-1].weight, 0)
            nn.init.constant_(block.pixelwise_adaln.mlp[-1].bias, 0)

        for final in [self.u_final_layer, self.v_final_layer]:
            nn.init.constant_(final.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(final.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(final.linear.weight, 0)
            nn.init.constant_(final.linear.bias, 0)

        if isinstance(self.null_sem_token, nn.Parameter):
            nn.init.normal_(self.null_sem_token, std=0.02)

    def patchify_pixels(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        return rearrange(x, "b d (h p1) (w p2) -> b (h w) (p1 p2) d", p1=p, p2=p)

    def unpatchify_pixels(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        p = self.patch_size
        hp, wp = h // p, w // p
        return rearrange(x, "b (h w) (p1 p2) c -> b c (h p1) (w p2)", h=hp, w=wp, p1=p, p2=p)

    def get_null_sem_tokens(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tok = self.null_sem_token.to(device=device, dtype=dtype)
        return tok.expand(int(batch_size), int(seq_len), -1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        sem_tokens: torch.Tensor,
        sem_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, c, height, width = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected input channels={self.in_channels}, got {c}")
        if sem_tokens.ndim != 3:
            raise ValueError(f"sem_tokens must be rank-3, got shape={tuple(sem_tokens.shape)}")

        p = self.patch_size
        l = (height // p) * (width // p)

        e_t = self.t_embedder(t)
        e_h = self.h_embedder(h)
        c_time = e_t + e_h

        sem = self.sem_proj(sem_tokens)

        s = self.patch_embed(x)
        s = s.flatten(2).transpose(1, 2)
        s = s + self.pos_embed

        for block in self.patch_blocks:
            s, sem = block(s, sem, c_time, rope=self.patch_rope, sem_mask=sem_mask)

        s_cond = s + c_time.unsqueeze(1)

        p_embed = self.pixel_embed(x)
        p_tokens = self.patchify_pixels(p_embed)
        pos_embed_patches = self.pixel_pos_embed.view(1, l, p * p, self.pixel_dim)
        p_tokens = p_tokens + pos_embed_patches

        for block in self.shared_pixel_blocks:
            p_tokens = block(p_tokens, s_cond, rope=self.patch_rope)

        u_tokens = p_tokens
        v_tokens = p_tokens

        for block in self.u_head_blocks:
            u_tokens = block(u_tokens, s_cond, rope=self.patch_rope)
        for block in self.v_head_blocks:
            v_tokens = block(v_tokens, s_cond, rope=self.patch_rope)

        u_flat = u_tokens.view(b, -1, self.pixel_dim)
        v_flat = v_tokens.view(b, -1, self.pixel_dim)

        u_out = self.u_final_layer(u_flat, c_time).view(b, l, p * p, c)
        v_out = self.v_final_layer(v_flat, c_time).view(b, l, p * p, c)

        x_hat_u = self.unpatchify_pixels(u_out, height, width)
        x_hat_v = self.unpatchify_pixels(v_out, height, width)

        return {
            "x_hat_u": x_hat_u,
            "x_hat_v": x_hat_v,
            "c_time": c_time,
            "s_cond": s_cond,
        }


__all__ = ["PixelDiTT2IPMF", "PixelDiTOutput"]
