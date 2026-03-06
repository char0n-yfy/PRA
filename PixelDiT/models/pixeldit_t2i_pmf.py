from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .pixeldit_blocks import (
    CrossAttentionPooler,
    MMDiTBlock,
    PixelFinalLayer,
    PixelTransformerBlock,
    RMSNorm,
    TimestepEmbedder,
    VisionRotaryEmbeddingFast,
    get_2d_sincos_pos_embed,
)

import math


@dataclass
class PixelDiTOutput:
    x_hat_u: torch.Tensor
    x_hat_v: torch.Tensor


def _is_power_of_two(x: int) -> bool:
    x = int(x)
    return x > 0 and (x & (x - 1)) == 0


class SpatialNet(nn.Module):
    """Lightweight CNN that downsamples an edge map to patch-grid features."""

    def __init__(self, patch_size: int, out_channels: int = 64):
        super().__init__()
        p = int(patch_size)
        if not _is_power_of_two(p):
            raise ValueError(f"SpatialNet requires power-of-2 patch_size, got {patch_size}")
        steps = int(round(math.log2(p)))
        if steps <= 0:
            raise ValueError(f"SpatialNet expects patch_size>=2, got {patch_size}")

        out_c = int(out_channels)
        if steps == 1:
            ch_list = [out_c]
        elif steps == 2:
            ch_list = [32, out_c]
        else:
            # e.g., patch_size=16 -> 4 steps: 1->32->64->64->out_c
            ch_list = [32, 64] + [64] * (steps - 3) + [out_c]

        layers = []
        in_c = 1
        for i, c in enumerate(ch_list):
            layers.append(nn.Conv2d(in_c, int(c), kernel_size=3, stride=2, padding=1))
            if i != len(ch_list) - 1:
                layers.append(nn.SiLU())
            in_c = int(c)
        self.net = nn.Sequential(*layers)

    def forward(self, edge_map: torch.Tensor) -> torch.Tensor:
        if edge_map.ndim != 4 or edge_map.shape[1] != 1:
            raise ValueError(f"edge_map must be (B,1,H,W), got {tuple(edge_map.shape)}")
        return self.net(edge_map)


class PixelDiTT2IPMF(nn.Module):
    """PixelDiT-T2I backbone adapted for pMF strategy-CFG training.

    Model conditioning is restricted to (t, h). Guidance parameters (omega/interval)
    stay outside the model and are used only in inference.
    """

    def __init__(
        self,
        # Default aligns with PixelDiT-T2I 1B capacity (D=1536, N=14, M=2, heads=24),
        # but uses 512x512 inputs by default for this project.
        input_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 1536,
        pixel_dim: int = 16,
        patch_depth: int = 14,  # N
        pixel_depth: int = 2,   # M
        pixel_head_depth: int = 1,
        num_heads: int = 24,
        pixel_num_heads: int = 24,
        mlp_ratio: float = 4.0,
        sem_in_dim: int = 768,
        sem_num_tokens: int = 64,
        sem_pool_num_heads: int = 12,
        enable_edge_cond: bool = False,
        edge_channels: int = 64,
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
        self.sem_in_dim = int(sem_in_dim)
        self.sem_num_tokens = int(sem_num_tokens)
        self.enable_edge_cond = bool(enable_edge_cond)

        self.num_patches = (self.input_size // self.patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            self.in_channels,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_size), requires_grad=False)

        # Step 1) Learnable-query attention pooling from raw DINO tokens -> fixed slots.
        self.sem_pooler = CrossAttentionPooler(
            dim=self.sem_in_dim,
            num_heads=int(sem_pool_num_heads),
            num_queries=self.sem_num_tokens,
            qkv_bias=True,
            qk_norm=bool(use_qknorm),
            use_rmsnorm=bool(use_rmsnorm),
        )
        # Step 2) Token adapter to model hidden size.
        self.sem_adapter = nn.Sequential(
            RMSNorm(self.sem_in_dim),
            nn.Linear(self.sem_in_dim, self.hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
        )
        # Step 3) Slot embeddings to disambiguate pooled semantic slots.
        self.sem_slot_embed = nn.Parameter(torch.zeros(1, self.sem_num_tokens, self.hidden_size))

        # Optional spatial (edge) condition injected into patch-level image tokens.
        if self.enable_edge_cond:
            self.spatial_net = SpatialNet(patch_size=self.patch_size, out_channels=int(edge_channels))
            self.edge_proj = nn.Linear(int(edge_channels), self.hidden_size, bias=True)
            # Per-layer scalar gates (0-init) so training starts identical to "no edge".
            self.edge_gate = nn.Parameter(torch.zeros(int(patch_depth)))
        else:
            self.spatial_net = None
            self.edge_proj = None
            self.edge_gate = None

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

        # CFG null semantic tokens live in the model semantic space (hidden_size) after pooling/adaptation.
        if self.null_token_learnable:
            self.null_sem_token = nn.Parameter(torch.zeros(1, self.sem_num_tokens, self.hidden_size))
        else:
            self.register_buffer(
                "null_sem_token",
                torch.zeros(1, self.sem_num_tokens, self.hidden_size),
                persistent=False,
            )

        self.initialize_weights()

    def _maybe_checkpoint(self, fn, *args):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return torch_checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _encode_edge(self, edge_map: torch.Tensor, expected_seq_len: int, dtype: torch.dtype) -> torch.Tensor:
        if not self.enable_edge_cond:
            raise RuntimeError("_encode_edge called while enable_edge_cond=False")
        assert self.spatial_net is not None and self.edge_proj is not None

        feat = self.spatial_net(edge_map)
        # (B, C_e, H/p, W/p) -> (B, L_img, C_e)
        tok = feat.flatten(2).transpose(1, 2)
        if tok.shape[1] != int(expected_seq_len):
            raise ValueError(
                f"edge tokens seq_len mismatch: got {tok.shape[1]}, expected {int(expected_seq_len)}"
            )
        res = self.edge_proj(tok)
        return res.to(dtype=dtype)

    def _encode_semantic(
        self,
        sem_tokens: torch.Tensor,
        sem_drop_mask: Optional[torch.Tensor],
        sem_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Convert raw DINO tokens (B, N, sem_in_dim) into model semantic tokens (B, sem_num_tokens, hidden_size).
        """
        if sem_tokens.ndim != 3 or sem_tokens.shape[-1] != self.sem_in_dim:
            raise ValueError(
                f"sem_tokens must be (B,N,{self.sem_in_dim}), got {tuple(sem_tokens.shape)}"
            )
        b = sem_tokens.shape[0]
        device = sem_tokens.device
        dtype = sem_tokens.dtype

        if sem_mask is not None:
            if sem_mask.ndim != 2 or sem_mask.shape != (b, sem_tokens.shape[1]):
                raise ValueError(
                    f"sem_mask must be (B,N)=({b},{sem_tokens.shape[1]}), got {tuple(sem_mask.shape)}"
                )
            sem_mask = sem_mask.to(device=device)

        if sem_drop_mask is not None:
            if sem_drop_mask.ndim != 1 or sem_drop_mask.shape[0] != b:
                raise ValueError(f"sem_drop_mask must be (B,), got {tuple(sem_drop_mask.shape)}")
            sem_drop_mask = sem_drop_mask.to(device=device)

        sem_raw = self.sem_pooler(sem_tokens, tokens_mask=sem_mask)  # (B, T, sem_in_dim)
        sem_hid = self.sem_adapter(sem_raw)  # (B, T, hidden_size)
        sem_hid = sem_hid + self.sem_slot_embed.to(device=device, dtype=sem_hid.dtype)

        if sem_drop_mask is not None:
            null = self.get_null_sem_tokens(batch_size=b, device=device, dtype=sem_hid.dtype)
            # Keep both cond/uncond branches in the graph for DDP consistency.
            # `torch.where` with a hard boolean mask can disconnect one branch
            # entirely on a rank (e.g., local_batch=1), causing per-rank unused
            # parameter divergence. Arithmetic mixing preserves graph topology.
            mask = sem_drop_mask[:, None, None].to(dtype=sem_hid.dtype)
            sem_hid = sem_hid * (1.0 - mask) + null * mask
        return sem_hid

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
        nn.init.normal_(self.sem_pooler.queries, std=0.02)
        nn.init.normal_(self.sem_slot_embed, std=0.02)

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
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tok = self.null_sem_token.to(device=device, dtype=dtype)
        tok = tok.expand(int(batch_size), self.sem_num_tokens, -1)
        return tok + self.sem_slot_embed.to(device=device, dtype=tok.dtype)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        sem_tokens: torch.Tensor,
        sem_mask: Optional[torch.Tensor] = None,
        sem_drop_mask: Optional[torch.Tensor] = None,
        edge_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, c, height, width = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected input channels={self.in_channels}, got {c}")

        p = self.patch_size
        l = (height // p) * (width // p)

        e_t = self.t_embedder(t)
        e_h = self.h_embedder(h)
        c_time = e_t + e_h

        sem = self._encode_semantic(sem_tokens, sem_drop_mask=sem_drop_mask, sem_mask=sem_mask)

        s = self.patch_embed(x)
        s = s.flatten(2).transpose(1, 2)
        s = s + self.pos_embed

        edge_res = None
        if self.enable_edge_cond:
            if edge_map is None:
                raise ValueError("edge_map must be provided when spatial.enable_edge=true")
            if edge_map.shape[0] != b or edge_map.shape[2] != height or edge_map.shape[3] != width:
                raise ValueError(
                    f"edge_map shape must be (B,1,H,W)=({b},1,{height},{width}), got {tuple(edge_map.shape)}"
                )
            edge_res = self._encode_edge(edge_map.to(device=x.device), expected_seq_len=l, dtype=s.dtype)

        for i, block in enumerate(self.patch_blocks):
            if edge_res is not None:
                # Pre-block edge injection with per-layer 0-init gate.
                gate = self.edge_gate[i].to(dtype=s.dtype)
                s = s + gate * edge_res
            # sem_mask is only for raw-token pooling; pooled semantic slots are always fixed-length valid tokens.
            s, sem = self._maybe_checkpoint(
                lambda s_in, sem_in, c_in: block(s_in, sem_in, c_in, rope=self.patch_rope, sem_mask=None),
                s,
                sem,
                c_time,
            )

        # DDP note:
        # The final patch block updates `sem` but only `s` is consumed downstream.
        # Attach a zero-weight semantic anchor so the final semantic branch remains
        # in the autograd graph without changing numerical behavior.
        sem_anchor = sem.mean(dim=1, keepdim=True)
        s_cond = s + c_time.unsqueeze(1) + sem_anchor * 0.0

        p_embed = self.pixel_embed(x)
        p_tokens = self.patchify_pixels(p_embed)
        pos_embed_patches = self.pixel_pos_embed.view(1, l, p * p, self.pixel_dim)
        p_tokens = p_tokens + pos_embed_patches

        for block in self.shared_pixel_blocks:
            p_tokens = self._maybe_checkpoint(
                lambda p_in, s_in: block(p_in, s_in, rope=self.patch_rope),
                p_tokens,
                s_cond,
            )

        u_tokens = p_tokens
        v_tokens = p_tokens

        for block in self.u_head_blocks:
            u_tokens = self._maybe_checkpoint(
                lambda p_in, s_in: block(p_in, s_in, rope=self.patch_rope),
                u_tokens,
                s_cond,
            )
        for block in self.v_head_blocks:
            v_tokens = self._maybe_checkpoint(
                lambda p_in, s_in: block(p_in, s_in, rope=self.patch_rope),
                v_tokens,
                s_cond,
            )

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
