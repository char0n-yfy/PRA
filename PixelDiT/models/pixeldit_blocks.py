import math
from math import pi
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: Optional[int] = None, act_layer=None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        act_layer = act_layer or nn.GELU
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer() if callable(act_layer) and not isinstance(act_layer, nn.Module) else act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=True)
        self.w3 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


def broadcat(tensors, dim=-1):
    shape_lens = set(map(lambda t: len(t.shape), tensors))
    assert len(shape_lens) == 1
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(map(lambda t: len(set(t[1])) <= 2, expandable_dims))
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * len(tensors)), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim: int,
        pt_seq_len: int = 16,
        ft_seq_len: Optional[int] = None,
        custom_freqs=None,
        freqs_for: str = "lang",
        theta: int = 10000,
        max_freq: int = 10,
        num_freqs: int = 1,
    ):
        super().__init__()
        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        self.register_buffer("freqs_cos", freqs.cos().view(-1, freqs.shape[-1]))
        self.register_buffer("freqs_sin", freqs.sin().view(-1, freqs.shape[-1]))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


def modulate(x: torch.Tensor, shift: Optional[torch.Tensor], scale: torch.Tensor) -> torch.Tensor:
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift


def pixel_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rmsnorm: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope=None, rope_segments: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            if rope_segments is None:
                q = rope(q)
                k = rope(k)
            else:
                seg_total = int(sum(rope_segments))
                if seg_total > n:
                    raise ValueError(f"rope_segments sum ({seg_total}) exceeds sequence length ({n})")
                q_parts, k_parts = [], []
                start = 0
                for seg_len in rope_segments:
                    end = start + int(seg_len)
                    q_parts.append(rope(q[:, :, start:end, :]))
                    k_parts.append(rope(k[:, :, start:end, :]))
                    start = end
                if start < n:
                    q_parts.append(q[:, :, start:, :])
                    k_parts.append(k[:, :, start:, :])
                q = torch.cat(q_parts, dim=2)
                k = torch.cat(k_parts, dim=2)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """Cross attention: queries attend to key/value tokens.

    Shapes:
      q:  (B, T, D)
      kv: (B, N, D)
      out:(B, T, D)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        use_rmsnorm: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads

        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, d = q.shape
        _, n, _ = kv.shape

        q = self.q_proj(q).reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,hd)
        k = self.k_proj(kv).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,N,hd)
        v = self.v_proj(kv).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,N,hd)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_mask = None
        if kv_mask is not None:
            # F.scaled_dot_product_attention expects boolean masks with True = keep, False = masked.
            if kv_mask.ndim != 2 or kv_mask.shape != (b, n):
                raise ValueError(f"kv_mask must be (B,N), got {tuple(kv_mask.shape)}")
            attn_mask = kv_mask[:, None, None, :].expand(b, self.num_heads, t, n)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.out_proj(out)


class CrossAttentionPooler(nn.Module):
    """Attention pooling with learnable queries (no extra transformer depth)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_queries: int,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_queries = int(num_queries)
        self.queries = nn.Parameter(torch.zeros(1, self.num_queries, self.dim))
        self.attn = CrossAttention(
            dim=self.dim,
            num_heads=int(num_heads),
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            use_rmsnorm=use_rmsnorm,
        )

    def forward(self, tokens: torch.Tensor, tokens_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be (B,N,D), got {tuple(tokens.shape)}")
        b = tokens.shape[0]
        q = self.queries.expand(b, -1, -1)
        return self.attn(q, tokens, kv_mask=tokens_mask)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class MMDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        norm_layer = RMSNorm if use_rmsnorm else lambda d: nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)

        self.norm1_img = norm_layer(hidden_size)
        self.norm2_img = norm_layer(hidden_size)
        self.qkv_img = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj_img = nn.Linear(hidden_size, hidden_size)

        self.norm1_sem = norm_layer(hidden_size)
        self.norm2_sem = norm_layer(hidden_size)
        self.qkv_sem = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj_sem = nn.Linear(hidden_size, hidden_size)

        if use_qknorm:
            self.q_norm = norm_layer(self.head_dim)
            self.k_norm = norm_layer(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp_img = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
            self.mlp_sem = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp_img = Mlp(hidden_size, mlp_hidden_dim, act_layer=lambda: nn.GELU(approximate="tanh"))
            self.mlp_sem = Mlp(hidden_size, mlp_hidden_dim, act_layer=lambda: nn.GELU(approximate="tanh"))

        self.adaLN_modulation_img = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_modulation_sem = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(
        self,
        img: torch.Tensor,
        sem: torch.Tensor,
        c: torch.Tensor,
        rope=None,
        sem_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = img.shape[0]
        l_img = img.shape[1]
        l_sem = sem.shape[1]

        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i = self.adaLN_modulation_img(c).chunk(6, dim=1)
        shift_msa_s, scale_msa_s, gate_msa_s, shift_mlp_s, scale_mlp_s, gate_mlp_s = self.adaLN_modulation_sem(c).chunk(6, dim=1)

        img_norm = modulate(self.norm1_img(img), shift_msa_i.unsqueeze(1), scale_msa_i.unsqueeze(1))
        sem_norm = modulate(self.norm1_sem(sem), shift_msa_s.unsqueeze(1), scale_msa_s.unsqueeze(1))

        qkv_img = self.qkv_img(img_norm).reshape(b, l_img, 3, self.num_heads, self.head_dim)
        qkv_sem = self.qkv_sem(sem_norm).reshape(b, l_sem, 3, self.num_heads, self.head_dim)

        qkv = torch.cat([qkv_img, qkv_sem], dim=1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q_img = rope(q[:, :, :l_img, :])
            k_img = rope(k[:, :, :l_img, :])
            q = torch.cat([q_img, q[:, :, l_img:, :]], dim=2)
            k = torch.cat([k_img, k[:, :, l_img:, :]], dim=2)

        attn_mask = None
        if sem_mask is not None:
            img_mask = torch.ones(b, l_img, dtype=torch.bool, device=img.device)
            combined = torch.cat([img_mask, sem_mask], dim=1)
            l_total = l_img + l_sem
            attn_mask = combined[:, None, None, :].expand(-1, -1, l_total, -1)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(b, l_img + l_sem, self.hidden_size)
        attn_img = attn_out[:, :l_img, :]
        attn_sem = attn_out[:, l_img:, :]

        img = img + gate_msa_i.unsqueeze(1) * self.proj_img(attn_img)
        sem = sem + gate_msa_s.unsqueeze(1) * self.proj_sem(attn_sem)

        img = img + gate_mlp_i.unsqueeze(1) * self.mlp_img(modulate(self.norm2_img(img), shift_mlp_i.unsqueeze(1), scale_mlp_i.unsqueeze(1)))
        sem = sem + gate_mlp_s.unsqueeze(1) * self.mlp_sem(modulate(self.norm2_sem(sem), shift_mlp_s.unsqueeze(1), scale_mlp_s.unsqueeze(1)))
        return img, sem


class PixelwiseAdaLN(nn.Module):
    def __init__(self, semantic_dim: int, pixel_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.pixel_dim = pixel_dim
        self.num_pixels = patch_size * patch_size
        out_dim = self.num_pixels * 6 * pixel_dim
        self.mlp = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim, bias=True),
            nn.SiLU(),
            nn.Linear(semantic_dim, out_dim, bias=True),
        )

    def forward(self, s_cond: torch.Tensor):
        theta = self.mlp(s_cond)
        theta = theta.view(-1, self.num_pixels, 6 * self.pixel_dim)
        return theta.chunk(6, dim=-1)


class PixelTokenCompaction(nn.Module):
    def __init__(self, pixel_dim: int, semantic_dim: int, patch_size: int):
        super().__init__()
        self.num_pixels = patch_size * patch_size
        self.pixel_dim = pixel_dim
        self.semantic_dim = semantic_dim
        self.compress = nn.Linear(self.num_pixels * pixel_dim, semantic_dim)
        self.expand = nn.Linear(semantic_dim, self.num_pixels * pixel_dim)

    def compact(self, x: torch.Tensor) -> torch.Tensor:
        b_l = x.shape[0]
        return self.compress(x.view(b_l, -1)).unsqueeze(1)

    def decompact(self, x: torch.Tensor) -> torch.Tensor:
        b_l = x.shape[0]
        return self.expand(x.squeeze(1)).view(b_l, self.num_pixels, self.pixel_dim)


class PixelTransformerBlock(nn.Module):
    def __init__(
        self,
        pixel_dim: int,
        semantic_dim: int,
        num_heads: int,
        patch_size: int,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = True,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.pixel_dim = pixel_dim

        self.pixelwise_adaln = PixelwiseAdaLN(semantic_dim, pixel_dim, patch_size)
        self.compaction = PixelTokenCompaction(pixel_dim, semantic_dim, patch_size)

        if use_rmsnorm:
            self.norm1 = RMSNorm(pixel_dim)
            self.norm2 = RMSNorm(pixel_dim)
        else:
            self.norm1 = nn.LayerNorm(pixel_dim, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(pixel_dim, elementwise_affine=False, eps=1e-6)

        self.attn = Attention(
            semantic_dim,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
        )

        mlp_hidden_dim = int(pixel_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=pixel_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )

    def forward(self, x: torch.Tensor, s_cond: torch.Tensor, rope=None) -> torch.Tensor:
        b, l, p2, d_pix = x.shape
        d_sem = self.semantic_dim

        x = x.reshape(b * l, p2, d_pix)
        s_cond_flat = s_cond.reshape(b * l, -1)

        shift1, scale1, gate1, shift2, scale2, gate2 = self.pixelwise_adaln(s_cond_flat)

        x_mod = pixel_modulate(self.norm1(x), shift1, scale1)
        x_compact = self.compaction.compact(x_mod).reshape(b, l, d_sem)
        x_attn = self.attn(x_compact, rope=rope).reshape(b * l, 1, d_sem)
        x_attn = self.compaction.decompact(x_attn)

        x = x + gate1 * x_attn
        x_mod = pixel_modulate(self.norm2(x), shift2, scale2)
        x = x + gate2 * self.mlp(x_mod)

        return x.reshape(b, l, p2, d_pix)


class PixelFinalLayer(nn.Module):
    def __init__(self, semantic_dim: int, pixel_dim: int, out_channels: int = 3, use_rmsnorm: bool = True):
        super().__init__()
        self.norm_final = RMSNorm(pixel_dim) if use_rmsnorm else nn.LayerNorm(pixel_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(pixel_dim, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(semantic_dim, 2 * pixel_dim, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift.unsqueeze(1), scale.unsqueeze(1))
        return self.linear(x)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed
