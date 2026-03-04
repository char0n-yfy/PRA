import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embedder import BottleneckPatchEmbedder, TimestepEmbedder
from models.torch_models import RMSNorm, SwiGLUMlp, TorchLinear


def unsqueeze(t: torch.Tensor, dim: int):
    return t.unsqueeze(dim)


class RoPEAttention(nn.Module):
    """Multi-head self-attention with RoPE and QK RMSNorm."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        weight_init: str = "scaled_variance",
        weight_init_constant: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        init_kwargs = dict(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=False,
            weight_init=weight_init,
            init_constant=weight_init_constant,
        )
        self.q_proj = TorchLinear(**init_kwargs)
        self.k_proj = TorchLinear(**init_kwargs)
        self.v_proj = TorchLinear(**init_kwargs)
        self.out_proj = TorchLinear(**init_kwargs)

        self.head_dim = self.hidden_size // self.num_heads
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor):
        b, s, _ = x.shape
        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, s, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rotary_pos_emb(q, rope_freqs)
        k = apply_rotary_pos_emb(k, rope_freqs)

        query = q / math.sqrt(self.head_dim)
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", query, k)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        attn = attn.reshape(b, s, self.hidden_size)

        return self.out_proj(attn)


class QwenGatedRoPEAttention(RoPEAttention):
    """
    RoPE attention with a Qwen-style output gate.

    The gate is predicted from the attention input and applied per-head on the
    attention output before the final output projection.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        weight_init: str = "scaled_variance",
        weight_init_constant: float = 1.0,
        gate_bias_init: float = 4.0,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            weight_init=weight_init,
            weight_init_constant=weight_init_constant,
        )
        self.gate_proj = TorchLinear(
            in_features=self.hidden_size,
            out_features=self.num_heads,
            bias=False,
            weight_init="zeros",
            init_constant=1.0,
        )
        self.gate_bias = nn.Parameter(torch.full((self.num_heads,), float(gate_bias_init)))

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor):
        b, s, _ = x.shape
        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, s, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rotary_pos_emb(q, rope_freqs)
        k = apply_rotary_pos_emb(k, rope_freqs)

        query = q / math.sqrt(self.head_dim)
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", query, k)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)

        gate = self.gate_proj(x).reshape(b, s, self.num_heads, 1)
        gate = torch.sigmoid(gate + self.gate_bias.view(1, 1, self.num_heads, 1))
        attn = attn * gate.to(dtype=attn.dtype)
        attn = attn.reshape(b, s, self.hidden_size)
        return self.out_proj(attn)


class TransformerBlock(nn.Module):
    """Transformer block with zero-initialized residual gates."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 8 / 3,
        weight_init: str = "scaled_variance",
        weight_init_constant: float = 1.0,
        attention_impl: str = "standard",
        attention_gate_bias_init: float = 4.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        attention_impl = str(attention_impl).lower()
        if attention_impl in {"standard", "rope", "default"}:
            self.attn = RoPEAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                weight_init=weight_init,
                weight_init_constant=weight_init_constant,
            )
        elif attention_impl in {"qwen_gated", "gated", "gated_attention"}:
            self.attn = QwenGatedRoPEAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                weight_init=weight_init,
                weight_init_constant=weight_init_constant,
                gate_bias_init=attention_gate_bias_init,
            )
        else:
            raise ValueError(
                f"Unsupported attention_impl={attention_impl}. "
                "Expected one of ['standard', 'qwen_gated']."
            )
        self.norm2 = RMSNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if hidden_size > 1024:
            mlp_hidden_dim = (mlp_hidden_dim + 7) // 8 * 8
        self.mlp = SwiGLUMlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            weight_init=weight_init,
            weight_init_constant=weight_init_constant,
        )
        self.attn_scale = nn.Parameter(torch.zeros(hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor):
        x = x + self.attn(self.norm1(x), rope_freqs) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x


class FinalLayer(nn.Module):
    """RMSNorm + zero-init projection to patch space."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = TorchLinear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
            weight_init="zeros",
            bias_init="zeros",
        )

    def forward(self, x: torch.Tensor):
        return self.linear(self.norm(x))


class pmfDiT(nn.Module):
    """Improved MeanFlow DiT with in-context semantic conditioning."""

    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 16,
        num_heads: int = 12,
        mlp_ratio: float = 8 / 3,
        num_classes: int = 1000,  # unused, kept for compatibility
        clip_feature_dim: int = 1024,
        dino_feature_dim: int = 768,
        aux_head_depth: int = 8,
        num_clip_tokens: int = 4,
        num_dino_tokens: int = 4,
        num_time_tokens: int = 4,
        num_cfg_tokens: int = 4,
        num_interval_tokens: int = 2,
        token_init_constant: float = 1.0,
        embedding_init_constant: float = 1.0,
        weight_init_constant: float = 0.32,
        attention_impl: str = "standard",
        attention_gate_bias_init: float = 4.0,
        eval: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.clip_feature_dim = clip_feature_dim
        self.dino_feature_dim = dino_feature_dim

        self.aux_head_depth = aux_head_depth
        self.num_clip_tokens = num_clip_tokens
        self.num_dino_tokens = num_dino_tokens
        self.num_time_tokens = num_time_tokens
        self.num_cfg_tokens = num_cfg_tokens
        self.num_interval_tokens = num_interval_tokens
        self.eval = eval
        self.attention_impl = str(attention_impl).lower()
        self.attention_gate_bias_init = float(attention_gate_bias_init)
        self.out_channels = self.in_channels

        self.x_embedder = BottleneckPatchEmbedder(
            input_size=self.input_size,
            initial_patch_size=self.patch_size,
            pca_channels=128 if self.hidden_size <= 1024 else 256,
            in_channels=self.in_channels,
            hidden_size=self.hidden_size,
            bias=True,
        )

        embed_kwargs = dict(
            hidden_size=self.hidden_size,
            weight_init="scaled_variance",
            init_constant=embedding_init_constant,
        )
        self.h_embedder = TimestepEmbedder(**embed_kwargs)
        self.omega_embedder = TimestepEmbedder(**embed_kwargs)
        self.cfg_t_start_embedder = TimestepEmbedder(**embed_kwargs)
        self.cfg_t_end_embedder = TimestepEmbedder(**embed_kwargs)
        self.clip_embedder = TorchLinear(
            self.clip_feature_dim,
            self.hidden_size,
            bias=True,
            weight_init="scaled_variance",
            init_constant=embedding_init_constant,
            bias_init="zeros",
        )
        self.dino_embedder = TorchLinear(
            self.dino_feature_dim,
            self.hidden_size,
            bias=True,
            weight_init="scaled_variance",
            init_constant=embedding_init_constant,
            bias_init="zeros",
        )

        token_initializer = partial(
            nn.init.normal_, std=token_init_constant / math.sqrt(self.hidden_size)
        )
        self.time_tokens = nn.Parameter(
            token_initializer(torch.empty(1, self.num_time_tokens, self.hidden_size))
        )
        self.clip_tokens = nn.Parameter(
            token_initializer(torch.empty(1, self.num_clip_tokens, self.hidden_size))
        )
        self.dino_tokens = nn.Parameter(
            token_initializer(torch.empty(1, self.num_dino_tokens, self.hidden_size))
        )
        self.omega_tokens = nn.Parameter(
            token_initializer(torch.empty(1, self.num_cfg_tokens, self.hidden_size))
        )
        self.t_min_tokens = nn.Parameter(
            token_initializer(torch.empty(1, self.num_interval_tokens, self.hidden_size))
        )
        self.t_max_tokens = nn.Parameter(
            token_initializer(torch.empty(1, self.num_interval_tokens, self.hidden_size))
        )

        total_tokens = (
            self.x_embedder.num_patches
            + self.num_clip_tokens
            + self.num_dino_tokens
            + self.num_cfg_tokens
            + 2 * self.num_interval_tokens
            + self.num_time_tokens
        )
        self.prefix_tokens = (
            self.num_clip_tokens
            + self.num_dino_tokens
            + self.num_cfg_tokens
            + 2 * self.num_interval_tokens
            + self.num_time_tokens
        )
        self.head_dim = self.hidden_size // self.num_heads
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs_2d(self.head_dim, self.x_embedder.num_patches),
            persistent=False,
        )
        self.pos_embed = nn.Parameter(
            nn.init.normal_(torch.empty(1, total_tokens, self.hidden_size), std=0.02)
        )
        head_depth = self.aux_head_depth
        shared_depth = self.depth - head_depth
        block_kwargs = dict(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            weight_init="scaled_variance",
            weight_init_constant=weight_init_constant,
            attention_impl=self.attention_impl,
            attention_gate_bias_init=self.attention_gate_bias_init,
        )
        self.shared_blocks = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(shared_depth)]
        )
        self.u_heads = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(head_depth)]
        )
        self.v_heads = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(head_depth if not self.eval else 0)]
        )

        self.u_final_layer = FinalLayer(self.hidden_size, self.patch_size, self.out_channels)
        if self.eval:
            self.v_final_layer = None
        else:
            self.v_final_layer = FinalLayer(self.hidden_size, self.patch_size, self.out_channels)

    def unpatchify(self, x: torch.Tensor):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def _build_sequence(self, x, h, w, t_min, t_max, y):
        x_embed = self.x_embedder(x)
        h_embed = self.h_embedder(h)
        omega_embed = self.omega_embedder(1.0 - 1.0 / w)
        t_min_embed = self.cfg_t_start_embedder(t_min)
        t_max_embed = self.cfg_t_end_embedder(t_max)
        clip_embed = self.clip_embedder(y["clip"])
        dino_input = y["dino"]
        if dino_input.ndim == 2:
            dino_embed = self.dino_embedder(dino_input).unsqueeze(1).expand(
                -1, self.num_dino_tokens, -1
            )
        elif dino_input.ndim == 3:
            if dino_input.shape[1] != self.num_dino_tokens:
                raise ValueError(
                    f"dino token count mismatch: got {dino_input.shape[1]}, expected {self.num_dino_tokens}"
                )
            dino_embed = self.dino_embedder(dino_input)
        else:
            raise ValueError(
                f"Unsupported dino input rank {dino_input.ndim}. Expected 2 or 3."
            )

        time_tokens = self.time_tokens + unsqueeze(h_embed, 1)
        omega_tokens = self.omega_tokens + unsqueeze(omega_embed, 1)
        t_min_tokens = self.t_min_tokens + unsqueeze(t_min_embed, 1)
        t_max_tokens = self.t_max_tokens + unsqueeze(t_max_embed, 1)
        clip_tokens = self.clip_tokens + unsqueeze(clip_embed, 1)
        dino_tokens = self.dino_tokens + dino_embed

        seq = torch.cat(
            [
                clip_tokens,
                dino_tokens,
                omega_tokens,
                t_min_tokens,
                t_max_tokens,
                time_tokens,
                x_embed,
            ],
            dim=1,
        )
        return seq + self.pos_embed

    def forward(self, x, t, h, w, t_min, t_max, y):
        seq = self._build_sequence(x, h, w, t_min, t_max, y)

        for blk in self.shared_blocks:
            seq = blk(seq, self.rope_freqs)

        u_seq = seq
        v_seq = seq
        for blk in self.u_heads:
            u_seq = blk(u_seq, self.rope_freqs)
        for blk in self.v_heads:
            v_seq = blk(v_seq, self.rope_freqs)

        u_tokens = u_seq[:, self.prefix_tokens:]
        v_tokens = v_seq[:, self.prefix_tokens:]

        u = self.unpatchify(self.u_final_layer(u_tokens))
        if self.v_final_layer is None:
            v = torch.zeros_like(u)
        else:
            v = self.unpatchify(self.v_final_layer(v_tokens))

        t = t.reshape((-1, 1, 1, 1))
        u = (x - u) / torch.clamp(t, min=0.05)
        v = (x - v) / torch.clamp(t, min=0.05)
        return u, v


def precompute_rope_freqs_2d(dim: int, seq_len: int, theta: float = 10000.0):
    dim = dim // 2
    t = int(seq_len ** 0.5)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(t, dtype=torch.float32)
    freqs_h = torch.einsum("i,j->ij", positions, freqs)
    freqs_w = torch.einsum("i,j->ij", positions, freqs)
    freqs = torch.cat(
        [
            torch.tile(freqs_h[:, None, :], (1, t, 1)),
            torch.tile(freqs_w[None, :, :], (t, 1, 1)),
        ],
        dim=-1,
    )
    real = torch.cos(freqs).reshape(seq_len, dim)
    imag = torch.sin(freqs).reshape(seq_len, dim)
    return torch.complex(real, imag)


def apply_rotary_pos_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    x_float = x.to(torch.float32)
    x_complex = torch.view_as_complex(
        x_float.reshape(*x_float.shape[:-1], -1, 2).contiguous()
    )
    freqs = unsqueeze(unsqueeze(freqs_cis, 0), 2).to(x_complex.device)
    num_patch_tokens = freqs.shape[1]

    x_rotated = x_complex.clone()
    x_rotated[:, -num_patch_tokens:, :] = (
        x_complex[:, -num_patch_tokens:, :] * freqs
    )
    return torch.view_as_real(x_rotated).flatten(-2).to(x.dtype)


pmfDiT_B_16 = partial(
    pmfDiT,
    input_size=256,
    depth=16,
    hidden_size=768,
    patch_size=16,
    num_heads=12,
    aux_head_depth=8,
)

pmfDiT_B_32 = partial(
    pmfDiT,
    input_size=512,
    depth=16,
    hidden_size=768,
    patch_size=32,
    num_heads=12,
    aux_head_depth=8,
)

pmfDiT_L_16 = partial(
    pmfDiT,
    input_size=256,
    depth=32,
    hidden_size=1024,
    patch_size=16,
    num_heads=16,
    aux_head_depth=8,
)

pmfDiT_L_32 = partial(
    pmfDiT,
    input_size=512,
    depth=32,
    hidden_size=1024,
    patch_size=32,
    num_heads=16,
    aux_head_depth=8,
)

pmfDiT_H_16 = partial(
    pmfDiT,
    input_size=256,
    depth=48,
    hidden_size=1280,
    patch_size=16,
    num_heads=16,
    aux_head_depth=8,
)

pmfDiT_H_32 = partial(
    pmfDiT,
    input_size=512,
    depth=48,
    hidden_size=1280,
    patch_size=32,
    num_heads=16,
    aux_head_depth=8,
)
