import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models import pmfDiT


def _expand_scalar(value, bsz: int, device, dtype):
    if torch.is_tensor(value):
        t = value.to(device=device, dtype=dtype).reshape(-1)
        if t.numel() == 1:
            t = t.expand(bsz)
        return t
    return torch.full((bsz,), float(value), device=device, dtype=dtype)


class pixelMeanFlow(nn.Module):
    """pixel MeanFlow in pure PyTorch."""

    @staticmethod
    def _normalize_condition_mode(mode) -> str:
        mode = "clip_dino" if mode is None else str(mode).lower()
        aliases = {
            "none": "uncond",
            "unconditional": "uncond",
            "null": "uncond",
            "clip+dino": "clip_dino",
        }
        mode = aliases.get(mode, mode)
        valid_modes = {"uncond", "clip", "dino", "clip_dino"}
        if mode not in valid_modes:
            raise ValueError(f"Unsupported condition_mode={mode}. Expected one of {sorted(valid_modes)}.")
        return mode

    def __init__(
        self,
        model_str: str,
        dtype: torch.dtype = torch.float32,
        num_classes: int = 1000,  # kept for compatibility
        clip_feature_dim: int = 1024,
        dino_feature_dim: int = 768,
        num_clip_tokens: int = 4,
        num_dino_tokens: int = 4,
        P_mean: float = -0.4,
        P_std: float = 1.0,
        cfg_max: float = 7.0,
        noise_scale: float = 1.0,
        data_proportion: float = 0.5,
        cfg_beta: float = 1.0,
        class_dropout_prob: float = 0.1,
        cond_drop_high_noise_only: bool = True,
        schedule_t0: float = 0.5,
        lambda_perc_max: float = 1.0,
        lambda_perc_gamma: float = 4.0,
        lambda_perc_hard_t0: bool = True,
        lambda_sem_max: float = 1.0,
        lambda_sem_beta: float = 2.0,
        lambda_sem_eps: float = 1e-4,
        lambda_sem_gate: str = "sigmoid",
        lambda_sem_gate_k: float = 12.0,
        lambda_sem_hard_t0: bool = True,
        norm_p: float = 1.0,
        norm_eps: float = 0.01,
        eval: bool = False,
        lpips: bool = False,
        lpips_lambda: float = 1.0,
        convnext: bool = False,
        convnext_model_name: str = "facebook/convnextv2-base-22k-224",
        convnext_lambda: float = 0.0,
        perceptual_max_t: float = 1.0,
        tr_uniform: bool = False,
        condition_mode: str = "clip_dino",
        enable_semantic_loss: bool = True,
        attention_impl: str = "standard",
        attention_gate_bias_init: float = 4.0,
    ):
        super().__init__()
        self.model_str = model_str
        self.dtype = dtype
        self.num_classes = num_classes
        self.clip_feature_dim = clip_feature_dim
        self.dino_feature_dim = dino_feature_dim
        self.num_clip_tokens = num_clip_tokens
        self.num_dino_tokens = num_dino_tokens

        self.P_mean = P_mean
        self.P_std = P_std
        self.cfg_max = cfg_max
        self.noise_scale = noise_scale

        self.data_proportion = data_proportion
        self.cfg_beta = cfg_beta
        self.class_dropout_prob = class_dropout_prob
        self.cond_drop_high_noise_only = cond_drop_high_noise_only

        self.schedule_t0 = schedule_t0
        self.lambda_perc_max = lambda_perc_max
        self.lambda_perc_gamma = lambda_perc_gamma
        self.lambda_perc_hard_t0 = lambda_perc_hard_t0
        self.lambda_sem_max = lambda_sem_max
        self.lambda_sem_beta = lambda_sem_beta
        self.lambda_sem_eps = lambda_sem_eps
        self.lambda_sem_gate = lambda_sem_gate
        self.lambda_sem_gate_k = lambda_sem_gate_k
        self.lambda_sem_hard_t0 = lambda_sem_hard_t0

        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.eval_mode = eval

        self.lpips = lpips
        self.lpips_lambda = lpips_lambda
        self.convnext = convnext
        # Aux-loss configuration key; accepted here so `config.model` can be passed
        # wholesale into pixelMeanFlow without filtering.
        self.convnext_model_name = str(convnext_model_name)
        self.convnext_lambda = convnext_lambda
        self.perceptual_max_t = perceptual_max_t
        self.tr_uniform = tr_uniform
        self.condition_mode = self._normalize_condition_mode(condition_mode)
        self.use_clip_condition = self.condition_mode in {"clip", "clip_dino"}
        self.use_dino_condition = self.condition_mode in {"dino", "clip_dino"}
        self.enable_semantic_loss = bool(enable_semantic_loss)
        self.attention_impl = str(attention_impl).lower()
        self.attention_gate_bias_init = float(attention_gate_bias_init)

        net_fn = getattr(pmfDiT, self.model_str)
        self.net: pmfDiT.pmfDiT = net_fn(
            num_classes=self.num_classes,
            clip_feature_dim=self.clip_feature_dim,
            dino_feature_dim=self.dino_feature_dim,
            num_clip_tokens=self.num_clip_tokens,
            num_dino_tokens=self.num_dino_tokens,
            attention_impl=self.attention_impl,
            attention_gate_bias_init=self.attention_gate_bias_init,
            eval=self.eval_mode,
        )

        null_clip_std = 1.0 / math.sqrt(self.clip_feature_dim)
        null_dino_std = 1.0 / math.sqrt(self.dino_feature_dim)
        self.null_clip_embedding = nn.Parameter(
            torch.randn(self.clip_feature_dim) * null_clip_std
        )
        self.null_dino_embedding = nn.Parameter(
            torch.randn(self.num_dino_tokens, self.dino_feature_dim) * null_dino_std
        )

    def null_condition(self, bz: int, device, dtype=None):
        if dtype is None:
            dtype = self.dtype
        null_clip = self.null_clip_embedding.to(device=device, dtype=dtype).reshape(1, -1)
        null_dino = self.null_dino_embedding.to(device=device, dtype=dtype).reshape(
            1, self.num_dino_tokens, self.dino_feature_dim
        )
        return {
            "clip": null_clip.expand(bz, self.clip_feature_dim),
            "dino": null_dino.expand(bz, self.num_dino_tokens, self.dino_feature_dim),
        }

    def normalize_condition(self, cond_embeddings, bz: int, device, dtype=None):
        if dtype is None:
            dtype = self.dtype
        if cond_embeddings is None:
            return self.null_condition(bz, device=device, dtype=dtype)
        if not isinstance(cond_embeddings, dict):
            raise TypeError("cond_embeddings must be a dict with keys 'clip' and 'dino'.")
        if "clip" not in cond_embeddings or "dino" not in cond_embeddings:
            raise KeyError("cond_embeddings must contain both 'clip' and 'dino'.")

        clip = torch.as_tensor(cond_embeddings["clip"], device=device, dtype=dtype)
        dino = torch.as_tensor(cond_embeddings["dino"], device=device, dtype=dtype)
        if clip.ndim == 1:
            clip = clip.unsqueeze(0)
        if dino.ndim == 1:
            dino = dino.unsqueeze(0).unsqueeze(1)
        elif dino.ndim == 2:
            dino = dino.unsqueeze(1)
        elif dino.ndim != 3:
            raise ValueError(
                f"dino embedding rank mismatch: got rank={dino.ndim}, expected 2 or 3."
            )
        if clip.shape[-1] != self.clip_feature_dim:
            raise ValueError(
                f"clip feature dim mismatch: got {clip.shape[-1]}, expected {self.clip_feature_dim}"
            )
        if dino.shape[2] != self.dino_feature_dim:
            raise ValueError(
                f"dino feature dim mismatch: got {dino.shape[2]}, expected {self.dino_feature_dim}"
            )
        if clip.shape[0] == 1 and bz > 1:
            clip = clip.expand(bz, self.clip_feature_dim)
        if dino.shape[0] == 1 and bz > 1:
            dino = dino.expand(bz, dino.shape[1], dino.shape[2])
        if dino.shape[1] == 1 and self.num_dino_tokens > 1:
            dino = dino.expand(dino.shape[0], self.num_dino_tokens, self.dino_feature_dim)
        if dino.shape[1] != self.num_dino_tokens:
            raise ValueError(
                f"dino token count mismatch: got {dino.shape[1]}, expected {self.num_dino_tokens}"
            )
        if clip.shape[0] != bz or dino.shape[0] != bz:
            raise ValueError(
                f"condition batch size mismatch: clip={clip.shape[0]}, dino={dino.shape[0]}, expected={bz}"
            )
        return {"clip": clip, "dino": dino}

    #######################################################
    # Solver
    #######################################################

    @torch.no_grad()
    def sample_one_step(
        self,
        z_t: torch.Tensor,
        cond_embeddings: Optional[Dict[str, torch.Tensor]],
        i: int,
        t_steps: torch.Tensor,
        omega,
        t_min,
        t_max,
    ):
        t = t_steps[i]
        r = t_steps[i + 1]
        bsz = z_t.shape[0]
        device = z_t.device
        dtype = z_t.dtype

        t = torch.full((bsz,), float(t), device=device, dtype=dtype)
        r = torch.full((bsz,), float(r), device=device, dtype=dtype)
        omega = _expand_scalar(omega, bsz, device, dtype)
        t_min = _expand_scalar(t_min, bsz, device, dtype)
        t_max = _expand_scalar(t_max, bsz, device, dtype)
        cond_embeddings = self.normalize_condition(cond_embeddings, bsz, device, dtype)

        u = self.u_fn(z_t, t, t - r, omega, t_min, t_max, y=cond_embeddings)[0]
        return z_t - (t - r).view(-1, 1, 1, 1) * u

    @torch.no_grad()
    def generate(
        self,
        n_sample: int,
        num_steps: int,
        omega,
        t_min,
        t_max,
        cond_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        generator: Optional[torch.Generator] = None,
        image_size: Optional[int] = None,
        image_channels: int = 3,
    ):
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = self.dtype
        if image_size is None:
            image_size = self.net.input_size

        z_t = (
            torch.randn(
                n_sample,
                image_channels,
                image_size,
                image_size,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            * self.noise_scale
        )
        t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
        for i in range(num_steps):
            z_t = self.sample_one_step(z_t, cond_embeddings, i, t_steps, omega, t_min, t_max)
        return z_t

    #######################################################
    # Schedules
    #######################################################

    def logit_normal_dist(self, bz: int, device, dtype):
        rnd_normal = torch.randn((bz, 1, 1, 1), device=device, dtype=dtype)
        return torch.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def sample_tr(self, bz: int, device, dtype):
        t = self.logit_normal_dist(bz, device, dtype)
        r = self.logit_normal_dist(bz, device, dtype)

        if self.tr_uniform:
            unif_mask = (torch.rand((bz, 1, 1, 1), device=device, dtype=dtype) < 0.1)
            t_uniform = torch.rand((bz, 1, 1, 1), device=device, dtype=dtype)
            r_uniform = torch.rand((bz, 1, 1, 1), device=device, dtype=dtype)
            t = torch.where(unif_mask, t_uniform, t)
            r = torch.where(unif_mask, r_uniform, r)

        data_size = int(bz * self.data_proportion)
        fm_mask = (torch.arange(bz, device=device) < data_size).view(bz, 1, 1, 1)
        r = torch.where(fm_mask, t, r)
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        return t, r, fm_mask

    def sample_cfg_scale(self, bz: int, device, dtype, s_max: float = 7.0):
        u = torch.rand((bz, 1, 1, 1), device=device, dtype=torch.float32)
        if self.cfg_beta == 1.0:
            s = torch.exp(u * torch.log1p(torch.tensor(s_max, device=device, dtype=torch.float32)))
        else:
            smax = torch.tensor(s_max, device=device, dtype=torch.float32)
            b = torch.tensor(self.cfg_beta, device=device, dtype=torch.float32)
            log_base = (1.0 - b) * torch.log1p(smax)
            log_inner = torch.log1p(u * torch.expm1(log_base))
            s = torch.exp(log_inner / (1.0 - b))
        return s.to(dtype=dtype)

    def sample_cfg_interval(self, bz: int, device, dtype, fm_mask=None):
        t_min = torch.rand((bz, 1, 1, 1), device=device, dtype=dtype) * 0.5
        t_max = 0.5 + torch.rand((bz, 1, 1, 1), device=device, dtype=dtype) * 0.5
        if fm_mask is not None:
            t_min = torch.where(fm_mask, torch.zeros_like(t_min), t_min)
            t_max = torch.where(fm_mask, torch.ones_like(t_max), t_max)
        return t_min, t_max

    def adaptive_loss_weights(self, t: torch.Tensor, h: torch.Tensor):
        t_vec = t.reshape(-1)
        h_vec = h.reshape(-1)

        lambda_perc = self.lambda_perc_max * (1.0 - t_vec) * torch.exp(
            -self.lambda_perc_gamma * h_vec
        )
        if self.lambda_perc_hard_t0:
            lambda_perc = torch.where(
                t_vec < self.schedule_t0,
                lambda_perc,
                torch.zeros_like(lambda_perc),
            )

        if self.lambda_sem_gate == "linear":
            denom = max(1.0 - self.schedule_t0, self.lambda_sem_eps)
            gate_t = torch.clamp((t_vec - self.schedule_t0) / denom, 0.0, 1.0)
        elif self.lambda_sem_gate == "sigmoid":
            gate_t = torch.sigmoid(self.lambda_sem_gate_k * (t_vec - self.schedule_t0))
        else:
            raise ValueError(f"Unsupported lambda_sem_gate: {self.lambda_sem_gate}")

        h_ratio = h_vec / torch.clamp(t_vec + self.lambda_sem_eps, min=self.lambda_sem_eps)
        h_ratio = torch.clamp(h_ratio, 0.0, 1.0)
        lambda_sem = self.lambda_sem_max * gate_t * (h_ratio ** self.lambda_sem_beta)
        if self.lambda_sem_hard_t0:
            lambda_sem = torch.where(
                t_vec >= self.schedule_t0,
                lambda_sem,
                torch.zeros_like(lambda_sem),
            )
        return lambda_perc, lambda_sem

    #######################################################
    # Guidance helpers
    #######################################################

    def u_fn(self, x, t, h, omega, t_min, t_max, y):
        bz = x.shape[0]
        return self.net(
            x,
            t.reshape(bz),
            h.reshape(bz),
            omega.reshape(bz),
            t_min.reshape(bz),
            t_max.reshape(bz),
            y,
        )

    def v_cond_fn(self, x, t, omega, y):
        h = torch.zeros_like(t)
        t_min = torch.zeros_like(t)
        t_max = torch.ones_like(t)
        return self.u_fn(x, t, h, omega, t_min, t_max, y=y)[1]

    def v_fn(self, x, t, omega, y):
        bz = x.shape[0]
        x2 = torch.cat([x, x], dim=0)
        y_null = self.null_condition(bz, device=x.device, dtype=x.dtype)
        y2 = {
            "clip": torch.cat([y["clip"], y_null["clip"]], dim=0),
            "dino": torch.cat([y["dino"], y_null["dino"]], dim=0),
        }
        t2 = torch.cat([t, t], dim=0)
        w2 = torch.cat([omega, torch.ones_like(omega)], dim=0)
        out = self.v_cond_fn(x2, t2, w2, y2)
        return torch.chunk(out, 2, dim=0)

    def cond_drop(self, v_t, v_g, conditions, drop_enable_mask=None):
        bz = v_t.shape[0]
        drop_mask = torch.rand((bz,), device=v_t.device) < self.class_dropout_prob
        if drop_enable_mask is not None:
            drop_mask = drop_mask & drop_enable_mask.reshape(bz)
        drop_mask_image = drop_mask.view(-1, 1, 1, 1)
        drop_mask_feat = drop_mask.view(-1, 1)
        drop_mask_dino = drop_mask.view(-1, 1, 1)

        null_cond = self.null_condition(bz, device=v_t.device, dtype=v_t.dtype)
        conditions = {
            "clip": torch.where(drop_mask_feat, null_cond["clip"], conditions["clip"]),
            "dino": torch.where(drop_mask_dino, null_cond["dino"], conditions["dino"]),
        }
        v_g = torch.where(drop_mask_image, v_t, v_g)
        return conditions, v_g, drop_mask

    def guidance_fn(self, v_t, z_t, t, r, y, fm_mask, w, t_min, t_max):
        v_c, v_u = self.v_fn(z_t, t, w, y=y)
        v_g_fm = v_t + (1.0 - 1.0 / w).view(-1, 1, 1, 1) * (v_c - v_u)

        w_interval = torch.where((t >= t_min) & (t <= t_max), w, torch.ones_like(w))
        v_c_interval = self.v_cond_fn(z_t, t, w_interval, y=y)
        v_g = v_t + (1.0 - 1.0 / w_interval).view(-1, 1, 1, 1) * (v_c_interval - v_u)
        v_g = torch.where(fm_mask, v_g_fm, v_g)
        return v_g, v_c_interval

    #######################################################
    # Forward and losses
    #######################################################

    def forward(
        self,
        images: torch.Tensor,
        cond_embeddings: Optional[Dict[str, torch.Tensor]],
        aux_fn=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = images.to(dtype=self.dtype)
        bsz = x.shape[0]
        device = x.device
        cond_embeddings = self.normalize_condition(
            cond_embeddings, bsz, device=device, dtype=x.dtype
        )
        gt_sem_targets = {
            "clip": cond_embeddings["clip"],
            "dino": cond_embeddings["dino"],
        }

        t, r, fm_mask = self.sample_tr(bsz, device=device, dtype=x.dtype)
        h = t - r
        t_flat = t.reshape(-1)
        high_noise_mask = t_flat >= self.schedule_t0

        e = torch.randn_like(x) * self.noise_scale
        z_t = (1.0 - t) * x + t * e
        v_t = (z_t - x) / torch.clamp(t, min=0.05)

        t_min, t_max = self.sample_cfg_interval(bsz, device=device, dtype=x.dtype, fm_mask=fm_mask)
        omega = self.sample_cfg_scale(bsz, device=device, dtype=x.dtype, s_max=self.cfg_max)
        lambda_perc, lambda_sem = self.adaptive_loss_weights(t, h)

        null_cond = self.null_condition(bsz, device=device, dtype=x.dtype)
        clip_cond = cond_embeddings["clip"] if self.use_clip_condition else null_cond["clip"]
        dino_cond = cond_embeddings["dino"] if self.use_dino_condition else null_cond["dino"]
        cond_embeddings = {
            "clip": torch.where(
                high_noise_mask[:, None], clip_cond, null_cond["clip"]
            ),
            "dino": torch.where(
                high_noise_mask[:, None, None], dino_cond, null_cond["dino"]
            ),
        }
        lambda_sem = torch.where(high_noise_mask, lambda_sem, torch.zeros_like(lambda_sem))

        # CFG targets (v_g) and the tangent velocity (v_c) are used as training
        # targets / JVP tangents only; they are stop-grad in the original pMF
        # formulation, so we avoid building an autograd graph here.
        with torch.no_grad():
            v_g, v_c = self.guidance_fn(
                v_t,
                z_t,
                t.reshape(-1),
                r.reshape(-1),
                cond_embeddings,
                fm_mask,
                omega.reshape(-1),
                t_min.reshape(-1),
                t_max.reshape(-1),
            )

        drop_enable_mask = high_noise_mask if self.cond_drop_high_noise_only else None
        cond_embeddings, v_g, dropped_mask = self.cond_drop(
            v_t, v_g, cond_embeddings, drop_enable_mask=drop_enable_mask
        )
        lambda_sem = torch.where(dropped_mask, torch.zeros_like(lambda_sem), lambda_sem)

        omega_flat = omega.reshape(-1)
        t_min_flat = t_min.reshape(-1)
        t_max_flat = t_max.reshape(-1)

        def u_only(z_in, t_in, r_in):
            u_out = self.u_fn(
                z_in,
                t_in.reshape(-1),
                (t_in - r_in).reshape(-1),
                omega_flat,
                t_min_flat,
                t_max_flat,
                y=cond_embeddings,
            )[0]
            return u_out

        dtdt = torch.ones_like(t)
        dtdr = torch.zeros_like(t)

        # Compute primal u/v.
        u, v = self.u_fn(
            z_t,
            t.reshape(-1),
            (t - r).reshape(-1),
            omega_flat,
            t_min_flat,
            t_max_flat,
            y=cond_embeddings,
        )
        # JVP du/dt using same tangent setting as the original code.
        _, du_dt = torch.autograd.functional.jvp(
            u_only,
            (z_t, t, r),
            (v_c.detach(), dtdt, dtdr),
            # Keep training-time behavior aligned with the original implementation.
            # `autograd.functional.jvp(create_graph=False)` can trigger a DDP reducer
            # "unfinished reduction / unused params" failure in some PyTorch builds,
            # because it may use an internal backward-based path during forward.
            # We still detach `du_dt` below to preserve the pMF stop-grad semantics.
            create_graph=self.training,
            strict=False,
        )

        V = u + (t - r) * du_dt.detach()
        pred_x = z_t - t * u
        v_g = v_g.detach()

        def adp_wt_fn(loss_vec):
            adp_wt = (loss_vec + self.norm_eps) ** self.norm_p
            return loss_vec / adp_wt.detach()

        loss_u = torch.sum((V - v_g) ** 2, dim=(1, 2, 3))
        loss_u = adp_wt_fn(loss_u)

        loss_v = torch.sum((v - v_g) ** 2, dim=(1, 2, 3))
        loss_v = adp_wt_fn(loss_v)

        use_semantic_loss = (
            self.enable_semantic_loss
            and self.lambda_sem_max > 0.0
            and (self.use_clip_condition or self.use_dino_condition)
        )
        need_aux = self.convnext or self.lpips or use_semantic_loss
        aux_loss_lpips = torch.zeros((bsz,), device=device, dtype=x.dtype)
        aux_loss_convnext = torch.zeros((bsz,), device=device, dtype=x.dtype)
        loss_sem_raw = torch.zeros((bsz,), device=device, dtype=x.dtype)
        if need_aux:
            if aux_fn is None:
                raise ValueError("auxiliary loss function is not provided.")

            perc_enabled = bool(self.convnext or self.lpips)
            perc_mask = (
                (t_flat < self.perceptual_max_t) & (lambda_perc > 0)
                if perc_enabled
                else torch.zeros_like(t_flat, dtype=torch.bool)
            )
            sem_mask = (
                (lambda_sem > 0)
                if use_semantic_loss
                else torch.zeros_like(t_flat, dtype=torch.bool)
            )

            # Partition masks to avoid duplicate aux forwards when perc/sem overlap.
            both_mask = perc_mask & sem_mask
            perc_only_mask = perc_mask & (~sem_mask)
            sem_only_mask = sem_mask & (~perc_mask)

            def _scatter_1d(idx: torch.Tensor, values: torch.Tensor):
                base = torch.zeros((bsz,), device=device, dtype=values.dtype)
                return base.scatter(0, idx, values)

            if both_mask.any():
                idx = torch.nonzero(both_mask, as_tuple=False).squeeze(1)
                lp_both, cv_both, sem_both = aux_fn(
                    pred_x[idx],
                    x[idx],
                    compute_perc=True,
                    compute_sem=True,
                    gt_sem_targets={
                        "clip": gt_sem_targets["clip"][idx],
                        "dino": gt_sem_targets["dino"][idx],
                    },
                )
                aux_loss_lpips = aux_loss_lpips + _scatter_1d(idx, lp_both.to(device=device, dtype=x.dtype))
                aux_loss_convnext = aux_loss_convnext + _scatter_1d(
                    idx, cv_both.to(device=device, dtype=x.dtype)
                )
                loss_sem_raw = loss_sem_raw + _scatter_1d(idx, sem_both.to(device=device, dtype=x.dtype))

            if perc_only_mask.any():
                idx = torch.nonzero(perc_only_mask, as_tuple=False).squeeze(1)
                lp_perc, cv_perc, _ = aux_fn(
                    pred_x[idx], x[idx], compute_perc=True, compute_sem=False
                )
                aux_loss_lpips = aux_loss_lpips + _scatter_1d(
                    idx, lp_perc.to(device=device, dtype=x.dtype)
                )
                aux_loss_convnext = aux_loss_convnext + _scatter_1d(
                    idx, cv_perc.to(device=device, dtype=x.dtype)
                )

            if sem_only_mask.any():
                idx = torch.nonzero(sem_only_mask, as_tuple=False).squeeze(1)
                _, _, sem_only = aux_fn(
                    pred_x[idx],
                    x[idx],
                    compute_perc=False,
                    compute_sem=True,
                    gt_sem_targets={
                        "clip": gt_sem_targets["clip"][idx],
                        "dino": gt_sem_targets["dino"][idx],
                    },
                )
                loss_sem_raw = loss_sem_raw + _scatter_1d(
                    idx, sem_only.to(device=device, dtype=x.dtype)
                )

        if self.convnext or self.lpips:
            perc_lpips = adp_wt_fn(aux_loss_lpips) * self.lpips_lambda
            perc_convnext = adp_wt_fn(aux_loss_convnext) * self.convnext_lambda
            loss_perc = lambda_perc * (perc_lpips + perc_convnext)
        else:
            loss_perc = torch.zeros((bsz,), device=device, dtype=x.dtype)

        if use_semantic_loss:
            loss_sem = lambda_sem * loss_sem_raw
        else:
            loss_sem = torch.zeros((bsz,), device=device, dtype=x.dtype)
            loss_sem_raw = torch.zeros((bsz,), device=device, dtype=x.dtype)

        loss = (loss_u + loss_v + loss_perc + loss_sem).mean()

        metrics = {
            "loss": loss.detach(),
            "loss_u": torch.mean((V - v_g) ** 2).detach(),
            "loss_v": torch.mean((v - v_g) ** 2).detach(),
            "loss_perc": torch.mean(loss_perc).detach(),
            "loss_sem": torch.mean(loss_sem).detach(),
            "loss_sem_raw": torch.mean(loss_sem_raw).detach(),
            "aux_loss_lpips": torch.mean(aux_loss_lpips).detach(),
            "aux_loss_convnext": torch.mean(aux_loss_convnext).detach(),
            "lambda_perc": torch.mean(lambda_perc).detach(),
            "lambda_sem": torch.mean(lambda_sem).detach(),
            "high_noise_ratio": torch.mean(high_noise_mask.float()).detach(),
            "cond_drop_ratio": torch.mean(dropped_mask.float()).detach(),
            "h_over_t": torch.mean(
                torch.clamp(
                    h.reshape(-1)
                    / torch.clamp(t.reshape(-1) + self.lambda_sem_eps, min=self.lambda_sem_eps),
                    0.0,
                    1.0,
                )
            ).detach(),
        }
        return loss, metrics
