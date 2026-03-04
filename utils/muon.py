import torch


def _muon_zeropower_via_newton_schulz_5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    """
    Approximate orthogonalization used by Muon for matrix-like parameters.
    Input is flattened to 2D and processed in fp32 for stability.
    """
    x = g.float()
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    elif x.ndim == 1:
        x = x.unsqueeze(0)

    transpose = False
    if x.shape[0] > x.shape[1]:
        x = x.t()
        transpose = True

    x = x / (torch.linalg.norm(x) + eps)

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(max(int(steps), 1)):
        a_mat = x @ x.t()
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x

    if transpose:
        x = x.t()
    return x.reshape_as(g).to(dtype=g.dtype)


class Muon(torch.optim.Optimizer):
    """
    Minimal PyTorch Muon optimizer.
    - 2D tensors (ndim == 2): momentum + orthogonalized update.
    - non-2D tensors: Adam-style update fallback.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        adam_b2: float = 0.95,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        ns_steps: int = 5,
        ns_eps: float = 1e-7,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid lr={lr}")
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"Invalid beta1={beta1}")
        if not (0.0 <= adam_b2 < 1.0):
            raise ValueError(f"Invalid adam_b2={adam_b2}")
        if eps <= 0:
            raise ValueError(f"Invalid eps={eps}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps={ns_steps}")

        defaults = dict(
            lr=lr,
            beta1=beta1,
            adam_b2=adam_b2,
            weight_decay=weight_decay,
            eps=eps,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1 = float(group["beta1"])
            beta2 = float(group["adam_b2"])
            wd = float(group["weight_decay"])
            eps = float(group["eps"])
            ns_steps = int(group["ns_steps"])
            ns_eps = float(group["ns_eps"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if p.ndim != 2:
                        state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                step = int(state["step"])

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # 2D parameters: Muon update.
                if p.ndim == 2:
                    m = state["m"]
                    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    m_hat = m / (1.0 - beta1**step)
                    upd = _muon_zeropower_via_newton_schulz_5(
                        m_hat,
                        steps=ns_steps,
                        eps=ns_eps,
                    )
                    p.add_(upd, alpha=-lr)
                # non-2D parameters: Adam-style fallback.
                else:
                    m = state["m"]
                    v = state["v"]
                    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    m_hat = m / (1.0 - beta1**step)
                    v_hat = v / (1.0 - beta2**step)
                    p.addcdiv_(m_hat, torch.sqrt(v_hat) + eps, value=-lr)

        return loss


class MuonWithAuxAdamW(torch.optim.Optimizer):
    """
    Composite optimizer:
    - 2D parameters: torch.optim.Muon
    - non-2D parameters: torch.optim.AdamW

    This mirrors optax.contrib.muon behavior used in the original pMF code.
    """

    def __init__(
        self,
        muon_params,
        adamw_params,
        muon_kwargs: dict,
        adamw_kwargs: dict,
    ):
        muon_params = list(muon_params)
        adamw_params = list(adamw_params)
        if len(muon_params) == 0 and len(adamw_params) == 0:
            raise ValueError("MuonWithAuxAdamW received no trainable parameters.")

        param_groups = []
        self._muon_group_idx = None
        self._adamw_group_idx = None

        if len(muon_params) > 0:
            self._muon_group_idx = len(param_groups)
            param_groups.append(
                {
                    "params": muon_params,
                    "lr": float(muon_kwargs.get("lr", 1e-3)),
                    "weight_decay": float(muon_kwargs.get("weight_decay", 0.0)),
                    "_opt_kind": "muon",
                }
            )
        if len(adamw_params) > 0:
            self._adamw_group_idx = len(param_groups)
            param_groups.append(
                {
                    "params": adamw_params,
                    "lr": float(adamw_kwargs.get("lr", 1e-3)),
                    "weight_decay": float(adamw_kwargs.get("weight_decay", 0.0)),
                    "betas": tuple(adamw_kwargs.get("betas", (0.9, 0.999))),
                    "eps": float(adamw_kwargs.get("eps", 1e-8)),
                    "_opt_kind": "adamw",
                }
            )

        # torch.optim.Optimizer.__init__ expects positional args in newer PyTorch
        # versions (e.g. 2.9), and may reject keyword `param_groups`.
        super().__init__(param_groups, {})

        self.muon_opt = (
            torch.optim.Muon(muon_params, **muon_kwargs) if len(muon_params) > 0 else None
        )
        self.adamw_opt = (
            torch.optim.AdamW(adamw_params, **adamw_kwargs) if len(adamw_params) > 0 else None
        )

    def _sync_outer_to_inner(self):
        if self.muon_opt is not None:
            g_out = self.param_groups[self._muon_group_idx]
            g_in = self.muon_opt.param_groups[0]
            g_in["lr"] = float(g_out["lr"])
            g_in["weight_decay"] = float(g_out.get("weight_decay", g_in.get("weight_decay", 0.0)))
        if self.adamw_opt is not None:
            g_out = self.param_groups[self._adamw_group_idx]
            g_in = self.adamw_opt.param_groups[0]
            g_in["lr"] = float(g_out["lr"])
            g_in["weight_decay"] = float(g_out.get("weight_decay", g_in.get("weight_decay", 0.0)))
            if "betas" in g_out:
                g_in["betas"] = tuple(g_out["betas"])
            if "eps" in g_out:
                g_in["eps"] = float(g_out["eps"])

    def _sync_inner_to_outer(self):
        if self.muon_opt is not None:
            g_out = self.param_groups[self._muon_group_idx]
            g_in = self.muon_opt.param_groups[0]
            g_out["lr"] = float(g_in["lr"])
            g_out["weight_decay"] = float(g_in.get("weight_decay", g_out.get("weight_decay", 0.0)))
        if self.adamw_opt is not None:
            g_out = self.param_groups[self._adamw_group_idx]
            g_in = self.adamw_opt.param_groups[0]
            g_out["lr"] = float(g_in["lr"])
            g_out["weight_decay"] = float(g_in.get("weight_decay", g_out.get("weight_decay", 0.0)))
            g_out["betas"] = tuple(g_in.get("betas", g_out.get("betas", (0.9, 0.999))))
            g_out["eps"] = float(g_in.get("eps", g_out.get("eps", 1e-8)))

    @torch.no_grad()
    def step(self, closure=None):
        self._sync_outer_to_inner()

        loss = None
        if self.muon_opt is not None:
            loss = self.muon_opt.step(closure=closure)
            if self.adamw_opt is not None:
                self.adamw_opt.step()
        elif self.adamw_opt is not None:
            loss = self.adamw_opt.step(closure=closure)

        self._sync_inner_to_outer()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        if self.muon_opt is not None:
            self.muon_opt.zero_grad(set_to_none=set_to_none)
        if self.adamw_opt is not None:
            self.adamw_opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "muon": None if self.muon_opt is None else self.muon_opt.state_dict(),
            "adamw": None if self.adamw_opt is None else self.adamw_opt.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if "muon" in state_dict or "adamw" in state_dict:
            muon_state = state_dict.get("muon", None)
            adamw_state = state_dict.get("adamw", None)
            if self.muon_opt is not None and muon_state is not None:
                self.muon_opt.load_state_dict(muon_state)
            if self.adamw_opt is not None and adamw_state is not None:
                self.adamw_opt.load_state_dict(adamw_state)
            self._sync_inner_to_outer()
            return

        # Backward compatibility for old checkpoints with a single optimizer state.
        if self.muon_opt is not None and self.adamw_opt is None:
            self.muon_opt.load_state_dict(state_dict)
        elif self.adamw_opt is not None and self.muon_opt is None:
            self.adamw_opt.load_state_dict(state_dict)
        else:
            raise ValueError(
                "Incompatible optimizer checkpoint for MuonWithAuxAdamW. "
                "Expected keys ['muon', 'adamw']."
            )
        self._sync_inner_to_outer()
