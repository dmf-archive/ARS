from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer


@torch.jit.script
def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype=torch.bfloat16) if G.dtype == torch.float32 else G

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(p=2.0, dim=[-2, -1], keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X.to(G.dtype)


@torch.jit.script
def _adamw_step_kernel(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    beta1: float,
    beta2: float,
    step: int,
    lr: float,
    weight_decay: float,
    eps: float
):
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
    step_size = lr / bias_correction1

    if weight_decay != 0:
        param.mul_(1 - lr * weight_decay)

    param.addcdiv_(exp_avg, denom, value=-step_size)


class ARS2CSAGA(Optimizer):
    """
    ARS2C-SAGA: Sharpening-Aware Geometric Adaptation.

    ARS2C-SAGA is the flagship optimizer of the ARS family, unifying three
    geometric mechanisms into a single self-contained update law:

    1. **Energy-Geometry Decoupling** (ARS): Adam-style preconditioned gradients
       are decomposed into scalar energy and unit direction via Newton-Schulz
       orthogonalization, enabling structured natural-gradient-like updates.

    2. **Christoffel-Aware Dynamic Beta** (ARS2C): The HVP implicitly sampled
       during SAM sync steps is reused to construct a Christoffel matrix C.
       The geometric alignment between the orthogonalized Christoffel direction
       c_ortho and the update direction s_ortho drives dynamic momentum decay
       rates beta1, beta2: high alignment (curvature structure present) → strong
       filtering; low alignment → fast adaptation.

    3. **Sharpening-Aware Dynamic Rho** (SAGA): The SAM perturbation radius rho
       is no longer a fixed hyperparameter but a per-parameter dynamic state
       modulated by alignment. When alignment is high (update direction aligns
       with Christoffel curvature), rho regresses toward its steady-state mu
       (trust NGD). When alignment is low (curvature structure absent or
       chaotic), rho increases to apply flatness pressure and expose hidden
       structure — the mechanism underlying spontaneous Grokking.

    The optimizer employs a dual-track design: 2D+ parameters are routed to
    the full SAGA track (orthogonalized update + dynamic beta + dynamic rho),
    while 1D parameters (bias, LayerNorm, etc.) use fixed-beta AdamW.

    Args:
        params: parameter groups; each must contain a 'params' key and an
            optional 'is_rmsuon_group' flag.
        lr: learning rate, default 1e-3.
        betas: fallback Adam beta tuple (beta1, beta2) for 1D params and
            initial steps, default (0.9, 0.999).
        eps: Adam epsilon for numerical stability, default 1e-8.
        weight_decay: weight-decay coefficient, default 0.01.
        ns_steps: Newton-Schulz iteration steps, default 5.
        rho: initial SAM perturbation radius, default 0.1.
        k: SAM mode; k=0 disables SAM, k=1 synchronous, k>1 delayed, default 1.
        alpha: base shear-force injection strength, default 0.1.
        adaptive_sync: enable AGA sync mode, default False.
        adaptive_beta: EMA coefficient for AGA geometric noise tracking,
            default 0.99.
        adaptive_lambda: AGA sensitivity for dynamic threshold, default 2.0.
        adaptive_gamma: AGA exponent for alpha amplification, default 2.0.
        beta1_min, beta1_max: dynamic range for beta1, default (0.5, 0.95).
        beta2_min, beta2_max: dynamic range for beta2, default (0.9, 0.9995).
        rho_kappa: regression rate toward rho_target, default 0.01.
        rho_eta: riseup rate when alignment is low, default 0.1.
        rho_min: lower bound for dynamic rho, default 0.01.
        rho_max: upper bound for dynamic rho, default 1.0.
    """

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01, ns_steps: int = 5,
                 rho: float = 0.1, k: int = 1, alpha: float = 0.1,
                 adaptive_sync: bool = False,
                 adaptive_beta: float = 0.99, adaptive_lambda: float = 2.0, adaptive_gamma: float = 2.0,
                 beta1_min: float = 0.5, beta1_max: float = 0.95,
                 beta2_min: float = 0.9, beta2_max: float = 0.9995,
                 rho_kappa: float = 0.01, rho_eta: float = 0.1,
                 rho_min: float = 0.01, rho_max: float = 1.0):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            ns_steps=ns_steps, rho=rho, k=k, alpha=alpha,
            adaptive_sync=adaptive_sync,
            adaptive_beta=adaptive_beta, adaptive_lambda=adaptive_lambda, adaptive_gamma=adaptive_gamma,
            beta1_min=beta1_min, beta1_max=beta1_max,
            beta2_min=beta2_min, beta2_max=beta2_max,
            rho_kappa=rho_kappa, rho_eta=rho_eta,
            rho_min=rho_min, rho_max=rho_max,
        )
        super().__init__(params, defaults)
        self.state: dict[str, Any] = self.state
        self.state['step'] = 0
        self.state['phi_t'] = 1.0
        self.state['phi_var'] = 0.0
        self.state['threshold'] = 1.0
        self.state['alpha_t'] = alpha
        self.state['sync_steps'] = 0
        self.state['last_sync_step'] = 0

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        if closure is None:
            raise ValueError("ARS2C-SAGA requires a closure.")

        self.state['step'] += 1
        global_step = self.state['step']

        group0 = self.param_groups[0]
        k = group0.get('k', 1)
        adaptive_sync = group0.get('adaptive_sync', False)
        is_sam_enabled = k > 0 or adaptive_sync

        with torch.enable_grad():
            loss = closure()
            loss.backward()

        is_sync_step = False

        if is_sam_enabled:
            if adaptive_sync:
                phi_t = self._calculate_global_phi()
                self.state['phi_t'] = phi_t

                beta = group0.get('adaptive_beta', 0.99)
                diff_sq = phi_t ** 2

                if global_step == 1:
                    self.state['phi_var'] = diff_sq
                else:
                    if global_step - self.state.get('last_sync_step', 0) > 1:
                        self.state['phi_var'] = beta * self.state['phi_var'] + (1 - beta) * diff_sq

                std = self.state['phi_var'] ** 0.5
                threshold = - group0.get('adaptive_lambda', 2.0) * std
                self.state['threshold'] = threshold

                steps_since_sync = global_step - self.state.get('last_sync_step', 0)
                is_drift = phi_t < threshold
                is_sync_step = is_drift or (global_step == 1) or (k > 1 and steps_since_sync >= k)

                gamma = group0.get('adaptive_gamma', 2.0)
                alpha_max = group0.get('alpha', 0.1)
                self.state['alpha_t'] = alpha_max * ((1.0 + max(0.0, phi_t)) ** gamma)

            else:
                is_sync_step = (global_step % k == 1 if k > 1 else True)
                self.state['alpha_t'] = group0.get('alpha', 0.1)

        if is_sam_enabled and is_sync_step:
            self.state['last_sync_step'] = global_step
            self.state['sync_steps'] += 1
            for group in self.param_groups:
                rho_base = group.get('rho', 0.1)
                if rho_base <= 0:
                    continue
                eps = group.get('eps', 1e-8)
                beta2 = group.get('betas', (0.9, 0.999))[1]

                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if 'exp_avg_sq' not in state:
                        state['exp_avg_sq'] = torch.zeros_like(p)
                        state['rho'] = rho_base

                    current_rho = state.get('rho', rho_base)
                    v_hat = state['exp_avg_sq'] / (1 - beta2 ** max(1, global_step - 1) + 1e-12)
                    g_nat = p.grad / (v_hat.sqrt() + eps)
                    g_nat = g_nat * p.abs()

                    norm = g_nat.norm() + 1e-12
                    perturb = g_nat * (current_rho / norm)

                    state['last_perturbation'] = perturb
                    state['base_grad'] = p.grad.clone()
                    p.add_(perturb)

            self.zero_grad()
            with torch.enable_grad():
                loss_adv = closure()
                loss_adv.backward()

            for group in self.param_groups:
                rho_base = group.get('rho', 0.1)
                if rho_base <= 0:
                    continue
                _eps = group.get('eps', 1e-8)
                _beta2 = group.get('betas', (0.9, 0.999))[1]
                _rho_kappa = group.get('rho_kappa', 0.01)
                _rho_eta = group.get('rho_eta', 0.1)
                _rho_min = group.get('rho_min', 0.01)
                _rho_max = group.get('rho_max', 1.0)
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    p.sub_(state['last_perturbation'])

                    if k > 1 or adaptive_sync:
                        g_base = state['base_grad']
                        g_adv = p.grad
                        dot = (g_adv * g_base).sum()
                        base_norm_sq = (g_base * g_base).sum() + 1e-12
                        state['shear_force'] = g_adv - (dot / base_norm_sq) * g_base

                    if p.ndim >= 2:
                        _v_hat = state['exp_avg_sq'] / (1 - _beta2 ** max(1, global_step - 1) + 1e-12)
                        _g_base = state['base_grad']
                        _g_adv = p.grad
                        _delta_g = _g_adv - _g_base
                        _g_hat = _g_base / (_v_hat.sqrt() + _eps)
                        _C = _delta_g / (state['rho'] * _g_hat.abs() + _eps)
                        _c_flat = _C.view(_C.size(0), -1)
                        state['c_magnitude'] = float(_c_flat.norm())
                        state['c_ortho'] = _zeropower_via_newtonschulz5(_c_flat)

                        current_rho = state['rho']
                        prev_alignment = state.get('alignment_matrix', None)
                        alignment_val = prev_alignment.mean().item() if prev_alignment is not None else 0.5
                        rho_target = _rho_min + (_rho_max - _rho_min) * alignment_val
                        new_rho = current_rho + _rho_kappa * (rho_target - current_rho)
                        state['rho'] = max(_rho_min, min(_rho_max, new_rho))

        elif is_sam_enabled and not is_sync_step:
            current_alpha = self.state.get('alpha_t', 0.1)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if 'shear_force' in state:
                        v = state['shear_force']
                        g_norm = p.grad.norm()
                        v_norm = v.norm() + 1e-12
                        p.grad.add_(v, alpha=current_alpha * (g_norm / v_norm))

        for group in self.param_groups:
            if group.get('is_rmsuon_group', False):
                self._ars2_update(group, global_step)
            else:
                self._adamw_update(group, global_step)

        return loss

    def _apply_ars2_kernel(self, p, beta1, beta2, lr, eps, weight_decay, ns_steps):
        if p.grad is None:
            return

        state = self.state[p]
        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(p)
            state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] = 0
            for group in self.param_groups:
                if any(p is param for param in group['params']):
                    state['beta1_min'] = group.get('beta1_min', 0.5)
                    state['beta1_max'] = group.get('beta1_max', 0.95)
                    state['beta2_min'] = group.get('beta2_min', 0.9)
                    state['beta2_max'] = group.get('beta2_max', 0.9995)
                    break

        state['step'] += 1
        step = state['step']
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

        alignment_matrix = state.get('alignment_matrix', None)
        b1_min = state.get('beta1_min', 0.5)
        b1_max = state.get('beta1_max', 0.95)
        b2_min = state.get('beta2_min', 0.9)
        b2_max = state.get('beta2_max', 0.9995)

        if alignment_matrix is not None and p.ndim >= 2:
            beta1_matrix = b1_min + (b1_max - b1_min) * alignment_matrix
            beta2_matrix = b2_min + (b2_max - b2_min) * alignment_matrix

            exp_avg.mul_(beta1_matrix)
            exp_avg.add_(p.grad * (1 - beta1_matrix))
            exp_avg_sq.mul_(beta2_matrix)
            exp_avg_sq.add_(p.grad * p.grad * (1 - beta2_matrix))

            bias_correction1 = 1 - torch.pow(beta1_matrix, step)
            bias_correction2 = 1 - torch.pow(beta2_matrix, step)

            m_hat = exp_avg / bias_correction1
            v_hat = exp_avg_sq / bias_correction2
        else:
            exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            m_hat = exp_avg / bias_correction1
            v_hat = exp_avg_sq / bias_correction2

        m_scaled = m_hat / (v_hat.sqrt() + eps)
        energy = m_scaled.norm()

        original_shape = m_scaled.shape
        m_scaled_flat = m_scaled.view(m_scaled.size(0), -1) if p.ndim == 4 else m_scaled

        s_ortho = _zeropower_via_newtonschulz5(m_scaled_flat, steps=ns_steps)
        if p.ndim == 4:
            s_ortho = s_ortho.view(original_shape)

        if 'c_ortho' in state:
            c_ortho = state.pop('c_ortho')
            c_magnitude = state.pop('c_magnitude', 0.0)
            state['c_magnitude'] = c_magnitude

            if p.ndim >= 2:
                c_reshaped = c_ortho.view(p.shape) if c_ortho.numel() == p.numel() else c_ortho

                c_row_norm = c_reshaped / (c_reshaped.norm(dim=1, keepdim=True) + 1e-12)
                s_row_norm = s_ortho / (s_ortho.norm(dim=1, keepdim=True) + 1e-12)
                row_alignment = (c_row_norm * s_row_norm).sum(dim=1, keepdim=True)
                row_alignment = (row_alignment + 1) / 2

                c_col_norm = c_reshaped / (c_reshaped.norm(dim=0, keepdim=True) + 1e-12)
                s_col_norm = s_ortho / (s_ortho.norm(dim=0, keepdim=True) + 1e-12)
                col_alignment = (c_col_norm * s_col_norm).sum(dim=0, keepdim=True)
                col_alignment = (col_alignment + 1) / 2

                state['alignment_matrix'] = torch.sqrt(row_alignment * col_alignment)

        update = energy * s_ortho

        if weight_decay != 0:
            p.mul_(1 - lr * weight_decay)

        p.add_(update, alpha=-lr)

    def _ars2_update(self, group: dict, global_step: int):
        beta1, beta2 = group['betas']
        lr = group['lr']
        eps = group['eps']
        weight_decay = group['weight_decay']
        ns_steps = group['ns_steps']

        is_distributed = dist.is_available() and dist.is_initialized()
        params = group['params']

        if is_distributed:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)

            for base_i in range(len(params))[::world_size]:
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    self._apply_ars2_kernel(p, beta1, beta2, lr, eps, weight_decay, ns_steps)
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
        else:
            for p in params:
                self._apply_ars2_kernel(p, beta1, beta2, lr, eps, weight_decay, ns_steps)

    def _adamw_update(self, group: dict, global_step: int):
        beta1, beta2 = group['betas']
        lr = group['lr']
        eps = group['eps']
        weight_decay = group['weight_decay']

        for p in group['params']:
            if p.grad is None:
                continue

            state = self.state[p]
            if 'exp_avg' not in state:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] = 0

            state['step'] += 1
            _adamw_step_kernel(
                p, p.grad, state['exp_avg'], state['exp_avg_sq'],
                beta1, beta2, state['step'], lr, weight_decay, eps
            )

    def _calculate_global_phi(self) -> float:
        num = 0.0
        den_g = 0.0
        den_v = 0.0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'shear_force' not in state:
                    continue

                g = p.grad
                v = state['shear_force']

                num += (g * v).sum().item()
                den_g += (g * g).sum().item()
                den_v += (v * v).sum().item()

        if dist.is_available() and dist.is_initialized():
            device = self.param_groups[0]['params'][0].device
            t = torch.tensor([num, den_g, den_v], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            num, den_g, den_v = t.tolist()

        phi = num / ((den_g * den_v) ** 0.5 + 1e-12)
        return float(phi)

    @property
    def diagnostics(self) -> dict:
        total_steps = self.state.get('step', 1)
        sync_steps = self.state.get('sync_steps', 0)
        d: dict[str, Any] = {
            'phi_t': self.state.get('phi_t', 1.0),
            'phi_std': self.state.get('phi_var', 0.0) ** 0.5,
            'threshold': self.state.get('threshold', 0.0),
            'alpha_t': self.state.get('alpha_t', 0.1),
            'effective_k': total_steps / max(1, sync_steps),
        }

        alignment_means = []
        beta1s = []
        beta2s = []
        rhos = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state.get(p, {})
                if 'alignment_matrix' not in state:
                    continue
                a_matrix = state['alignment_matrix']
                a_mean = a_matrix.mean().item()
                alignment_means.append(a_mean)
                b1_min = state.get('beta1_min', 0.5)
                b1_max = state.get('beta1_max', 0.95)
                b2_min = state.get('beta2_min', 0.9)
                b2_max = state.get('beta2_max', 0.9995)
                beta1_matrix = b1_min + (b1_max - b1_min) * a_matrix
                beta2_matrix = b2_min + (b2_max - b2_min) * a_matrix
                beta1s.append(beta1_matrix.mean().item())
                beta2s.append(beta2_matrix.mean().item())
                if 'rho' in state:
                    rhos.append(state['rho'])

        if alignment_means:
            d['alignment'] = sum(alignment_means) / len(alignment_means)
            d['beta1_dynamic'] = sum(beta1s) / len(beta1s)
            d['beta2_dynamic'] = sum(beta2s) / len(beta2s)
        else:
            d['alignment'] = 0.0
            d['beta1_dynamic'] = self.defaults.get('beta1_max', 0.95)
            d['beta2_dynamic'] = self.defaults.get('beta2_max', 0.9995)

        if rhos:
            d['rho_mean'] = sum(rhos) / len(rhos)
            d['rho_std'] = (sum((r - d['rho_mean']) ** 2 for r in rhos) / len(rhos)) ** 0.5
            d['rho_min_obs'] = min(rhos)
            d['rho_max_obs'] = max(rhos)
        else:
            d['rho_mean'] = self.defaults.get('rho', 0.1)
            d['rho_std'] = 0.0
            d['rho_min_obs'] = self.defaults.get('rho', 0.1)
            d['rho_max_obs'] = self.defaults.get('rho', 0.1)

        return d


class SingleDeviceARS2CSAGA(ARS2CSAGA):
    """
    Single-device version of ARS2C-SAGA optimizer.

    Removes DDP-related synchronization logic for non-distributed training.
    """

    def _ars2_update(self, group: dict, global_step: int):
        beta1, beta2 = group['betas']
        lr = group['lr']
        eps = group['eps']
        weight_decay = group['weight_decay']
        ns_steps = group['ns_steps']

        for p in group['params']:
            self._apply_ars2_kernel(p, beta1, beta2, lr, eps, weight_decay, ns_steps)
