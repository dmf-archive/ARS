from collections.abc import Callable

import torch

from optimizer.ars2_neo import ARS2Neo, zeropower_via_newtonschulz5


class ARS2C(ARS2Neo):
    """
    ARS2C: Christoffel-Aware Dynamic Beta Optimization.

    ARS2C extends ARS2-Neo by replacing fixed momentum decay rates beta1, beta2 with
    Christoffel-symbol-driven dynamic betas derived from the Fisher information
    manifold. The HVP implicitly sampled during SAM sync steps is reused at zero
    additional forward/backward cost to compute a structured Christoffel matrix
    c_ortho via Newton-Schulz orthogonalization. The geometric alignment between
    c_ortho and the update direction s_ortho then drives beta: high alignment
    (update direction aligns with rapid curvature change) -> high beta (strong filtering);
    low alignment -> low beta (fast adaptation).

    Args:
    - params: parameter groups; 2D+ params use dynamic-beta ARS2 track, 1D params use fixed-beta AdamW.
    - lr: learning rate, default 1e-3.
    - betas: fallback Adam beta tuple (beta1, beta2) for 1D params and initial steps, default (0.9, 0.999).
    - eps: Adam epsilon, default 1e-8.
    - weight_decay: weight-decay coefficient, default 0.01.
    - ns_steps: Newton-Schulz iteration steps, default 5.
    - rho: SAM perturbation radius, default 0.1.
    - k: SAM mode; k=0 disables SAM, k=1 synchronous, k>1 delayed, default 1.
    - alpha: base shear-force injection strength, default 0.1.
    - adaptive_sync: enable AGA sync mode, default False.
    - adaptive_beta: EMA coefficient for AGA, default 0.99.
    - adaptive_lambda: AGA sensitivity, default 2.0.
    - adaptive_gamma: AGA exponent, default 2.0.
    - beta1_min, beta1_max: dynamic range for beta1, default (0.5, 0.9995).
    - beta2_min, beta2_max: dynamic range for beta2, default (0.5, 0.9995).
    """

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01, ns_steps: int = 5,
                 rho: float = 0.1, k: int = 1, alpha: float = 0.1,
                 adaptive_sync: bool = False,
                 adaptive_beta: float = 0.99, adaptive_lambda: float = 2.0, adaptive_gamma: float = 2.0,
                 beta1_min: float = 0.5, beta1_max: float = 0.95,
                 beta2_min: float = 0.9, beta2_max: float = 0.9995):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                         ns_steps=ns_steps, rho=rho, k=k, alpha=alpha,
                         adaptive_sync=adaptive_sync,
                         adaptive_beta=adaptive_beta, adaptive_lambda=adaptive_lambda,
                         adaptive_gamma=adaptive_gamma)
        self.defaults['beta1_min'] = beta1_min
        self.defaults['beta1_max'] = beta1_max
        self.defaults['beta2_min'] = beta2_min
        self.defaults['beta2_max'] = beta2_max
        for group in self.param_groups:
            group.setdefault('beta1_min', beta1_min)
            group.setdefault('beta1_max', beta1_max)
            group.setdefault('beta2_min', beta2_min)
            group.setdefault('beta2_max', beta2_max)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        if closure is None:
            raise ValueError("ARS2C requires a closure.")

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
                rho = group.get('rho', 0.1)
                if rho <= 0:
                    continue
                eps = group.get('eps', 1e-8)
                beta2 = group.get('betas', (0.9, 0.999))[1]

                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if 'exp_avg_sq' not in state:
                        state['exp_avg_sq'] = torch.zeros_like(p)

                    v_hat = state['exp_avg_sq'] / (1 - beta2 ** max(1, global_step - 1) + 1e-12)
                    g_nat = p.grad / (v_hat.sqrt() + eps)
                    g_nat = g_nat * p.abs()

                    norm = g_nat.norm() + 1e-12
                    perturb = g_nat * (rho / norm)

                    state['last_perturbation'] = perturb
                    state['base_grad'] = p.grad.clone()
                    p.add_(perturb)

            self.zero_grad()
            with torch.enable_grad():
                loss_adv = closure()
                loss_adv.backward()

            for group in self.param_groups:
                rho = group.get('rho', 0.1)
                if rho <= 0:
                    continue
                _eps = group.get('eps', 1e-8)
                _beta2 = group.get('betas', (0.9, 0.999))[1]
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
                        _C = _delta_g / (rho * _g_hat.abs() + _eps)
                        _c_flat = _C.view(_C.size(0), -1)
                        state['c_magnitude'] = float(_c_flat.norm())
                        state['c_ortho'] = zeropower_via_newtonschulz5(_c_flat)

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

        alignment = state.get('alignment', 0.0)
        b1_min = state.get('beta1_min', 0.5)
        b1_max = state.get('beta1_max', 0.95)
        b2_min = state.get('beta2_min', 0.9)
        b2_max = state.get('beta2_max', 0.9995)
        beta1_d = b1_min + (b1_max - b1_min) * alignment
        beta2_d = b2_min + (b2_max - b2_min) * alignment

        exp_avg.mul_(beta1_d).add_(p.grad, alpha=1 - beta1_d)
        exp_avg_sq.mul_(beta2_d).addcmul_(p.grad, p.grad, value=1 - beta2_d)

        bias_correction1 = 1 - beta1_d ** step
        bias_correction2 = 1 - beta2_d ** step

        m_hat = exp_avg / bias_correction1
        v_hat = exp_avg_sq / bias_correction2

        m_scaled = m_hat / (v_hat.sqrt() + eps)
        energy = m_scaled.norm()

        original_shape = m_scaled.shape
        m_scaled_flat = m_scaled.view(m_scaled.size(0), -1) if p.ndim == 4 else m_scaled

        s_ortho = zeropower_via_newtonschulz5(m_scaled_flat, steps=ns_steps)
        if p.ndim == 4:
            s_ortho = s_ortho.view(original_shape)

        if 'c_ortho' in state:
            c_ortho = state.pop('c_ortho')
            c_magnitude = state.pop('c_magnitude', 0.0)
            s_flat = s_ortho.view(s_ortho.size(0), -1) if s_ortho.ndim >= 2 else s_ortho
            s_unit = s_flat / (s_flat.norm() + 1e-12)
            alignment_raw = float((c_ortho * s_unit).sum().abs())
            mag_gate = float(torch.sigmoid(torch.tensor(c_magnitude, device=p.device)))
            state['alignment_raw'] = alignment_raw
            state['c_magnitude'] = c_magnitude
            state['alignment'] = alignment_raw * mag_gate

        update = energy * s_ortho

        if weight_decay != 0:
            p.mul_(1 - lr * weight_decay)

        p.add_(update, alpha=-lr)

    @property
    def diagnostics(self) -> dict:
        d = super().diagnostics
        alignments = []
        beta1s = []
        beta2s = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state.get(p, {})
                if 'alignment' not in state:
                    continue
                a = state['alignment']
                alignments.append(a)
                b1_min = state.get('beta1_min', 0.5)
                b1_max = state.get('beta1_max', 0.95)
                b2_min = state.get('beta2_min', 0.9)
                b2_max = state.get('beta2_max', 0.9995)
                beta1s.append(b1_min + (b1_max - b1_min) * a)
                beta2s.append(b2_min + (b2_max - b2_min) * a)

        if alignments:
            d['alignment'] = sum(alignments) / len(alignments)
            d['beta1_dynamic'] = sum(beta1s) / len(beta1s)
            d['beta2_dynamic'] = sum(beta2s) / len(beta2s)
        else:
            d['alignment'] = 0.0
            d['beta1_dynamic'] = self.defaults.get('beta1_max', 0.95)
            d['beta2_dynamic'] = self.defaults.get('beta2_max', 0.9995)

        return d


class SingleDeviceARS2C(ARS2C):
    """
    Single-device version of ARS2C optimizer.

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
