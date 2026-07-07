from collections import deque
from collections.abc import Callable
from typing import Any

import torch

from optimizer.ars2_neo import ARS2Neo

_EPS_MAP = {
    torch.float32: 1.19e-7,
    torch.float16: 6.10e-4,
    torch.bfloat16: 7.81e-3,
}
_BETA1_GAM = 0.2
_BETA2_GAM = 0.1


class EVPThreshold:
    def __init__(self, window_size: int, n_paths: int = 3):
        self.window: deque[float] = deque(maxlen=window_size)
        self._n_paths = max(n_paths, 2)
        self._min_samples = max(10, window_size // 20)

    def update(self, value: float) -> None:
        self.window.append(value)

    @property
    def threshold(self) -> float:
        if len(self.window) < self._min_samples:
            return float('inf')
        sorted_w = sorted(self.window)
        idx = int(len(sorted_w) * (1.0 - 1.0 / self._n_paths))
        return sorted_w[min(idx, len(sorted_w) - 1)]

    @property
    def is_ready(self) -> bool:
        return len(self.window) >= self._min_samples


class ARS2NeoAGAM(ARS2Neo):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
        rho: float = 0.1,
        k: int = 1,
        alpha: float = 0.1,
        adaptive_sync: bool = True,
        adaptive_beta: float = 0.99,
        adaptive_lambda: float = 2.0,
        adaptive_gamma: float = 2.0,
        agam_window: int = 1000,
        agam_n_paths: int = 3,
        agam_beta1: float = _BETA1_GAM,
        agam_beta2: float = _BETA2_GAM,
        agam_rho_small: float | None = None,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            rho=rho,
            k=k,
            alpha=alpha,
            adaptive_sync=adaptive_sync,
            adaptive_beta=adaptive_beta,
            adaptive_lambda=adaptive_lambda,
            adaptive_gamma=adaptive_gamma,
        )
        _nparams = sum(p.numel() for pg in self.param_groups for p in pg['params'])
        self._mps_tau_min = _EPS_MAP.get(torch.float32, 1.19e-7) * (_nparams ** 0.5)
        self.defaults['agam_window'] = agam_window
        self.defaults['agam_n_paths'] = agam_n_paths
        self.defaults['agam_beta1'] = agam_beta1
        self.defaults['agam_beta2'] = agam_beta2
        self.defaults['agam_rho_small'] = agam_rho_small if agam_rho_small is not None else min(rho / 10.0, 0.1)
        _as: dict[str, Any] = self.state.setdefault('agam', {})
        _as.setdefault('evp_alpha1', EVPThreshold(agam_window, agam_n_paths))
        _as.setdefault('evp_alpha2', EVPThreshold(agam_window, agam_n_paths))
        _as.setdefault('theta_sam', float('inf'))
        _as.setdefault('theta_gam', float('inf'))
        _as.setdefault('last_alpha2', 0.0)
        _as.setdefault('path_counts', {'sam': 0, 'middle': 0, 'gam': 0})

    @property
    def diagnostics(self) -> dict[str, Any]:
        base = super().diagnostics
        _as = self.state.get('agam', {})
        base['agam_theta_sam'] = _as.get('theta_sam', float('inf'))
        base['agam_theta_gam'] = _as.get('theta_gam', float('inf'))
        base['agam_last_alpha2'] = _as.get('last_alpha2', 0.0)
        base['agam_path_counts'] = dict(_as.get('path_counts', {}))
        return base

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        if closure is None:
            raise ValueError('ARS2NeoAGAM requires a closure.')
        self.state['step'] += 1
        global_step = self.state['step']
        group0 = self.param_groups[0]
        k = group0.get('k', 1)
        adaptive_sync = group0.get('adaptive_sync', True)
        is_sam_enabled = k > 0 or adaptive_sync

        with torch.enable_grad():
            loss = closure()
            loss.backward()

        is_sync_step = False
        if is_sam_enabled and adaptive_sync:
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
            threshold = -group0.get('adaptive_lambda', 2.0) * std
            self.state['threshold'] = threshold
            steps_since_sync = global_step - self.state.get('last_sync_step', 0)
            is_drift = phi_t < threshold
            is_sync_step = is_drift or (global_step == 1) or (k > 1 and steps_since_sync >= k)
            gamma = group0.get('adaptive_gamma', 2.0)
            alpha_max = group0.get('alpha', 0.1)
            self.state['alpha_t'] = alpha_max * ((1.0 + max(0.0, phi_t)) ** gamma)
        elif is_sam_enabled:
            is_sync_step = global_step % k == 1 if k > 1 else True
            self.state['alpha_t'] = group0.get('alpha', 0.1)

        if is_sam_enabled and is_sync_step:
            self.state['last_sync_step'] = global_step
            self.state['sync_steps'] += 1
            _as: dict[str, Any] = self.state.setdefault('agam', {})
            evp_a1: EVPThreshold = _as['evp_alpha1']
            evp_a2: EVPThreshold = _as['evp_alpha2']
            if evp_a1.is_ready and evp_a2.is_ready:
                _as['theta_sam'] = max(evp_a1.threshold, self._mps_tau_min)
                _as['theta_gam'] = max(evp_a2.threshold, self._mps_tau_min)
            theta_sam = _as.get('theta_sam', float('inf'))
            theta_gam = _as.get('theta_gam', float('inf'))

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

            alpha1 = phi_t if adaptive_sync else 1.0
            evp_a1.update(alpha1)

            if alpha1 > theta_sam:
                _as['path_counts']['sam'] += 1
                for group in self.param_groups:
                    rho = group.get('rho', 0.1)
                    if rho <= 0:
                        continue
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        state = self.state[p]
                        p.sub_(state['last_perturbation'])
                        g_base = state['base_grad']
                        g_adv = p.grad
                        dot = (g_adv * g_base).sum()
                        base_norm_sq = (g_base * g_base).sum() + 1e-12
                        state['shear_force'] = g_adv - (dot / base_norm_sq) * g_base
            else:
                agam_rho_small = group0.get('agam_rho_small', 0.01)
                agam_beta1 = group0.get('agam_beta1', _BETA1_GAM)
                agam_beta2 = group0.get('agam_beta2', _BETA2_GAM)
                _g_adv_per_param: dict[torch.Tensor, torch.Tensor] = {}
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        _g_adv_per_param[p] = p.grad.clone()
                        state = self.state[p]
                        p.sub_(state['last_perturbation'])
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        state = self.state[p]
                        delta_g = _g_adv_per_param[p] - state['base_grad']
                        d_norm = delta_g.norm() + 1e-12
                        perturb2 = delta_g * (agam_rho_small / d_norm)
                        state['second_perturbation'] = perturb2
                        p.add_(perturb2)
                self.zero_grad()
                with torch.enable_grad():
                    loss_adv2 = closure()
                    loss_adv2.backward()

                alpha2: float = 0.0
                _num = 0.0
                _den_a = 0.0
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        state = self.state[p]
                        g_base = state['base_grad']
                        g_adv = _g_adv_per_param[p]
                        g_adv2 = p.grad.clone()
                        delta_g = g_adv - g_base
                        delta_g2 = g_adv2 - g_adv
                        n1 = delta_g.norm() + 1e-12
                        n2 = delta_g2.norm() + 1e-12
                        _num += float((delta_g * delta_g2).sum().abs())
                        _den_a += float(n1 * n2)
                if _den_a > 0:
                    alpha2 = min(_num / _den_a, 1.0)
                evp_a2.update(alpha2)
                _as['last_alpha2'] = alpha2

                if alpha2 > theta_gam:
                    _as['path_counts']['middle'] += 1
                    for group in self.param_groups:
                        for p in group['params']:
                            if p.grad is None:
                                continue
                            state = self.state[p]
                            p.sub_(state['second_perturbation'])
                            p.grad.copy_(_g_adv_per_param[p])
                            g_base = state['base_grad']
                            g_adv = _g_adv_per_param[p]
                            dot = (g_adv * g_base).sum()
                            base_norm_sq = (g_base * g_base).sum() + 1e-12
                            state['shear_force'] = g_adv - (dot / base_norm_sq) * g_base
                else:
                    _as['path_counts']['gam'] += 1
                    for group in self.param_groups:
                        for p in group['params']:
                            if p.grad is None:
                                continue
                            state = self.state[p]
                            g_base = state['base_grad']
                            g_adv = _g_adv_per_param[p]
                            g_adv2 = p.grad.clone()
                            delta_g = g_adv - g_base
                            delta_g2 = g_adv2 - g_adv
                            dg_dot = (g_base * delta_g).sum()
                            dg_norm_sq = (delta_g * delta_g).sum() + 1e-12
                            parallel = delta_g * (dg_dot / dg_norm_sq)
                            vertical = g_base - parallel
                            p.grad = vertical + agam_beta1 * delta_g + agam_beta2 * delta_g2
                            p.sub_(state['second_perturbation'])
                            dot = (g_adv * g_base).sum()
                            base_norm_sq = (g_base * g_base).sum() + 1e-12
                            state['shear_force'] = g_adv - (dot / base_norm_sq) * g_base
        elif is_sam_enabled and not is_sync_step:
            current_alpha = self.state.get('alpha_t', 0.1)
            _as = self.state.get('agam', {})
            last_alpha2 = _as.get('last_alpha2', 0.0)
            if last_alpha2 > 0:
                current_alpha = current_alpha * (1.0 + (1.0 - last_alpha2))
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


class SingleDeviceARS2NeoAGAM(ARS2NeoAGAM):
    def _ars2_update(self, group: dict, global_step: int):
        beta1, beta2 = group['betas']
        lr = group['lr']
        eps = group['eps']
        weight_decay = group['weight_decay']
        ns_steps = group['ns_steps']
        for p in group['params']:
            self._apply_ars2_kernel(p, beta1, beta2, lr, eps, weight_decay, ns_steps)
