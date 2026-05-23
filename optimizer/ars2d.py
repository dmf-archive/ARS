from collections.abc import Callable

import torch

from optimizer.ars2_neo import ARS2Neo, zeropower_via_newtonschulz5


class ARS2D(ARS2Neo):
    """
    ARS2D: Bidirectional Orthogonalization for Energy-Geometry Decoupling.

    ARS2D extends ARS2-Neo by applying Newton-Schulz orthogonalization twice:
    first on the raw gradient matrix (row orthogonalization, UU^T ≈ I_m),
    then on its transpose (column orthogonalization, W^T W ≈ I_n).
    This yields a double-sided update that more closely approximates
    K-FAC-style natural gradient descent without explicitly computing
    Kronecker factors.

    Update chain:
        G_nat = m_hat / sqrt(v_hat + eps)       (Adam pre-whitening)
        E = ||G_nat||_F                          (energy extraction)
        U = NS(G_nat)                            (row orthogonalization)
        W = NS(U^T)^T                            (column orthogonalization)
        Δθ = -η · E · W                          (energy-injected update)

    All SAM/AGA logic is inherited from ARS2-Neo unchanged.
    No Christoffel dynamic beta is introduced (cf. ARS2C).

    Args:
        Same as ARS2Neo.
    """

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        if closure is None:
            raise ValueError("ARS2D requires a closure.")

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

        state['step'] += 1
        step = state['step']
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

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

        # First NS: row orthogonalization  =>  UU^T ≈ I_m
        U = zeropower_via_newtonschulz5(m_scaled_flat, steps=ns_steps)

        # Second NS: column orthogonalization via transpose  =>  W^T W ≈ I_n
        # W = NS(U^T)^T
        W = zeropower_via_newtonschulz5(U.mT, steps=ns_steps).mT

        if p.ndim == 4:
            W = W.view(original_shape)

        update = energy * W

        if weight_decay != 0:
            p.mul_(1 - lr * weight_decay)

        p.add_(update, alpha=-lr)

    @property
    def diagnostics(self) -> dict:
        d = super().diagnostics
        d['bidirectional_ns'] = True
        return d


class SingleDeviceARS2D(ARS2D):
    """
    Single-device version of ARS2D optimizer.

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
