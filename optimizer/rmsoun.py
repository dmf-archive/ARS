import torch
from typing import Any, Dict, Optional, List


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute orthogonalization of G.
    Includes transpose optimization for non-square matrices.
    Adapted from Muon implementation.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype == torch.float32 else G

    # Transpose optimization: if rows > cols, work on the transposed matrix
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Perform NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    # Transpose back if needed
    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


class RMSoun(torch.optim.Optimizer):
    """
    RMSoun: Energy-Geometry Decoupled Optimizer with internal parameter grouping.

    Decouples optimization into:
    1. Geometry: Determined by Muon (Newton-Schulz orthogonalization of momentum)
    2. Energy: Determined by AdamW (Adaptive RMS amplitude)

    Internally, parameters are split into two groups:
    - 2D weights (e.g., Linear.weight, Conv2d.weight): Optimized with RMSoun.
    - 1D/Other parameters (e.g., biases, embeddings, norm layers): Optimized with AdamW.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate for RMSoun group (2D weights).
        betas: (beta1, beta2) for momentum and second moment.
        eps: Epsilon for numerical stability.
        weight_decay: Weight decay coefficient for RMSoun group.
        ns_steps: Newton-Schulz iteration steps (default: 5).
        aux_lr: Learning rate for AdamW group (1D/Other params). Defaults to `lr`.
        aux_betas: Betas for AdamW group. Defaults to `betas`.
        aux_eps: Eps for AdamW group. Defaults to `eps`.
        aux_weight_decay: Weight decay for AdamW group. Defaults to `weight_decay`.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
        aux_lr: Optional[float] = None,
        aux_betas: Optional[tuple] = None,
        aux_eps: Optional[float] = None,
        aux_weight_decay: Optional[float] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        aux_lr = aux_lr if aux_lr is not None else lr
        aux_betas = aux_betas if aux_betas is not None else betas
        aux_eps = aux_eps if aux_eps is not None else eps
        aux_weight_decay = aux_weight_decay if aux_weight_decay is not None else weight_decay

        # The input `params` can be a list of dicts (from get_optimizer) or a list of Parameters.
        # We need to handle both cases and flatten them into a single list of parameters.
        all_params = []
        for p_or_group in params:
            if isinstance(p_or_group, dict):
                # It's a parameter group dict, extract the 'params' list
                all_params.extend(p_or_group['params'])
            else:
                # It's a single parameter
                all_params.append(p_or_group)

        # Sort parameters into groups
        rmsoun_params = []
        adamw_params = []
        for p in all_params:
            if p.requires_grad:
                # Muon/RMSoun should only be applied to hidden weights (Linear, Conv).
                # Embeddings (vocab size > 10000) should be optimized with AdamW.
                if p.ndim >= 2 and max(p.shape) < 10000:
                    rmsoun_params.append(p)
                else:  # 1D, scalar, or large embedding parameters
                    adamw_params.append(p)

        # Create parameter groups
        param_groups = []
        if rmsoun_params:
            param_groups.append({
                'params': rmsoun_params,
                'is_rmsoun_group': True,
                'lr': lr,
                'betas': betas,
                'eps': eps,
                'weight_decay': weight_decay,
                'ns_steps': ns_steps,
            })
        if adamw_params:
            param_groups.append({
                'params': adamw_params,
                'is_rmsoun_group': False,
                'lr': aux_lr,
                'betas': aux_betas,
                'eps': aux_eps,
                'weight_decay': aux_weight_decay,
            })

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']

            is_rmsoun_group = group.get('is_rmsoun_group', False)
            ns_steps = group.get('ns_steps', 5)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)  # First moment
                    state['exp_avg_sq'] = torch.zeros_like(p)  # Second moment

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                # Update moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                m_hat = exp_avg / bias_correction1

                if is_rmsoun_group and p.ndim >= 2:
                    # --- RMSoun logic for 2D weights ---
                    # 1. Energy Calculation (AdamW Amplitude)
                    denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)
                    adam_update = m_hat.div(denom)
                    energy = adam_update.norm()
                    # CRITICAL: Free memory immediately before Newton-Schulz
                    del denom, adam_update

                    # 2. Geometry Calculation (Muon Direction)
                    original_shape = m_hat.shape
                    if p.ndim == 4:  # Conv: [out_c, in_c, k, k]
                        m_hat_flat = m_hat.view(m_hat.size(0), -1)
                    else:
                        m_hat_flat = m_hat

                    # Newton-Schulz orthogonalization
                    O = zeropower_via_newtonschulz5(m_hat_flat, steps=ns_steps)

                    # Reshape back
                    if len(original_shape) == 4:
                        O = O.view(original_shape)

                    # 3. Energy Injection
                    base_energy = O.norm() + 1e-10 # Avoid division by zero
                    scale = energy / base_energy
                    final_update = O.mul_(scale)
                else:
                    # --- AdamW logic for 1D/Other parameters ---
                    denom_1d = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)
                    final_update = m_hat.div(denom_1d)

                # Final update with weight decay
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                p.add_(final_update, alpha=-lr)

        return loss