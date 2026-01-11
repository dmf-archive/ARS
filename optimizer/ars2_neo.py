import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable


@torch.jit.script
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    
    This implementation uses a quintic iteration whose coefficients are selected to maximize the slope at zero.
    For the purpose of minimizing steps, it turns out to be empirically effective to keep increasing the slope
    at zero even beyond the point where the iteration no longer converges all the way to one everywhere on the interval.
    
    This iteration does not produce UV^T but rather something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5).
    Experiments show this approximation does not hurt model performance at all relative to the exact SVD decomposition USV^T.
    
    The function supports batched processing and can efficiently run on GPU in bfloat16 precision with numerical stability.
    """
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
def adamw_step_kernel(
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


class ARS2Neo(Optimizer):
    """
    ARS2-Neo: Adaptive Riemannian Stiefel Optimization with Neo-SAM。
    
    ARS2-Neo is a hybrid optimizer that merges the geometric optimization power of AdaRMSuon with the flatness constraint of GSAM.  
    
    It attains rapid convergence on Riemannian manifolds via energy–geometry decoupling while leveraging manifold-aware SAM to escape sharp minima.

    The optimizer employs a dual-track design: 2D+ parameters are automatically routed to the ARS2 track (orthogonalized optimization) and 1D parameters to the AdamW track.  
    Distributed training (DDP) is supported, and three SAM modes are provided: disabled (k=0), synchronous SAM (k=1), and delayed SAM (k>1).

    Args:
    - params: list of parameter groups; each must contain a 'params' key and an optional 'is_rmsuon_group' flag.  
    - lr: learning rate, default 1e-3.  
    - betas: Adam beta tuple (beta1, beta2), default (0.9, 0.999).  
    - eps: Adam epsilon for numerical stability, default 1e-8.  
    - weight_decay: weight-decay coefficient, default 0.01.  
    - ns_steps: Newton–Schulz iteration steps, default 5.  
    - rho: SAM perturbation radius controlling flatness strength, default 0.1.  
    - k: SAM mode parameter; k=0 disables SAM, k=1 gives synchronous mode, k>1 gives delayed mode, default 1.  
    - alpha: shear-force injection strength in delayed mode, default 0.1.  
    - adaptive: use adaptive perturbation (scaled by param magnitude), default True.
    """
    
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01, ns_steps: int = 5,
                 rho: float = 0.1, k: int = 1, alpha: float = 0.1, 
                 adaptive: bool = True):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            ns_steps=ns_steps, rho=rho, k=k, alpha=alpha, adaptive=adaptive
        )
        super().__init__(params, defaults)
        self.state['step'] = 0

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        if closure is None:
            raise ValueError("ARS2-Neo requires a closure.")
            
        self.state['step'] += 1
        global_step = self.state['step']
        
        # 1. Determine SAM mode for this step (using first group as reference)
        k = self.param_groups[0].get('k', 1)
        is_sam_enabled = k > 0
        is_sync_step = is_sam_enabled and (global_step % k == 1 if k > 1 else True)
        
        # 2. First Backward (Base Gradient)
        with torch.enable_grad():
            loss = closure()
            loss.backward()
            
        # 3. SAM Logic (Global across groups)
        if is_sam_enabled and is_sync_step:
            # Perturb
            for group in self.param_groups:
                rho = group.get('rho', 0.1)
                if rho <= 0: continue
                adaptive = group.get('adaptive', True)
                eps = group.get('eps', 1e-8)
                beta2 = group.get('betas', (0.9, 0.999))[1]
                
                for p in group['params']:
                    if p.grad is None: continue
                    state = self.state[p]
                    if 'exp_avg_sq' not in state:
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    
                    v_hat = state['exp_avg_sq'] / (1 - beta2 ** max(1, global_step - 1) + 1e-12)
                    g_nat = p.grad / (v_hat.sqrt() + eps)
                    if adaptive: g_nat = g_nat * p.abs()
                    norm = g_nat.norm() + 1e-12
                    perturb = g_nat * (rho / norm)
                    
                    state['last_perturbation'] = perturb
                    state['base_grad'] = p.grad.clone()
                    p.add_(perturb)
            
            # Second Backward
            self.zero_grad()
            with torch.enable_grad():
                loss_adv = closure()
                loss_adv.backward()
                
            # Restore and Shear Force
            for group in self.param_groups:
                rho = group.get('rho', 0.1)
                if rho <= 0: continue
                for p in group['params']:
                    if p.grad is None: continue
                    state = self.state[p]
                    p.sub_(state['last_perturbation'])
                    
                    if k > 1:
                        g_base = state['base_grad']
                        g_adv = p.grad
                        dot = (g_adv * g_base).sum()
                        base_norm_sq = (g_base * g_base).sum() + 1e-12
                        state['shear_force'] = g_adv - (dot / base_norm_sq) * g_base
        elif is_sam_enabled and not is_sync_step:
            # Lazy SAM: Inject shear force
            for group in self.param_groups:
                alpha = group.get('alpha', 0.1)
                for p in group['params']:
                    if p.grad is None: continue
                    state = self.state[p]
                    if 'shear_force' in state:
                        v = state['shear_force']
                        g_norm = p.grad.norm()
                        v_norm = v.norm() + 1e-12
                        p.grad.add_(v, alpha=alpha * (g_norm / v_norm))
                        
        # 4. Final Updates
        for group in self.param_groups:
            if group.get('is_rmsuon_group', False):
                self._ars2_update(group, global_step)
            else:
                self._adamw_update(group, global_step)
                
        return loss

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

    def _apply_ars2_kernel(self, p, beta1, beta2, lr, eps, weight_decay, ns_steps):
        if p.grad is None: return
        
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
        
        s_ortho = zeropower_via_newtonschulz5(m_scaled_flat, steps=ns_steps)
        if p.ndim == 4: s_ortho = s_ortho.view(original_shape)
        
        update = energy * s_ortho
        
        if weight_decay != 0:
            p.mul_(1 - lr * weight_decay)
        
        p.add_(update, alpha=-lr)

    def _adamw_update(self, group: dict, global_step: int):
        beta1, beta2 = group['betas']
        lr = group['lr']
        eps = group['eps']
        weight_decay = group['weight_decay']
        
        for p in group['params']:
            if p.grad is None: continue
            
            state = self.state[p]
            if 'exp_avg' not in state:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] = 0
            
            state['step'] += 1
            adamw_step_kernel(
                p, p.grad, state['exp_avg'], state['exp_avg_sq'],
                beta1, beta2, state['step'], lr, weight_decay, eps
            )


class SingleDeviceARS2Neo(ARS2Neo):
    """
    单设备版本的 ARS2-Neo 优化器。
    
    该变体专为非分布式训练环境设计，移除了 DDP 相关的同步逻辑。
    """
    def _ars2_update(self, group: dict, global_step: int):
        beta1, beta2 = group['betas']
        lr = group['lr']
        eps = group['eps']
        weight_decay = group['weight_decay']
        ns_steps = group['ns_steps']
        
        for p in group['params']:
            self._apply_ars2_kernel(p, beta1, beta2, lr, eps, weight_decay, ns_steps)
