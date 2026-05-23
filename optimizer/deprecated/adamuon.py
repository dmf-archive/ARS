import torch
from torch.optim.optimizer import Optimizer


@torch.jit.script
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype=torch.bfloat16) if G.dtype == torch.float32 else G

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X.div(X.norm(p=2.0, dim=[-2, -1], keepdim=True).add(1e-7))

    for _ in range(steps):
        A = X.matmul(X.mT)
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a, alpha=1.0)

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X.to(G.dtype)


class AdaMuon(Optimizer):
    def __init__(self, params, lr=0.02, betas=(0.95, 0.95), eps=1e-8, weight_decay=0.01, ns_steps=5):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # 1. Momentum update (Nesterov style as in official AdaMuon)
                exp_avg.mul_(beta1).add_(grad)
                m_nesterov = grad.add(exp_avg, alpha=beta1)

                if p.ndim >= 2:
                    # 2. Sign-stabilized Orthogonalization
                    original_shape = p.shape
                    m_flat = m_nesterov.view(m_nesterov.size(0), -1) if p.ndim == 4 else m_nesterov

                    # AdaMuon core: sign(m) -> NewtonSchulz
                    s = torch.sign(m_flat)
                    o = zeropower_via_newtonschulz5(s, steps=ns_steps)

                    if p.ndim == 4:
                        o = o.view(original_shape)

                    # 3. Element-wise Adaptivity on Orthogonal Direction
                    # Official AdaMuon applies v_buffer to the orthogonalized direction 'o'
                    exp_avg_sq.mul_(beta2).addcmul_(o, o, value=1 - beta2)

                    # Note: Official implementation uses v.sqrt() without bias correction for v
                    # but uses a specific scale.
                    update = o.div(exp_avg_sq.sqrt().add(eps))

                    # 4. RMS-aligned Rescaling
                    # scale = 0.2 * sqrt(min_dim * max_dim) / norm(update)
                    m, n = original_shape[0], original_shape[1:].numel()
                    scale = 0.2 * (m * n)**0.5 / (update.norm().add(eps))
                    update.mul_(scale)
                else:
                    # Standard AdamW for 1D params
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                    update = (exp_avg / bias_correction1) / denom

                # 5. Weight Decay and Final Update
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                p.add_(update, alpha=-lr)

        return loss
