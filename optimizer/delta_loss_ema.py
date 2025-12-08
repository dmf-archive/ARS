import torch
from torch.optim.optimizer import Optimizer


class DeltaLossEMA(Optimizer):
    def __init__(self, params, **kwargs):
        defaults = {
            'lr': 1e-3,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.01,
        }
        super().__init__(params, defaults)
        self.prev_loss = None
        self.w_min = torch.tensor(1.0)
        self.w_max = torch.tensor(1.0)

    @torch.no_grad()
    def step(self, loss: torch.Tensor):
        w_t_raw = 0.0
        if self.prev_loss is not None:
            w_t_raw = max(0.0, self.prev_loss.item() - loss.item())
        self.prev_loss = loss.clone().detach()

        if w_t_raw > 0:
            self.w_min = torch.minimum(self.w_min, torch.tensor(w_t_raw))
            self.w_max = torch.maximum(self.w_max, torch.tensor(w_t_raw))

        w_t_norm = w_t_raw
        if self.w_max.item() > self.w_min.item():
            w_t_norm = (w_t_raw - self.w_min.item()) / (self.w_max.item() - self.w_min.item() + 1e-8)

        for group in self.param_groups:
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
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # Dynamic beta based on normalized delta loss
                beta_dynamic = 1.0 - w_t_norm
                # Clamp to avoid beta becoming 0 or 1, which can cause instability
                beta_dynamic = torch.clamp(torch.tensor(beta_dynamic), 0.001, 0.9999).item()


                lr = group['lr']
                weight_decay = group['weight_decay']
                eps = group['eps']

                exp_avg.mul_(beta_dynamic).add_(grad, alpha=1 - beta_dynamic)
                exp_avg_sq.mul_(beta_dynamic).addcmul_(grad, grad, value=1 - beta_dynamic)

                bias_correction1 = 1 - beta_dynamic ** state['step']
                bias_correction2 = 1 - beta_dynamic ** state['step']
                
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                step_size = lr / bias_correction1

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss