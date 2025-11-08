import torch

class PICalculator:
    """A helper class to calculate Predictive Integrity (PI)."""
    def __init__(self, gamma: float, alpha: float, ema_beta: float | None = None, eps: float = 1e-8):
        self.gamma = gamma
        self.alpha = alpha
        self.ema_beta = ema_beta
        self.eps = eps
        self.exp_avg_pi = 0.0
        self.pi_step = 0

    def calculate_pi(self, entropy: torch.Tensor, grad_norm: float) -> tuple[float, float]:
        """Calculates the instantaneous and optionally smoothed PI."""
        instant_pi = torch.exp(-(self.alpha * entropy + self.gamma * grad_norm)).item()

        if self.ema_beta is not None:
            self.pi_step += 1
            self.exp_avg_pi = self.exp_avg_pi * self.ema_beta + instant_pi * (1 - self.ema_beta)
            bias_correction = 1 - self.ema_beta ** self.pi_step
            smoothed_pi = self.exp_avg_pi / bias_correction
            return instant_pi, smoothed_pi

        return instant_pi, instant_pi

def compute_grad_norm(model: torch.nn.Module) -> float:
    """Computes the total L2 norm of the gradients of a model."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5