import math

import torch
import torch.optim as optim

from .kfac_utils import update_running_stat
from .muon import zeropower_via_newtonschulz5


class DiagKFACMuonOptimizer(optim.Optimizer):
    """
    DiagKFAC-Muon: Fisher-aware structural regularization for natural gradients.
    
    This optimizer correctly applies Muon-style orthogonalization to DiagKFAC natural
    gradients by adapting the orthogonalization strength based on Fisher information
    geometry, avoiding the space mismatch problem of naive operator composition.
    """

    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=10,
                 TInv=100,
                 muon_momentum=0.95,
                 muon_min_steps=3,
                 muon_max_steps=7,
                 batch_averaged=True):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, damping=damping, weight_decay=weight_decay)
        super().__init__(model.parameters(), defaults)

        self.known_modules = {'Linear', 'Conv2d'}
        self.modules = []
        self.model = model
        self._prepare_model()

        self.steps = 0
        self.m_aa, self.m_gg = {}, {}
        self.stat_decay = stat_decay
        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TInv = TInv
        self.muon_momentum = muon_momentum
        self.muon_min_steps = muon_min_steps
        self.muon_max_steps = muon_max_steps
        self.batch_averaged = batch_averaged

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            a = input[0].data
            a = a.reshape(-1, a.size(-1))
            if module.bias is not None:
                a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)

            aa_diag = (a * a).sum(dim=0)

            if self.steps == 0:
                self.m_aa[module] = torch.ones_like(aa_diag)
            update_running_stat(aa_diag, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        if self.steps % self.TCov == 0:
            g = grad_output[0].data
            g = g.reshape(-1, g.size(-1))

            gg_diag = (g * g).sum(dim=0)

            if self.steps == 0:
                self.m_gg[module] = torch.ones_like(gg_diag)
            update_running_stat(gg_diag, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad_output)

    def _get_natural_grad(self, m, p_grad_mat, damping):
        """Compute natural gradient with diagonal Fisher approximation"""
        A_inv_diag = 1.0 / (self.m_aa[m] + damping)
        G_inv_diag = 1.0 / (self.m_gg[m] + damping)

        # Natural gradient: F_inv @ grad
        v = p_grad_mat * (G_inv_diag.unsqueeze(1) @ A_inv_diag.unsqueeze(0))

        if m.bias is not None:
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]

        return v

    def _adaptive_muon_update(self, nat_grad, fisher_info, momentum_buffer):
        """
        Fisher-aware Muon update that adapts orthogonalization based on statistical geometry.
        
        Args:
            nat_grad: Natural gradient (F_inv @ g)
            fisher_info: Fisher information strength indicator
            momentum_buffer: Momentum buffer for this parameter
        """
        # 1. Compute Fisher strength for adaptive control
        fisher_strength = torch.sqrt(fisher_info.mean() + 1e-8)

        # 2. Apply momentum in natural gradient space
        momentum_buffer.lerp_(nat_grad, 1 - self.muon_momentum)
        update = nat_grad.lerp_(momentum_buffer, self.muon_momentum)

        # 3. Skip orthogonalization for 1D parameters
        if update.ndim == 1:
            return update

        # 4. Adaptive orthogonalization based on Fisher strength
        if update.ndim >= 2:
            # Adaptive steps: more Fisher info -> more orthogonalization
            adaptive_steps = int(
                self.muon_min_steps +
                (self.muon_max_steps - self.muon_min_steps) *
                min(1.0, fisher_strength)
            )

            # Reshape for convolutional layers
            original_shape = update.shape
            if update.ndim == 4:
                update = update.view(update.size(0), -1)

            # Apply Newton-Schulz orthogonalization
            update = zeropower_via_newtonschulz5(update, steps=adaptive_steps)

            # Fisher-aware spectral scaling
            spectral_scale = max(1, nat_grad.size(-2) / nat_grad.size(-1))**0.5
            fisher_scale = (fisher_strength / (fisher_strength + 1.0))**0.5
            update *= spectral_scale * fisher_scale

            # Restore original shape
            update = update.reshape(original_shape)

        return update

    def _kl_clip_and_update_grad(self, updates, lr):
        """KL clipping for natural gradients"""
        vg_sum = 0
        for m in self.modules:
            v = updates[m]
            vg_sum += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
            if m.bias is not None:
                vg_sum += (v[1] * m.bias.grad.data * lr ** 2).sum().item()

        nu = min(1.0, math.sqrt(self.kl_clip / (vg_sum + 1e-8)))

        for m in self.modules:
            v = updates[m]
            m.weight.grad.data.copy_(v[0])
            m.weight.grad.data.mul_(nu)
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
                m.bias.grad.data.mul_(nu)

    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """Convert gradient to matrix form for natural gradient computation"""
        if classname == 'Conv2d':
            p_grad_mat = m.weight.grad.data.reshape(m.weight.grad.data.size(0), -1)
        else:
            p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def step(self, closure=None):
        """Main optimization step with Fisher-aware structural regularization"""
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        weight_decay = group['weight_decay']
        momentum = group['momentum']

        # Step 1: Compute natural gradients
        updates = {}
        fisher_info = {}

        for m in self.modules:
            classname = m.__class__.__name__
            p_grad_mat = self._get_matrix_form_grad(m, classname)

            # Get natural gradient
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v

            # Store Fisher information for adaptive control
            fisher_info[m] = {
                'A_diag': self.m_aa[m],
                'G_diag': self.m_gg[m]
            }

        # Step 2: Apply KL clipping
        self._kl_clip_and_update_grad(updates, lr)

        # Step 3: Apply Fisher-aware Muon structural regularization
        for m in self.modules:
            if m not in updates:
                continue

            # Apply structural regularization to each parameter
            for i, p in enumerate(m.parameters()):
                if p.grad is None or p.ndim < 2:
                    continue

                state = self.state[p]

                # Initialize momentum buffer if needed
                if 'muon_momentum_buffer' not in state:
                    state['muon_momentum_buffer'] = torch.zeros_like(p.grad)

                # Compute composite Fisher strength for this parameter
                if i == 0:  # weight
                    param_fisher = fisher_info[m]['A_diag'][:p.shape[1]].mean()
                else:  # bias
                    param_fisher = fisher_info[m]['A_diag'][-1:].mean()

                # Apply adaptive Muon update to the natural gradient
                nat_grad = p.grad.data.clone()
                struct_grad = self._adaptive_muon_update(
                    nat_grad,
                    torch.tensor([param_fisher], device=nat_grad.device),
                    state['muon_momentum_buffer']
                )

                # Update gradient with structurally regularized natural gradient
                p.grad.data.copy_(struct_grad)

        # Step 4: Standard SGD update with momentum
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                # Weight decay
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                # Momentum
                if momentum != 0:
                    param_state = self.state[p]
                    if 'sgd_momentum_buffer' not in param_state:
                        buf = param_state['sgd_momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['sgd_momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1)
                    d_p = buf

                # Parameter update
                p.data.add_(d_p, alpha=-lr)

        self.steps += 1
