import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from typing import cast


def disable_running_stats(model: nn.Module):
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            setattr(module, 'backup_momentum', module.momentum)
            module.momentum = 0

def enable_running_stats(model: nn.Module):
    for module in model.modules():
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = cast(float, getattr(module, 'backup_momentum'))
