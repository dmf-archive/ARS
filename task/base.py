from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

class BaseTask(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        pass

    @abstractmethod
    def get_model(self) -> nn.Module:
        pass

    @abstractmethod
    def get_criterion(self) -> nn.Module:
        pass

    @abstractmethod
    def train_step(self, model: nn.Module, batch: Any, criterion: nn.Module,
                   optimizer: torch.optim.Optimizer, device: torch.device,
                   needs_second_order: bool, accepts_pi_signal: bool,
                   eff_gamma: float | None) -> Tuple[torch.Tensor, float, Dict[str, float]]:
        """
        Performs a single training step.
        Returns the model's output logits, the calculated loss, and a dictionary of step-specific metrics.
        """
        pass

    @abstractmethod
    def validate_epoch(self, model: nn.Module, test_loader: DataLoader, 
                       criterion: nn.Module, device: torch.device) -> Dict[str, float]:
        """
        Performs a full validation epoch.
        Returns a dictionary of metrics (e.g., {"loss": 0.5, "accuracy": 95.0}).
        """
        pass