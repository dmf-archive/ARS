from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from utils.data import MetricStore, StepMetric

class Callback(ABC):
    """
    Abstract base class for creating callbacks. Callbacks are used to hook into
    the training process to perform actions at various stages (e.g., logging,
    saving checkpoints, etc.).
    """

    @abstractmethod
    def on_train_begin(self, store: "MetricStore", **kwargs):
        """Called at the beginning of training."""
        pass

    @abstractmethod
    def on_train_end(self, store: "MetricStore", **kwargs):
        """Called at the end of training."""
        pass

    @abstractmethod
    def on_epoch_begin(self, epoch: int, total_steps: int, **kwargs):
        """Called at the beginning of an epoch."""
        pass

    @abstractmethod
    def on_epoch_end(self, store: "MetricStore", **kwargs):
        """Called at the end of an epoch."""
        pass

    @abstractmethod
    def on_step_begin(self, step: int, **kwargs):
        """Called at the beginning of a training step."""
        pass

    @abstractmethod
    def on_step_end(self, step_metric: "StepMetric", total_steps: int, **kwargs):
        """Called at the end of a training step."""
        pass

    @abstractmethod
    def save(self, epoch: int, model: "torch.nn.Module", optimizer: "torch.optim.Optimizer",
             scheduler: "torch.optim.lr_scheduler._LRScheduler | None", store: "MetricStore", **kwargs):
        """Called to save a checkpoint."""
        pass

    @abstractmethod
    def load(self, path: str, model: "torch.nn.Module", optimizer: "torch.optim.Optimizer",
             scheduler: "torch.optim.lr_scheduler._LRScheduler | None", **kwargs) -> dict | None:
        """Called to load a checkpoint."""
        return None
