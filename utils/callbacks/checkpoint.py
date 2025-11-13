from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .base import Callback

if TYPE_CHECKING:
    from utils.data import MetricStore, StepMetric


class CheckpointSaver(Callback):
    def __init__(self, output_dir: Path, max_checkpoints: int = 3):
        self.output_dir = output_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoint_files: list[Path] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_train_begin(self, store: "MetricStore", **kwargs):
        pass

    def on_train_end(self, store: "MetricStore", **kwargs):
        pass

    def on_epoch_begin(self, epoch: int, total_steps: int, **kwargs):
        pass

    def on_epoch_end(self, store: "MetricStore", **kwargs):
        pass

    def on_step_begin(self, step: int, **kwargs):
        pass

    def on_step_end(self, step_metric: "StepMetric", total_steps: int, **kwargs):
        pass

    def save(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             scheduler: torch.optim.lr_scheduler._LRScheduler | None, store: "MetricStore", **kwargs):

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "store": store
        }

        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)

        self.checkpoint_files.append(checkpoint_path)
        if len(self.checkpoint_files) > self.max_checkpoints:
            oldest_checkpoint = self.checkpoint_files.pop(0)
            if oldest_checkpoint.exists():
                oldest_checkpoint.unlink()

        latest_path = self.output_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

    def load(self, path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             scheduler: torch.optim.lr_scheduler._LRScheduler | None, **kwargs) -> dict | None:

        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            return None

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint
