from dataclasses import dataclass, field
from typing import Any, Dict, List
import time

@dataclass(frozen=True)
class StepMetric:
    task_name: str
    global_step: int
    task_epoch: int
    step_in_epoch: int
    loss: float
    learning_rate: float
    timestamp: float = field(default_factory=time.time, init=False)

@dataclass(frozen=True)
class TaskMetrics:
    metrics: Dict[str, Any]

@dataclass(frozen=True)
class EpochMetric:
    task_name: str
    task_epoch: int
    global_epoch: int
    avg_train_loss: float
    task_metrics: TaskMetrics
    avg_pi: float | None
    avg_effective_gamma: float | None
    avg_entropy: float | None
    grad_norm: float | None
    learning_rate: float
    timestamp: float = field(default_factory=time.time, init=False)

class MetricStore:
    def __init__(self):
        self._step_history: List[StepMetric] = []
        self._epoch_history: Dict[str, List[EpochMetric]] = {}

    def add_step(self, step_metric: StepMetric):
        self._step_history.append(step_metric)

    def add_epoch(self, epoch_metric: EpochMetric):
        task_name = epoch_metric.task_name
        if task_name not in self._epoch_history:
            self._epoch_history[task_name] = []
        
        latest_global_epoch = self.get_latest_epoch_for_task(task_name).global_epoch if self.get_latest_epoch_for_task(task_name) else -1
        if epoch_metric.global_epoch != latest_global_epoch + 1 and self._epoch_history[task_name]:
             raise ValueError(f"Global epoch mismatch for task '{task_name}'. Expected {latest_global_epoch + 1}, got {epoch_metric.global_epoch}")

        self._epoch_history[task_name].append(epoch_metric)

    def get_full_step_history(self) -> List[StepMetric]:
        return self._step_history

    def get_steps_for_epoch(self, task_name: str, task_epoch: int) -> List[StepMetric]:
        return [s for s in self._step_history if s.task_name == task_name and s.task_epoch == task_epoch]

    def get_history_for_task(self, task_name: str) -> List[EpochMetric]:
        return self._epoch_history.get(task_name, [])

    def get_full_epoch_history(self) -> Dict[str, List[EpochMetric]]:
        return self._epoch_history
    
    def get_flat_epoch_history(self) -> List[EpochMetric]:
        flat_list = []
        for task_epochs in self._epoch_history.values():
            flat_list.extend(task_epochs)
        return sorted(flat_list, key=lambda x: (x.task_name, x.global_epoch))

    def get_latest_epoch_for_task(self, task_name: str) -> EpochMetric | None:
        if task_name in self._epoch_history and self._epoch_history[task_name]:
            return self._epoch_history[task_name][-1]
        return None