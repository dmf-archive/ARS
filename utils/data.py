from dataclasses import dataclass, field
from typing import Dict, List

@dataclass(frozen=True)
class StepMetric:
    global_step: int
    epoch: int
    step_in_epoch: int
    task_name: str
    loss: float
    eval_metrics: Dict[str, float]
    grad_norm: float | None
    learning_rate: float
    pi: float | None
    effective_gamma: float | None
    entropy: float | None
    timestamp: float

class SingleTaskStore:
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.history: List[StepMetric] = []

    def add_step(self, step_metric: StepMetric):
        if step_metric.task_name != self.task_name:
            raise ValueError("Mismatched task name for SingleTaskStore")
        self.history.append(step_metric)

class CLStore:
    def __init__(self):
        self._tasks: Dict[str, SingleTaskStore] = {}
        self._full_history: List[StepMetric] = []

    @property
    def tasks(self) -> List[SingleTaskStore]:
        return list(self._tasks.values())

    def add_step(self, step_metric: StepMetric):
        task_name = step_metric.task_name
        if task_name not in self._tasks:
            self._tasks[task_name] = SingleTaskStore(task_name)
        
        self._tasks[task_name].add_step(step_metric)
        self._full_history.append(step_metric)

    def get_task_history(self, task_name: str) -> SingleTaskStore | None:
        return self._tasks.get(task_name)

    def get_full_history(self) -> List[StepMetric]:
        return self._full_history