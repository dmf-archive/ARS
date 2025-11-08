from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from typing import Any, Dict

from utils.data import StepMetric, CLStore

class ConsoleLogger:
    def __init__(self, config: Dict[str, Any]):
        self.console = Console()
        self.config = config
        self.progress = None
        self.step_task_id = None

    def on_train_begin(self, output_dir: str):
        self.console.print(Panel.fit(
            f"[bold cyan]Tasks:[/bold cyan] {', '.join(self.config['experiment']['tasks'])}\n"
            f"[bold cyan]Model:[/bold cyan] {self.config['model']['arch']}\n"
            f"[bold cyan]Optimizer:[/bold cyan] {self.config['optimizer']['name']}\n"
            f"[bold cyan]Epochs:[/bold cyan] {self.config['train']['epochs']}\n"
            f"[bold cyan]Device:[/bold cyan] {self.config['experiment']['device']}\n"
            f"[bold cyan]Output:[/bold cyan] {output_dir}",
            title="[bold]F3EO-Bench Training[/bold]",
            border_style="cyan"
        ))
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=self.console
        )
        self.progress.start()

    def on_epoch_begin(self, epoch: int, total_steps: int):
        self.step_task_id = self.progress.add_task(f"Epoch {epoch+1}", total=total_steps)

    def on_step_end(self, metric: StepMetric, total_steps: int):
        self.progress.update(self.step_task_id, advance=1)
        if metric.step_in_epoch % 10 != 0:
            return

        msg = f"Epoch {metric.epoch+1} | Step {metric.step_in_epoch}/{total_steps} | Loss: {metric.loss:.4f}"
        task_name = metric.task_name
        if 'wikitext2' in task_name:
            ppl = metric.eval_metrics.get(task_name, 0.0)
            msg += f" | PPL: {ppl:.2f}"
        else:
            acc = metric.eval_metrics.get(task_name, 0.0)
            msg += f" | Acc: {acc:.2f}%"

        if metric.grad_norm is not None:
            msg += f" | Grad: {metric.grad_norm:.4f}"
        if metric.pi is not None:
            msg += f" | PI: {metric.pi:.3f}"
        if metric.effective_gamma is not None:
            msg += f" | β: {metric.effective_gamma:.3f}"
        if metric.entropy is not None:
            msg += f" | H: {metric.entropy:.3f}"
        
        self.console.print(msg)

    def on_epoch_end(self, store: CLStore):
        last_metric = store.get_full_history()[-1]
        
        self.progress.remove_task(self.step_task_id)

        table = Table(title=f"Epoch {last_metric.epoch+1} Results")
        table.add_column("Task", style="cyan")
        table.add_column("Loss", justify="right", style="magenta")
        table.add_column("Metric", justify="right", style="green")

        for task_name, metrics in last_metric.eval_metrics.items():
            is_ppl = 'wikitext2' in task_name
            metric_str = f"{metrics:.2f}" if is_ppl else f"{metrics:.2f}%"
            table.add_row(task_name, "N/A", metric_str)

        self.console.print(table)
        self.console.print(f"[dim]LR: {last_metric.learning_rate:.6f}[/dim]")

    def on_train_end(self, store: CLStore):
        self.progress.stop()
        self.console.print("\n[bold green]Training completed![/bold green]")