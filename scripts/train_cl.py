"""
极简持续学习训练入口
仅支持 MNIST → FashionMNIST，各 5 epoch，总 10 epoch
旁路 CLObserver，零 checkpoint，零早停，10 分钟跑完
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from utils.cl_observer import CLObserver
from utils.training_monitor import TrainingMonitor

console = Console()


def load_config(config_path: Path) -> dict:
    import toml
    with open(config_path) as f:
        return toml.load(f)


def get_task_module(task_name: str):
    try:
        import importlib
        module = importlib.import_module(f"task.{task_name}")
        task_class = getattr(module, f"{task_name.capitalize()}Task")
        return task_class
    except (ImportError, AttributeError) as e:
        console.print(f"[red]Error loading task module '{task_name}': {e}[/red]")
        sys.exit(1)


def create_optimizer_scheduler(model, config):
    from optimizer import get_optimizer
    opt_name = config["optimizer"]["name"]
    opt_config = config["optimizer"].copy()
    opt_config.pop("name", None)
    if opt_name == "AdaFisher":
        opt_config["model"] = model
    optimizer, tags, pi_config = get_optimizer(opt_name, model.parameters(), **opt_config)
    scheduler = None
    if "scheduler" in config:
        sched_name = config["scheduler"].get("name", "none")
        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["scheduler"]["T_max"]
            )
    return optimizer, scheduler, tags, pi_config


def run_cl(config_path: Path):
    config = load_config(config_path)
    task_name = config["experiment"]["task"]
    task_class = get_task_module(task_name)
    task = task_class(config)

    # 持续学习数据流
    train_loaders, valid_loaders = task.get_dataloaders()
    model = task.get_model().to(config["experiment"]["device"])
    criterion = task.get_criterion()
    optimizer, scheduler, opt_tags, pi_config = create_optimizer_scheduler(model, config)

    # 旁路观测器
    cl_obs = CLObserver(output_dir=Path("outputs") / task_name / config_path.stem)
    task.cl_observer = cl_obs  # 注入到任务内部，方便后续钩子

    epochs_per_task = 5
    num_tasks = 2
    total_epochs = epochs_per_task * num_tasks

    # 轻量 TrainingMonitor（仅用于 PI 计算与 Rich 日志）
    monitor = TrainingMonitor(config, cl_obs.output_dir, pi_config=pi_config)

    console.print(f"[bold cyan]CL Quick Run: {num_tasks} tasks × {epochs_per_task} epochs[/bold cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task_prog = progress.add_task("Task Progress", total=total_epochs)

        for task_id in range(num_tasks):
            console.print(f"\n[yellow]>>> Task {task_id} (MNIST→FashionMNIST)[/yellow]")
            for epoch in range(epochs_per_task):
                global_epoch = task_id * epochs_per_task + epoch

                monitor.start_epoch(global_epoch, len(train_loaders[task_id]))

                # 训练一个 epoch
                task.train_epoch(
                    model, train_loaders, optimizer, criterion,
                    monitor, optimizer_tags=opt_tags, task_id=task_id
                )

                # 验证所有任务（遗忘曲线数据源）
                accs = [res["accuracy"] for res in task.validate_all(model, valid_loaders, criterion)]

                # 旁路记录
                last_metrics = monitor.metrics_history[-1] if monitor.metrics_history else None
                cl_obs.log_epoch(
                    epoch=global_epoch,
                    task_id=task_id,
                    train_loss=monitor.epoch_metrics_history[-1]["train_loss"] if monitor.epoch_metrics_history else 0.0,
                    accuracies=accs,
                    pi=last_metrics.pi if last_metrics else None
                )

                monitor.end_epoch(
                    {"loss": monitor.epoch_metrics_history[-1]["train_loss"], "accuracy": accs[task_id]},
                    {"loss": 0.0, "accuracy": accs[task_id]},  # 占位
                    optimizer.param_groups[0]["lr"]
                )

                progress.update(task_prog, advance=1)

                if scheduler:
                    scheduler.step()

    # 生成 CL 报告
    cl_obs.finalize()
    console.print("\n[bold green]CL quick run finished.[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="10-min CL sandbox")
    parser.add_argument("--config", type=Path, required=True, help="TOML config")
    args = parser.parse_args()
    run_cl(args.config)