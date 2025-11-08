import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import toml
import torch

from utils.data import CLStore, StepMetric
from utils.observers.console import ConsoleLogger
from utils.observers.markdown import MDLogger
from utils.observers.checkpoint import CheckpointSaver
from utils.early_stop import EarlyStop
from utils.metrics import PICalculator, compute_grad_norm # Import from new metrics module

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return toml.load(f)

def get_task_class(task_name: str):
    try:
        module = importlib.import_module(f"task.{task_name}")
        return getattr(module, f"{task_name.capitalize()}Task")
    except (ImportError, AttributeError) as e:
        print(f"Error loading task module '{task_name}': {e}")
        sys.exit(1)

def create_optimizer(model, config):
    from optimizer import get_optimizer
    opt_name = config["optimizer"]["name"]
    opt_config = {k: v for k, v in config["optimizer"].items() if k != "name"}
    if opt_name == "AdaFisher":
        opt_config["model"] = model
    return get_optimizer(opt_name, model.parameters(), **opt_config)

def create_scheduler(optimizer, config):
    if "scheduler" not in config:
        return None
    sched_name = config["scheduler"].get("name")
    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["scheduler"]["T_max"])
    elif sched_name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["scheduler"]["milestones"], gamma=config["scheduler"]["gamma"])
    return None

def train(config: Dict[str, Any], config_name: str):
    device = torch.device(config["experiment"]["device"])
    torch.manual_seed(config["experiment"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["experiment"]["seed"])

    output_dir = Path("outputs") / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    task_names = config["experiment"]["tasks"]
    tasks = {name: get_task_class(name)(config) for name in task_names}
    # Assume a single model and criterion for now
    model = tasks[task_names[0]].get_model().to(device)
    criterion = tasks[task_names[0]].get_criterion()
    optimizer, optimizer_tags, pi_config = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # --- New Unified Architecture ---
    store = CLStore()
    console_logger = ConsoleLogger(config)
    md_logger = MDLogger(config, output_dir)
    ckpt_saver = CheckpointSaver(output_dir)

    # --- Force-enable PI calculation for all tasks ---
    # If pi_config is not provided by the optimizer, use defaults.
    pi_gamma = pi_config.get("gamma", 1.0) if pi_config else 1.0
    pi_alpha = pi_config.get("alpha", 1.0) if pi_config else 1.0
    pi_ema_beta = pi_config.get("ema_beta") if pi_config else None
    pi_calculator = PICalculator(gamma=pi_gamma, alpha=pi_alpha, ema_beta=pi_ema_beta)
    # ---

    console_logger.on_train_begin(str(output_dir))

    start_epoch = 0
    checkpoint = ckpt_saver.load(output_dir / "latest_checkpoint.pt", model, optimizer, scheduler)
    if checkpoint:
        start_epoch = checkpoint["epoch"] + 1
        store = checkpoint["store"]
        print(f"Resuming training from epoch {start_epoch}")
    else:
        store = CLStore()
    
    # Ensure get_dataloaders() is called only once per task to avoid duplicate logs
    train_loaders = {}
    valid_loaders = {}
    for name, task in tasks.items():
        train_loader, valid_loader = task.get_dataloaders()
        train_loaders[name] = train_loader
        valid_loaders[name] = valid_loader
    
    global_step = 0
    epochs = config["train"]["epochs"]

    for task_name in task_names:
        current_task = tasks[task_name]
        current_train_loader = train_loaders[task_name]
        total_steps = len(current_train_loader)
        
        for epoch in range(start_epoch, epochs):
            model.train()
            console_logger.on_epoch_begin(epoch, total_steps)

            for step, batch in enumerate(current_train_loader):
                logits, loss, step_metrics = current_task.train_step(model, batch, criterion, optimizer, pi_config)
                
                grad_norm = compute_grad_norm(model)
                
                pi, eff_gamma, entropy = None, None, None
                if pi_calculator and logits is not None:
                    with torch.no_grad():
                        probas = torch.softmax(logits, dim=-1)
                        log_probas = torch.log_softmax(logits, dim=-1)
                        entropy_tensor = -(probas * log_probas).sum(dim=-1).mean()
                        entropy = entropy_tensor.item()
                        _, pi = pi_calculator.calculate_pi(entropy_tensor, grad_norm)
                        if pi is not None:
                            eff_gamma = -torch.log(1.0 - torch.tensor(pi) + pi_calculator.eps).item()

                metric = StepMetric(
                    global_step=global_step,
                    epoch=epoch,
                    step_in_epoch=step,
                    task_name=task_name,
                    loss=loss,
                    eval_metrics=step_metrics,
                    grad_norm=grad_norm,
                    learning_rate=optimizer.param_groups[0]['lr'],
                    pi=pi, effective_gamma=eff_gamma, entropy=entropy,
                    timestamp=time.time()
                )
                store.add_step(metric)
                console_logger.on_step_end(metric, total_steps)
                global_step += 1

            model.eval()
            all_eval_metrics = {}
            with torch.no_grad():
                for eval_task_name, eval_loader in valid_loaders.items():
                    eval_task = tasks[eval_task_name]
                    results = eval_task.validate_epoch(model, eval_loader, criterion)
                    metric_key = "perplexity" if "wikitext2" in eval_task_name else "accuracy"
                    all_eval_metrics[eval_task_name] = results[metric_key]
            
            last_metric = store.get_full_history()[-1]
            last_metric.eval_metrics.update(all_eval_metrics)
            
            console_logger.on_epoch_end(store)
            
            ckpt_saver.save(epoch, model, optimizer, scheduler, store)
            
            if scheduler:
                scheduler.step()

    md_logger.on_train_end(store)
    console_logger.on_train_end(store)

def main():
    parser = argparse.ArgumentParser(description="Unified F3EO-Bench Training Framework")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML configuration file")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
        
    config = load_config(config_path)
    config_name = config_path.stem
    train(config, config_name)

if __name__ == "__main__":
    main()
