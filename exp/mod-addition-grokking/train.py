import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import toml
from model import get_model
from optimizer import get_optimizer
from utils.callbacks.console import ConsoleLogger
from utils.callbacks.markdown import MDLogger
from utils.callbacks.checkpoint import CheckpointSaver
from utils.callbacks.context import TrainerContext
from utils.data import MetricStore, StepMetric, EpochMetric, TaskMetrics
from utils.metrics import PICalculator, compute_grad_norm

def get_dataloaders(config):
    task_cfg = config["task"]
    data_cfg = config["data"]
    p = task_cfg.get("p", 113)
    fraction = task_cfg.get("fraction", 0.3)
    batch_size = data_cfg.get("batch_size", 512)
    seed = config["experiment"].get("seed", 42)
    
    torch.manual_seed(seed)
    equals_token = p
    x, y = torch.meshgrid(torch.arange(p), torch.arange(p), indexing='ij')
    x, y = x.flatten(), y.flatten()
    equals = torch.ones(x.shape, dtype=torch.int64) * equals_token
    
    prompts = torch.stack([x, y, equals], dim=1)
    answers = (x + y) % p
    
    dataset = TensorDataset(prompts, answers)
    train_size = int(fraction * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    return train_loader, test_loader

def train_step(model, batch, criterion, device, **kwargs):
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)
    logits = model(inputs)[:, -1]
    loss = criterion(logits, targets)
    return logits.detach(), loss

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)[:, -1]
            loss = criterion(logits, targets)
            total_loss += loss.item() * inputs.size(0)
            correct += (torch.argmax(logits, dim=-1) == targets).sum().item()
            total += targets.size(0)
    return {"loss": total_loss / total, "accuracy": 100.0 * correct / total}

def main():
    parser = argparse.ArgumentParser(description="High-Fidelity Atomic Grokking Training")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = toml.load(args.config)
    device = torch.device(config["experiment"]["device"])
    torch.manual_seed(config["experiment"]["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["experiment"]["seed"])
    
    output_dir = Path("outputs") / Path(args.config).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_dataloaders(config)
    
    p = config["task"].get("p", 113)
    model_config = config["model"].copy()
    arch = model_config.pop("arch")
    model = get_model(arch, num_classes=p + 1, **model_config).to(device)
    criterion = nn.CrossEntropyLoss()

    hidden_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
    other_params = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "embed" not in n)]
    param_groups = [{'params': hidden_params, 'use_muon': True}, {'params': other_params, 'use_muon': False}]

    opt_config = config["optimizer"].copy()
    opt_name = opt_config.pop("name")
    smart_opt = get_optimizer(opt_name, param_groups, model=model, criterion=criterion, device=device, **opt_config)

    store = MetricStore()
    context = TrainerContext(config=config, output_dir=output_dir, device=device, model=model, 
                             optimizer=smart_opt.optimizer, store=store, total_epochs=config["experiment"]["epochs"])

    pi_cfg = config.get("pi", {"gamma": 1.0, "alpha": 1.0, "ema_beta": 0.9})
    pi_calculator = PICalculator(gamma=pi_cfg.get("gamma", 1.0), alpha=pi_cfg.get("alpha", 1.0), ema_beta=pi_cfg.get("ema_beta"))

    callbacks = [ConsoleLogger(), MDLogger(), CheckpointSaver()]
    def broadcast(event):
        for cb in callbacks: getattr(cb, event)(context)

    broadcast("on_train_begin")

    for epoch in range(config["experiment"]["epochs"]):
        context.current_epoch, context.current_task_name, context.is_training = epoch, "mod_addition", True
        context.total_steps_in_epoch = len(train_loader)
        model.train()
        broadcast("on_epoch_begin")
        
        epoch_start_time = time.time()
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
            
        epoch_loss_sum, epoch_entropy_sum, epoch_correct, num_samples, epoch_grad_norm_list = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), 0, 0, []

        for step, batch in enumerate(train_loader):
            context.current_step_in_epoch = step
            broadcast("on_step_begin")
            
            logits, loss = smart_opt.step(batch, train_step)
            
            with torch.no_grad():
                loss_val = loss.item()
                epoch_loss_sum += loss * batch[1].size(0)
                if logits is not None:
                    probas = torch.softmax(logits, dim=-1)
                    epoch_entropy_sum += -(probas * torch.log_softmax(logits, dim=-1)).sum()
                    epoch_correct += (torch.argmax(logits, dim=-1) == batch[1].to(device)).sum().item()
                num_samples += batch[1].size(0)
                gn = compute_grad_norm(model, return_tensor=True)
                if gn is not None: epoch_grad_norm_list.append(gn)

            store.add_step(StepMetric(task_name="mod_addition", global_step=context.global_step, task_epoch=epoch, 
                                      step_in_epoch=step, loss=loss_val, learning_rate=smart_opt.param_groups[0]['lr']))
            broadcast("on_step_end")
            context.global_step += 1

        context.is_training = False
        val_metrics = validate_epoch(model, test_loader, criterion, device)
        val_metrics["train_accuracy"] = 100.0 * epoch_correct / num_samples if num_samples > 0 else 0.0
        
        avg_train_loss = (epoch_loss_sum / num_samples).item() if num_samples > 0 else 0.0
        avg_gn_tensor = torch.stack(epoch_grad_norm_list).mean() if epoch_grad_norm_list else None
        avg_gn = avg_gn_tensor.item() if avg_gn_tensor is not None else None
        
        avg_entropy, avg_pi = None, None
        if num_samples > 0:
            avg_entropy_tensor = epoch_entropy_sum / num_samples
            avg_entropy = avg_entropy_tensor.item()
            if avg_gn is not None: _, avg_pi = pi_calculator.calculate_pi(avg_entropy_tensor, avg_gn)

        diagnostics = smart_opt.diagnostics
        if diagnostics:
            import copy
            diagnostics = copy.deepcopy(diagnostics)
        else:
            diagnostics = {}
            
        for i, group in enumerate(smart_opt.param_groups):
            name = "muon" if group.get("use_muon") or group.get("is_rmsuon_group") else "adam"
            norms = [p.norm().item() for p in group['params']]
            if norms: diagnostics[f"group_{i}_{name}_avg_norm"] = sum(norms) / len(norms)

        store.add_epoch(EpochMetric(task_name="mod_addition", task_epoch=epoch, global_epoch=epoch, avg_train_loss=avg_train_loss,
                                    task_metrics=TaskMetrics(metrics=val_metrics), avg_pi_obj=avg_pi, avg_entropy=avg_entropy,
                                    grad_norm=avg_gn, learning_rate=smart_opt.param_groups[0]['lr'], diagnostics=diagnostics,
                                    epoch_time_s=time.time() - epoch_start_time, peak_gpu_mem_mb=torch.cuda.max_memory_allocated() / (1024**2) if device.type == 'cuda' else None))
        
        broadcast("on_epoch_end")
        broadcast("save")
        if val_metrics["accuracy"] >= 99.5:
            print(f"Early stopping at epoch {epoch} as test accuracy reached {val_metrics['accuracy']:.2f}%")
            break


    broadcast("on_train_end")

if __name__ == "__main__":
    main()
