import argparse
import time
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import toml

from model import get_model
from optimizer import get_optimizer
from utils.callbacks.console import ConsoleLogger
from utils.callbacks.markdown import MDLogger
from utils.callbacks.checkpoint import CheckpointSaver
from utils.callbacks.context import TrainerContext
from utils.data import MetricStore, StepMetric, EpochMetric, TaskMetrics
from utils.metrics import PICalculator, compute_grad_norm

class Cutout:
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length
    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y, x = random.randint(0, h-1), random.randint(0, w-1)
            y1, y2 = np.clip(y - self.length // 2, 0, h), np.clip(y + self.length // 2, 0, h)
            x1, x2 = np.clip(x - self.length // 2, 0, w), np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask

def get_dataloaders(config):
    data_cfg = config.get("data", {})
    batch_size = data_cfg.get("batch_size", 128)
    num_workers = data_cfg.get("num_workers", 2)
    
    train_transform_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    if data_cfg.get("cutout", False):
        train_transform_list.append(Cutout(n_holes=data_cfg.get("n_holes", 1), length=data_cfg.get("cutout_length", 16)))
    
    train_transform = transforms.Compose(train_transform_list)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def train_step(model, batch, criterion, device, **kwargs):
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    return outputs.detach(), loss

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return {"loss": total_loss / len(loader), "accuracy": 100.0 * correct / total}

def main():
    parser = argparse.ArgumentParser(description="Atomic CIFAR-10 Training")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = toml.load(args.config)
    
    exp_cfg = config.get("experiment", {})
    train_cfg = config.get("train", {})
    device = torch.device(exp_cfg.get("device", "cuda"))
    seed = exp_cfg.get("seed", 42)
    epochs = train_cfg.get("epochs", exp_cfg.get("epochs", 10))
    
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    
    output_dir = Path("outputs") / Path(args.config).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_dataloaders(config)
    model = get_model(config["model"]["arch"], num_classes=config["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()

    param_groups = [{'params': model.parameters()}]
    
    opt_cfg = config["optimizer"].copy()
    smart_opt = get_optimizer(opt_cfg.pop("name"), param_groups, model=model, criterion=criterion, device=device, **opt_cfg)

    scheduler = None
    if "scheduler" in config:
        sched_cfg = config["scheduler"].copy()
        name = sched_cfg.pop("name")
        if name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(smart_opt.optimizer, **sched_cfg)

    store = MetricStore()
    context = TrainerContext(config=config, output_dir=output_dir, device=device, model=model, 
                             optimizer=smart_opt.optimizer, store=store, total_epochs=epochs)

    pi_cfg = config.get("pi", {"gamma": 1.0, "alpha": 1.0, "ema_beta": 0.9})
    pi_calculator = PICalculator(gamma=pi_cfg.get("gamma", 1.0), alpha=pi_cfg.get("alpha", 1.0), ema_beta=pi_cfg.get("ema_beta"))
    callbacks = [ConsoleLogger(), MDLogger(), CheckpointSaver()]
    
    def broadcast(event):
        for cb in callbacks: getattr(cb, event)(context)

    broadcast("on_train_begin")
    for epoch in range(epochs):
        context.current_epoch, context.current_task_name, context.is_training = epoch, "cifar10", True
        context.total_steps_in_epoch = len(train_loader)
        model.train()
        broadcast("on_epoch_begin")
        
        epoch_start_time = time.time()
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
        epoch_loss_sum, epoch_entropy_sum, num_samples, epoch_grad_norm_list = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), 0, []

        for step, batch in enumerate(train_loader):
            context.current_step_in_epoch = step
            broadcast("on_step_begin")
            
            logits, loss = smart_opt.step(batch, train_step)
            
            with torch.no_grad():
                epoch_loss_sum += loss
                if logits is not None:
                    probas = torch.softmax(logits, dim=-1)
                    epoch_entropy_sum += -(probas * torch.log_softmax(logits, dim=-1)).sum()
                    num_samples += logits.size(0)
                gn = compute_grad_norm(model, return_tensor=True)
                if gn is not None: epoch_grad_norm_list.append(gn)

            store.add_step(StepMetric(task_name="cifar10", global_step=context.global_step, task_epoch=epoch, 
                                      step_in_epoch=step, loss=loss.item(), learning_rate=smart_opt.param_groups[0]['lr']))
            broadcast("on_step_end")
            context.global_step += 1

        context.is_training = False
        val_metrics = validate_epoch(model, test_loader, criterion, device)
        avg_train_loss = (epoch_loss_sum / len(train_loader)).item()
        avg_gn = torch.stack(epoch_grad_norm_list).mean().item() if epoch_grad_norm_list else None
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

        store.add_epoch(EpochMetric(task_name="cifar10", task_epoch=epoch, global_epoch=epoch, avg_train_loss=avg_train_loss,
                                    task_metrics=TaskMetrics(metrics=val_metrics), avg_pi_obj=avg_pi, avg_entropy=avg_entropy,
                                    grad_norm=avg_gn, learning_rate=smart_opt.param_groups[0]['lr'], diagnostics=diagnostics,
                                    epoch_time_s=time.time() - epoch_start_time, peak_gpu_mem_mb=torch.cuda.max_memory_allocated() / (1024**2) if device.type == 'cuda' else None))
        broadcast("on_epoch_end")
        broadcast("save")
        
        if scheduler:
            scheduler.step()

        if val_metrics["accuracy"] >= train_cfg.get("early_stop_threshold", 99.5):
            print(f"Early stopping at epoch {epoch} as accuracy reached {val_metrics['accuracy']}%")
            break
            
    broadcast("on_train_end")

if __name__ == "__main__":
    main()
