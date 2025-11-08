import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# 旁路 CL 观测器
from utils.cl_observer import CLObserver


class MNISTCLTask:
    """
    极简持续学习沙盒：MNIST → FashionMNIST
    每任务 5 epoch，评估指标 = 平均准确率 + 遗忘率
    适配 F3EO-Bench 框架，重用现有 TrainingMonitor 与 PI 轨道
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"]["num_workers"]
        self.device = config["experiment"]["device"]
        self.num_classes = 10  # 两数据集均为 10 类
        self.sequence_length = 28 * 28  # 展平后当语言模型用，方便 PI 计算
        # 旁路观测器：由外部脚本注入，训练结束再 finalize
        self.cl_observer: CLObserver | None = None

    def _get_transform(self):
        """28×28 灰度图 → 展平 784 向量，归一化到 [0,1]"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))  # 展平
        ])

    def _build_dataloader(self, dataset: Dataset, shuffle: bool = True):
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True, drop_last=True
        )

    def get_dataloaders(self) -> tuple[list[DataLoader], list[DataLoader]]:
        """
        返回两个任务的数据流：
        train_loaders[0] = MNIST, train_loaders[1] = FashionMNIST
        valid_loaders 同理
        """
        root = Path("./data")
        transform = self._get_transform()

        # Task 0: MNIST
        mnist_train = datasets.MNIST(root, train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root, train=False, transform=transform)

        # Task 1: FashionMNIST
        fashion_train = datasets.FashionMNIST(root, train=True, download=True, transform=transform)
        fashion_test = datasets.FashionMNIST(root, train=False, transform=transform)

        train_loaders = [
            self._build_dataloader(mnist_train, shuffle=True),
            self._build_dataloader(fashion_train, shuffle=True)
        ]
        valid_loaders = [
            self._build_dataloader(mnist_test, shuffle=False),
            self._build_dataloader(fashion_test, shuffle=False)
        ]
        return train_loaders, valid_loaders

    def get_model(self) -> nn.Module:
        """
        返回 Micro-SwinViT：
        4 层, dim=96, head=4, patch=4, 输入 28×28 灰度图
        总参数量 ≈ 80 K，10 分类输出
        """
        from model.swin import swin_t
        model = swin_t(
            num_classes=self.num_classes,
            hidden_dim=96,
            layers=(2, 2, 2, 2),  # 每层 2 个 block，共 4 层
            heads=(3, 6, 12, 24),
            downscaling_factors=(4, 2, 2, 2),  # 首下采样 4→14→7→4→2
            window_size=4,  # 7 太大，改用 4
            channels=1  # 灰度
        )
        return model

    def get_criterion(self) -> nn.Module:
        """
        返回 NLLLoss，方便与 PI 计算对接（softmax 后取 log）
        """
        return nn.NLLLoss()

    def _compute_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        pred = logits.argmax(dim=1)
        return (pred == targets).float().mean().item() * 100.0

    def train_epoch(self, model: nn.Module, train_loaders: list[DataLoader],
                   optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   monitor: Any, progress_callback=None, optimizer_tags=None,
                   task_id: int = 0) -> dict[str, float]:
        """
        仅训练指定 task_id 的数据流
        其余任务数据完全不可见，模拟真实 CL
        """
        model.train()
        loader = train_loaders[task_id]
        total_loss = 0.0
        total_samples = 0
        last_callback_time = time.time()

        accepts_pi_signal = optimizer_tags.get("accepts_pi_signal", False) if optimizer_tags else False
        needs_second_order = optimizer_tags.get("requires_second_order", False) if optimizer_tags else False

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()

            logits = model(data.unsqueeze(1))  # (B,784) → (B,10)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = criterion(log_probs, target)

            loss.backward(create_graph=needs_second_order)

            # PI 计算与 step
            metrics = monitor.end_step(model, loss.item(), optimizer.param_groups[0]['lr'], logits)
            step_args = {}
            if accepts_pi_signal:
                step_args['effective_gamma'] = metrics.effective_gamma
            optimizer.step(**step_args)

            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

            if progress_callback and (batch_idx + 1) % 10 == 0:
                acc = self._compute_accuracy(logits, target)
                current_time = time.time()
                steps_processed = 10
                time_elapsed = current_time - last_callback_time
                steps_per_sec = steps_processed / time_elapsed if time_elapsed > 0 else 0.0
                last_callback_time = current_time

                progress_callback(
                    epoch=monitor.current_epoch,
                    step=batch_idx + 1,
                    total_steps=len(loader),
                    loss=loss.item(),
                    metric=acc,
                    grad_norm=metrics.grad_norm,
                    items_per_sec=steps_per_sec,
                    pi=metrics.pi,
                    effective_gamma=metrics.effective_gamma,
                    entropy=metrics.entropy
                )

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        # 返回训练集准确率（用于 Rich 表格），验证集准确率由外部 validate_all 负责
        train_acc = self._validate(model, loader, criterion)
        return {"loss": avg_loss, "accuracy": train_acc}

    def validate_all(self, model: nn.Module, valid_loaders: list[DataLoader],
                    criterion: nn.Module) -> list[dict[str, float]]:
        """
        返回每个任务当前的验证准确率
        用于画遗忘曲线
        """
        model.eval()
        results = []
        with torch.no_grad():
            for loader in valid_loaders:
                acc = self._validate(model, loader, criterion)
                results.append({"accuracy": acc})
        return results

    def _validate(self, model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
        total_loss = 0.0
        total_samples = 0
        correct = 0
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            logits = model(data.unsqueeze(1))
            log_probs = torch.log_softmax(logits, dim=1)
            loss = criterion(log_probs, target)
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
        return correct / total_samples * 100.0 if total_samples > 0 else 0.0