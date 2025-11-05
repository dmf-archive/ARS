import math
import os
from typing import Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.data_utils import download_file


class Wikitext2Dataset:
    def __init__(self, root: str, split: str = "train", max_length: int = 256):
        self.root = os.path.expanduser(root)
        self.split = split
        self.max_length = max_length

        filename = os.path.join(self.root, f"wiki.{split}.npz")
        with open(filename, "rb") as f:
            data = np.load(f)
            self._tokens = torch.from_numpy(data["tokens"].astype(np.int64))
            self._sizes = tuple(data["sizes"])

        indices = self._get_split_indices(self._tokens.numel())
        self._examples = [
            self._tokens[start:start + length]
            for (start, length) in indices
            if length > 1
        ]

    def _get_split_indices(self, num_tokens: int) -> list:
        num_chunks = (num_tokens - 1) // self.max_length
        indices = [
            (i * self.max_length, self.max_length + 1) for i in range(num_chunks)
        ]
        full_chunks = num_chunks * self.max_length + (num_chunks > 0)
        remainder = num_tokens - full_chunks
        if remainder:
            indices.append((full_chunks, remainder + (num_chunks > 0)))
        return indices

    def __getitem__(self, index: int) -> dict:
        example = self._examples[index]
        example_size = example.numel() - 1

        if example_size < self.max_length:
            remainder = self.max_length - example_size
            source_tokens = nn.functional.pad(example[:-1], (0, remainder))
            target_tokens = nn.functional.pad(example[1:], (0, remainder))
            mask = torch.zeros(self.max_length, dtype=torch.float)
            mask[:example_size] = 1.0
        else:
            source_tokens, target_tokens = example[:-1], example[1:]
            mask = torch.ones(self.max_length, dtype=torch.float)

        return {"source": source_tokens, "target": target_tokens, "mask": mask}

    def __len__(self) -> int:
        return len(self._examples)


class Wikitext2Task:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.vocabulary_size = config["model"]["vocabulary_size"]
        self.embedding_size = config["model"]["embedding_size"]
        self.sequence_length = config["model"]["sequence_length"]
        self.num_heads = config["model"]["num_heads"]
        self.num_layers = config["model"]["num_layers"]
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"]["num_workers"]
        self.device = config["experiment"]["device"]

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        train_dataset = Wikitext2Dataset("./data", split="train", max_length=self.sequence_length)
        valid_dataset = Wikitext2Dataset("./data", split="validation", max_length=self.sequence_length)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )

        return train_loader, valid_loader

    def get_model(self) -> nn.Module:
        from model.nano_gpt import MiniGPT1
        
        # 下载预训练嵌入文件
        embeddings_url = "https://github.com/ElementAI/duvenaud-gpt-code/raw/master/data/embeddings.npz"
        embeddings_path = download_file(embeddings_url, Path("./data"), "embeddings.npz")

        # 使用预训练嵌入加载模型
        model = MiniGPT1.load_embeddings_from(
            filename=str(embeddings_path),
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            learn_embeddings=True  # AdaFisher 默认会学习嵌入
        )
        return model

    def get_criterion(self) -> nn.Module:
        return nn.NLLLoss()

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   monitor: Any, progress_callback=None) -> dict[str, float]:
        model.train()
        total_loss = 0.0
        total_tokens = 0

        # 检测是否使用需要二阶梯度的优化器
        needs_second_order = hasattr(optimizer, '__class__') and optimizer.__class__.__name__ in ['F3EO', 'F3EL', 'F3EW', 'AdaHessian']

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            optimizer.zero_grad()
            log_probas = model(batch["source"])
            # 使用模型内置的损失函数，因为它正确处理了掩码
            loss = model.loss(log_probas, batch["target"], batch["mask"])

            # 根据优化器类型决定是否创建计算图
            if needs_second_order:
                if optimizer.__class__.__name__ in ['F3EL']:
                    loss.backward(create_graph=True)
                    optimizer.step(loss=loss)
                else:
                    loss.backward(create_graph=True)
                    optimizer.step()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch["mask"].sum().item()
            total_tokens += batch["mask"].sum().item()

            # 每10个batch更新一次进度
            if progress_callback and (batch_idx + 1) % 10 == 0:
                current_ppl = math.exp(loss.item())
                grad_norm = monitor.compute_grad_norm(model)
                progress_callback(batch_idx + 1, len(train_loader), loss.item(), current_ppl, grad_norm, 0.0) # it/s is handled by train.py

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        return {"loss": avg_loss, "perplexity": perplexity}

    def validate_epoch(self, model: nn.Module, valid_loader: DataLoader,
                      criterion: nn.Module) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                log_probas = model(batch["source"])
                loss = model.loss(log_probas, batch["target"], batch["mask"])

                total_loss += loss.item() * batch["mask"].sum().item()
                total_tokens += batch["mask"].sum().item()

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        return {"loss": avg_loss, "perplexity": perplexity}
