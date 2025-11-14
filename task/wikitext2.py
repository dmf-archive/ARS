import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, Dataset

from .base import BaseTask


def get_or_train_tokenizer(config: dict[str, Any]) -> Tokenizer:
    """
    Loads a pre-trained tokenizer or trains a new one from the wikitext dataset.
    """
    tokenizer_path = Path(config["data"]["tokenizer_path"])
    vocab_size = config["model"]["vocabulary_size"]

    if tokenizer_path.exists():
        # Load existing tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        # Train a new tokenizer
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

        def get_training_corpus() -> Iterator[list[str]]:
            for i in range(0, len(dataset), 1000):
                # Correctly access the list of texts from the sliced dictionary
                batch_texts = dataset[i : i + 1000]['text']
                # Filter out empty or whitespace-only strings
                yield [text for text in batch_texts if text.strip()]

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, special_tokens=["<|endoftext|>", "<pad>"]
        )
        tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))

    return tokenizer


def split_into_sentences(text: str) -> list[str]:
    """
    最小启发式：以句点+空格作为句子边界。
    返回非空句子列表，末尾保留句点。
    """
    if not text.strip():
        return []
    # 简单 split，保留句点
    sents = [s.strip() + '.' for s in text.split('. ') if s.strip()]
    # 最后一句若已以句点结尾则去重
    if sents and sents[-1].endswith('..'):
        sents[-1] = sents[-1][:-1]
    return sents


def build_line_mode_samples(tokenizer: Tokenizer, lines: list[str], seq_len: int, eos_id: int, pad_id: int = 0, ignore_index: int = -100) -> list[torch.Tensor]:
    """
    按句 tokenize → EOF 拼接 → 切块到 seq_len，补零。
    返回 List[Tensor(seq_len)]]
    关键点：将填充部分的标签设置为 ignore_index（-100），以符合 HF 最佳实践。
    """
    samples, current = [], []
    for sent in lines:
        ids = tokenizer.encode(sent).ids + [eos_id]   # 句末加 EOF
        # 确保单句长度不超过 seq_len，因为我们已经统计过最大句子长度为 242 < 256
        # 如果出现超长，则说明分词或统计有误，此处不应截断，而是应该抛出异常或重新评估
        if len(ids) > seq_len:
            # 理论上此处不应该被触发，如果触发，需要检查数据或seq_len配置
            print(f"Warning: Sentence token length {len(ids)} exceeds sequence_length {seq_len}. Skipping or re-evaluating.")
            continue # 跳过这个超长句子，或者可以考虑更复杂的处理

        # 缓存拼接
        # 尝试将当前句子完整地放入 current，如果放不下，则先产出 current
        if len(current) + len(ids) > seq_len:
            # 当前样本已满或不足以容纳整个新句子，先产出当前样本
            pad_len = seq_len - len(current)
            # 将填充部分的标签设置为 ignore_index（-100），以符合 HF 最佳实践
            current.extend([ignore_index] * pad_len)  # 使用 ignore_index 作为填充标签
            samples.append(torch.tensor(current, dtype=torch.long))
            current = [] # 清空缓存，准备新样本

        current.extend(ids) # 将整个句子放入 current
    # 末尾不足 (seq_len-1) 的补零（使用 ignore_index），因为移位后有效长度是 seq_len-1
    if current:
        pad_len = (seq_len - 1) - len(current)
        current.extend([ignore_index] * pad_len)
        samples.append(torch.tensor(current, dtype=torch.long))
    return samples


class LineModeWikitext2Dataset(Dataset):
    """
    行模式数据集：每个样本尽量落在自然句边界内，
    再通过 EOF 拼接填满 256，最大限度减少语义断裂。
    """
    def __init__(self, samples: list[torch.Tensor]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.samples[idx]          # 长度已固定为 seq_len
        # 自回归移位：source 取[:-1]，target 取[1:]
        source = seq[:-1]
        target = seq[1:]
        # mask 只覆盖有效位置，长度同步减 1
        mask = torch.ones_like(source, dtype=torch.float)
        return {"source": source, "target": target, "mask": mask}


class Wikitext2Task(BaseTask):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.sequence_length = config["model"]["sequence_length"]
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"]["num_workers"]
        self.tokenizer = get_or_train_tokenizer(config)
        self.config["model"]["vocabulary_size"] = self.tokenizer.get_vocab_size()

    def _prepare_dataset(self, split: str) -> LineModeWikitext2Dataset:
        """
        缓存键加入行模式标识，避免旧缓存干扰。
        步骤：
        1. 按文章遍历；
        2. 每篇文章按句切分；
        3. 全局收集所有句子；
        4. 统一做 EOF 拼接 → 256 块；
        5. 返回行模式数据集。
        """
        cache_dir = Path("./data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"wikitext2_{split}_line_mode_ids.pt"

        if cache_file.exists():
            print(f"Loading cached line-mode IDs for '{split}' split...")
            samples = torch.load(cache_file)
        else:
            print(f"No cache found. Building line-mode '{split}' split...")
            raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            eos_id = self.tokenizer.token_to_id("<|endoftext|>")
            if eos_id is None:
                eos_id = self.tokenizer.get_vocab_size() - 1

            all_sentences = []
            for item in raw_dataset:
                text = item['text']
                if not text or text.isspace():
                    continue
                # 按句切分并收集
                all_sentences.extend(split_into_sentences(text))

            # 统一做 EOF 拼接，使用 ignore_index=-100 作为填充标签
            pad_id = 0  # 假设 0 是 pad_id
            samples = build_line_mode_samples(self.tokenizer, all_sentences, self.sequence_length, eos_id, pad_id=pad_id, ignore_index=-100)
            print(f"Saving {len(samples)} line-mode samples to cache: {cache_file}")
            torch.save(samples, cache_file)

        return LineModeWikitext2Dataset(samples)

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        train_dataset = self._prepare_dataset("train")
        valid_dataset = self._prepare_dataset("validation")

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=True
        )
        return train_loader, valid_loader

    def get_model(self) -> nn.Module:
        from model.nano_gpt import MiniGPT1
        model = MiniGPT1(
            vocabulary_size=self.tokenizer.get_vocab_size(),
            embedding_size=self.config["model"]["embedding_size"],
            sequence_length=self.sequence_length,
            num_heads=self.config["model"]["num_heads"],
            num_layers=self.config["model"]["num_layers"],
            learn_embeddings=True,
        )
        return model

    def get_criterion(self) -> nn.Module:
        # 使用 ignore_index=-100 来忽略填充部分，以符合 Hugging Face 最佳实践
        return nn.NLLLoss(ignore_index=-100)

    def get_param_groups(self, model: nn.Module) -> list[dict]:
        """
        保持与旧版一致的参数分组逻辑。
        """
        hidden_weights = [
            p for n, p in model.named_parameters()
            if p.ndim >= 2 and 'transformer.h' in n
        ]
        non_hidden_weights = [
            p for n, p in model.named_parameters()
            if not (p.ndim >= 2 and 'transformer.h' in n)
        ]
        param_groups = [
            {'params': hidden_weights, 'use_diag_fog': True},
            {'params': non_hidden_weights, 'use_diag_fog': False},
        ]
        return param_groups

    @contextmanager
    def _maybe_efficient_attention(self, needs_second_order: bool):
        if needs_second_order:
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                yield
        else:
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                yield

    def train_step(self, model: nn.Module, batch: Any, criterion: nn.Module,
                   optimizer: torch.optim.Optimizer, device: torch.device,
                   needs_second_order: bool) -> tuple[torch.Tensor, float, dict[str, float]]:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        with self._maybe_efficient_attention(needs_second_order):
            log_probas = model(batch["source"])
            # 移位后 log_probas 长度=seq_len-1，与 target 对齐
            batch_size, seq_len_minus_1, vocab_size = log_probas.shape
            log_probas_flat = log_probas.view(-1, vocab_size)
            target_flat = batch["target"].view(-1)
            loss = criterion(log_probas_flat, target_flat)
            
            # Step-level 断言：如果损失为0或NaN，直接报错退出
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"NaN/Inf loss detected: {loss.item()}")
            if loss.item() == 0.0:
                raise RuntimeError(f"Zero loss detected: {loss.item()}")
                
        loss.backward(create_graph=needs_second_order)
        optimizer.step()
        return log_probas.detach(), loss.item(), {}

    def validate_epoch(self, model: nn.Module, valid_loader: DataLoader,
                       criterion: nn.Module, device: torch.device) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                log_probas = model(batch["source"])
                # 移位后长度同步减 1（变量名已体现）
                batch_size, seq_len_minus_1, vocab_size = log_probas.shape
                log_probas_flat = log_probas.view(-1, vocab_size)
                target_flat = batch["target"].view(-1)
                loss = criterion(log_probas_flat, target_flat)
                
                # Step-level 断言：如果损失为0或NaN，直接报错退出
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError(f"NaN/Inf loss detected in validation: {loss.item()}")
                if loss.item() == 0.0:
                    raise RuntimeError(f"Zero loss detected in validation: {loss.item()}")
                    
                # 只统计非 -100 的 token 数，与 NLLLoss 内部口径一致
                valid_mask = (target_flat != -100)
                total_loss += loss.item() * valid_mask.sum().item()
                total_tokens += valid_mask.sum().item()
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss > 0 else float('inf')
        return {"loss": avg_loss, "perplexity": perplexity}
