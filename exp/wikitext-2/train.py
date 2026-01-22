import argparse
import time
import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.nn.attention import SDPBackend, sdpa_kernel
import toml
from optimizer import get_optimizer
from utils.callbacks.console import ConsoleLogger
from utils.callbacks.markdown import MDLogger
from utils.callbacks.checkpoint import CheckpointSaver
from utils.callbacks.context import TrainerContext
from utils.data import MetricStore, StepMetric, EpochMetric, TaskMetrics
from utils.metrics import PICalculator, compute_grad_norm

def get_or_train_tokenizer(config):
    tokenizer_path = Path(config["data"]["tokenizer_path"])
    vocab_size = config["model"]["vocabulary_size"]
    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    def get_training_corpus():
        for i in range(0, len(dataset), 1000):
            yield [text for text in dataset[i : i + 1000]['text'] if text.strip()]
            
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>", "<pad>"])
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    return tokenizer

def pack_sequences_greedy(texts, tokenizer, max_length):
    all_sentences = []
    for text in texts:
        if not text or not text.strip(): continue
        for sent in text.replace('\n', ' ').split('.'):
            sent = sent.strip()
            if sent:
                tokens = tokenizer.encode(sent + ".").ids + [tokenizer.token_to_id("<|endoftext|>")]
                if 1 < len(tokens) <= max_length: all_sentences.append(tokens)
    all_sentences.sort(key=len, reverse=True)
    packs, current_pack = [], []
    for tokens in all_sentences:
        if len(current_pack) + len(tokens) <= max_length: current_pack.extend(tokens)
        else:
            if current_pack: packs.append(current_pack)
            current_pack = list(tokens)
    if current_pack: packs.append(current_pack)
    return packs

class Wikitext2Dataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return self.samples.size(0)
    def __getitem__(self, idx):
        seq = self.samples[idx]
        return {"source": seq[:-1], "target": seq[1:], "mask": torch.ones(seq.size(0)-1, dtype=torch.float)}

def get_dataloaders(config, tokenizer):
    seq_len = config["model"]["sequence_length"]
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    
    cache_dir = Path("./data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    loaders = []
    for split in ["train", "validation"]:
        cache_file = cache_dir / f"wikitext2_{split}_line_pack_ids_v3.pt"
        if cache_file.exists():
            samples = torch.load(cache_file)
        else:
            raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            texts = [item['text'] for item in raw if item['text'] and not item['text'].isspace()]
            packed = pack_sequences_greedy(texts, tokenizer, seq_len + 1)
            pad_id = tokenizer.token_to_id("<pad>") or 0
            samples_list = []
            for seq in packed:
                seq = seq[:seq_len+1]
                samples_list.append(torch.tensor(seq + [pad_id] * ((seq_len+1) - len(seq)), dtype=torch.long))
            samples = torch.stack(samples_list) if samples_list else torch.empty(0, seq_len+1, dtype=torch.long)
            torch.save(samples, cache_file)
        loaders.append(DataLoader(Wikitext2Dataset(samples), batch_size=batch_size, shuffle=(split=="train"), num_workers=num_workers, pin_memory=True, drop_last=True))
    return loaders[0], loaders[1]

def train_step(model, batch, criterion, device, needs_second_order=False, **kwargs):
    batch = {k: v.to(device) for k, v in batch.items()}
    backends = [SDPBackend.MATH] if needs_second_order else [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    with sdpa_kernel(backends=backends):
        logits = model(batch["source"])
        loss = criterion(logits.transpose(1, 2), batch["target"])
        if torch.isnan(loss) or torch.isinf(loss): raise RuntimeError(f"NaN/Inf loss: {loss.item()}")
    return logits, loss

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["source"])
            loss = criterion(logits.transpose(1, 2), batch["target"])
            total_loss += loss.item() * batch["target"].numel()
            total_tokens += batch["target"].numel()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return {"loss": avg_loss, "perplexity": math.exp(avg_loss) if avg_loss > 0 else float('inf')}

def main():
    parser = argparse.ArgumentParser(description="High-Fidelity Atomic WikiText-2 Training")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = toml.load(args.config)
    device = torch.device(config["experiment"]["device"])
    torch.manual_seed(config["experiment"]["seed"])
    if torch.cuda.is_available(): torch.cuda.manual_seed(config["experiment"]["seed"])
    
    output_dir = Path("outputs") / Path(args.config).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = get_or_train_tokenizer(config)
    config["model"]["vocabulary_size"] = tokenizer.get_vocab_size()
    train_loader, valid_loader = get_dataloaders(config, tokenizer)
    
    model_type = config["model"].get("type", "nano_gpt")
    if model_type == "rope":
        from model.qwen3_rope import Qwen3RoPEWrapper
        model = Qwen3RoPEWrapper(vocabulary_size=tokenizer.get_vocab_size(), hidden_size=config["model"]["embedding_size"],
                                 num_hidden_layers=config["model"]["num_layers"], num_attention_heads=config["model"]["num_heads"],
                                 num_key_value_heads=config["model"]["num_heads"], max_position_embeddings=config["model"]["sequence_length"],
                                 rope_theta=config["model"]["rope_theta"], intermediate_size=config["model"]["intermediate_size"],
                                 tie_word_embeddings=config["model"]["tie_word_embeddings"])
    else:
        from model.nano_gpt import MiniGPT1
        model = MiniGPT1(vocabulary_size=tokenizer.get_vocab_size(), embedding_size=config["model"]["embedding_size"],
                         sequence_length=config["model"]["sequence_length"], num_heads=config["model"]["num_heads"],
                         num_layers=config["model"]["num_layers"], learn_embeddings=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    h_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and ('transformer.h' in n or 'model.layers' in n)]
    nh_params = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and ('transformer.h' in n or 'model.layers' in n))]
    param_groups = [{'params': h_params, 'use_muon': True}, {'params': nh_params, 'use_muon': False}]

    opt_cfg = config["optimizer"].copy()
    smart_opt = get_optimizer(opt_cfg.pop("name"), param_groups, model=model, criterion=criterion, device=device, **opt_cfg)

    store = MetricStore()
    context = TrainerContext(config=config, output_dir=output_dir, device=device, model=model, 
                             optimizer=smart_opt.optimizer, store=store, total_epochs=config["experiment"]["epochs"])
    
    pi_cfg = config.get("pi", {"gamma": 1.0, "alpha": 1.0, "ema_beta": 0.9})
    pi_calculator = PICalculator(gamma=pi_cfg.get("gamma", 1.0), alpha=pi_cfg.get("alpha", 1.0), ema_beta=pi_cfg.get("ema_beta"))
    callbacks = [ConsoleLogger(), MDLogger(), CheckpointSaver()]
    def broadcast(event):
        for cb in callbacks: getattr(cb, event)(context)

    resumed = False
    for cb in callbacks:
        if isinstance(cb, CheckpointSaver):
            if cb.load(context):
                resumed = True
                break
    
    broadcast("on_train_begin")
    start_epoch = context.current_epoch + 1 if resumed else 0
    for epoch in range(start_epoch, config["experiment"]["epochs"]):
        context.current_epoch, context.current_task_name, context.is_training = epoch, "wikitext2", True
        context.total_steps_in_epoch = len(train_loader)
        model.train()
        broadcast("on_epoch_begin")
        
        epoch_start_time = time.time()
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
        epoch_loss_sum, epoch_entropy_sum, num_tokens, epoch_grad_norm_list = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), 0, []

        for step, batch in enumerate(train_loader):
            context.current_step_in_epoch = step
            broadcast("on_step_begin")
            logits, loss = smart_opt.step(batch, train_step)
            with torch.no_grad():
                nt = batch["target"].numel()
                epoch_loss_sum += loss * nt
                if logits is not None:
                    lgt_flat = logits.view(-1, logits.size(-1))
                    probas = torch.softmax(lgt_flat, dim=-1)
                    epoch_entropy_sum += -(probas * torch.log_softmax(lgt_flat, dim=-1)).sum()
                num_tokens += nt
                gn = compute_grad_norm(model, return_tensor=True)
                if gn is not None: epoch_grad_norm_list.append(gn)

            context.store.add_step(StepMetric(task_name="wikitext2", global_step=context.global_step, task_epoch=epoch,
                                              step_in_epoch=step, loss=loss.item(), learning_rate=smart_opt.param_groups[0]['lr']))
            broadcast("on_step_end")
            context.global_step += 1

        context.is_training = False
        val_metrics = validate_epoch(model, valid_loader, criterion, device)
        avg_train_loss = (epoch_loss_sum / num_tokens).item() if num_tokens > 0 else 0.0
        avg_gn = torch.stack(epoch_grad_norm_list).mean().item() if epoch_grad_norm_list else None
        avg_entropy, avg_pi = None, None
        if num_tokens > 0:
            avg_entropy_tensor = epoch_entropy_sum / num_tokens
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

        context.store.add_epoch(EpochMetric(task_name="wikitext2", task_epoch=epoch, global_epoch=epoch, avg_train_loss=avg_train_loss,
                                            task_metrics=TaskMetrics(metrics=val_metrics), avg_pi_obj=avg_pi, avg_entropy=avg_entropy,
                                            grad_norm=avg_gn, learning_rate=smart_opt.param_groups[0]['lr'], diagnostics=diagnostics,
                                            epoch_time_s=time.time() - epoch_start_time, peak_gpu_mem_mb=torch.cuda.max_memory_allocated() / (1024**2) if device.type == 'cuda' else None))
        broadcast("on_epoch_end")
        broadcast("save")
    broadcast("on_train_end")

if __name__ == "__main__":
    main()
