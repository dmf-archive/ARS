# F3EO-Bench Experiment Report

## Configuration Summary
```json
{
  "experiment": {
    "tasks": [
      "wikitext2"
    ],
    "seed": 42,
    "device": "cuda"
  },
  "model": {
    "arch": "nano_gpt",
    "vocabulary_size": 40479,
    "embedding_size": 768,
    "sequence_length": 255,
    "num_heads": 12,
    "num_layers": 4
  },
  "data": {
    "batch_size": 8,
    "num_workers": 0,
    "tokenizer_path": "./data/wikitext2_tokenizer.json"
  },
  "optimizer": {
    "name": "RMSoun",
    "lr": 0.0001,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.1,
    "ns_steps": 5
  },
  "train": {
    "epochs": 10,
    "log_every": 10,
    "ckpt_every": 2
  }
}
```

## Training Results
| Epoch | Task | Train Loss | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Eval Loss | Eval Perplexity |
|-------|------|------------|----|----|------------|---------|-----------|----------------|-------------------|-----------|-----------------|
| 1 | wikitext2 | 14.9802 | 0.001000 | 0.000 | N/A | 5.653 | 34.9703 | 692.78 | 2984.5 | 8.29 | 3991.17 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 8.29, wikitext2 Perplexity: 3991.17
- **Final Validation Metrics**: wikitext2: {"loss": 8.291840475181054, "perplexity": 3991.173093951617}
