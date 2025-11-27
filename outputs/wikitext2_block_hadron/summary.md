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
    "name": "BlockHadron",
    "lr": 0.0001,
    "momentum": 0.9,
    "stat_decay": 0.95,
    "damping": 0.001,
    "kl_clip": 0.001,
    "weight_decay": 0.1,
    "TCov": 10,
    "TInv": 100,
    "muon_momentum": 0.95,
    "block_size": 64,
    "adam_lr": 0.001,
    "adam_weight_decay": 0.1,
    "adam_betas": [
      0.9,
      0.95
    ]
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
| 1 | wikitext2 | 15.3642 | 0.000100 | 0.000 | N/A | 5.759 | 14.5125 | 416.34 | 3007.8 | 8.46 | 4727.01 |
| 2 | wikitext2 | 7.8221 | 0.000100 | 0.000 | N/A | 7.237 | 2.3984 | 418.05 | 3007.8 | 7.73 | 2280.52 |
| 3 | wikitext2 | 7.5523 | 0.000100 | 0.000 | N/A | 7.455 | 2.1736 | 434.17 | 3007.8 | 7.57 | 1948.24 |
| 4 | wikitext2 | 7.5222 | 0.000100 | 0.000 | N/A | 7.471 | 2.2000 | 409.32 | 3007.8 | 7.60 | 2004.19 |
| 5 | wikitext2 | 7.5040 | 0.000100 | 0.000 | N/A | 7.476 | 1.6824 | 417.10 | 3007.8 | 7.56 | 1926.93 |
| 6 | wikitext2 | 7.4959 | 0.000100 | 0.000 | N/A | 7.475 | 1.2798 | 432.07 | 3007.8 | 7.56 | 1918.99 |
| 7 | wikitext2 | 7.4911 | 0.000100 | 0.000 | N/A | 7.475 | 1.1229 | 435.18 | 3007.8 | 7.56 | 1918.61 |
| 8 | wikitext2 | 7.4888 | 0.000100 | 0.000 | N/A | 7.475 | 1.0348 | 405.31 | 3007.8 | 7.55 | 1904.42 |
| 9 | wikitext2 | 7.4862 | 0.000100 | 0.000 | N/A | 7.475 | 0.9193 | 406.06 | 3007.8 | 7.56 | 1917.53 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 8.46, wikitext2 Perplexity: 1904.42
- **Final Validation Metrics**: wikitext2: {"loss": 7.558793709195894, "perplexity": 1917.5310176435894}
