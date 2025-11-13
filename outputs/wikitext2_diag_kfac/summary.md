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
    "sequence_length": 256,
    "num_heads": 12,
    "num_layers": 4
  },
  "data": {
    "batch_size": 8,
    "num_workers": 0,
    "tokenizer_path": "./data/wikitext2_tokenizer.json"
  },
  "optimizer": {
    "name": "DiagKFAC",
    "lr": 0.0001,
    "momentum": 0.9,
    "stat_decay": 0.95,
    "damping": 0.001,
    "kl_clip": 0.001,
    "weight_decay": 0.1,
    "TCov": 10,
    "TInv": 100
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
| 1 | wikitext2 | 74.8298 | 0.000100 | 0.000 | N/A | 4.923 | 530.6935 | 257.61 | 2563.5 | 9.11 | 9068.63 |
| 2 | wikitext2 | 8.8098 | 0.000100 | 0.000 | N/A | 8.669 | 479.9306 | 263.13 | 2563.5 | 8.66 | 5742.19 |
| 3 | wikitext2 | 8.3377 | 0.000100 | 0.000 | N/A | 8.317 | 432.7895 | 262.79 | 2563.5 | 8.30 | 4018.46 |
| 4 | wikitext2 | 8.0601 | 0.000100 | 0.000 | N/A | 8.048 | 387.0759 | 263.18 | 2563.5 | 8.10 | 3278.64 |
| 5 | wikitext2 | 7.8907 | 0.000100 | 0.000 | N/A | 7.881 | 345.6761 | 263.69 | 2563.5 | 7.99 | 2952.44 |
| 6 | wikitext2 | 7.8121 | 0.000100 | 0.000 | N/A | 7.807 | 309.7520 | 263.64 | 2563.5 | 7.93 | 2776.93 |
| 7 | wikitext2 | 7.7568 | 0.000100 | 0.000 | N/A | 7.758 | 278.5189 | 320.20 | 2563.5 | 7.88 | 2645.00 |
| 8 | wikitext2 | 7.7107 | 0.000100 | 0.000 | N/A | 7.717 | 251.1847 | 321.37 | 2563.5 | 7.84 | 2540.74 |
| 9 | wikitext2 | 7.6690 | 0.000100 | 0.000 | N/A | 7.681 | 228.2978 | 321.03 | 2563.5 | 7.80 | 2440.76 |
| 10 | wikitext2 | 7.6266 | 0.000100 | 0.000 | N/A | 7.643 | 210.1726 | 320.80 | 2563.5 | 7.76 | 2345.99 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 9.11, wikitext2 Perplexity: 2345.99
- **Final Validation Metrics**: wikitext2: {"loss": 7.760462646810417, "perplexity": 2345.9897189050816}
