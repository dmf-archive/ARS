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
    "name": "Muon",
    "lr": 0.0001,
    "weight_decay": 0.1,
    "momentum": 0.95,
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
| 1 | wikitext2 | 15.5168 | 0.000100 | 0.000 | N/A | 5.753 | 14.4058 | 348.25 | 2985.0 | 8.46 | 4728.78 |
| 2 | wikitext2 | 7.8485 | 0.000100 | 0.000 | N/A | 7.257 | 2.8864 | 348.09 | 2985.0 | 7.63 | 2062.82 |
| 3 | wikitext2 | 7.5480 | 0.000100 | 0.000 | N/A | 7.460 | 2.0731 | 343.64 | 2985.0 | 7.65 | 2092.02 |
| 4 | wikitext2 | 7.5205 | 0.000100 | 0.000 | N/A | 7.472 | 2.3874 | 343.76 | 2985.0 | 7.59 | 1983.32 |
| 5 | wikitext2 | 7.5035 | 0.000100 | 0.000 | N/A | 7.476 | 1.7627 | 343.16 | 2985.0 | 7.56 | 1918.75 |
| 6 | wikitext2 | 7.4956 | 0.000100 | 0.000 | N/A | 7.476 | 1.1886 | 341.87 | 2985.0 | 7.56 | 1918.32 |
| 7 | wikitext2 | 7.4917 | 0.000100 | 0.000 | N/A | 7.475 | 1.1041 | 341.30 | 2985.0 | 7.56 | 1924.59 |
| 8 | wikitext2 | 7.3612 | 0.000100 | 0.000 | N/A | 7.346 | 0.9498 | 340.50 | 2985.0 | 7.41 | 1646.62 |
| 9 | wikitext2 | 7.3810 | 0.000100 | 0.000 | N/A | 7.367 | 1.0397 | 342.86 | 2985.0 | 7.46 | 1733.97 |
| 10 | wikitext2 | 7.3711 | 0.000100 | 0.000 | N/A | 7.358 | 1.0586 | 341.86 | 2985.0 | 7.46 | 1738.48 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 8.46, wikitext2 Perplexity: 1646.62
- **Final Validation Metrics**: wikitext2: {"loss": 7.4607657522990785, "perplexity": 1738.4787919091063}
