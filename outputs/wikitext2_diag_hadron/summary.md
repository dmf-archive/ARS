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
    "name": "DiagHadron",
    "lr": 0.0001,
    "momentum": 0.9,
    "stat_decay": 0.95,
    "damping": 0.001,
    "kl_clip": 0.001,
    "weight_decay": 0.1,
    "TCov": 10,
    "TInv": 100,
    "muon_momentum": 0.95,
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
| 1 | wikitext2 | 15.4198 | 0.000100 | 0.000 | N/A | 5.747 | 14.5289 | 254.71 | 2984.8 | 8.46 | 4729.79 |
| 2 | wikitext2 | 7.9381 | 0.000100 | 0.000 | N/A | 7.238 | 2.4334 | 261.91 | 2984.8 | 7.72 | 2253.24 |
| 3 | wikitext2 | 7.5554 | 0.000100 | 0.000 | N/A | 7.455 | 2.2296 | 386.38 | 2984.8 | 7.57 | 1939.01 |
| 4 | wikitext2 | 7.5209 | 0.000100 | 0.000 | N/A | 7.472 | 2.2238 | 342.51 | 2985.3 | 7.61 | 2021.46 |
| 5 | wikitext2 | 7.4982 | 0.000100 | 0.000 | N/A | 7.464 | 1.9460 | 342.57 | 2985.3 | 7.58 | 1955.16 |
| 6 | wikitext2 | 7.4862 | 0.000100 | 0.000 | N/A | 7.457 | 1.6242 | 342.80 | 2985.3 | 7.61 | 2019.93 |
| 7 | wikitext2 | 7.4959 | 0.000100 | 0.000 | N/A | 7.477 | 1.2138 | 342.54 | 2985.3 | 7.56 | 1920.04 |
| 8 | wikitext2 | 7.4897 | 0.000100 | 0.000 | N/A | 7.477 | 0.9340 | 342.54 | 2985.3 | 7.55 | 1908.71 |
| 9 | wikitext2 | 7.4886 | 0.000100 | 0.000 | N/A | 7.475 | 0.9861 | 340.96 | 2985.3 | 7.56 | 1916.40 |
| 10 | wikitext2 | 7.4860 | 0.000100 | 0.000 | N/A | 7.475 | 0.8917 | 341.82 | 2985.3 | 7.56 | 1922.31 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 8.46, wikitext2 Perplexity: 1908.71
- **Final Validation Metrics**: wikitext2: {"loss": 7.561282832047035, "perplexity": 1922.3099331038097}
