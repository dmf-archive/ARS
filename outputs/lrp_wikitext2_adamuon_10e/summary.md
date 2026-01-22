# F3EO-Bench Experiment Report

## Configuration Summary
```json
{
  "experiment": {
    "tasks": [
      "wikitext2"
    ],
    "seed": 42,
    "device": "cuda",
    "epochs": 10
  },
  "model": {
    "type": "rope",
    "vocabulary_size": 40479,
    "embedding_size": 512,
    "sequence_length": 255,
    "num_heads": 4,
    "num_layers": 3,
    "rope_theta": 10000.0,
    "intermediate_size": 2048,
    "tie_word_embeddings": true
  },
  "data": {
    "batch_size": 8,
    "num_workers": 0,
    "tokenizer_path": "./data/wikitext2_tokenizer.json"
  },
  "optimizer": {
    "name": "AdaMuon",
    "lr": 0.0001,
    "betas": [
      0.95,
      0.95
    ],
    "eps": 1e-08,
    "weight_decay": 0.1,
    "ns_steps": 5,
    "adam_lr": 0.001,
    "adam_weight_decay": 0.01,
    "adam_betas": [
      0.9,
      0.999
    ]
  },
  "train": {
    "log_every": 10,
    "ckpt_every": 5
  }
}
```

## Training Results
| Epoch | Task | Train Loss | Min Loss | Min Step | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Diag group_0_muon_avg_norm | Diag group_1_adam_avg_norm | Eval Loss | Eval Perplexity |
|-------|------|------------|----------|----------|----|----|------------|---------|-----------|----------------|-------------------|----------------------------|----------------------------|-----------|-----------------|
| 1 | wikitext2 | 6.7315 | N/A | N/A | 0.000100 | 0.000 | N/A | 7.222 | 3.1583 | 508.01 | 2673.4 | 14.4852 | 25.0562 | 5.64 | 280.46 |
| 2 | wikitext2 | 5.0898 | N/A | N/A | 0.000100 | 0.000 | N/A | 5.199 | 4.7492 | 521.60 | 2673.4 | 14.3840 | 26.1852 | 5.20 | 180.87 |
| 3 | wikitext2 | 4.4618 | N/A | N/A | 0.000100 | 0.000 | N/A | 4.625 | 5.2807 | 517.78 | 2673.4 | 14.3203 | 27.8346 | 5.10 | 163.70 |
| 4 | wikitext2 | 3.9677 | N/A | N/A | 0.000100 | 0.000 | N/A | 4.188 | 6.0789 | 518.63 | 2673.4 | 14.2817 | 29.5643 | 5.17 | 175.81 |
| 5 | wikitext2 | 3.5473 | N/A | N/A | 0.000100 | 0.000 | N/A | 3.813 | 7.1368 | 532.31 | 2673.4 | 14.2566 | 31.0549 | 5.36 | 212.95 |
| 6 | wikitext2 | 3.1786 | N/A | N/A | 0.000100 | 0.000 | N/A | 3.481 | 8.2339 | 528.34 | 2673.4 | 14.2378 | 32.2219 | 5.63 | 277.47 |
| 7 | wikitext2 | 2.8643 | N/A | N/A | 0.000100 | 0.000 | N/A | 3.192 | 9.3099 | 520.12 | 2673.4 | 14.2233 | 33.0934 | 5.91 | 369.53 |
| 8 | wikitext2 | 2.5991 | N/A | N/A | 0.000100 | 0.000 | N/A | 2.939 | 10.3826 | 522.76 | 2673.4 | 14.2100 | 33.7493 | 6.23 | 505.91 |
| 9 | wikitext2 | 2.3953 | N/A | N/A | 0.000100 | 0.000 | N/A | 2.737 | 11.4099 | 518.91 | 2673.4 | 14.1975 | 34.2083 | 6.48 | 649.85 |
| 10 | wikitext2 | 2.2459 | N/A | N/A | 0.000100 | 0.000 | N/A | 2.581 | 12.2799 | 516.15 | 2673.4 | 14.1847 | 34.5026 | 6.70 | 815.46 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 6.70, wikitext2 Perplexity: 163.70
- **Final Validation Metrics**: wikitext2: {"loss": 6.703746856148563, "perplexity": 815.4555027123806}
