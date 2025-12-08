# F3EO-Bench Experiment Report

## Configuration Summary
```json
{
  "experiment": {
    "tasks": [
      "wikitext2_line"
    ],
    "seed": 42,
    "device": "cuda"
  },
  "model": {
    "type": "rope",
    "vocabulary_size": 40479,
    "embedding_size": 512,
    "sequence_length": 255,
    "num_heads": 6,
    "num_layers": 4,
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
    "name": "AdaRMSuon",
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
  },
  "adaptive_wd": {
    "enabled": true,
    "mode": "pcwd",
    "ema_beta": 0.95,
    "alpha": 0.1,
    "lambda_min": 0.1,
    "lambda_max": 10
  }
}
```

## Training Results
| Epoch | Task | Train Loss | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Eval Loss | Eval Perplexity |
|-------|------|------------|----|----|------------|---------|-----------|----------------|-------------------|-----------|-----------------|
| 1 | wikitext2_line | 5.7146 | 0.000100 | 0.422 | N/A | 0.000 | 0.8629 | 483.92 | 2787.7 | 5.29 | 198.15 |
| 2 | wikitext2_line | 4.9304 | 0.000100 | 0.515 | N/A | 0.000 | 0.6633 | 535.30 | 2786.9 | 4.94 | 140.08 |
| 3 | wikitext2_line | 4.6117 | 0.000100 | 0.506 | N/A | 0.000 | 0.6816 | 516.31 | 2786.9 | 4.77 | 117.44 |
| 4 | wikitext2_line | 4.3630 | 0.000100 | 0.475 | N/A | 0.000 | 0.7433 | 490.45 | 2786.9 | 4.72 | 112.47 |
| 5 | wikitext2_line | 4.2389 | 0.000100 | 0.428 | N/A | 0.000 | 0.8491 | 486.69 | 2786.9 | 4.71 | 110.58 |
| 6 | wikitext2_line | 4.0827 | 0.000100 | 0.378 | N/A | 0.000 | 0.9722 | 485.64 | 2786.9 | 4.69 | 108.78 |
| 7 | wikitext2_line | 3.9183 | 0.000100 | 0.324 | N/A | 0.000 | 1.1260 | 483.88 | 2786.9 | 4.74 | 114.91 |
| 8 | wikitext2_line | 3.8502 | 0.000100 | 0.267 | N/A | 0.000 | 1.3198 | 484.33 | 2786.9 | 4.81 | 122.43 |
| 9 | wikitext2_line | 3.7266 | 0.000100 | 0.222 | N/A | 0.000 | 1.5060 | 483.04 | 2786.9 | 4.86 | 129.14 |
| 10 | wikitext2_line | 3.6111 | 0.000100 | 0.185 | N/A | 0.000 | 1.6855 | 482.10 | 2786.9 | 4.93 | 137.85 |

## Performance Summary
- **Best Validation Metrics**: wikitext2_line Loss: 5.29, wikitext2_line Perplexity: 108.78
- **Final Validation Metrics**: wikitext2_line: {"loss": 4.926202153092, "perplexity": 137.85496484292494}
