# F3EO-Bench Experiment Report

## Configuration Summary
| Parameter | Value |
|-|-------|
| Task | wikitext2 |
| Model | nano_gpt |
| Optimizer | F3EO_raw |
| Learning Rate | 0.0001 |
| Weight Decay | 0.1 |
| Epochs | 30 |
| Batch Size | 1 |
| Device | cuda |
| Seed | 42 |

## Training Results
| Epoch | Train Loss | Valid Loss | Train Perplexity | Valid Perplexity | Learning Rate | PI | Eff. Gamma | Entropy | Time |
|-----|--|-----|-----|-----|-----|----|---|---|------|
| 0 | 11.7755 | 7.5550 | 130031.46 | 1910.23 | 0.000100 | N/A | N/A | N/A | 2068.49s |
| 2 | 6.9130 | 6.7685 | 1005.27 | 869.98 | 0.000100 | 0.000 | 0.000 | 6.157 | 2158.81s |

## Performance Summary
- **Best Validation Perplexity**: 869.98
- **Final Validation Perplexity**: 869.98
- **Total Training Time**: 2161.58s
- **Average Epoch Time**: 2113.65s

## Configuration Details
```toml
{
  "experiment": {
    "task": "wikitext2",
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
    "batch_size": 1,
    "num_workers": 4,
    "tokenizer_path": "./data/wikitext2_tokenizer.json"
  },
  "optimizer": {
    "name": "F3EO_raw",
    "lr": 0.0001,
    "weight_decay": 0.1,
    "betas": [
      0.9,
      0.999
    ],
    "gamma": 1.0,
    "alpha": 1.0
  },
  "train": {
    "epochs": 30,
    "log_every": 10,
    "ckpt_every": 2
  },
  "early_stop": {
    "patience": 10,
    "threshold": 1.0
  }
}
```
