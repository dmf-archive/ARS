# F3EO-Bench Experiment Report

## Configuration Summary
| Parameter | Value |
|-|-------|
| Task | wikitext2 |
| Model | nano_gpt |
| Optimizer | AdamW |
| Learning Rate | 0.0001 |
| Weight Decay | 0.01 |
| Epochs | 60 |
| Batch Size | 8 |
| Device | cuda |
| Seed | 42 |

## Training Results
| Epoch | Train Loss | Valid Loss | Train Perplexity | Valid Perplexity | Learning Rate | Log(PI) | Time |
|-----|--|-----|-----|-----|-----|--------|------|
| 1 | 25.0486 | 13.1442 | 75588240482.44 | 511060.04 | 0.000100 | N/A | 294.22s |
| 2 | 10.8652 | 9.3341 | 52321.06 | 11317.42 | 0.000100 | N/A | 294.11s |
| 3 | 8.4716 | 8.1507 | 4777.01 | 3465.76 | 0.000100 | N/A | 293.78s |
| 4 | 7.6229 | 7.5500 | 2044.52 | 1900.82 | 0.000100 | N/A | 293.68s |
| 5 | 7.1459 | 7.1916 | 1268.90 | 1328.18 | 0.000100 | N/A | 293.49s |
| 6 | 6.8192 | 6.9609 | 915.22 | 1054.54 | 0.000100 | N/A | 293.31s |
| 7 | 6.5844 | 6.7984 | 723.75 | 896.41 | 0.000100 | N/A | 293.85s |

## Performance Summary
- **Best Validation Perplexity**: 896.41
- **Final Validation Perplexity**: 896.41
- **Total Training Time**: 2075.20s
- **Average Epoch Time**: 293.78s

## Configuration Details
```toml
{
  "experiment": {
    "task": "wikitext2",
    "seed": 42,
    "device": "cuda",
    "config_name": "wikitext2_adamw_classic"
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
    "num_workers": 4,
    "tokenizer_path": "./data/wikitext2_tokenizer.json"
  },
  "optimizer": {
    "name": "AdamW",
    "lr": 0.0001,
    "weight_decay": 0.01
  },
  "train": {
    "epochs": 60,
    "log_every": 10,
    "ckpt_every": 2
  },
  "early_stop": {
    "patience": 10,
    "threshold": 1.0
  }
}
```
