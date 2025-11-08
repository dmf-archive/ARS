# F3EO-Bench Experiment Report

## Configuration Summary
| Parameter | Value |
|-|-------|
| Task | wikitext2 |
| Model | nano_gpt |
| Optimizer | F3EWD |
| Learning Rate | 0.0001 |
| Weight Decay | 0.1 |
| Epochs | 30 |
| Batch Size | 8 |
| Device | cuda |
| Seed | 42 |

## Training Results
| Epoch | Train Loss | Valid Loss | Train Perplexity | Valid Perplexity | Learning Rate | PI | Eff. Gamma | Entropy | Time |
|-----|--|-----|-----|-----|-----|----|---|---|------|
| 1 | 26.0941 | 13.0434 | 215047956080.83 | 462046.52 | 0.000100 | 0.000 | 0.000 | 6.835 | 806.65s |
| 2 | 10.8916 | 9.3808 | 53721.42 | 11858.20 | 0.000100 | 0.000 | -0.000 | 6.086 | 853.50s |
| 3 | 8.6966 | 8.3644 | 5982.45 | 4291.69 | 0.000100 | 0.000 | -0.000 | 6.125 | 857.78s |
| 4 | 7.7712 | 8.0244 | 2371.33 | 3054.56 | 0.000100 | 0.000 | -0.000 | 5.305 | 859.21s |
| 5 | 7.2867 | 7.2302 | 1460.76 | 1380.49 | 0.000100 | 0.000 | 0.000 | 6.179 | 863.95s |

## Performance Summary
- **Best Validation Perplexity**: 1380.49
- **Final Validation Perplexity**: 1380.49
- **Total Training Time**: 4254.64s
- **Average Epoch Time**: 848.22s

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
    "batch_size": 8,
    "num_workers": 4,
    "tokenizer_path": "./data/wikitext2_tokenizer.json"
  },
  "optimizer": {
    "name": "F3EWD",
    "lr": 0.0001,
    "weight_decay": 0.1,
    "betas": [
      0.5,
      0.999
    ],
    "gamma": 0.5
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
