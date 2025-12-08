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
    "name": "DeltaLossEMA",
    "lr": 0.0001,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.1,
    "ema_decay": 0.999
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
| 1 | wikitext2_line | 11.5435 | 0.001000 | 0.145 | N/A | 0.000 | 1.9309 | 356.51 | 2787.3 | 6.84 | 938.91 |
| 2 | wikitext2_line | 6.4871 | 0.001000 | 0.637 | N/A | 0.000 | 0.4503 | 380.53 | 2786.4 | 6.33 | 562.29 |

## Performance Summary
- **Best Validation Metrics**: wikitext2_line Loss: 6.84, wikitext2_line Perplexity: 562.29
- **Final Validation Metrics**: wikitext2_line: {"loss": 6.332019307720127, "perplexity": 562.2908865527282}
