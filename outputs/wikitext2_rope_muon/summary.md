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
    "num_workers": 4,
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
    "epochs": 5,
    "log_every": 10,
    "ckpt_every": 2
  }
}
```

## Training Results
| Epoch | Task | Train Loss | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Eval Loss | Eval Perplexity |
|-------|------|------------|----|----|------------|---------|-----------|----------------|-------------------|-----------|-----------------|
| 1 | wikitext2 | 6.4466 | 0.000100 | 0.001 | N/A | 6.382 | 1.1270 | 622.42 | 2908.4 | 6.16 | 472.96 |
| 2 | wikitext2 | 5.7147 | 0.000100 | 0.001 | N/A | 5.840 | 1.2335 | 668.55 | 2904.3 | 5.98 | 396.29 |
| 3 | wikitext2 | 5.4238 | 0.000100 | 0.001 | N/A | 5.603 | 1.1529 | 671.47 | 2904.3 | 5.86 | 349.06 |
| 4 | wikitext2 | 5.2731 | 0.000100 | 0.001 | N/A | 5.471 | 1.2401 | 675.78 | 2904.3 | 5.82 | 336.56 |
| 5 | wikitext2 | 5.1617 | 0.000100 | 0.001 | N/A | 5.363 | 1.3198 | 668.91 | 2904.3 | 5.80 | 330.07 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 6.16, wikitext2 Perplexity: 330.07
- **Final Validation Metrics**: wikitext2: {"loss": 5.799294689605976, "perplexity": 330.066678333455}
