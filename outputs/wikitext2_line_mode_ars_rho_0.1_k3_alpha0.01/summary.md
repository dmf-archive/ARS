# ARS-Bench Experiment Report

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
    "name": "ARS",
    "lr": 0.0001,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.1,
    "ns_steps": 5,
    "rho": 0.1,
    "k": 3,
    "alpha": 0.01
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
| 1 | wikitext2_line | 5.5372 | 0.000100 | 0.433 | N/A | 0.000 | 0.8374 | 588.71 | 3108.9 | 5.06 | 157.84 |
| 2 | wikitext2_line | 4.6057 | 0.000100 | 0.495 | N/A | 0.000 | 0.7040 | 587.96 | 3583.5 | 4.70 | 110.12 |
| 3 | wikitext2_line | 4.1910 | 0.000100 | 0.479 | N/A | 0.000 | 0.7368 | 588.49 | 3412.5 | 4.53 | 93.08 |
| 4 | wikitext2_line | 3.8589 | 0.000100 | 0.454 | N/A | 0.000 | 0.7888 | 589.80 | 3577.7 | 4.46 | 86.16 |
| 5 | wikitext2_line | 3.5514 | 0.000100 | 0.426 | N/A | 0.000 | 0.8531 | 592.49 | 3110.4 | 4.44 | 84.71 |

## Performance Summary
- **Best Validation Metrics**: wikitext2_line Loss: 5.06, wikitext2_line Perplexity: 84.71
- **Final Validation Metrics**: wikitext2_line: {"loss": 4.439290528866782, "perplexity": 84.7148176305487}
