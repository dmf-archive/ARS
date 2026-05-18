# ARS-Bench Experiment Report

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
    "embedding_size": 128,
    "sequence_length": 128,
    "num_heads": 4,
    "num_layers": 2,
    "rope_theta": 10000.0,
    "intermediate_size": 256,
    "tie_word_embeddings": true
  },
  "data": {
    "batch_size": 16,
    "num_workers": 0,
    "tokenizer_path": "./data/wikitext2_tokenizer.json"
  },
  "optimizer": {
    "name": "AdamW",
    "lr": 0.0005,
    "betas": [
      0.9,
      0.95
    ],
    "eps": 1e-08,
    "weight_decay": 0.1
  }
}
```

## Training Results
| Epoch | Task | Train Loss | Min Loss | Min Step | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Diag group_0_muon_avg_norm | Diag group_1_adam_avg_norm | Eval Loss | Eval Perplexity |
|-------|------|------------|----------|----------|----|----|------------|---------|-----------|----------------|-------------------|----------------------------|----------------------------|-----------|-----------------|
| 1 | wikitext2 | 6.1804 | N/A | N/A | 0.000500 | 0.000 | N/A | 6.722 | 0.9875 | 151.28 | 4063.3 | 5.0747 | 30.9561 | 5.55 | 257.24 |
| 2 | wikitext2 | 5.2535 | N/A | N/A | 0.000500 | 0.001 | N/A | 5.511 | 1.0867 | 151.24 | 4063.3 | 5.4806 | 38.7445 | 5.25 | 190.13 |
| 3 | wikitext2 | 4.9577 | N/A | N/A | 0.000500 | 0.001 | N/A | 5.243 | 1.1386 | 151.19 | 4063.3 | 5.8489 | 43.2807 | 5.09 | 162.58 |
| 4 | wikitext2 | 4.7665 | N/A | N/A | 0.000500 | 0.001 | N/A | 5.070 | 1.2083 | 151.32 | 4063.3 | 6.1830 | 46.0465 | 4.99 | 146.23 |
| 5 | wikitext2 | 4.6261 | N/A | N/A | 0.000500 | 0.002 | N/A | 4.943 | 1.3070 | 151.38 | 4063.3 | 6.4872 | 47.9002 | 4.93 | 138.29 |
| 6 | wikitext2 | 4.5185 | N/A | N/A | 0.000500 | 0.002 | N/A | 4.846 | 1.4101 | 151.36 | 4063.3 | 6.7708 | 49.2706 | 4.90 | 133.92 |
| 7 | wikitext2 | 4.4289 | N/A | N/A | 0.000500 | 0.002 | N/A | 4.766 | 1.4736 | 151.31 | 4063.3 | 7.0290 | 50.2672 | 4.86 | 129.52 |
| 8 | wikitext2 | 4.3560 | N/A | N/A | 0.000500 | 0.002 | N/A | 4.700 | 1.5674 | 151.33 | 4063.3 | 7.2725 | 51.0137 | 4.84 | 125.90 |
| 9 | wikitext2 | 4.2953 | N/A | N/A | 0.000500 | 0.002 | N/A | 4.645 | 1.6740 | 151.25 | 4063.3 | 7.4960 | 51.6368 | 4.83 | 125.19 |
| 10 | wikitext2 | 4.2434 | N/A | N/A | 0.000500 | 0.002 | N/A | 4.598 | 1.7505 | 151.34 | 4063.3 | 7.7060 | 52.1267 | 4.82 | 124.17 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 5.55, wikitext2 Perplexity: 124.17
- **Final Validation Metrics**: wikitext2: {"loss": 4.821660660985691, "perplexity": 124.17112579947879}
