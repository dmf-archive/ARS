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
    "name": "Muon",
    "lr": 0.02,
    "momentum": 0.95,
    "ns_steps": 5
  }
}
```

## Training Results
| Epoch | Task | Train Loss | Min Loss | Min Step | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Diag group_0_muon_avg_norm | Diag group_1_adam_avg_norm | Eval Loss | Eval Perplexity |
|-------|------|------------|----------|----------|----|----|------------|---------|-----------|----------------|-------------------|----------------------------|----------------------------|-----------|-----------------|
| 1 | wikitext2 | 6.3160 | N/A | N/A | 0.020000 | 0.000 | N/A | 7.195 | 0.6415 | 171.34 | 4060.6 | 41.7240 | 27.1420 | 5.52 | 248.96 |
| 2 | wikitext2 | 5.2221 | N/A | N/A | 0.020000 | 0.002 | N/A | 5.534 | 0.4309 | 170.83 | 4060.6 | 57.5148 | 34.9801 | 5.22 | 185.09 |
| 3 | wikitext2 | 4.9500 | N/A | N/A | 0.020000 | 0.002 | N/A | 5.250 | 0.4506 | 170.77 | 4060.6 | 70.7243 | 39.9569 | 5.06 | 158.16 |
| 4 | wikitext2 | 4.7722 | N/A | N/A | 0.020000 | 0.003 | N/A | 5.073 | 0.4814 | 171.66 | 4060.6 | 82.1969 | 43.2174 | 4.97 | 144.27 |
| 5 | wikitext2 | 4.6412 | N/A | N/A | 0.020000 | 0.003 | N/A | 4.945 | 0.5129 | 171.47 | 4060.6 | 92.6585 | 45.6184 | 4.92 | 136.49 |
| 6 | wikitext2 | 4.5371 | N/A | N/A | 0.020000 | 0.003 | N/A | 4.844 | 0.5511 | 170.77 | 4060.6 | 102.2433 | 47.5793 | 4.87 | 130.81 |
| 7 | wikitext2 | 4.4519 | N/A | N/A | 0.020000 | 0.004 | N/A | 4.762 | 0.5914 | 170.41 | 4060.6 | 111.1683 | 49.1776 | 4.85 | 127.95 |
| 8 | wikitext2 | 4.3800 | N/A | N/A | 0.020000 | 0.004 | N/A | 4.693 | 0.6348 | 171.10 | 4060.6 | 119.5626 | 50.5198 | 4.84 | 126.33 |
| 9 | wikitext2 | 4.3169 | N/A | N/A | 0.020000 | 0.004 | N/A | 4.633 | 0.6795 | 170.49 | 4060.6 | 127.5582 | 51.7175 | 4.83 | 125.41 |
| 10 | wikitext2 | 4.2624 | N/A | N/A | 0.020000 | 0.004 | N/A | 4.581 | 0.7218 | 170.75 | 4060.6 | 135.1197 | 52.8392 | 4.83 | 124.81 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 5.52, wikitext2 Perplexity: 124.81
- **Final Validation Metrics**: wikitext2: {"loss": 4.826786432693254, "perplexity": 124.80923264088545}
