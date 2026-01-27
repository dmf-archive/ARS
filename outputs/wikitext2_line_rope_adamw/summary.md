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
    "name": "AdamW",
    "lr": 0.0001,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.1
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
| 1 | wikitext2_line | 5.8043 | 0.000100 | 0.277 | N/A | 0.000 | 1.2848 | 424.57 | 2786.7 | 5.32 | 204.85 |
| 2 | wikitext2_line | 4.9271 | 0.000100 | 0.282 | N/A | 0.000 | 1.2657 | 400.24 | 2786.7 | 4.94 | 140.16 |
| 3 | wikitext2_line | 4.5272 | 0.000100 | 0.243 | N/A | 0.000 | 1.4132 | 364.08 | 2785.7 | 4.75 | 115.78 |
| 4 | wikitext2_line | 4.2019 | 0.000100 | 0.190 | N/A | 0.000 | 1.6619 | 371.35 | 2785.7 | 4.66 | 105.48 |
| 5 | wikitext2_line | 3.8928 | 0.000100 | 0.129 | N/A | 0.000 | 2.0482 | 415.75 | 2785.7 | 4.65 | 104.68 |
| 6 | wikitext2_line | 3.5737 | 0.000100 | 0.080 | N/A | 0.000 | 2.5302 | 441.31 | 2785.7 | 4.70 | 109.94 |
| 7 | wikitext2_line | 3.2376 | 0.000100 | 0.046 | N/A | 0.000 | 3.0784 | 443.74 | 2785.7 | 4.84 | 125.95 |
| 8 | wikitext2_line | 2.8918 | 0.000100 | 0.027 | N/A | 0.000 | 3.6189 | 692.47 | 2785.7 | 5.03 | 153.61 |
| 9 | wikitext2_line | 2.5544 | 0.000100 | 0.017 | N/A | 0.000 | 4.0951 | 442.55 | 2785.7 | 5.27 | 194.53 |
| 10 | wikitext2_line | 2.2361 | 0.000100 | 0.011 | N/A | 0.000 | 4.4762 | 444.55 | 2785.7 | 5.52 | 250.82 |

## Performance Summary
- **Best Validation Metrics**: wikitext2_line Loss: 5.52, wikitext2_line Perplexity: 104.68
- **Final Validation Metrics**: wikitext2_line: {"loss": 5.524745809498118, "perplexity": 250.82257320121664}
