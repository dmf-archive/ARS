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
    "k": 5,
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
| 1 | wikitext2_line | 5.5333 | 0.000100 | 0.427 | N/A | 0.000 | 0.8508 | 545.79 | 3108.8 | 5.06 | 158.17 |
| 2 | wikitext2_line | 4.6167 | 0.000100 | 0.486 | N/A | 0.000 | 0.7209 | 546.13 | 3576.3 | 4.71 | 111.52 |
| 3 | wikitext2_line | 4.2040 | 0.000100 | 0.472 | N/A | 0.000 | 0.7514 | 547.29 | 3581.4 | 4.54 | 93.94 |
| 4 | wikitext2_line | 3.8742 | 0.000100 | 0.447 | N/A | 0.000 | 0.8055 | 548.40 | 3107.0 | 4.47 | 87.00 |
| 5 | wikitext2_line | 3.5690 | 0.000100 | 0.417 | N/A | 0.000 | 0.8740 | 549.33 | 3415.0 | 4.45 | 85.69 |

## Performance Summary
- **Best Validation Metrics**: wikitext2_line Loss: 5.06, wikitext2_line Perplexity: 85.69
- **Final Validation Metrics**: wikitext2_line: {"loss": 4.450732042540365, "perplexity": 85.68964951656343}
