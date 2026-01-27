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
    "alpha": 0.1
  },
  "train": {
    "epochs": 5,
    "log_every": 10,
    "ckpt_every": 1
  }
}
```

## Training Results
| Epoch | Task | Train Loss | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Diag cos_pos_0 | Diag cos_pos_2 | Diag cos_pos_3 | Diag cos_pos_4 | Eval Loss | Eval Perplexity |
|-------|------|------------|----|----|------------|---------|-----------|----------------|-------------------|----------------|----------------|----------------|----------------|-----------|-----------------|
| 1 | wikitext2_line | 5.5468 | 0.000100 | 0.445 | N/A | 0.000 | 0.8098 | 551.23 | 3108.8 | 0.0330 | 0.0930 | 0.0632 | 0.0404 | 5.06 | 158.21 |
| 2 | wikitext2_line | 4.6198 | 0.000100 | 0.504 | N/A | 0.000 | 0.6852 | 567.00 | 3576.3 | 0.0266 | 0.0849 | 0.0788 | 0.0749 | 4.71 | 110.95 |
| 3 | wikitext2_line | 4.2055 | 0.000100 | 0.493 | N/A | 0.000 | 0.7081 | 561.56 | 3581.4 | 0.0621 | 0.0934 | 0.0792 | 0.0722 | 4.53 | 92.84 |
| 4 | wikitext2_line | 3.8773 | 0.000100 | 0.473 | N/A | 0.000 | 0.7484 | 560.82 | 3576.3 | 0.0662 | 0.0843 | 0.0709 | 0.0549 | 4.44 | 85.19 |
| 5 | wikitext2_line | 3.5778 | 0.000100 | 0.450 | N/A | 0.000 | 0.7986 | 563.85 | 3573.5 | 0.0664 | 0.0643 | 0.0705 | 0.0621 | 4.41 | 82.10 |

## Performance Summary
- **Best Validation Metrics**: wikitext2_line Loss: 5.06, wikitext2_line Perplexity: 82.10
- **Final Validation Metrics**: wikitext2_line: {"loss": 4.407930149960873, "perplexity": 82.09935416309474}
