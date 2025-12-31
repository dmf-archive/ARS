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
    "name": "AdaMuon",
    "lr": 0.0001,
    "betas": [
      0.95,
      0.95
    ],
    "eps": 1e-08,
    "weight_decay": 0.1,
    "ns_steps": 5,
    "adam_lr": 0.001,
    "adam_weight_decay": 0.01,
    "adam_betas": [
      0.9,
      0.999
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
| 1 | wikitext2_line | 5.8332 | 0.000100 | 0.023 | N/A | 0.000 | 3.7761 | 605.19 | 2784.7 | 5.11 | 165.69 |
| 2 | wikitext2_line | 4.5365 | 0.000100 | 0.012 | N/A | 0.000 | 4.3980 | 607.05 | 2783.9 | 4.87 | 130.65 |
| 3 | wikitext2_line | 4.0999 | 0.000100 | 0.009 | N/A | 0.000 | 4.6596 | 605.49 | 2783.9 | 4.83 | 125.46 |
| 4 | wikitext2_line | 3.7914 | 0.000100 | 0.006 | N/A | 0.000 | 5.0746 | 609.61 | 2783.9 | 4.88 | 131.83 |
| 5 | wikitext2_line | 3.5452 | 0.000100 | 0.004 | N/A | 0.000 | 5.6042 | 603.60 | 2783.9 | 4.99 | 147.60 |

## Performance Summary
- **Best Validation Metrics**: wikitext2_line Loss: 5.11, wikitext2_line Perplexity: 125.46
- **Final Validation Metrics**: wikitext2_line: {"loss": 4.994485225250472, "perplexity": 147.59694664279547}
