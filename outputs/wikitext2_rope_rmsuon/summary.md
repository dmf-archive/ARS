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
    "name": "RMSuon",
    "lr": 0.0001,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.1,
    "ns_steps": 5
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
| 1 | wikitext2 | 6.2467 | 0.000100 | 0.000 | N/A | 6.615 | 1.8024 | 683.69 | 2984.5 | 5.69 | 297.26 |
| 2 | wikitext2 | 5.0733 | 0.000100 | 0.001 | N/A | 5.479 | 2.1205 | 721.41 | 2984.5 | 5.32 | 204.06 |
| 3 | wikitext2 | 4.4127 | 0.000100 | 0.000 | N/A | 4.925 | 2.6901 | 698.74 | 2984.5 | 5.25 | 189.76 |
| 4 | wikitext2 | 3.8043 | 0.000100 | 0.000 | N/A | 4.438 | 3.4547 | 705.99 | 2984.5 | 5.35 | 211.53 |
| 5 | wikitext2 | 3.2102 | 0.000100 | 0.000 | N/A | 3.963 | 4.2650 | 704.69 | 2984.5 | 5.60 | 270.23 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 5.69, wikitext2 Perplexity: 189.76
- **Final Validation Metrics**: wikitext2: {"loss": 5.599276789303484, "perplexity": 270.23090285969965}
