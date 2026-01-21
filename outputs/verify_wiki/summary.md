# F3EO-Bench Experiment Report

## Configuration Summary
```json
{
  "experiment": {
    "tasks": [
      "wikitext2_line"
    ],
    "seed": 42,
    "device": "cuda",
    "epochs": 1
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
    "name": "ARS2-Neo",
    "lr": 0.0001,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.1,
    "ns_steps": 5,
    "k": 0
  },
  "train": {
    "epochs": 1,
    "log_every": 10,
    "ckpt_every": 2
  }
}
```

## Training Results
| Epoch | Task | Train Loss | Min Loss | Min Step | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Diag group_0_muon_avg_norm | Diag group_1_adam_avg_norm | Eval Loss | Eval Perplexity |
|-------|------|------------|----------|----------|----|----|------------|---------|-----------|----------------|-------------------|----------------------------|----------------------------|-----------|-----------------|
| 1 | wikitext2 | 5.5089 | N/A | N/A | 0.000100 | 0.001 | N/A | 6.064 | 0.8962 | 463.95 | 2943.2 | 41.1725 | 22.3575 | 5.06 | 157.03 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 5.06, wikitext2 Perplexity: 157.03
- **Final Validation Metrics**: wikitext2: {"loss": 5.056428699351069, "perplexity": 157.02871698443553}
