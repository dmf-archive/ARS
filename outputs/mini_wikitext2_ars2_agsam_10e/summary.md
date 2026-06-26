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
    "name": "ARS2-Neo",
    "lr": 0.001,
    "betas": [
      0.9,
      0.95
    ],
    "rho": 0.3,
    "k": 1,
    "alpha": 0.0,
    "adaptive_sync": true,
    "adaptive_lambda": 0.5,
    "adaptive_gamma": 2.0,
    "adaptive_beta": 0.9
  }
}
```

## Training Results
| Epoch | Task | Train Loss | Min Loss | Min Step | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Diag alpha_t | Diag effective_k | Diag group_0_muon_avg_norm | Diag group_1_adam_avg_norm | Diag phi_std | Diag phi_t | Diag threshold | Eval Loss | Eval Perplexity |
|-------|------|------------|----------|----------|----|----|------------|---------|-----------|----------------|-------------------|--------------|------------------|----------------------------|----------------------------|--------------|------------|----------------|-----------|-----------------|
| 1 | wikitext2 | 5.6780 | N/A | N/A | 0.001000 | 0.001 | N/A | 6.130 | 0.4390 | 219.34 | 4131.5 | 0.0000 | 3.0000 | 43.0571 | 23.4622 | 0.1018 | 0.0639 | -0.0509 | 5.08 | 160.74 |
| 2 | wikitext2 | 4.6556 | N/A | N/A | 0.001000 | 0.003 | N/A | 4.885 | 0.3456 | 218.35 | 4199.0 | 0.0000 | 2.9856 | 61.3809 | 26.1053 | 0.0749 | -0.1444 | -0.0375 | 4.78 | 119.58 |
| 3 | wikitext2 | 4.2708 | N/A | N/A | 0.001000 | 0.005 | N/A | 4.477 | 0.3786 | 215.90 | 4199.0 | 0.0000 | 3.0293 | 75.1492 | 28.1837 | 0.0681 | 0.0316 | -0.0340 | 4.67 | 107.02 |
| 4 | wikitext2 | 4.0014 | N/A | N/A | 0.001000 | 0.007 | N/A | 4.190 | 0.4112 | 214.91 | 4199.0 | 0.0000 | 3.0819 | 86.1011 | 29.9501 | 0.0551 | 0.0360 | -0.0275 | 4.64 | 103.59 |
| 5 | wikitext2 | 3.7962 | N/A | N/A | 0.001000 | 0.008 | N/A | 3.964 | 0.4433 | 212.41 | 4199.0 | 0.0000 | 3.1300 | 95.0588 | 31.4690 | 0.0532 | 0.0303 | -0.0266 | 4.67 | 106.29 |
| 6 | wikitext2 | 3.6351 | N/A | N/A | 0.001000 | 0.009 | N/A | 3.782 | 0.4709 | 210.46 | 4199.0 | 0.0000 | 3.1846 | 102.6303 | 32.7896 | 0.0508 | 0.0452 | -0.0254 | 4.72 | 112.28 |
| 7 | wikitext2 | 3.5039 | N/A | N/A | 0.001000 | 0.011 | N/A | 3.638 | 0.4970 | 210.18 | 4199.0 | 0.0000 | 3.2224 | 109.0048 | 33.9564 | 0.0572 | 0.0765 | -0.0286 | 4.78 | 119.57 |
| 8 | wikitext2 | 3.3974 | N/A | N/A | 0.001000 | 0.012 | N/A | 3.520 | 0.5202 | 210.99 | 4131.5 | 0.0000 | 3.2471 | 114.4817 | 35.0013 | 0.0313 | -0.0184 | -0.0156 | 4.86 | 128.42 |
| 9 | wikitext2 | 3.3051 | N/A | N/A | 0.001000 | 0.013 | N/A | 3.421 | 0.5370 | 211.50 | 4199.0 | 0.0000 | 3.2608 | 119.2761 | 35.9469 | 0.0516 | -0.0209 | -0.0258 | 4.92 | 136.39 |
| 10 | wikitext2 | 3.2257 | N/A | N/A | 0.001000 | 0.014 | N/A | 3.335 | 0.5527 | 210.96 | 4199.1 | 0.0000 | 3.2805 | 123.4441 | 36.8037 | 0.0395 | 0.0144 | -0.0197 | 4.99 | 146.67 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 5.08, wikitext2 Perplexity: 103.59
- **Final Validation Metrics**: wikitext2: {"loss": 4.988181647969715, "perplexity": 146.66948412013716}
