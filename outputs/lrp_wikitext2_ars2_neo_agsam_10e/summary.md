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
    "embedding_size": 512,
    "sequence_length": 255,
    "num_heads": 4,
    "num_layers": 3,
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
| 1 | wikitext2 | 5.1291 | N/A | N/A | 0.001000 | 0.003 | N/A | 5.140 | 0.6334 | 561.47 | 3055.8 | 0.0000 | 3.2454 | 380.7543 | 31.3022 | 0.0543 | -0.1185 | -0.0271 | 4.78 | 118.76 |
| 2 | wikitext2 | 4.1603 | N/A | N/A | 0.001000 | 0.006 | N/A | 4.193 | 0.5334 | 553.59 | 3435.2 | 0.0000 | 3.5113 | 523.5838 | 34.4526 | 0.0441 | -0.0350 | -0.0220 | 4.55 | 94.24 |
| 3 | wikitext2 | 3.6632 | N/A | N/A | 0.001000 | 0.009 | N/A | 3.718 | 0.5420 | 544.69 | 3435.6 | 0.0000 | 3.6487 | 622.2999 | 37.3500 | 0.0311 | 0.0370 | -0.0156 | 4.54 | 93.23 |
| 4 | wikitext2 | 3.2565 | N/A | N/A | 0.001000 | 0.012 | N/A | 3.335 | 0.5729 | 547.60 | 3435.6 | 0.0000 | 3.7299 | 697.5419 | 40.0726 | 0.0251 | -0.0389 | -0.0126 | 4.62 | 101.94 |
| 5 | wikitext2 | 2.8790 | N/A | N/A | 0.001000 | 0.016 | N/A | 2.984 | 0.6130 | 533.22 | 3436.0 | 0.0000 | 3.7553 | 756.3189 | 42.5769 | 0.0193 | 0.0221 | -0.0097 | 4.80 | 121.00 |
| 6 | wikitext2 | 2.5306 | N/A | N/A | 0.001000 | 0.020 | N/A | 2.654 | 0.6496 | 534.40 | 3436.7 | 0.0000 | 3.7309 | 802.0593 | 44.7968 | 0.0197 | 0.0268 | -0.0099 | 5.03 | 152.26 |
| 7 | wikitext2 | 2.2281 | N/A | N/A | 0.001000 | 0.026 | N/A | 2.363 | 0.6789 | 538.08 | 3308.2 | 0.0000 | 3.6806 | 838.0904 | 46.7361 | 0.0119 | 0.0170 | -0.0060 | 5.27 | 195.16 |
| 8 | wikitext2 | 1.9712 | N/A | N/A | 0.001000 | 0.032 | N/A | 2.111 | 0.7000 | 542.47 | 3436.5 | 0.0000 | 3.6107 | 866.3318 | 48.4406 | 0.0081 | -0.0138 | -0.0041 | 5.54 | 254.97 |
| 9 | wikitext2 | 1.7619 | N/A | N/A | 0.001000 | 0.038 | N/A | 1.908 | 0.7147 | 548.02 | 3435.2 | 0.0000 | 3.4970 | 888.5332 | 49.9627 | 0.0236 | -0.0497 | -0.0118 | 5.78 | 322.41 |
| 10 | wikitext2 | 1.5906 | N/A | N/A | 0.001000 | 0.046 | N/A | 1.738 | 0.7219 | 556.29 | 3436.2 | 0.0000 | 3.3860 | 906.2640 | 51.3313 | 0.0170 | -0.0091 | -0.0085 | 6.03 | 414.83 |

## Performance Summary
- **Best Validation Metrics**: wikitext2 Loss: 6.03, wikitext2 Perplexity: 93.23
- **Final Validation Metrics**: wikitext2: {"loss": 6.027876894865463, "perplexity": 414.8333589393056}
