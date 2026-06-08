# ARS-Bench Experiment Report — Grokking Milestone Mode

## Configuration Summary
```json
{
  "experiment": {
    "seed": 42,
    "device": "cuda",
    "tasks": [
      "mod_addition"
    ],
    "epochs": 400,
    "save_interval": 100
  },
  "task": {
    "p": 113,
    "fraction": 0.3
  },
  "data": {
    "batch_size": 512,
    "num_workers": 0
  },
  "model": {
    "arch": "grokking_transformer",
    "num_layers": 1,
    "d_model": 128,
    "d_mlp": 512,
    "d_head": 32,
    "num_heads": 4,
    "n_ctx": 3,
    "act_type": "ReLU"
  },
  "optimizer": {
    "name": "ARS2C-SAGA",
    "lr": 0.001,
    "rho": 0.1,
    "adaptive_sync": true,
    "adaptive_lambda": 0.1,
    "k": 10,
    "ns_steps": 5,
    "betas": [
      0.9,
      0.98
    ],
    "weight_decay": 1.0,
    "beta1_min": 0.9,
    "beta1_max": 0.9995,
    "beta2_min": 0.9,
    "beta2_max": 0.9995,
    "rho_kappa": 0.01,
    "rho_eta": 0.1,
    "rho_min": 0.01,
    "rho_max": 1.0
  }
}
```

## Grokking Milestones
| Milestone | Epoch | Train Loss | Eval Acc | Train Acc | PI | Grad Norm |
|-----------|-------|------------|----------|-----------|----|-----------|
| Train Loss < 0.5   (Fitting) | 13 | 0.3433 | 26.45709810940821 | 94.69973890339426 | 0.024424054334307665 | 1.5347 |
| Train Loss < 0.1   (Overfit) | 18 | 0.0878 | 26.468285043069695 | 98.09399477806788 | 0.10690420343508802 | 1.3657 |
| Train Acc > 99%    (Memorization) | 53 | 0.0329 | 31.032553976954915 | 99.24281984334203 | 0.3758232940886116 | 0.6928 |
| Eval Acc > 10%     (Early Signal) | 9 | 2.2029 | 11.846962747510908 | 48.067885117493475 | 0.006754835219582743 | 1.5999 |
| Eval Acc > 50%     (Above Chance) | 70 | 0.0185 | 52.91419621881642 | 99.47780678851174 | 0.4764846265528957 | 0.7698 |
| Eval Acc > 80%     (Strong Signal) | 85 | 0.0142 | 80.98221277547825 | 99.71279373368147 | 0.5568726562619847 | 0.4728 |
| Eval Acc > 90%     (Grokking) | 98 | 0.0101 | 90.58060185703098 | 99.73890339425587 | 0.6096852002394505 | 0.5124 |
| Eval Acc > 95%     (Near Convergence) | 119 | 0.0049 | 95.25674012753105 | 99.94778067885117 | 0.6921873141983804 | 0.3215 |
| Eval Acc > 98%     (Convergence) | 152 | 0.0033 | 98.16534287951673 | 99.9738903394256 | 0.6875201882395615 | 0.2394 |
| Eval Acc > 99%     (Full Convergence) | — | — | — | — | — | — |

## Performance Summary
- **Best Eval Acc**: 98.86% at Epoch 228
- **Total Epochs**: 267

## Last 5 Epochs (Raw Snapshot)
| Epoch | Train Loss | Eval Acc | Train Acc | PI | Grad Norm |
|-------|------------|----------|-----------|----|-----------|
| 263 | 0.0037 | 98.43382928739233 | 99.9738903394256 | 0.736137477310058 | 0.2817 |
| 264 | 0.0035 | 98.3219599507775 | 100.0 | 0.7388897786303392 | 0.2552 |
| 265 | 0.0031 | 98.42264235373084 | 99.9738903394256 | 0.7378852590707561 | 0.3047 |
| 266 | 0.0027 | 98.35552075176194 | 99.9738903394256 | 0.7466261766661124 | 0.1824 |
| 267 | 0.0041 | 98.51213782302271 | 99.89556135770235 | 0.7497667117991095 | 0.2393 |
