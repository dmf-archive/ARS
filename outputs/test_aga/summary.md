# F3EO-Bench Experiment Report

## Configuration Summary
```json
{
  "experiment": {
    "name": "test_aga_smoke",
    "device": "cuda",
    "seed": 42,
    "tasks": [
      "cifar10"
    ]
  },
  "data": {
    "batch_size": 256,
    "num_workers": 0,
    "cutout": false
  },
  "model": {
    "arch": "resnet18",
    "num_classes": 10
  },
  "train": {
    "epochs": 1
  },
  "optimizer": {
    "name": "ARS2-Neo",
    "lr": 0.001,
    "weight_decay": 0.05,
    "betas": [
      0.9,
      0.95
    ],
    "eps": 1e-08,
    "ns_steps": 5,
    "rho": 0.1,
    "k": 5,
    "alpha": 0.1,
    "adaptive_sync": true,
    "adaptive_beta": 0.9,
    "adaptive_lambda": 0.1,
    "adaptive_gamma": 2.0
  },
  "scheduler": {
    "name": "cosine",
    "T_max": 1,
    "eta_min": 1e-05
  }
}
```

## Training Results
| Epoch | Task | Train Loss | Min Loss | Min Step | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Diag alpha_t | Diag effective_k | Diag group_0_muon_avg_norm | Diag group_1_adam_avg_norm | Diag phi_mean | Diag phi_t | Diag threshold | Eval Accuracy | Eval Loss |
|-------|------|------------|----------|----------|----|----|------------|---------|-----------|----------------|-------------------|--------------|------------------|----------------------------|----------------------------|---------------|------------|----------------|---------------|-----------|
| 1 | cifar10 | 1.4855 | N/A | N/A | 0.001000 | 0.017 | N/A | 1.486 | 2.6108 | 67.13 | 460.4 | 0.0933 | 1.9406 | 57.4770 | 7.1155 | -0.0545 | 0.0341 | -0.0687 | 61.08 | 1.15 |

## Performance Summary
- **Best Validation Metrics**: cifar10 Accuracy: 61.08, cifar10 Loss: 1.15
- **Final Validation Metrics**: cifar10: {"loss": 1.1459078371524811, "accuracy": 61.08}
