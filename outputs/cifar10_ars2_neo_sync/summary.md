# F3EO-Bench Experiment Report

## Configuration Summary
```json
{
  "experiment": {
    "tasks": [
      "cifar10"
    ],
    "seed": 42,
    "device": "cuda",
    "epochs": 10
  },
  "model": {
    "arch": "resnet18_cifar",
    "num_classes": 10
  },
  "data": {
    "batch_size": 256,
    "num_workers": 0,
    "aug": true,
    "cutout": true,
    "n_holes": 1,
    "cutout_length": 16
  },
  "optimizer": {
    "name": "ARS2-Neo",
    "lr": 0.001,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "weight_decay": 0.0005,
    "ns_steps": 5,
    "rho": 0.1,
    "k": 1,
    "alpha": 0.7,
    "adaptive": true
  },
  "scheduler": {
    "name": "cosine",
    "T_max": 200
  },
  "train": {
    "log_every": 10,
    "ckpt_every": 10
  }
}
```

## Training Results
| Epoch | Task | Train Loss | Min Loss | Min Step | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Eval Accuracy | Eval Loss |
|-------|------|------------|----------|----------|----|----|------------|---------|-----------|----------------|-------------------|---------------|-----------|
| 1 | cifar10 | 1.5135 | N/A | N/A | 0.001000 | 0.070 | N/A | 0.160 | 2.5001 | 107.78 | 1830.4 | 61.36 | 1.11 |
| 2 | cifar10 | 0.9342 | N/A | N/A | 0.001000 | 0.208 | N/A | 0.104 | 1.4658 | 114.50 | 1915.4 | 76.42 | 0.68 |
| 3 | cifar10 | 0.6950 | N/A | N/A | 0.001000 | 0.313 | N/A | 0.079 | 1.0819 | 117.40 | 1916.1 | 82.12 | 0.51 |
| 4 | cifar10 | 0.5815 | N/A | N/A | 0.000999 | 0.388 | N/A | 0.067 | 0.8793 | 120.50 | 2003.5 | 84.20 | 0.48 |
| 5 | cifar10 | 0.5043 | N/A | N/A | 0.000999 | 0.447 | N/A | 0.059 | 0.7468 | 120.96 | 2088.3 | 85.58 | 0.43 |
| 6 | cifar10 | 0.4461 | N/A | N/A | 0.000998 | 0.484 | N/A | 0.053 | 0.6720 | 120.84 | 2001.8 | 87.28 | 0.36 |
| 7 | cifar10 | 0.4043 | N/A | N/A | 0.000998 | 0.522 | N/A | 0.048 | 0.6017 | 120.86 | 2088.8 | 88.14 | 0.36 |
| 8 | cifar10 | 0.3731 | N/A | N/A | 0.000997 | 0.552 | N/A | 0.045 | 0.5487 | 121.56 | 2004.4 | 89.48 | 0.31 |
| 9 | cifar10 | 0.3394 | N/A | N/A | 0.000996 | 0.583 | N/A | 0.041 | 0.4987 | 122.76 | 2089.7 | 90.04 | 0.29 |
| 10 | cifar10 | 0.3178 | N/A | N/A | 0.000995 | 0.600 | N/A | 0.039 | 0.4710 | 119.59 | 2174.0 | 90.70 | 0.28 |

## Performance Summary
- **Best Validation Metrics**: cifar10 Accuracy: 90.70, cifar10 Loss: 1.11
- **Final Validation Metrics**: cifar10: {"loss": 0.27783818654716014, "accuracy": 90.7}
