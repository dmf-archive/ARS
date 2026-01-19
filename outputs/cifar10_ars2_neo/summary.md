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
    "epochs": 30
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
    "k": 0
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
| 1 | cifar10 | 1.3437 | N/A | N/A | 0.001000 | 0.117 | N/A | 0.131 | 2.0110 | 71.37 | 1742.6 | 70.21 | 0.88 |
| 2 | cifar10 | 0.8162 | N/A | N/A | 0.001000 | 0.167 | N/A | 0.081 | 1.7073 | 72.86 | 1742.7 | 79.37 | 0.61 |
| 3 | cifar10 | 0.6466 | N/A | N/A | 0.001000 | 0.210 | N/A | 0.064 | 1.4981 | 79.11 | 1742.7 | 81.06 | 0.56 |
| 4 | cifar10 | 0.5547 | N/A | N/A | 0.000999 | 0.258 | N/A | 0.055 | 1.2992 | 81.30 | 1742.7 | 81.15 | 0.61 |
| 5 | cifar10 | 0.4912 | N/A | N/A | 0.000999 | 0.289 | N/A | 0.048 | 1.1941 | 81.33 | 1742.7 | 84.91 | 0.45 |
| 6 | cifar10 | 0.4476 | N/A | N/A | 0.000998 | 0.312 | N/A | 0.044 | 1.1204 | 84.61 | 1742.7 | 85.27 | 0.45 |
| 7 | cifar10 | 0.4094 | N/A | N/A | 0.000998 | 0.342 | N/A | 0.040 | 1.0327 | 82.36 | 1742.7 | 88.19 | 0.37 |
| 8 | cifar10 | 0.3815 | N/A | N/A | 0.000997 | 0.368 | N/A | 0.037 | 0.9616 | 82.59 | 1742.7 | 88.37 | 0.35 |
| 9 | cifar10 | 0.3509 | N/A | N/A | 0.000996 | 0.392 | N/A | 0.034 | 0.9026 | 84.65 | 1742.7 | 89.85 | 0.32 |
| 10 | cifar10 | 0.3316 | N/A | N/A | 0.000995 | 0.396 | N/A | 0.032 | 0.8940 | 83.19 | 1742.7 | 90.13 | 0.32 |

## Performance Summary
- **Best Validation Metrics**: cifar10 Accuracy: 90.13, cifar10 Loss: 0.88
- **Final Validation Metrics**: cifar10: {"loss": 0.3228240296244621, "accuracy": 90.13}
