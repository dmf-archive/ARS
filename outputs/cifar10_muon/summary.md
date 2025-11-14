# F3EO-Bench Experiment Report

## Configuration Summary
```json
{
  "experiment": {
    "tasks": [
      "cifar10"
    ],
    "seed": 42,
    "device": "cuda"
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
    "name": "Muon",
    "lr": 0.001,
    "weight_decay": 0.0005,
    "momentum": 0.95
  },
  "scheduler": {
    "name": "cosine",
    "T_max": 200
  },
  "train": {
    "epochs": 10,
    "log_every": 10,
    "ckpt_every": 50
  }
}
```

## Training Results
| Epoch | Task | Train Loss | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Eval Accuracy | Eval Loss |
|-------|------|------------|----|----|------------|---------|-----------|----------------|-------------------|---------------|-----------|
| 1 | cifar10 | 1.7438 | 0.001000 | 0.239 | N/A | 0.178 | 1.2540 | 83.87 | 1697.8 | 55.06 | 1.25 |
| 2 | cifar10 | 1.2040 | 0.001000 | 0.337 | N/A | 0.125 | 0.9642 | 86.96 | 1697.8 | 65.56 | 0.97 |
| 3 | cifar10 | 0.9721 | 0.001000 | 0.333 | N/A | 0.102 | 0.9970 | 87.24 | 1697.8 | 70.76 | 0.83 |
| 4 | cifar10 | 0.8159 | 0.000999 | 0.347 | N/A | 0.086 | 0.9721 | 89.37 | 1697.8 | 76.65 | 0.67 |
| 5 | cifar10 | 0.7098 | 0.000999 | 0.354 | N/A | 0.074 | 0.9655 | 89.82 | 1697.8 | 80.46 | 0.57 |
| 6 | cifar10 | 0.6347 | 0.000998 | 0.363 | N/A | 0.067 | 0.9463 | 90.90 | 1697.8 | 82.32 | 0.50 |
| 7 | cifar10 | 0.5744 | 0.000998 | 0.373 | N/A | 0.060 | 0.9271 | 88.62 | 1697.8 | 83.23 | 0.49 |
| 8 | cifar10 | 0.5300 | 0.000997 | 0.380 | N/A | 0.056 | 0.9125 | 88.78 | 1697.8 | 85.03 | 0.44 |
| 9 | cifar10 | 0.4844 | 0.000996 | 0.396 | N/A | 0.051 | 0.8751 | 90.54 | 1697.8 | 84.59 | 0.45 |
| 10 | cifar10 | 0.4543 | 0.000995 | 0.399 | N/A | 0.048 | 0.8702 | 91.04 | 1697.8 | 87.05 | 0.39 |

## Performance Summary
- **Best Validation Metrics**: cifar10 Accuracy: 87.05, cifar10 Loss: 1.25
- **Final Validation Metrics**: cifar10: {"loss": 0.3854283049702644, "accuracy": 87.05}
