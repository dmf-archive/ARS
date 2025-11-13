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
| 1 | cifar10 | 1.7482 | 0.001000 | 0.048 | N/A | 1.777 | 1.2692 | N/A | N/A | 54.60 | 1.26 |
| 2 | cifar10 | 1.1947 | 0.001000 | 0.113 | N/A | 1.244 | 0.9387 | N/A | N/A | 65.53 | 0.99 |
| 3 | cifar10 | 0.9535 | 0.001000 | 0.140 | N/A | 1.002 | 0.9654 | 79.16 | 1699.1 | 73.34 | 0.76 |
| 4 | cifar10 | 0.7853 | 0.001000 | 0.163 | N/A | 0.835 | 0.9772 | 80.03 | 1703.6 | 77.02 | 0.67 |
| 5 | cifar10 | 0.7157 | 0.000999 | 0.179 | N/A | 0.742 | 0.9764 | 79.77 | 1703.6 | 78.33 | 0.61 |
| 6 | cifar10 | 0.6358 | 0.000999 | 0.202 | N/A | 0.668 | 0.9340 | 79.95 | 1703.9 | 81.80 | 0.53 |
| 7 | cifar10 | 0.5740 | 0.000998 | 0.217 | N/A | 0.600 | 0.9298 | 83.03 | 1703.6 | 83.57 | 0.48 |
| 8 | cifar10 | 0.5299 | 0.000998 | 0.229 | N/A | 0.557 | 0.9174 | 84.85 | 1703.6 | 84.47 | 0.45 |
| 9 | cifar10 | 0.4896 | 0.000997 | 0.249 | N/A | 0.516 | 0.8740 | 83.14 | 1703.9 | 85.33 | 0.43 |
| 10 | cifar10 | 0.4590 | 0.000996 | 0.258 | N/A | 0.485 | 0.8688 | 81.98 | 1703.6 | 85.94 | 0.41 |

## Performance Summary
- **Best Validation Metrics**: cifar10 Accuracy: 85.94, cifar10 Loss: 1.26
- **Final Validation Metrics**: cifar10: {"loss": 0.406997362151742, "accuracy": 85.94}
