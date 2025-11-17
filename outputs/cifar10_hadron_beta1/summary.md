# F3EO-Bench Experiment Report

## Configuration Summary
```json
{
  "experiment": {
    "tasks": [
      "cifar10"
    ],
    "seed": 42,
    "device": "cuda:0"
  },
  "model": {
    "arch": "resnet18_cifar",
    "num_classes": 10
  },
  "data": {
    "batch_size": 128,
    "num_workers": 0,
    "aug": true,
    "cutout": true,
    "n_holes": 1,
    "cutout_length": 16
  },
  "optimizer": {
    "name": "Hadron",
    "lr": 0.001,
    "momentum": 0.9,
    "stat_decay": 1.0,
    "damping": 0.001,
    "kl_clip": 0.001,
    "weight_decay": 0.0005,
    "TCov": 10,
    "TInv": 100,
    "muon_momentum": 0.95
  },
  "scheduler": {
    "name": "multistep",
    "milestones": [
      25,
      40
    ],
    "gamma": 0.1
  },
  "train": {
    "epochs": 10,
    "log_every": 10,
    "ckpt_every": 10
  }
}
```

## Training Results
| Epoch | Task | Train Loss | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Eval Accuracy | Eval Loss |
|-------|------|------------|----|----|------------|---------|-----------|----------------|-------------------|---------------|-----------|
| 1 | cifar10 | 1.5897 | 0.001000 | 0.000 | N/A | 0.161 | 61.4542 | 158.09 | 1660.1 | 61.74 | 1.08 |
| 2 | cifar10 | 1.0469 | 0.001000 | 0.000 | N/A | 0.112 | 62.0276 | 156.59 | 1659.2 | 72.69 | 0.78 |
| 3 | cifar10 | 0.8219 | 0.001000 | 0.000 | N/A | 0.090 | 62.0485 | 156.72 | 1660.4 | 78.41 | 0.61 |
| 4 | cifar10 | 0.6898 | 0.001000 | 0.000 | N/A | 0.076 | 62.0441 | 156.92 | 1658.9 | 81.94 | 0.52 |
| 5 | cifar10 | 0.6054 | 0.001000 | 0.000 | N/A | 0.067 | 62.0220 | 156.15 | 1659.8 | 84.10 | 0.47 |
| 6 | cifar10 | 0.5374 | 0.001000 | 0.000 | N/A | 0.060 | 61.9987 | 156.03 | 1659.9 | 85.18 | 0.43 |
| 7 | cifar10 | 0.4871 | 0.001000 | 0.000 | N/A | 0.055 | 61.9564 | 156.03 | 1659.5 | 86.89 | 0.39 |
| 8 | cifar10 | 0.4451 | 0.001000 | 0.000 | N/A | 0.050 | 61.9180 | 155.88 | 1661.2 | 87.80 | 0.37 |
| 9 | cifar10 | 0.4074 | 0.001000 | 0.000 | N/A | 0.047 | 61.8782 | 156.05 | 1660.8 | 88.25 | 0.35 |
| 10 | cifar10 | 0.3720 | 0.001000 | 0.000 | N/A | 0.043 | 61.8365 | 156.01 | 1660.8 | 88.74 | 0.34 |

## Performance Summary
- **Best Validation Metrics**: cifar10 Accuracy: 88.74, cifar10 Loss: 1.08
- **Final Validation Metrics**: cifar10: {"loss": 0.3358959106704857, "accuracy": 88.74}
