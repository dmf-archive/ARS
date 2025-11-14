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
    "name": "FOG",
    "lr": 0.001,
    "momentum": 0.9,
    "stat_decay": 0.95,
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
    "epochs": 20,
    "log_every": 10,
    "ckpt_every": 10
  }
}
```

## Training Results
| Epoch | Task | Train Loss | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Eval Accuracy | Eval Loss |
|-------|------|------------|----|----|------------|---------|-----------|----------------|-------------------|---------------|-----------|
| 1 | cifar10 | 1.5833 | 0.001000 | 0.000 | N/A | 0.163 | 60.8862 | 177.78 | 1659.9 | 61.45 | 1.10 |
| 2 | cifar10 | 1.0486 | 0.001000 | 0.000 | N/A | 0.112 | 61.5979 | 143.18 | 1660.5 | 72.97 | 0.77 |
| 3 | cifar10 | 0.8159 | 0.001000 | 0.000 | N/A | 0.089 | 61.4157 | 150.03 | 1660.7 | 79.14 | 0.60 |
| 4 | cifar10 | 0.6819 | 0.001000 | 0.000 | N/A | 0.074 | 60.9542 | 151.45 | 1661.0 | 82.32 | 0.51 |
| 5 | cifar10 | 0.5845 | 0.001000 | 0.000 | N/A | 0.063 | 60.6873 | 152.02 | 1660.4 | 85.07 | 0.44 |
| 6 | cifar10 | 0.5160 | 0.001000 | 0.000 | N/A | 0.055 | 60.5741 | 151.83 | 1660.7 | 86.20 | 0.40 |
| 7 | cifar10 | 0.4692 | 0.001000 | 0.000 | N/A | 0.050 | 60.5749 | 153.70 | 1660.7 | 86.89 | 0.38 |
| 8 | cifar10 | 0.4262 | 0.001000 | 0.000 | N/A | 0.046 | 60.6326 | 152.80 | 1660.3 | 87.99 | 0.36 |
| 9 | cifar10 | 0.3930 | 0.001000 | 0.000 | N/A | 0.042 | 60.6090 | 149.88 | 1660.3 | 88.82 | 0.34 |
| 10 | cifar10 | 0.3697 | 0.001000 | 0.000 | N/A | 0.039 | 60.5865 | 152.04 | 1658.8 | 88.91 | 0.33 |

## Performance Summary
- **Best Validation Metrics**: cifar10 Accuracy: 88.91, cifar10 Loss: 1.10
- **Final Validation Metrics**: cifar10: {"loss": 0.33160251756257647, "accuracy": 88.91}
