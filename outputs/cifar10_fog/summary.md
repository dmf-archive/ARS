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
    "epochs": 10,
    "log_every": 10,
    "ckpt_every": 50
  }
}
```

## Training Results
| Epoch | Task | Train Loss | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Eval Accuracy | Eval Loss |
|-------|------|------------|----|----|------------|---------|-----------|----------------|-------------------|---------------|-----------|
| 1 | cifar10 | 1.6327 | 0.001000 | 0.000 | N/A | 1.550 | 58.6630 | 159.53 | 2505.5 | 62.84 | 1.04 |
| 2 | cifar10 | 0.9694 | 0.001000 | 0.000 | N/A | 0.984 | 59.6003 | 167.15 | 2511.5 | 76.27 | 0.69 |
| 3 | cifar10 | 0.9945 | 0.001000 | 0.000 | N/A | 0.925 | 59.3626 | 161.09 | 2512.1 | 78.98 | 0.63 |
| 4 | cifar10 | 0.6270 | 0.001000 | 0.000 | N/A | 0.649 | 60.6326 | 161.07 | 2514.4 | 84.04 | 0.47 |
| 5 | cifar10 | 0.5539 | 0.001000 | 0.000 | N/A | 0.554 | 60.7650 | 159.88 | 2517.7 | 85.07 | 0.43 |
| 6 | cifar10 | 0.5230 | 0.001000 | 0.000 | N/A | 0.525 | 59.8582 | 158.19 | 2515.9 | 86.26 | 0.40 |
| 7 | cifar10 | 0.4844 | 0.001000 | 0.000 | N/A | 0.484 | 59.0524 | 158.36 | 2515.1 | 86.98 | 0.37 |
| 8 | cifar10 | 0.4400 | 0.001000 | 0.000 | N/A | 0.444 | 58.7989 | 168.52 | 2517.6 | 88.97 | 0.32 |
| 9 | cifar10 | 0.4057 | 0.001000 | 0.000 | N/A | 0.407 | 58.7133 | 178.28 | 2517.5 | 88.79 | 0.33 |
| 10 | cifar10 | 0.3767 | 0.001000 | 0.000 | N/A | 0.382 | 58.6786 | 177.04 | 2515.5 | 90.30 | 0.29 |

## Performance Summary
- **Best Validation Metrics**: cifar10 Accuracy: 90.30, cifar10 Loss: 1.04
- **Final Validation Metrics**: cifar10: {"loss": 0.29328335067139394, "accuracy": 90.3}
