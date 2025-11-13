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
    "name": "KFAC",
    "lr": 0.001,
    "momentum": 0.9,
    "stat_decay": 0.95,
    "damping": 0.001,
    "kl_clip": 0.001,
    "weight_decay": 0.0005,
    "TCov": 10,
    "TInv": 100
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
| 1 | cifar10 | 1.7177 | 0.001000 | 0.000 | N/A | 1.711 | 22.7839 | 140.53 | 2461.8 | 51.39 | 1.38 |
| 2 | cifar10 | 1.3186 | 0.001000 | 0.000 | N/A | 1.275 | 184.4266 | 142.64 | 2468.4 | 63.35 | 1.12 |
| 3 | cifar10 | 1.0806 | 0.001000 | 0.000 | N/A | 1.020 | 399.5008 | 143.92 | 2466.8 | 71.68 | 0.86 |
| 4 | cifar10 | 0.9165 | 0.001000 | 0.000 | N/A | 0.843 | 414.6483 | 141.29 | 2468.1 | 75.26 | 0.77 |
| 5 | cifar10 | 0.7924 | 0.001000 | 0.000 | N/A | 0.738 | 392.9425 | 139.25 | 2468.6 | 78.93 | 0.64 |
| 6 | cifar10 | 0.6927 | 0.001000 | 0.000 | N/A | 0.642 | 368.8286 | 139.53 | 2466.7 | 81.66 | 0.55 |
| 7 | cifar10 | 0.6278 | 0.001000 | 0.000 | N/A | 0.584 | 346.3362 | 142.88 | 2467.4 | 82.41 | 0.54 |
| 8 | cifar10 | 0.5750 | 0.001000 | 0.000 | N/A | 0.536 | 331.0387 | 141.53 | 2466.5 | 85.33 | 0.45 |
| 9 | cifar10 | 0.5373 | 0.001000 | 0.000 | N/A | 0.500 | 315.1467 | 140.12 | 2467.4 | 85.56 | 0.43 |
| 10 | cifar10 | 0.4998 | 0.001000 | 0.000 | N/A | 0.464 | 308.0117 | 141.26 | 2467.1 | 85.76 | 0.43 |

## Performance Summary
- **Best Validation Metrics**: cifar10 Accuracy: 85.76, cifar10 Loss: 1.38
- **Final Validation Metrics**: cifar10: {"loss": 0.4295931117066854, "accuracy": 85.76}
