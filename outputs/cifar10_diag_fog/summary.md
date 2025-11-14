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
    "arch": "resnet18",
    "num_classes": 10
  },
  "data": {
    "batch_size": 128,
    "num_workers": 0
  },
  "optimizer": {
    "name": "DiagFOG",
    "lr": 0.001,
    "momentum": 0.9,
    "stat_decay": 0.95,
    "damping": 0.001,
    "kl_clip": 0.001,
    "weight_decay": 0.1,
    "TCov": 10,
    "TInv": 100,
    "muon_momentum": 0.95,
    "adam_lr": 0.001,
    "adam_weight_decay": 0.1,
    "adam_betas": [
      0.9,
      0.95
    ]
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
| 1 | cifar10 | 1.5496 | 0.001000 | 0.008 | N/A | 0.146 | 4.6412 | 60.40 | 273.6 | 51.59 | 1.33 |
| 2 | cifar10 | 1.1871 | 0.001000 | 0.016 | N/A | 0.115 | 4.0301 | 59.71 | 274.1 | 62.98 | 1.04 |
| 3 | cifar10 | 1.0274 | 0.001000 | 0.025 | N/A | 0.100 | 3.5835 | 59.51 | 274.1 | 64.56 | 1.03 |
| 4 | cifar10 | 0.9195 | 0.001000 | 0.034 | N/A | 0.090 | 3.2978 | 61.35 | 274.1 | 70.04 | 0.86 |
| 5 | cifar10 | 0.8423 | 0.001000 | 0.040 | N/A | 0.083 | 3.1363 | 58.57 | 274.1 | 74.08 | 0.76 |
| 6 | cifar10 | 0.7763 | 0.001000 | 0.047 | N/A | 0.076 | 2.9710 | 61.58 | 274.1 | 70.85 | 0.85 |
| 7 | cifar10 | 0.7285 | 0.001000 | 0.054 | N/A | 0.072 | 2.8398 | 59.21 | 274.1 | 74.55 | 0.73 |
| 8 | cifar10 | 0.6923 | 0.001000 | 0.061 | N/A | 0.068 | 2.7271 | 59.01 | 274.1 | 76.04 | 0.68 |
| 9 | cifar10 | 0.6568 | 0.001000 | 0.070 | N/A | 0.065 | 2.5957 | 60.15 | 274.1 | 75.95 | 0.69 |
| 10 | cifar10 | 0.6272 | 0.001000 | 0.079 | N/A | 0.062 | 2.4822 | 60.37 | 274.1 | 77.66 | 0.64 |

## Performance Summary
- **Best Validation Metrics**: cifar10 Accuracy: 77.66, cifar10 Loss: 1.33
- **Final Validation Metrics**: cifar10: {"loss": 0.643285346936576, "accuracy": 77.66}
