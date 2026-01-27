# ARS-Bench Experiment Report

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
    "T_max": 10
  },
  "train": {
    "epochs": 1,
    "log_every": 10,
    "ckpt_every": 50
  }
}
```

## Training Results
| Epoch | Task | Train Loss | Min Loss | Min Step | LR | PI | Eff. Gamma | Entropy | Grad Norm | Epoch Time (s) | Peak GPU Mem (MB) | Diag group_0_muon_avg_norm | Diag group_1_adam_avg_norm | Eval Accuracy | Eval Loss |
|-------|------|------------|----------|----------|----|----|------------|---------|-----------|----------------|-------------------|----------------------------|----------------------------|---------------|-----------|
| 1 | cifar10 | 1.7191 | N/A | N/A | 0.001000 | 0.057 | N/A | 1.765 | 1.1079 | 72.17 | 1698.2 | 8.0576 | 7.2421 | 54.83 | 1.26 |

## Performance Summary
- **Best Validation Metrics**: cifar10 Accuracy: 54.83, cifar10 Loss: 1.26
- **Final Validation Metrics**: cifar10: {"loss": 1.260988289117813, "accuracy": 54.83}
