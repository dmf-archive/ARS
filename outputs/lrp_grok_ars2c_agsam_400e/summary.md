# ARS-Bench Experiment Report — Grokking Milestone Mode

## Configuration Summary

```json
{
  "experiment": {
    "seed": 42,
    "device": "cuda",
    "tasks": [
      "mod_addition"
    ],
    "epochs": 400,
    "save_interval": 100
  },
  "task": {
    "p": 113,
    "fraction": 0.3
  },
  "data": {
    "batch_size": 512,
    "num_workers": 0
  },
  "model": {
    "arch": "grokking_transformer",
    "num_layers": 1,
    "d_model": 128,
    "d_mlp": 512,
    "d_head": 32,
    "num_heads": 4,
    "n_ctx": 3,
    "act_type": "ReLU"
  },
  "optimizer": {
    "name": "ARS2C",
    "lr": 0.001,
    "rho": 0.1,
    "adaptive_sync": true,
    "adaptive_lambda": 0.1,
    "k": 10,
    "ns_steps": 5,
    "betas": [
      0.9,
      0.98
    ],
    "weight_decay": 1.0,
    "beta1_min": 0.9,
    "beta1_max": 0.9995,
    "beta2_min": 0.9,
    "beta2_max": 0.9995
  }
}
```

## Grokking Milestones

| Milestone | Epoch | Train Loss | Eval Acc | Train Acc | PI | Grad Norm |
|-----------|-------|------------|----------|-----------|----|-----------|
| Train Loss < 0.5   (Fitting) | 13 | 0.3665 | 26.009620762948877 | 94.30809399477806 | 0.022420473611961023 | 1.5249 |
| Train Loss < 0.1   (Overfit) | 16 | 0.0782 | 27.3408658686654 | 99.08616187989556 | 0.07618491190249095 | 1.3683 |
| Train Acc > 99%    (Memorization) | 16 | 0.0782 | 27.3408658686654 | 99.08616187989556 | 0.07618491190249095 | 1.3683 |
| Eval Acc > 10%     (Early Signal) | 9 | 2.2005 | 11.768654211880524 | 48.38120104438642 | 0.006665547404129054 | 1.6582 |
| Eval Acc > 50%     (Above Chance) | 71 | 0.0146 | 51.69482044971473 | 99.66057441253264 | 0.4771378028068956 | 0.4892 |
| Eval Acc > 80%     (Strong Signal) | 83 | 0.0078 | 80.12081888354402 | 99.89556135770235 | 0.5425889543365967 | 0.4448 |
| Eval Acc > 90%     (Grokking) | 93 | 0.0083 | 90.37923705112429 | 99.84334203655352 | 0.5862532558543001 | 0.5941 |
| Eval Acc > 95%     (Near Convergence) | 102 | 0.0061 | 95.11130998993175 | 99.92167101827677 | 0.6327040674920817 | 0.3505 |
| Eval Acc > 98%     (Convergence) | 118 | 0.0031 | 98.27721221613156 | 100.0 | 0.6312364798586646 | 0.5326 |
| Eval Acc > 99%     (Full Convergence) | 137 | 0.0037 | 99.0602975724354 | 99.9738903394256 | 0.6666317234275867 | 0.3768 |

## Performance Summary

- **Best Eval Acc**: 99.06% at Epoch 137
- **Total Epochs**: 137

## Last 5 Epochs (Raw Snapshot)

| Epoch | Train Loss | Eval Acc | Train Acc | PI | Grad Norm |
|-------|------------|----------|-----------|----|-----------|
| 133 | 0.0026 | 98.40026848640788 | 100.0 | 0.662981198040892 | 0.3432 |
| 134 | 0.0044 | 98.48976395569974 | 99.92167101827677 | 0.6632741803380847 | 0.3899 |
| 135 | 0.0056 | 98.12059514487079 | 99.92167101827677 | 0.6672759602259537 | 0.3290 |
| 136 | 0.0044 | 98.76943729723683 | 99.94778067885117 | 0.6656887536497147 | 0.4095 |
| 137 | 0.0037 | 99.0602975724354 | 99.9738903394256 | 0.6666317234275867 | 0.3768 |
