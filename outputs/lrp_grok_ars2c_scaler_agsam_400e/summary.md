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
| Train Loss < 0.5   (Fitting) | 13 | 0.3521 | 24.90211433046202 | 93.83812010443864 | 0.023848993282024927 | 1.5015 |
| Train Loss < 0.1   (Overfit) | 23 | 0.0987 | 24.20852444345005 | 97.41514360313316 | 0.16263621470092576 | 1.0459 |
| Train Acc > 99%    (Memorization) | 46 | 0.0366 | 36.5700861393892 | 99.00783289817232 | 0.3453286809836869 | 0.6797 |
| Eval Acc > 10%     (Early Signal) | 9 | 2.2501 | 12.339187828616176 | 47.38903394255875 | 0.0065245766891715426 | 1.6554 |
| Eval Acc > 50%     (Above Chance) | 53 | 0.0253 | 52.36603646940374 | 99.39947780678851 | 0.3849277970130491 | 0.6900 |
| Eval Acc > 80%     (Strong Signal) | 64 | 0.0141 | 80.56829623000336 | 99.84334203655352 | 0.4891804511549941 | 0.5303 |
| Eval Acc > 90%     (Grokking) | 75 | 0.0078 | 90.78196666293769 | 99.86945169712794 | 0.5597256476920869 | 0.5286 |
| Eval Acc > 95%     (Near Convergence) | 81 | 0.0123 | 95.06656225528583 | 99.7911227154047 | 0.5695461451110763 | 0.4791 |
| Eval Acc > 98%     (Convergence) | 123 | 0.0081 | 98.10940821120931 | 99.76501305483029 | 0.6544329015416607 | 0.4526 |
| Eval Acc > 99%     (Full Convergence) | 172 | 0.0045 | 99.02673677145094 | 99.94778067885117 | 0.6780249621632084 | 0.3412 |

## Performance Summary

- **Best Eval Acc**: 99.03% at Epoch 172
- **Total Epochs**: 172

## Last 5 Epochs (Raw Snapshot)

| Epoch | Train Loss | Eval Acc | Train Acc | PI | Grad Norm |
|-------|------------|----------|-----------|----|-----------|
| 168 | 0.0091 | 98.43382928739233 | 99.81723237597912 | 0.6797069522619341 | 0.3426 |
| 169 | 0.0021 | 98.91486743483611 | 100.0 | 0.6762985657610124 | 0.4262 |
| 170 | 0.0043 | 98.47857702203827 | 99.9738903394256 | 0.6749559525288078 | 0.3978 |
| 171 | 0.0090 | 98.4562031547153 | 99.76501305483029 | 0.675633896924601 | 0.3597 |
| 172 | 0.0045 | 99.02673677145094 | 99.94778067885117 | 0.6780249621632084 | 0.3412 |
