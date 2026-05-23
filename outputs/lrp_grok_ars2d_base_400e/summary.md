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
    "name": "ARS2D",
    "lr": 0.001,
    "rho": 0.0,
    "k": 0,
    "ns_steps": 5,
    "betas": [
      0.9,
      0.98
    ],
    "weight_decay": 1.0
  }
}
```

## Grokking Milestones
| Milestone | Epoch | Train Loss | Eval Acc | Train Acc | PI | Grad Norm |
|-----------|-------|------------|----------|-----------|----|-----------|
| Train Loss < 0.5   (Fitting) | 11 | 0.4302 | 15.628146325092292 | 88.7467362924282 | 0.029120937408096147 | 1.5419 |
| Train Loss < 0.1   (Overfit) | 26 | 0.0986 | 21.478912630048104 | 97.02349869451697 | 0.2017525748313694 | 1.0455 |
| Train Acc > 99%    (Memorization) | 30 | 0.0354 | 24.387515382033783 | 99.08616187989556 | 0.29470176274178067 | 0.5448 |
| Eval Acc > 10%     (Early Signal) | 8 | 2.0448 | 11.119812059514487 | 50.469973890339425 | 0.007971207159726474 | 1.7167 |
| Eval Acc > 50%     (Above Chance) | 197 | 0.0000 | 51.001230562702766 | 100.0 | 0.9999970901780856 | 0.0000 |
| Eval Acc > 80%     (Strong Signal) | 232 | 0.1117 | 81.55274639221389 | 97.25848563968668 | 0.5812574404181627 | 1.5033 |
| Eval Acc > 90%     (Grokking) | 237 | 0.0005 | 90.37923705112429 | 100.0 | 0.6514639760903636 | 0.0172 |
| Eval Acc > 95%     (Near Convergence) | 250 | 0.0009 | 95.36860946414588 | 100.0 | 0.9023591186579646 | 0.0064 |
| Eval Acc > 98%     (Convergence) | 258 | 0.0010 | 98.07584741022485 | 100.0 | 0.9494308733577312 | 0.0049 |
| Eval Acc > 99%     (Full Convergence) | 264 | 0.0008 | 99.04911063877391 | 100.0 | 0.9671845411567807 | 0.0035 |

## Performance Summary
- **Best Eval Acc**: 99.05% at Epoch 264
- **Total Epochs**: 264

## Last 5 Epochs (Raw Snapshot)
| Epoch | Train Loss | Eval Acc | Train Acc | PI | Grad Norm |
|-------|------------|----------|-----------|----|-----------|
| 260 | 0.0009 | 98.47857702203827 | 100.0 | 0.956416886443323 | 0.0044 |
| 261 | 0.0009 | 98.62400715963754 | 100.0 | 0.9594544423172742 | 0.0043 |
| 262 | 0.0009 | 98.76943729723683 | 100.0 | 0.9622407900586638 | 0.0040 |
| 263 | 0.0008 | 98.98198903680502 | 100.0 | 0.9648063552941327 | 0.0038 |
| 264 | 0.0008 | 99.04911063877391 | 100.0 | 0.9671845411567807 | 0.0035 |
