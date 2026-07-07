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
    "name": "ARS2-Neo-AGAM",
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
    "weight_decay": 1.0
  }
}
```

## Grokking Milestones
| Milestone | Epoch | Train Loss | Eval Acc | Train Acc | PI | Grad Norm |
|-----------|-------|------------|----------|-----------|----|-----------|
| Train Loss < 0.5   (Fitting) | 11 | 0.4883 | 16.9705783644703 | 88.09399477806788 | 0.024210510127702235 | 1.7008 |
| Train Loss < 0.1   (Overfit) | 17 | 0.0985 | 17.955028526680838 | 97.38903394255874 | 0.17158789792359427 | 0.8910 |
| Train Acc > 99%    (Memorization) | 34 | 0.0402 | 24.92448819778499 | 99.00783289817232 | 0.28742175248454227 | 0.6797 |
| Eval Acc > 10%     (Early Signal) | 8 | 2.2756 | 10.96319498825372 | 46.37075718015666 | 0.007699274858915704 | 1.5586 |
| Eval Acc > 50%     (Above Chance) | 53 | 0.0687 | 50.00559346683074 | 97.911227154047 | 0.4114925881502144 | 0.8349 |
| Eval Acc > 80%     (Strong Signal) | 67 | 0.0138 | 80.02013648059066 | 99.68668407310705 | 0.6299272487384695 | 0.2922 |
| Eval Acc > 90%     (Grokking) | 81 | 0.0318 | 91.48674348361114 | 99.13838120104438 | 0.6044723872324275 | 0.5709 |
| Eval Acc > 95%     (Near Convergence) | 86 | 0.0059 | 95.18961852556214 | 99.76501305483029 | 0.6444942357845737 | 0.2600 |
| Eval Acc > 98%     (Convergence) | 131 | 0.0043 | 98.07584741022485 | 99.92167101827677 | 0.6967783359922426 | 0.2363 |
| Eval Acc > 99%     (Full Convergence) | 181 | 0.0008 | 99.00436290412797 | 100.0 | 0.7514696441035702 | 0.3827 |

## Performance Summary
- **Best Eval Acc**: 99.00% at Epoch 181
- **Total Epochs**: 181

## Last 5 Epochs (Raw Snapshot)
| Epoch | Train Loss | Eval Acc | Train Acc | PI | Grad Norm |
|-------|------------|----------|-----------|----|-----------|
| 177 | 0.0068 | 97.97516500727151 | 99.84334203655352 | 0.7319640581608547 | 0.2781 |
| 178 | 0.0046 | 98.16534287951673 | 99.86945169712794 | 0.7398114858951761 | 0.1998 |
| 179 | 0.0040 | 98.15415594585524 | 99.92167101827677 | 0.7488305860343896 | 0.1753 |
| 180 | 0.0018 | 98.3219599507775 | 99.9738903394256 | 0.7595325137029231 | 0.1486 |
| 181 | 0.0008 | 99.00436290412797 | 100.0 | 0.7514696441035702 | 0.3827 |
