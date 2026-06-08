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
    "name": "ARS2DC-SAGA",
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
    "beta2_max": 0.9995,
    "rho_kappa": 0.01,
    "rho_eta": 0.1,
    "rho_min": 0.01,
    "rho_max": 1.0,
    "is_dual": true,
    "is_aga": true,
    "is_saga": true,
    "is_christoffel": true
  }
}
```

## Grokking Milestones
| Milestone | Epoch | Train Loss | Eval Acc | Train Acc | PI | Grad Norm |
|-----------|-------|------------|----------|-----------|----|-----------|
| Train Loss < 0.5   (Fitting) | 13 | 0.4498 | 25.383152477905806 | 90.80939947780679 | 0.019552897203476857 | 1.7362 |
| Train Loss < 0.1   (Overfit) | 29 | 0.0865 | 25.517395681843606 | 97.83289817232377 | 0.19536087227812746 | 1.0747 |
| Train Acc > 99%    (Memorization) | 45 | 0.0481 | 31.155610247231234 | 99.03394255874673 | 0.30741176230485845 | 0.8061 |
| Eval Acc > 10%     (Early Signal) | 9 | 2.2522 | 11.231681396129321 | 46.91906005221932 | 0.006685968018129919 | 1.6653 |
| Eval Acc > 50%     (Above Chance) | 58 | 0.0248 | 51.30327777156281 | 99.3733681462141 | 0.4191254300424142 | 0.5252 |
| Eval Acc > 80%     (Strong Signal) | 70 | 0.0175 | 81.66461572882872 | 99.50391644908616 | 0.5058537760672687 | 0.5255 |
| Eval Acc > 90%     (Grokking) | 79 | 0.0090 | 90.06600290860275 | 99.84334203655352 | 0.5336049021423025 | 0.6365 |
| Eval Acc > 95%     (Near Convergence) | 89 | 0.0049 | 95.29030092851549 | 99.92167101827677 | 0.5618006529337644 | 0.5711 |
| Eval Acc > 98%     (Convergence) | 139 | 0.0044 | 98.1765298131782 | 99.9738903394256 | 0.6917422592289889 | 0.4079 |
| Eval Acc > 99%     (Full Convergence) | 204 | 0.0042 | 99.16097997538874 | 99.92167101827677 | 0.7109644379131861 | 0.3263 |

## Performance Summary
- **Best Eval Acc**: 99.16% at Epoch 204
- **Total Epochs**: 204

## Last 5 Epochs (Raw Snapshot)
| Epoch | Train Loss | Eval Acc | Train Acc | PI | Grad Norm |
|-------|------------|----------|-----------|----|-----------|
| 200 | 0.0032 | 98.56807249133013 | 99.94778067885117 | 0.699100990290321 | 0.3269 |
| 201 | 0.0037 | 98.47857702203827 | 99.92167101827677 | 0.7061994724272951 | 0.2487 |
| 202 | 0.0042 | 98.92605436849759 | 99.9738903394256 | 0.7101618635199599 | 0.2795 |
| 203 | 0.0032 | 98.9931759704665 | 99.9738903394256 | 0.7105939615557747 | 0.3227 |
| 204 | 0.0042 | 99.16097997538874 | 99.92167101827677 | 0.7109644379131861 | 0.3263 |
