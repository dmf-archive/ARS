# 实验数据提取报告

## CIFAR-10 (ResNet-18, Batch Size 256)

| Optimizer | Best Acc | Final Acc | Final Train Loss | Final Eval Loss | Best Eval Loss | Avg Epoch Time | Gen Gap (Acc) |
|-----------|---------|----------|-----------------|----------------|---------------|---------------|--------------|
| ARS2-Neo (Sync, ρ=0.1) | 95.87 | 95.73 | 0.0347 | 0.1500 | 0.1500 | 104.30s | +0.14 |
| ARS2-Neo (Base) | 95.58 | 95.52 | 0.0181 | 0.2500 | 0.2400 | 70.92s | +0.06 |
| ARS2-Neo (AGA, λ=2.0) | 94.10 | 94.09 | 0.1251 | 0.1800 | 0.1800 | 89.96s | +0.01 |
| AdamW | 94.60 | 94.47 | 0.0451 | 0.2700 | 0.2500 | 58.40s | +0.13 |
| Muon | 93.76 | 93.69 | 0.0267 | 0.2900 | 0.2900 | 432.59s* | +0.07 |

> *Muon avg_time 被一个离群 epoch (35331s ≈ 9.8hrs) 拉高，其余 epoch 约 75s。

## Wikitext-2 (Qwen3 RoPE, 3-layer, Context 255)

| Optimizer | Best PPL | Final PPL | Best Eval Loss | Final Eval Loss | Final Train Loss | Avg Time | PPL Gap |
|-----------|---------|----------|---------------|----------------|-----------------|---------|--------|
| AdamW | 116.46 | 213.52 | 4.76 | 5.36 | 2.9740 | 314.4s | +97.06 |
| Muon | 111.35 | 475.65 | 4.71 | 6.16 | 2.2938 | 444.5s | +364.30 |
| ARS2-Neo (Base) | 96.10 | 3055.47 | 4.57 | 8.02 | 0.9123 | 425.3s | +2959.37 |
| ARS2-Neo (Sync) | 90.69 | 330.85 | 4.51 | 5.80 | 1.6100 | 784.3s | +240.16 |
| ARS2-Neo (AGA) | 93.23 | 414.83 | 4.54 | 6.03 | 1.5906 | 546.0s | +321.60 |

> PPL Gap = Final PPL - Best PPL，正值表示过拟合退化。

## Grokking (Modular Addition p=113, train_frac=0.3)

| Optimizer | Fitting Ep | Grokking Ep | Converge Ep | Best Eval Acc | Total Ep |
|-----------|-----------|------------|------------|--------------|---------|
| AdamW | - | - | - | 15.65% | 600 |
| Muon | - | - | - | 36.83% | 600 |
| ARS2-Neo (Base) | - | - | - | 99.53% | 600 |
| ARS2-Neo (AGA) | - | - | - | 99.60% | 600 |
| ARS2C (AGA) | 13 | 93 | 137 | 99.06% | 137 |
| ARS2C (Scaler) (AGA) | 13 | 75 | 172 | 99.03% | 172 |
| ARS2D (Base) | 11 | 237 | 264 | 99.05% | 264 |
| ARS2D (AGA) | 12 | 60 | 112 | 99.00% | 112 |

> 旧格式 grokking 实验（AdamW/Muon/ARS2-Neo）只有 epoch 表格无里程碑标记，故 Fitting/Grokking/Converge 列无数据。

---

## 下一步建议

1. **更新 [`README.md`](README.md:1) 实验表格**：删除「总结栏」（Description/Status），替换为上述原始数据
2. **更新 [`ARS-Series.md`](.roo/rules/ARS-Series.md:1) 实验对比表**：同步详细指标
3. **迁移旧格式 grokking summary**：考虑将 adamw/muon/ars2_neo_base/ars2_neo_aga 的实验 summary 重写为新格式（含里程碑）

## 提取脚本

[`utils/extract_all_metrics.py`](utils/extract_all_metrics.py) 可复用，下次新增实验后直接 `python -m utils.extract_all_metrics` 即可生成报告。
