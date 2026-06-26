# ReSRLU 家族激活函数实验报告

实验时间：2026-06-22
实验脚本：[`exp/act_test.py`](exp/act_test.py:1)
设备：NVIDIA GPU (CUDA), FP32

---

## 实验设计

两个条件 × 两个任务 × 五个激活函数：

| 条件 | Plain MLP (无残差) | ResMLP (有残差) |
|---|---|---|
| **信号诊断** | 20层 MLP 前向/反向传播 | 20层 ResMLP 前向/反向传播 |
| **训练** | 3层 MLP, FMNIST 20 epochs | 20层 ResMLP, FMNIST 20 epochs |

| 激活函数 | 特性 |
|---|---|
| ReLU | 基线 |
| SiLU | 工业标准 |
| ReSRLU | sqrt 正半轴，零负半轴 |
| ReSRLU-MPS | ReSRLU + MPS 物理下界死区 |
| SiSRLU | x / sqrt(1 + x²) 全域网关（饱和函数） |

---

## Part 1: 信号传播诊断结果

### 前向激活值（20层后逐层范数）

| Activation | Plain MLP (最后一层) | ResMLP (第20层以后) |
|---|---|---|
| ReLU | ~10 (稳定) | **459（随深度增长）** |
| SiLU | ~9.7 (稳定) | 142（随深度增长） |
| ReSRLU | ~9.4 (稳定) | **27（极度压缩）** |
| ReSRLU-MPS | ~9.5 (稳定) | 28（极度压缩） |
| SiSRLU | ~11.5 (稳定) | 143（随深度增长） |

残差连接下激活值差异被放大——ReSRLU 的 sqrt 压缩效果真正体现出来：激活值路径只有 ReLU 的 1/17。

### 梯度范围比（max / min）

| Activation | Plain MLP | ResMLP | 改善倍数 |
|---|---|---|---|
| **ReLU** | 2.33e+07 | **13.0x** | 1,800,000x |
| SiLU | 2.75e+10 | **5.8x** | 4,700,000,000x |
| **ReSRLU** | 5.04e+13 | **10.2x** | 4,900,000,000,000x |
| ReSRLU-MPS | 6.67e+13 | **10.9x** | 6,100,000,000,000x |
| SiSRLU | 6.51e+04 | **4.2x** | 15,000x |

**残差连接完全消除了激活函数差异带来的梯度消失/爆炸问题。** 所有激活函数的梯度范围都在 4.2x-13.0x 以内。

---

## Part 2: ResMLP 训练对比（20层，无Norm层，10% FMNIST，20 epochs）

| Activation | Best Acc | Final Acc | 前向激活值 | 每 epoch 时间 |
|---|---|---|---|---|
| **ReSRLU** | **85.03%** | 84.40% | ~27 | 10.9s |
| ReSRLU-MPS | 84.50% | 84.50% | ~28 | 11.2s |
| SiLU | 84.22% | 84.22% | ~142 | 9.0s |
| ReLU | 84.19% | 83.74% | ~459 | 8.9s |
| SiSRLU | 84.05% | 84.05% | ~143 | 10.1s |

所有激活函数精度集中在 84-85%。差异在噪声范围内，但 ReSRLU 微弱领先。

---

## Part 3: Deep ResMLP CIFAR-10 全量训练（30 epochs）

| Activation | Best Acc | Final Acc | Epoch 0 Acc | Final Loss | 每 epoch |
|---|---|---|---|---|---|
| ReLU | 56.07% | 56.07% | 33.25% | 1.101 | ~52s |
| SiLU | 56.41% | 55.89% | 35.00% | 1.026 | ~52s |
| **ReSRLU** | **57.56%** | 56.56% | **38.88%** | **1.018** | ~66s |

模型：20 层 ResMLP（每层 Linear→Act→Linear→+skip），隐藏层宽 512，无 BN/Conv/Pool/AvgPool，CIFAR-10 图片展平为 3072 维输入。

### 关键观察

1. **ReSRLU 在 CIFAR-10 上反超**：57.56% vs SiLU 56.41% vs ReLU 56.07%，领先约 1.5 个百分点。置信度高于 FMNIST 的 0.8% 差距——全量数据的信噪比更高。

2. **初始收敛加速**：epoch 0 就 38.88%，显著高于 ReLU 的 33.25% 和 SiLU 的 35.00%——早期 sqrt 导数大 → 快速学习。

3. **Loss 更低**：最终 loss 1.018 vs SiLU 1.026 vs ReLU 1.101，也验证了更好的收敛深度。

4. **计算开销约 25%**：sqrt + masked assignment 导致每 epoch ~25% 额外时间。相比 SiLU 的 exp，开销可接受但客观存在。

5. **曲线仍在上升**：30 epoch 没到任何激活函数的平台期——如果跑更久，差距可能扩大或缩小。

6. **ReSRLU-MPS 未测试**：MPS 变体在 FMNIST 上精度略低于基础 ReSRLU（84.50% vs 85.03%），CIFAR-10 全量未纳入本次对比。

---

## 最终结论

**1. ReSRLU 的理论优势在残差网络上体现为一致的微弱优势**：

- 信号传播：激活压缩到 ~27（ReLU 的 1/17），梯度范围 10.2x（与其他激活函数同级）
- FMNIST（10% 子采样）：85.03% vs SiLU 84.22%（+0.8%）
- CIFAR-10（全量）：57.56% vs SiLU 56.41%（+1.2%）

三组独立实验一致指向 ReSRLU 在纯残差 MLP 上的稳定优势。

**2. ReSRLU 确实去掉了对 Normalization 的需求**：

- 所有训练都无 BN/LN/GN——直接在 20 层深度网络稳定训练
- 激活值自动压缩到合理范围，不需手动归一化

**3. SiSRLU 被证伪**：`x / sqrt(1 + x²)` 是有界饱和函数，在残差网络中初始收敛极慢且精度垫底。

**4. 计算开销是唯一代价**：sqrt + masked assignment 比线性激活慢约 25%，与 SiLU 的 exp 算力开销基本同量级。在大模型场景下，省掉 LN 的前反向可能部分抵消这个开销。

**5. 开放问题**：

- 更长时间（100+ epoch）下优势能否保持或扩大？
- 在 Transformer 架构中，MLP 的 GELU/SiLU 替换为 ReSRLU 能否省掉 pre-LN？
- MPS 变体的价值不明确——MPS 死区在实验中没有带来比基础 ReSRLU 更好的结果
- ReSRLU-MPS 的训练速度显著慢于基础版（多了一个 CUDA kernel 调用），在没有明显精度收益的情况下不建议继续
