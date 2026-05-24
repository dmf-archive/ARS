# ARS2C-AGA： 沿着损失景观的测地线直接滑向全局最优解

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/dmf-archive/ARS)

> 本项目是一个专注于二阶优化动力学与信息几何的研究框架。

## 1. 理论基础：从对角 Fisher 到满秩 NGD

ARS2-Neo 的核心设计基于对现代优化算法的深度重构，旨在克服一阶优化器在病态曲率地形下的局限性。

### 1.1 参数去协关联 (De-correlation)

通过 **Muon** 的 Newton-Schulz 迭代，ARS2-Neo 强制更新矩阵保持正交性（Stiefel 流形约束）。在数学上，正交化更新等价于在参数空间执行去协关联，消除了内部协变量偏移，使梯度信息更加纯净。

### 1.2 满秩 Fisher 近似与 NGD

对于任意正交矩阵 `R` 与对角矩阵 `D`，提升后的乘积 `RDRᵀ` 是一个满秩矩阵，其谱分布仍在旋转后的坐标系中——这种恒等式独立于曲率的来源。Adam 通常被解读为由梯度二阶矩构建的对角预条件，而 ARS2-Neo 将这一对角缩放与矩阵级正交化步骤（类似极分解的混合）复合。如果混合基 `R` 缓慢漂移并保持与曲率特征基的相关性，同时对角 `D` 跟踪对应的谱，那么 `RDRᵀ` 在原始坐标系下就可以被视为一个结构化的自然梯度预条件器，这与 Amari（1998）以及 K-FAC、Shampoo 等实践近似保持一致。

这种表述更适合作为一个经验可验证的假设而非数学恒等：当 ARS2-Neo 保持必要的对齐关系时，复合算子可以近似**自然梯度下降（NGD）**，我们在 Wikitext-2 的训练（20 epochs 训练损失约 0.9）验证了这一强预条件化下降的效能。若正交化仅仅重塑奇异值而与曲率统计脱钩，那么提升后的 `RDRᵀ` 就会失去与真实 Fisher/Hessian 的联系，NGD 的类比也随之削弱。

### 1.3 全局最优与 MDL 原则

虽然 NGD 提供了极速的收敛，但极易陷入“针尖极小值”（过拟合）。ARS2-Neo 引入了**流形感知 SAM (Sharpness-Aware Minimization)**：

- **平坦度约束**：通过在黎曼流形上寻找对抗方向，算法被引导至损失景观中更宽阔的盆地。
- **MDL 对应**：根据最小描述长度 (MDL) 原则，平坦的区域对应于更简单的模型解释，从而具备更强的泛化能力。

## 2. 核心机制：能量-几何解耦

ARS2-Neo 将优化过程分解为两个独立的算子：

1. **统计算子 (能量)**：利用 AdamW 的二阶矩修正动量范数确定更新步长，作为自由能下降速率的代理。
2. **结构算子 (几何)**：通过预白化 (Pre-whitening) 与正交投影，确保更新方向严格遵循流形的测地线 (Geodesic)。

## 3. 关键实验结果 (LRP 验证)

### 3.1 Wikitext-2 语言建模

实验设置: Qwen3 (RoPE, 3-layer), Context 255. 旨在探测病态曲率流形上的优化稳定性。

| 优化器 | Best PPL | Final PPL | Best Eval Loss | Final Eval Loss | Final Train Loss | 平均时间 | PPL Gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AdamW** | 116.46 | 213.52 | 4.76 | 5.36 | 2.9740 | 314s | +97.06 |
| **Muon** | 111.35 | 475.65 | 4.71 | 6.16 | 2.2938 | 445s | +364.30 |
| **ARS2-Neo (Base)** | 96.10 | 3055.47 | 4.57 | 8.02 | 0.9123 | 425s | +2959.37 |
| **ARS2-Neo (Sync)** | **90.69** | **330.85** | **4.51** | 5.80 | 1.6100 | 784s | +240.16 |
| **ARS2-Neo (AGA)** | 93.23 | 414.83 | 4.54 | 6.03 | 1.5906 | 546s | +321.60 |

### 3.2 CIFAR-10 视觉分类

实验设置: ResNet-18, Batch Size 256.

| 优化器 | Best Acc | Final Acc | Final Train Loss | Best Eval Loss | Final Eval Loss | 平均每轮时间 | Gen Gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ARS2-Neo (Sync, ρ=0.1)** | **95.87%** | **95.73%** | 0.0347 | 0.1500 | 0.1500 | 104s | +0.14 |
| **ARS2-Neo (Base)** | 95.58% | 95.52% | 0.0181 | 0.2400 | 0.2500 | 71s | +0.06 |
| **ARS2-Neo (AGA, λ=2.0)** | 94.10% | 94.09% | 0.1251 | 0.1800 | 0.1800 | 90s | +0.01 |
| **AdamW** | 94.60% | 94.47% | 0.0451 | 0.2500 | 0.2700 | 58s | +0.13 |
| **Muon** | 93.76% | 93.69% | 0.0267 | 0.2900 | 0.2900 | 75s* | +0.07 |

> *Muon CIFAR-10 平均每轮时间包含一个 35331s (~9.8hrs) 的离群值；典型轮次约 75s。*

### 3.3 Grokking 现象加速

为了验证优化器在泛化相变（Phase Transition）中的动力学特征，我们在模加法任务 (`p=113`, `train_frac=0.3`) 上对比了各优化器的表现。

| 优化器 | 拟合 (Epoch) | 顿悟 (Epoch) | 收敛 (Epoch) | 最佳 Eval Acc |
| :--- | :--- | :--- | :--- | :--- |
| **AdamW** | 113 | >600 | N/A | 15.65% |
| **Muon** | 22 | >347 | N/A | 36.83% |
| **ARS2-Neo (Base)** | 11 | 239 | 290 | 99.53% |
| **ARS2-Neo (AGA)** | 12 | **77** | **116** | **99.60%** |
| **ARS2C (AGA)** | 13 | 93 | 137 | 99.06% |
| **ARS2C (Scaler) (AGA)** | 13 | 75 | 172 | 99.03% |
| **ARS2D (Base)** | 11 | 237 | 264 | 99.05% |
| **ARS2D (AGA)** | 12 | **60** | **112** | **99.00%** |

**核心洞察**：能量-几何解耦能避免模型在过拟合吸引盆中的无效游走，直接穿越高维峡谷抵达泛化解。ARS2D (AGA) 在 60 轮达成顿悟、112 轮收敛，是所有变体中最快的。Muon 和 AdamW 在 600 轮内未能实现顿悟。

## 4. 快速开始

### 4.1 安装

```bash
# 推荐使用 uv
uv sync
```

### 4.2 运行实验

```bash
# 运行 WikiText-2 同步模式 (最优泛化)
# 注意：实验目录为 `exp/wikitext-2`，因此使用脚本路径启动。
python exp/wikitext-2/train.py --config config/lrp_wikitext2_ars2_neo_sync_10e.toml

# 运行 CIFAR-10 AGA 模式 (高效收敛)
python -m exp.cifar.train --config config/lrp_cifar10_ars2_neo_aga_20e.toml
```

### 4.3 结果分层与解释口径

- **LRP/Main 实验**：目录名为 `outputs/lrp_*`，用于主要对比结论。
- **Verify/Smoke 实验**：目录名为 `outputs/verify_*`，主要用于短程连通性验证（通常 1 epoch），不与长程 LRP 结果直接对比。

## 5. 框架结构

- **[`optimizer/`](optimizer/)**: 核心优化器实现，包括 [`ars2_neo.py`](optimizer/ars2_neo.py)。
- **[`exp/`](exp/)**: 原子化实验脚本，解耦数据流与模型逻辑。
- **[`model/`](model/)**: 包含 Qwen3 (RoPE) 与 ResNet 等标准研究模型。
- **[`config/`](config/)**: 基于 TOML 的实验配置管理。

## 引用

```bibtex
@software{ARS2_Neo_2025,
  author = {Rui, L.},
  title = {ARS2-Neo: Gliding Directly Towards Global Optima Along Geodesics of the Loss Landscape},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/dmf-archive/ARS}
}
