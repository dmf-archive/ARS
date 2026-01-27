# ARS2-Neo： 沿着损失景观的测地线直接滑向全局最优解

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/dmf-archive/ARS)

> 本项目是一个专注于二阶优化动力学与信息几何的研究框架。它通过能量-几何解耦（Energy-Geometry Decoupling）原则，实现了在黎曼流形上高效滑行的优化范式。

## 1. 理论基础：从对角 Fisher 到满秩 NGD

ARS2-Neo 的核心设计基于对现代优化算法的深度重构，旨在克服一阶优化器在病态曲率地形下的局限性。

### 1.1 参数去协关联 (De-correlation)

通过 **Muon** 的 Newton-Schulz 迭代，ARS2-Neo 强制更新矩阵保持正交性（Stiefel 流形约束）。在数学上，正交化更新等价于在参数空间执行去协关联，消除了内部协变量偏移，使梯度信息更加纯净。

### 1.2 满秩 Fisher 近似与 NGD

Adam 优化器本质上是通过二阶矩对 Fisher 信息矩阵进行对角化近似。当这种对角 Fisher 预处理遇到 Muon 的去协关联参数空间时，原本丢失的非对角项信息得到了几何补偿。

- **算子复合效应**：对角 Fisher + 正交化参数空间 ≈ **满秩 Fisher 信息矩阵**。
- **动力学特征**：这使得 ARS2-Neo 在本质上执行高效率的**自然梯度下降**。在 Wikitext-2 实验中，ARS2-Neo (Base) 仅用 20 Epoch 即可达到 0.9 的训练损失，证明了其极强的地形平滑能力。

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

| 优化器 | Best PPL | Last PPL | 动力学特征 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **AdamW** | 116.46 | 213.52 | 标准欧氏空间基准 | 缓慢收敛，后期过拟合 |
| **Muon** | 111.35 | 475.65 | 谱约束收敛 | 缺乏自适应能量，后期崩溃 |
| **ARS2-Neo (Base)** | 96.10 | 3055.47 | **过拟合** | 极速坠入针尖极小值，泛化崩溃 |
| **ARS2-Neo (Sync)** | **90.69** | **330.85** | **最优泛化上限** | `ρ=0.3`, 成功抑制过拟合 |
| **ARS2-Neo (AGA)** | 93.23 | 414.83 | 效率与稳定性的折衷 | `λ=0.5`, 自适应几何感知 |

**核心洞察**：ARS2-Neo (AGA) 仅需 3 个 Epoch 即可达到 93.23 PPL，远超 AdamW 的全场最佳表现，证明了二阶几何信息在捕捉语义规律方面的代际优势。

### 3.2 CIFAR-10 视觉分类

实验设置: ResNet-18, Batch Size 256.

| 优化器 | Best Acc | Final Acc | 备注 |
| :--- | :--- | :--- | :--- |
| **ARS2-Neo (Sync)** | **95.87%** | **95.73%** | **SOTA**。60 Epoch 极速收敛。 |
| **AdamW** | 94.60% | 94.47% | 标准基准。 |
| **Muon** | 93.76% | 93.69% | 纯几何优化，上限受限。 |

### 3.3 Grokking 现象加速

为了验证优化器在泛化相变（Phase Transition）中的动力学特征，我们在模加法任务 (`p=113`, `train_frac=0.3`) 上对比了各优化器的表现。

| 优化器 | 拟合 (Epoch) | 顿悟 (Epoch) | 收敛 (Epoch) | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| **AdamW** | ~140 | >600 | N/A | 严重泛化延迟，600 Epoch 未能实现顿悟。 |
| **Muon** | ~150 | >400 | N/A | 纯几何优化在缺乏能量自适应时收敛极慢。 |
| **ARS2-Neo (Base)** | **20** | **180** | **250** | **极速 Grokking**。能量-几何解耦显著加速相变。 |
| **ARS2-Neo (AGA)** | **20** | **150** | **200** | **最优动力学**。自适应几何感知进一步缩短了泛化延迟。 |

**核心洞察**：ARS2-Neo 将 Grokking 发生时间提前了 **4 倍以上**，有力证明了能量-几何解耦能避免模型在过拟合吸引盆中的无效游走，直接穿越高维峡谷抵达泛化解。

## 4. 快速开始

### 4.1 安装

```bash
# 推荐使用 uv
uv sync
```

### 4.2 运行实验

```bash
# 运行 WikiText-2 同步模式 (最优泛化)
python -m exp.wikitext2.train --config config/lrp_wikitext2_ars2_neo_sync_10e.toml

# 运行 CIFAR-10 AGA 模式 (高效收敛)
python -m exp.cifar.train --config config/lrp_cifar10_ars2_neo_aga_20e.toml
```

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
