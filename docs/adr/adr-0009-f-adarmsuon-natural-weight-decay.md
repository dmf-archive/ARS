---
title: "ADR-0009: F-AdaRMSuon: Natural Weight Decay Integration"
status: "Proposed"
date: "2025-12-23"
authors: "Ω Researcher"
tags: ["optimizer", "architecture", "fep"]
supersedes: ""
superseded_by: ""
---

# ADR-0009: F-AdaRMSuon: Natural Weight Decay Integration

## 状态 (Status)

**Proposed** | Accepted | Rejected | Superseded | Deprecated

## 背景 (Context)

当前的 `AdaRMSuon` 优化器实现了“能量-几何解耦”范式，它首先通过 Fisher 预白化计算出近似的“不变自然梯度”，然后将其分解为“能量”（标量范数）和“几何”（正交方向），最后再重新组合进行更新。

然而，其权重衰减机制 (`weight_decay`) 沿用了 AdamW 的标准解耦方法，即 `p.mul_(1 - lr * wd)`。这是一个在欧氏空间 (Euclidean space) 中的操作。这导致了优化器内部的几何不一致性：主要的参数更新发生在预条件化的信息流形上，而正则化项却在原始的、平坦的欧氏空间中施加。

FAdam 论文 (arXiv:2405.12807) 从第一性原理出发，为权重衰减提供了一个信息几何的解释。它将权重衰减视为参数上的高斯先验，在贝叶斯推断框架下，对先验的梯度同样需要通过 Fisher 信息矩阵进行预条件化，从而得到“自然权重衰减” (Natural Weight Decay)。其形式为 `g_wd = θ / sqrt(F)`，其中 `F` 是 Fisher 信息矩阵。

## 决策 (Decision)

我们将引入一个名为 `F-AdaRMSuon` 的新优化器变体（或直接在 `AdaRMSuon` 中实现），该变体将集成“自然权重衰减”机制。

具体的修改如下：

1. 废除 `p.mul_(1 - lr * wd)` 形式的欧氏权重衰减。
2. 将权重衰减项作为一个独立的、经过预条件化的“力”加入到最终的参数更新中。
3. 更新规则将从：
    `p.add_(update, alpha=-lr)`
    `p.mul_(1 - lr * wd)`
    修改为统一的更新步骤：
    `natural_wd = p * weight_decay / (v_hat.sqrt() + eps)`
    `p.add_(update, alpha=-lr)`
    `p.add_(natural_wd, alpha=-lr)`
    或者更简洁的：
    `p.add_(-lr, update + natural_wd)`

此决策旨在使正则化项与梯度更新项在几何上保持一致，共同在信息流形上进行优化。

## 后果 (Consequences)

### 积极 (Positive)

- **POS-001**: **理论一致性 (Theoretical Consistency)**: 解决了梯度更新与正则化项之间的几何不一致性，使整个优化过程在信息几何的框架下更加自洽。
- **POS-002**: **自适应正则化 (Adaptive Regularization)**: 实现了一种自适应的奥卡姆剃刀。对于 Fisher 信息量低（`v_hat` 小，通常对应不重要或冗余的）参数，施加更强的权重衰减；对于信息量高（`v_hat` 大）的参数，则予以保护。
- **POS-003**: **对齐 MDL 原则 (Alignment with MDL)**: 有望通过自动剪除不重要的参数来更有效地压缩模型的描述长度，从而可能提升泛化能力并发现更稀疏或低秩的解。

### 消极 (Negative)

- **NEG-001**: **计算开销 (Computational Cost)**: 投影操作 `(p * s_ortho).sum()` 引入了额外的矩阵元素乘和全局求和，略微增加了计算成本。
- **NEG-002**: **能量耦合 (Energy Coupling)**: 衰减能量现在与参数 `p` 和方向 `s_ortho` 的对齐程度有关，这是一种新的动态，其效果需要实验验证。

## 考虑的备选方案 (Alternatives Considered)

### 逐元素自然权重衰减 (F-AdaRMSuon)

- **ALT-001**: **描述 (Description)**: 实现一个逐元素的、由 `v_hat` 加权的衰减项。
- **ALT-002**: **拒绝理由 (Rejection Reason)**: **已被实验证伪**。该方法破坏了 Muon 更新的低秩几何结构，导致性能显著下降。

### 维持现状 (AdamW-style Decay)

- **ALT-003**: **描述 (Description)**: 继续使用当前的 AdamW 式解耦权重衰减。
- **ALT-004**: **拒绝理由 (Rejection Reason)**: 理论上不完备，放弃了利用已计算出的几何信息来指导正则化的机会。

## 实施注意事项 (Implementation Notes)

- **IMP-001**: 实现时需特别注意 `eps` 的使用，确保在 `v_hat` 极小的情况下分母不会为零。
- **IMP-002**: 应首先在 `wikitext-2` 等标准任务上进行消融实验，对比 `AdaRMSuon` 和 `F-AdaRMSuon` 在相同超参数下的性能差异，以评估其影响。
- **IMP-003**: 监控训练过程中参数的范数和 `v_hat` 的分布，以验证自然权重衰减是否按预期工作（即对 `v_hat` 小的参数施加更强衰减）。

## 参考文献 (References)

- **REF-001**: `docs/adr/adr-0006-energy-geometry-decoupling-analysis.md`
- **REF-002**: `FAdam: Adam is a natural gradient optimizer using diagonal empirical Fisher information` (arXiv:2405.12807)
- **REF-003**: `Decoupled Weight Decay Regularization` (AdamW, arXiv:1711.05101)
