---
created: 2026-06-26
landed: ~
status: proposed
---

# ARS2E (Einsteinium) — 离散联络与 EFE 驱动的 β 调度

## 0. 原理概述

ARS2E 在 [`1-ARS2`](.roo/rules/1-ARS2.md:1) 的能量-几何解耦之上，用离散 EFE 替代 Christoffel 行列余弦归一化，构造*保幅度的离散联络*以驱动动量衰减 β₁, β₂ 的动态调度。

其核心命题是：ARS2C 的 Christoffel 采样在原理上是正确的——梯度差 `δ_g = g_adv - g_base` 编码了 Hessian-向量积 `H·ĝ`，但后续的*行列余弦归一化*在归一化步骤中丢弃了离散导数的幅度信息，使 `alignment_mean` 在曲率剧烈变化时退化为恒定值。

ARS2E 从 IEST 重新推导离散联络构造，恢复幅度信息。E 代表 Einsteinium（锿），呼应 Einstein 场方程的理论源头。

## 1. ARS2C 的缺陷：行列余弦归一化杀死离散导数

### 1.1 Christoffel 构造（ARS2C 保留部分）

在 SAM 同步步中，梯度差 `δ_g = g_adv - g_base` 近似 Hessian-向量积：

`C = δ_g / (ρ · ĝ + ε)` — 离散 Christoffel 采样（正确）
`c_ortho = NS(C_flat)` — 正交化去除共线冗余（正确）

### 1.2 行列余弦归一化（缺陷所在）

ARS2C 后续实现中：

```
c_row_norm = C / ‖C‖_row
s_row_norm = s_ortho / ‖s_ortho‖_row
row_alignment = (c_row_norm * s_row_norm).sum(dim=1)
# 然后同理做列对齐，几何平均融合
```

*数学上等价于*：

- C 的 Frobenius 范数 `‖C‖_F` 编码了曲率变化的总量——这是 H·ĝ 的物理幅度
- 行归一化后，每行范数变为 1，通道间的曲率差异完全丢失
- 余弦对齐度 `[0, 1]` 只保留方向，丢弃了*这个方向上的曲率变化有多剧烈*

*实验证据*：Grokking 任务中实测 `c_magnitude_mean` 在 4.9→166.6 变化（35×）时，`alignment_mean` 仅从 0.507→0.514，相对变化 < 2%。

### 1.3 后果

`alignment_matrix` 实质上退化为了一个与曲率强度无关的常量调制器。动态 β 失去了对曲率变化的响应能力——β 的确在变化，但它的变化不是由曲率驱动的。

## 2. 离散 EFE 与优化器的映射

`H_ab[γ, Σ] + P_L · Σ_ab γ_ab = κ · T_P · Σ_ab`

各项的优化器映射：

- IEST 度量 γ_ab → 优化器对应 Fisher 信息 G_ij，对角近似 v̂_t → 物理含义 参数空间局部几何
- IEST 能动量 Σ_ab → 优化器对应 梯度流 ∇L → 物理含义 更新动力
- IEST 离散协变导数 Δ_𝒢 → 优化器对应 参数流形上的有限差分 → 物理含义 几何修正
- IEST 曲率 H_ab = Δ_𝒢 Σ_ab + ½ ℛ_𝒢[γ] Σ_ab → 优化器对应 联络调制梯度 → 物理含义 β 调度的几何来源
- IEST 普朗克压强 P_L · Σ_ab γ_ab → 优化器对应 MPS 截断项 ε·√d · g · v̂ → 物理含义 AGAM 触发阈值下界
- IEST 右端项 κ·T_P·Σ_ab → 优化器对应 学习率 × 梯度 η·g → 物理含义 参数更新步

*关键方程*：

`H_ab + P_L · Σ_ab γ_ab = η · g`

左侧两项联合决定了 β 调度（H_ab）和 AGAM 触发（P_L 项）。两者来自同一个方程——这正是等分布变分原理（EVP）同时驱动 ARS 侧和 AGAM 侧的根本原因。

## 3. 保幅度的离散联络构造

### 3.1 从度量差构造离散 Christoffel

替代行列余弦归一化的核心思路——从 Fisher 度量的变化率直接构造联络：

`Γ_ij = Δv̂_ij / (v̂_ij + ε)` — 度量变化的相对率，*保留幅度*

其中 `Δv̂_t = v̂_t - v̂_{t-1}` 是两时间步间 Fisher 对角度量的变化。

*物理含义*：

- `Γ_ij > 0`：该参数方向的曲率正在增大 → 需要更慢的 β（保留历史动量）
- `Γ_ij < 0`：该参数方向的曲率正在减小 → 需要更快的 β（快速适应）
- `|Γ_ij|` 编码了变化幅度

### 3.2 为什么度量差优于梯度差

- 信息来源：δ_g (ARS2C) 来自梯度空间 H·ĝ，Δv̂/v̂ (ARS2E) 来自度量空间 Fisher 变化率
- 幅度保真：ARS2C 行列归一化后丢失，ARS2E 天然保留
- 计算开销：ARS2C 需 SAM 双前向才有 δ_g，ARS2E 每步可算（v̂ 已存在）
- 噪声水平：ARS2C 随批量采样波动大，ARS2E 经 EMA 平滑后稳定
- 曲率符号：ARS2C 的 H·ĝ 只有方向，ARS2E 的 Γ 正负自然编码扩张/收缩

### 3.3 对齐度构造

保幅度的对齐度 = 方向一致 × 幅度调制：

`alignment_preserved = |⟨Γ_norm, v̂_norm⟩| · σ(‖H_ab‖_F / MPS_tau)`

其中：

- `Γ_norm = Γ / ‖Γ‖` — 全局归一化一次（非逐行），保留相对通道差异
- `v̂_norm = v̂ / ‖v̂‖` — 度量方向
- `σ` 是 sigmoid 函数，`‖H_ab‖_F = ‖Γ · g_nat‖_F` 是协变导数的强度
- MPS_tau = ε·√d 是 Model Planck Scale 截断

*关键行为*：

- 度量稳定（`‖Δv̂‖ → 0`）→ `‖H_ab‖_F → 0` → sigmoid → 0 → 对齐度自动衰减
- 度量剧变（Grokking 顿悟期）→ `‖H_ab‖_F` 飙升 → sigmoid → 1 → 对齐度全量激活
- 低曲率时即使方向一致，对齐度也被压低——避免经验性阈值调参

## 4. β 调度：从连续曲率到动态衰减

### 4.1 逐元素 β 调制

ARS2C 已经实现了逐元素 β 矩阵（`exp_avg.mul_(beta1_matrix)`），但 β 矩阵的构造依赖行列余弦归一化。ARS2E 替换为：

`β₁_t = β₁_min + (β₁_max - β₁_min) · alignment_preserved`
`β₂_t = β₂_min + (β₂_max - β₂_min) · alignment_preserved`

*物理含义*：

- alignment_preserved → 1 (高)：度量稳定，曲率方向与更新方向对齐 → β → β_max（强滤波，保留历史）
- alignment_preserved → 0 (低)：度量剧变，曲率方向与更新方向失配 → β → β_min（快速遗忘，适应突变）
- 介于之间：中等曲率变化，连续插值

### 4.2 与 ARS2C 的关键差异

- β 驱动信号：ARS2C 用 row_alignment（幅度死亡），ARS2E 用 ‖H_ab‖_F × cos_dir（幅度保真）
- 低曲率行为：ARS2C 的 alignment 仍为中间值 (0.5)，ARS2E 自动衰减到近 0
- 高曲率行为：ARS2C 的 alignment 仍为中间值 (0.5)，ARS2E 激活到近 1
- 动态范围：ARS2C < 2%，ARS2E > 100×
- 理论来源：ARS2C 来自 Christoffel 类比，ARS2E 来自 IEST

## 5. EVP 联结：ARS 侧与 AGAM 侧的同一来源

等分布变分原理（Equidistribution Variational Principle）同时驱动：

- *ARS 侧*（β 调度）：`H_ab` 项决定每个参数方向的遗忘速率——曲率高的方向快速遗忘，曲率低的方向保留记忆
- *AGAM 侧*（扰动深度）：`P_L · Σ_ab γ_ab` 项提供 MPS 截断——曲率变化不足以被数值精度分辨时，GAM 路径不激活

两者来自同一个离散 EFE 的*同一次展开*：

```
H_ab[γ, Σ]  →  离散联络项  →  β 调度 (ARS2E)
P_L · Σ_ab γ_ab  →  MPS 截断  →  AGAM 阈值 (2-AGAM.md)
```

这意味着 β 和 ρ 不是由两个独立的启发式驱动，而是由同一几何量的两个正交投影决定。

### 5.1 两层架构总结

- 层 ARS 侧，负责 NGD 测地线与 β 调度，机制为离散 EFE → 联络 → β 调制，文档 ARS2E.md (本文)
- 层 SAM/GAM 侧，负责 MDL 拉格朗日与扰动深度，机制为等分布 + MPS → AGAM 触发，文档 AGAM.md

两层通过 EVP 耦合，但设计上保持正交——修改一侧不改变另一侧的理论结构。

## 6. 理论验证要点

1. *保幅验证*：构造沿 Fisher 度量轨迹变化率逐渐放大的合成数据，验证：
   - 旧方案：`alignment_mean` 不随曲率变化
   - ARS2E：`alignment_preserved` 正确响应曲率变化 (>100× 动态范围)

2. *β 动态范围验证*：在 Grokking 顿悟期，验证 β 调度是否自动切换到*快速遗忘*模式

3. *EFE 残差验证*：在真实训练轨迹上计算 `‖H_ab + P_L·Σ·γ - η·g‖ / ‖η·g‖`，验证离散 EFE 的平衡条件

## 7. 与现有 ARS 家族的关系

```
ARS (能量-几何解耦)
  └─ ARS2 (+ SAM 平坦度)
       ├─ ARS2C (+ 行列余弦 β 调制) — 缺陷已诊断，被 ARS2E 取代
       ├─ ARS2E (+ 离散 EFE 联络，保幅度 β 调制) ← 本方案
       └─ AGAM (+ 等分布 + MPS 扰动深度切换) — 独立侧
            └─ 与 ARS2E 通过 EVP 耦合
```

ARS2E 不取代 AGAM。两者正交——ARS2E 处理*沿测地线滑行多远*（β，记忆维度），AGAM 处理*何时做昂贵扰动*（ρ，计算维度）。

## 8. 参考来源

- ARS2C 实现锚点：[`optimizer/ars2c.py:_apply_ars2_kernel`](optimizer/ars2c.py:194)
- AGAM 设计文档：[`2-AGAM.md`](.roo/rules/2-AGAM.md:1)
- AR-GSAM 设计文档：[`2-AR-GSAM.md`](.roo/rules/2-AR-GSAM.md:1)
- ARS2 设计文档：[`1-ARS2.md`](.roo/rules/1-ARS2.md:1)

*状态*：理论定稿，待数值验证。
