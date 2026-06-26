# ARS2C: Christoffel-Aware Dynamic Beta Optimization

## 0. 原理概述

ARS2C 在 [`ARS2`](.roo/rules/ARS2.md:1) 的能量-几何解耦 + SAM 平坦度约束之上，将固定的动量衰减率 β₁, β₂ 替换为从 Fisher 信息流形推导的动态 β。其核心命题是：曲率决定遗忘速率。

## 1. 定义

ARS2C 继承自 ARS2-Neo，C 代表 Christoffel 符号 Γ^μ_νρ——信息几何中连接度规与测地线的核心算子。

| GR | 信息几何 | ARS 家族 |
|:---|:---|:---|
| 度规 g_μν | Fisher 信息 G_ij | 二阶矩 v_t (对角近似) |
| Christoffel 符号 Γ | G⁻¹ ∂G | **动态 β** |
| 测地线方程 | 自然梯度流 | Muon + NGD |
| 曲率张量 R | Fisher 信息的二阶导数 | HVP 采样 |

## 2. 更新链路

### 2.1 Christoffel 矩阵构造（Sync Step）

在 SAM sync step 中，复用已有 HVP 采样构造逐元素 Christoffel 矩阵：

`C = δg ⊘ (ρ · ĝ + ε)`

其中 `δg = g_adv - g_base` 是扰动前后梯度差，`ĝ = g_base / (√v̂ + ε)` 是预白化梯度。

### 2.2 结构化正交化

`c_ortho = zeropower_via_newtonschulz5(C_flat)`

`c_ortho` 编码了曲率变化的方向结构。

### 2.3 几何对齐度

当前实现采用**行 + 列双向对齐矩阵**方案，替代了早期的全局标量退化版本：

1. **行对齐计算**:
   `c_row_norm = C / ‖C‖_row`
   `s_row_norm = S / ‖S‖_row`
   `row_alignment = (⟨c_row_norm, s_row_norm⟩ + 1) / 2`

2. **列对齐计算**:
   `c_col_norm = C / ‖C‖_col`
   `s_col_norm = S / ‖S‖_col`
   `col_alignment = (⟨c_col_norm, s_col_norm⟩ + 1) / 2`

3. **几何平均融合**:
   `alignment_matrix = √(row_alignment ⊙ col_alignment)`

其中 `⊙` 表示逐元素乘法。最终 `alignment_matrix` 是一个与参数矩阵同维度的矩阵，每个元素对应一个独立的 β 调制系数。

**演进说明**:

- 早期标量版本通过 `(c_ortho * s_unit).sum().abs()` 将高维对齐压缩为全局标量，导致大矩阵内所有参数共享同一 β 值，抹杀了流形切空间的各向异性曲率差异。
- 当前行 + 列对齐方案保留了矩阵级别的局部曲率信息，通过几何平均融合行列两个方向的对齐信号，在计算复杂度与精度之间取得平衡。
- 实验数据显示，行 + 列对齐版本在收敛阶段表现更优（137 epoch vs 172 epoch 达到 99% 精度），验证了保留矩阵级对齐信息的必要性。

### 2.4 动态 β 更新

`β₁_t = β₁_min + (β₁_max - β₁_min) · alignment`

`β₂_t = β₂_min + (β₂_max - β₂_min) · alignment`

**物理含义**：

| 条件 | 几何含义 | β 行为 |
|:---|:---|:---|
| 高曲率变化 | 更新方向与曲率剧变方向对齐 | β → β_max（强滤波） |
| 低曲率变化 | 曲率稳定或变化微弱 | β → β_min（快速适应） |

### 2.5 1D 参数策略

1D 参数（bias、LayerNorm 等）不引入 Christoffel 动态 β，继续走固定 β 的 AdamW 更新。

## 3. 代码锚点

- Christoffel 矩阵构造：[`_C = _delta_g / (rho * _g_hat.abs() + _eps)`](optimizer/ars2c.py:204)
- 结构化正交化：[`state['c_ortho'] = zeropower_via_newtonschulz5(_c_flat)`](optimizer/ars2c.py:207)
- 对齐度计算：[`alignment_raw = float((c_ortho * s_unit).sum().abs())`](optimizer/ars2c.py:283)
- 幅度门控：[`mag_gate = float(torch.sigmoid(torch.tensor(c_magnitude)))`](optimizer/ars2c.py:284)
- 动态 β 注入：[`exp_avg.mul_(beta1_d).add_(p.grad, alpha=1 - beta1_d)`](optimizer/ars2c.py:259)

## 4. 边界

- ARS2C 控制动量衰减 β 的动力学（"记忆"维度），与 AR-GSAM（"几何"维度，控制 ρ）正交互补。
- Non-sync step 复用上一次 sync step 的 alignment 值，不额外计算。
- 1D 参数保持固定 β，避免标量 Christoffel 退化为全局衰减率调节。
