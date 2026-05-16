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

`alignment_raw = |⟨c_ortho, s_unit⟩|`

`mag_gate = σ(‖C_flat‖)`

`alignment = alignment_raw · mag_gate`

其中 `s_unit` 是正交化更新方向，`σ` 是 Sigmoid 函数。

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

- ARS2C 控制动量衰减 β 的动力学（"记忆"维度），与 SAGA（"几何"维度，控制 ρ）正交互补。
- Non-sync step 复用上一次 sync step 的 alignment 值，不额外计算。
- 1D 参数保持固定 β，避免标量 Christoffel 退化为全局衰减率调节。
