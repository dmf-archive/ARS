---
created: 2026-05-16
landed: 2026-05-16
status: experimental
---

# ARS2C: Christoffel 动态 β

## 原理

ARS2C 在 ARS2 的能量-几何解耦之上，将固定的 β₁, β₂ 替换为从 Fisher 信息流形推导的动态 β。曲率决定遗忘速率。

类比 GR 对应：Fisher 信息 G_ij（ARS 二阶矩 v̂）→ GR 度规 g_μν；Christoffel 符号 Γ → 动态 β；测地线方程 → 自然梯度流。

## 更新链路

### Christoffel 矩阵构造（Sync Step）

复用 SAM 同步步的 HVP 采样：

`C = δg ⊘ (ρ · ĝ + ε)`，其中 `δg = g_adv - g_base`，`ĝ = g_base / (√v̂ + ε)`。

### 结构化正交化

`c_ortho = zeropower_via_newtonschulz5(C_flat)`，编码曲率变化的方向结构。

### 行列双向对齐度

`row_alignment = (⟨c_row_norm, s_row_norm⟩ + 1) / 2`
`col_alignment = (⟨c_col_norm, s_col_norm⟩ + 1) / 2`
`alignment_matrix = √(row_alignment ⊙ col_alignment)`

最终 alignment_matrix 与参数矩阵同维度，每个元素对应独立的 β 调制系数。

### 动态 β

`β₁_t = β₁_min + (β₁_max - β₁_min) · alignment`
`β₂_t = β₂_min + (β₂_max - β₂_min) · alignment`

高对齐 → β → β_max（强滤波）；低对齐 → β → β_min（快速适应）。

### 1D 参数策略

bias、LayerNorm 等标量参数不引入动态 β，继续固定 β 的 AdamW。

## 代码锚点

- Christoffel 矩阵：[`_C = _delta_g / (rho * _g_hat.abs() + _eps)`](optimizer/ars2c.py:204)
- 正交化：[`state['c_ortho'] = zeropower_via_newtonschulz5(_c_flat)`](optimizer/ars2c.py:207)
- 对齐度：[`alignment_raw = float((c_ortho * s_unit).sum().abs())`](optimizer/ars2c.py:283)
- 动态 β 注入：[`exp_avg.mul_(beta1_d).add_(p.grad, alpha=1 - beta1_d)`](optimizer/ars2c.py:259)

## 边界

- 控制 β（*记忆*维度），与 AR-GSAM 控制 ρ（*几何*维度）正交互补。
- Non-sync step 复用上次 sync step 的 alignment，不额外计算。
- 1D 参数保持固定 β。

*关联文档*：[`1-ARS.md`](.roo/rules/1-ARS.md:1), [`1-ARS2.md`](.roo/rules/1-ARS2.md:1), [`1-ARS2D.md`](.roo/rules/1-ARS2D.md:1)
