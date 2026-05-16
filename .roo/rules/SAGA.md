# SAGA：Sharpening-Aware Geometric Adaptation

## 0. 原理概述

SAGA 在 [`ARS2`](.roo/rules/ARS2.md:1) 的平坦度约束之上引入**曲率对齐调制**，其核心命题是：

- ARS2 的扰动半径 `ρ` 和动量衰减率 `β1, β2` 本应由地形的局部几何直接决定，而非启发式规则。
- ARS2C 的 `alignment` 指标提供了一个直接测量：更新方向与 Christoffel 曲率方向的内积对齐度。
- 由此可统一调制 `ρ` 与动量衰减率，消除基于 `dL/dt` 或代理间隙 `h_t` 的启发式反馈循环。

## 1. 对齐度 (Alignment)

在 ARS2C 的 SAM 同步步中，梯度差 `δ_g = g_adv - g_base` 近似于 Hessian-向量积。经归一化与 Newton-Schulz 正交化后得到曲率方向：

`c_ortho = NS(C_flat)`, 其中 `C = δ_g / (ρ · |ĝ| + ε)`

更新方向的正交化版本：

`s_ortho = NS(ĝ)`, 其中 `ĝ = m̂ / √v̂`

对齐度定义为两者之间的绝对余弦相似度，经曲率幅度门控：

`alignment = |⟨c_ortho, s_unit⟩| · σ(‖C_flat‖)`

## 2. 对齐度如何调制参数

| 信号 | 高 alignment | 低 alignment |
| :--- | :--- | :--- |
| `ρ` (扰动半径) | 回归稳态 `μ`（曲率结构良好，信任 NGD） | 增大（曲率混乱，以平坦压强强制规则化） |
| `β1, β2` (动量衰减) | 强滤波（对齐稳定，保留历史动量） | 弱滤波（曲率突变，快速适应） |

### 2.1 Rho 调制

`dρ/dt = κ(μ - ρ) + η·(1 - A_t)`

`A_t → 1` 时回归项主导，`ρ → μ`；`A_t → 0` 时 `(1-A_t)` 项主导，`ρ` 递增直至曲率结构显现。

### 2.2 动量调制

`β_d = β_min + (β_max - β_min) · A_t`

对应实现：[`_apply_ars2_kernel`](optimizer/ars2c.py:218-224)。

## 3. 平台期加压与 Grokking

当损失停滞（`dL/dt ≈ 0`）时，NGD 更新进入 Hessian 零空间，`s_ortho` 丧失曲率对齐 → `A_t` 自动下降 → `ρ` 自动增大 → SAM 扰动在地形中炸出新结构 → Grokking 的触发机制。

不需要显式的平台期检测或 `κ_t` 动态调度，对齐度天然承载了探索/利用切换信号。

## 4. SAGA 与连续 Kolmogorov 复杂度逼近

SAGA 提供了一个可操作的视角：优化器以 `ρ` 作为连续二分搜索的探针，逼近模型-数据集对的 Kolmogorov 压缩极限。

- 当 `A_t → 1` 时，更新始终与残差曲率对齐，说明已无未建模的结构剩余 → 系统已逼近压缩极限。
- 当 `A_t → 0` 时，存在未建模的曲率结构 → 需要增大 `ρ` 以平坦化地形，使结构暴露给 NGD 捕获。

因此 SAGA 的本质是：**以曲率对齐为探针的连续 Kolmogorov 复杂度逼近过程**。由于 K 复杂度在理论上不可判定，SAGA 通过连续流形上的几何调制收敛到其可计算代理值。

## 5. 代码锚点

- Christoffel 矩阵构造：[`_delta_g / (rho * _g_hat.abs() + eps)`](optimizer/ars2c.py:171)
- 曲率方向正交化：[`zeropower_via_newtonschulz5(_c_flat)`](optimizer/ars2c.py:174)
- 对齐度计算：[`alignment = alignment_raw * mag_gate`](optimizer/ars2c.py:254)
- 对齐调制 beta：[`beta1_d = b1_min + (b1_max - b1_min) * alignment`](optimizer/ars2c.py:223)
- 诊断输出：[`diagnostics`](optimizer/ars2c.py:264)
