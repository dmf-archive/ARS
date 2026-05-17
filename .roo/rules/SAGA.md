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

> **架构冻结公告**：本节描述的基于全局标量线性映射的 ρ 自适应方案已被验证存在重大缺陷，并正式废弃。目前 SAGA 实验分支已封存，以下记录仅用于历史存档、机制对比与理论分析。

| 信号 | 高 alignment | 低 alignment |
| :--- | :--- | :--- |
| `ρ` (扰动半径) | 回归稳态 `μ`（原线性版本为增大，在 redesign 假说中应指数收缩） | 增大（原线性版本为减小，在 redesign 假说中应呈指数级向外膨胀） |
| `β1, β2` (动量衰减) | 强滤波（对齐稳定，保留历史动量） | 弱滤波（曲率突变，快速适应） |

### 2.1 历史版本的缺陷与封存状态

1. **实验表现差异**：
   SAGA 在 `Modular Addition` Grokking 实验中取得了不错的加速效果（相比 ARS2C-AGA 提前 81 个 epoch 实现收敛）。但在 `Wikitext-2` 上表现不佳，其性能显著劣于固定 $\rho=0.3$ 的 ARS2C-Unlock 版本。
2. **失效根因分析**：
   Wikitext-2 的全局 alignment 全程偏低（$\approx 0.09 \sim 0.29$），在线性映射：
   `ρ_target = ρ_min + (ρ_max - ρ_min) · A_t`
   的作用下，$\rho$ 被死死锁死在很小的范围（$\rho_{mean} \approx 0.18 \sim 0.26$），从未被有效缩放到能炸开局部尖锐性的物理尺度。
3. **参数值域映射失配**：
   Alignment 的值域为 $[0, 1]$，但扰动半径 $\rho$ 的有效几何范围在不同数据尺度下具有各向异性的非线性特征（理论上与数据分布联合熵正相关）。使用粗暴的线性映射无法跨越不同尺度任务。

### 2.2 SAGA 指数自适应假说

为了使 SAGA 在低对齐的混沌地形中能够自适应膨胀，未来的重构版本应采用**非对称指数级软饱和发散响应曲线**：

`ρ_t = ρ_min * exp( κ * softplus( τ * (μ - A_bar) ) )`

- 其中 `A_bar = mean(A_scaled)` 为当前层参数 of 全局平均对齐度。
- 当 `A_bar → 1`（流形对齐极佳且平滑）时，`ρ_t` 快速回缩至稳态下界 `ρ_min`，完全信任 NGD 进行高效滑行。
- 当 `A_bar → 0`（几何结构发生漂移或突变）时，`ρ_t` 呈**指数级向上发散膨胀**，从而在局部地形中引入强大的平坦化压强，防止系统坠入尖锐极小值陷阱。
- 扰动调制因子应直接与 Christoffel 迹的 Frobenius 范数挂钩，而非使用静态超参数，以此捕获真实的流形相空间体积变动率。

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
