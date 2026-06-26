# ARS2D: Bidirectional Orthogonalization for Energy-Geometry Decoupling

## 0. 原理概述

ARS2D 在 [`ARS2`](.roo/rules/ARS2.md:1) 的能量-几何解耦基础上，将 Newton-Schulz 正交化从单边（行正交）扩展为双边（行列双正交）。其核心命题是：

> 当更新矩阵同时满足行正交和列正交时，它更接近 Natural Gradient Descent 的 K-FAC 近似 F⁻¹g ≈ A⁻¹ᐟ² G B⁻¹ᐟ²，从而在参数流形的切丛上实现真正的等距更新。

## 1. 为什么需要双边正交化

当前 ARS2 的更新方向为：

`U = NS(G),  UUᵀ ≈ I_m`

其中 G ∈ ℝ^(m×n) 是预白化梯度矩阵。这只解耦了输出通道（行方向），输入通道（列方向）仍存在耦合。

在深度网络中，输入通道之间的相关性（例如图像相邻通道、词嵌入维度）同样造成冗余更新。双边正交化通过第二轮 Newton-Schulz 强制 UᵀU ≈ I_n，使更新方向在行和列两个维度上都接近等距。

## 2. 更新链路

给定预白化梯度 G_nat ∈ ℝ^(m×n)（已经过 Adam 统计预条件处理）：

`E = ‖G_nat‖_F`

`U = NS(G_nat)  ⇒  UUᵀ ≈ I_m`

`W = NS(Uᵀ)ᵀ  ⇒  WᵀW ≈ I_n`

`Δθ = -η · E · W`

其中 NS 是 Newton-Schulz 迭代（通常 5 步）。当 m = n 时，W 是正交矩阵；当 m ≠ n 时，W 是双等距映射（行和列均近似等距）。

## 3. 与 Full-Rank NGD 的关系

在 K-FAC 框架中，Fisher 矩阵近似为 Kronecker 积 F ≈ A ⊗ B，自然梯度更新为：

`F⁻¹g ≈ A⁻¹ᐟ² G B⁻¹ᐟ²`

其中 A ∈ ℝ^(m×m) 对应输出协方差，B ∈ ℝ^(n×n) 对应输入协方差。

ARS2D 的两轮 NS 迭代恰好模拟了这一过程：

- 第一轮 NS：U = A⁻¹ᐟ² G（行白化）
- 第二轮 NS：W = U B⁻¹ᐟ²（列白化）

但 ARS2D 不需要显式计算 A 和 B，也不需要矩阵求逆或特征分解，完全通过矩阵乘法完成。

## 4. 与 ARS2 家族其他变体的关系

- ARS：能量-几何解耦（仅行正交），1× NS，1 单位开销
- ARS2：ARS + SAM 平坦度，1× NS，1 + SAM 开销
- ARS2C：ARS2 + 动态 β（Christoffel），1× NS，1 + SAM + 少量矩阵运算
- **ARS2D**：双边正交化（行列双等距），2× NS，2 单位开销
- ARS2CD：ARS2C + 双边正交化，2× NS，2 + 动态 β 开销

ARS2D 可与 A-GSAM、SAM、动态 β 正交组合，形成 ARS2CD 等变体。

## 5. 预期收益

- 收敛更快：消除输入通道耦合，减少无效迭代
- 泛化更稳：更逼近 Fisher 信息流形上的测地线，天然倾向于平坦极小值
- 超参数鲁棒：双边正交化降低了对学习率和动量衰减的敏感性
- 计算友好：Newton-Schulz 迭代仅涉及矩阵乘法，无需存储 Kronecker 因子，额外开销与参数量成线性关系

## 6. 边界

- 当 m=1 或 n=1（如 bias 或 LayerNorm 的增益参数）时，双边正交化退化为普通 NS，此时 ARS2D 行为等价于 ARS2
- 对于极扁平的矩阵（m ≪ n 或 n ≪ m），双边正交化仍能给出行等距或列等距，但无法同时达到双等距——此时应优先使用行正交或列正交中的较大者
- ARS2D 不解决 1D 参数（bias、LayerNorm 等）的优化问题，这些参数继续使用固定 β 的 AdamW 更新

---

**状态**：理论完成，实现待编码。
**关联文档**：[`ARS.md`](.roo/rules/ARS.md:1), [`ARS2.md`](.roo/rules/ARS2.md:1), [`ARS2C.md`](.roo/rules/ARS2C.md:1)
