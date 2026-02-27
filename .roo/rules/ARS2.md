# ARS2：AdaRMSuon SAM

## 0. 原理概述

ARS2 在 [`ARS`](.roo/rules/ARS.md:1) 的能量-几何解耦之上引入平坦度约束，其核心命题是：

- 当优化器更接近自然梯度的高效下降时，模型更容易快速进入训练损失的尖锐谷底。
- 因此必须引入更强的正则化机制，把轨迹从“训练集最快下降”改写为“可泛化下降”。

这就是 ARS2 引入 SAM 家族约束的动机。

## 1. 为什么“越逼近 NGD”越需要正则化

设预条件更新可写作：

`Δθ_t = -η_t · P_t · g_t`

当 `P_t` 与有效曲率子空间对齐程度提高时，训练损失下降会更快，但同时也更容易放大训练集特异方向。工程上常见后果是：

- 训练损失快速下降；
- 验证指标先改善后恶化；
- 最终出现尖锐极小值驱动的泛化崩溃。

对应证据可见 [`outputs/lrp_wikitext2_ars2_neo_base_20e/summary.md`](outputs/lrp_wikitext2_ars2_neo_base_20e/summary.md:1)。

## 2. SAM、ASAM、GSAM 在 ARS2 语境中的角色

### 2.1 SAM

SAM 的基本思想是在邻域内优化最坏方向损失：

`min_θ max_{‖ϵ‖≤ρ} L(θ+ϵ)`

它不是单纯“加噪”，而是显式惩罚尖锐解。

### 2.2 ASAM

ASAM 引入参数尺度感知，使扰动更匹配不同参数量级。在实现上，ARS2-Neo 采用自然梯度方向后再进行尺度调制：

- 扰动基础方向：[`g_nat = p.grad / (v_hat.sqrt() + eps)`](optimizer/ars2_neo.py:201)
- 尺度感知：[`g_nat = g_nat * p.abs()`](optimizer/ars2_neo.py:204)

### 2.3 GSAM

GSAM 的关键价值是把“平坦度约束”与“代理间隙控制”结合，在抑制尖锐性的同时尽量减少优化冲突。

ARS2 并不逐字复刻 GSAM，而是保留其核心工程思想：

- 双前向（base/adv）构造平坦度信号；
- 在非同步步注入正交剪切力，降低额外计算开销。

对应实现锚点：[`ARS2Neo.step()`](optimizer/ars2_neo.py:123)。

## 3. AGA：如何进一步压缩 GSAM 开销

静态 `k` 同步在不同任务曲率条件下开销-收益不稳定。AGA 通过全局干涉因子实现按需同步：

- 干涉因子计算：[`_calculate_global_phi()`](optimizer/ars2_neo.py:323)
- 漂移判定阈值：[`threshold = - adaptive_lambda * std`](optimizer/ars2_neo.py:165)
- 同步决策：[`is_sync_step = is_drift or ...`](optimizer/ars2_neo.py:171)
- 非同步剪切力注入：[`p.grad.add_(v, alpha=...)`](optimizer/ars2_neo.py:288)

简言之，AGA 把“每步都做昂贵对抗扰动”改写为“只在几何漂移时做完整同步，其余步复用剪切校正”。

## 4. SAM 与 MDL 的关系

在 MDL 视角中，模型泛化等价于更低的有效描述复杂度。尖锐极小值通常对应对参数扰动敏感、编码冗余高、描述长度不稳定。SAM 通过惩罚局部尖锐性，倾向于选择更平坦盆地，从而降低有效编码复杂度。

`更平坦局部几何  ⇒  更低敏感度  ⇒  更短有效描述长度`

因此 ARS2 中的平坦度约束可被视为对 MDL 原则的实现。

## 6. 代码锚点（ARS2-Neo）

- base 前向与反向：[`loss = closure(); loss.backward()`](optimizer/ars2_neo.py:139)
- 扰动与 adv 前向：[`loss_adv = closure(); loss_adv.backward()`](optimizer/ars2_neo.py:216)
- 剪切力构造：[`state['shear_force'] = g_adv - ...`](optimizer/ars2_neo.py:275)
- AGA 诊断输出：[`diagnostics`](optimizer/ars2_neo.py:353)
