---
created: 2026-01-11
landed: 2026-01-11
status: stable
---

# A-GSAM: Adaptive GSAM

## 0. 原理概述

A-GSAM (Adaptive GSAM) 在 [`1-ARS2`](.roo/rules/1-ARS2.md:1) 的 GSAM 风格平坦度约束之上，用*按需同步*替代固定周期 `k` 同步，其核心命题是：

- GSAM 的双前向（base/adv）计算开销与算子间隙控制收益之间存在 trade-off。
- 固定周期 `k` 无法适应动态变化的黎曼流形——曲率变化快时 `k` 太大，优化器在尖锐区域停留过久；曲率平稳时 `k` 太小，浪费计算资源。
- A-GSAM 通过全局干涉因子 `ϕ_t` 检测几何漂移，仅在必要时触发完整同步，非同步步复用剪切力维持平坦度压强。

A-GSAM 的 G 指 GSAM（Zhuang et al., ICLR 2022），而非 Generic/Geometric；A 指 Adaptive——同步决策自适应于流形几何变化，而非基于静态周期。

## 1. 全局干涉因子 `ϕ_t`

`ϕ_t` 测量当前梯度方向与上次同步步缓存的剪切力向量之间的全局余弦相似度：

`ϕ_t = (∑_{p ∈ Θ} ⟨g_{t,p}, v_{flat,p}⟩) / √(∑ ‖g_{t,p}‖² · ∑ ‖v_{flat,p}‖²)`

其中 `v_{flat}` 是上次同步步存储的正交剪切力向量。物理含义：

- `ϕ_t → 1`：当前梯度与剪切力方向一致，流形几何稳定，无需同步。
- `ϕ_t → 0`：梯度与剪切力正交，流形几何适度变化。
- `ϕ_t → -1`：梯度与剪切力反向，**几何漂移**，需触发同步重新采样曲率信息。

跨设备场景下，`ϕ_t` 通过 `all_reduce` 聚合各 rank 的 `num/den_g/den_v` 三标量实现，通信开销极低（单标量 all_reduce）。

## 2. 正交基准与动态阈值

在高维病态曲率流形中，梯度与剪切力天然倾向于正交。系统采用 **0.0 正交基准模型**：

```python
# 方差跟踪（仅在非同步步更新，避免同步步 bias）
ν_{ϕ, t} = β · ν_{ϕ, t-1} + (1-β) · (ϕ_t - 0.0)²

# 漂移判定
threshold = -λ · σ_{ϕ, t}   # σ = √ν
is_drift = ϕ_t < threshold
is_sync_step = is_drift or (global_step == 1) or (k > 1 and steps_since_sync >= k)
```

- **基准点** `μ = 0.0`：不做均值偏移估计，直接采用正交假设。
- **噪声 EMA** `β`：由 `adaptive_beta` 参数控制，建议 0.9~0.99。
- **灵敏度** `λ`：由 `adaptive_lambda` 控制，λ 越小越灵敏（更容易触发同步）。
- **回退机制**：`k > 1` 时即使无几何漂移，最多 `k` 步后强制同步。

## 3. 自适应强度放大

对齐良好时（`ϕ_t > 0`）"奖励"剪切力注入强度：

`α_t = α_max · (1 + max(0, ϕ_t))^γ`

其中 `γ` 由 `adaptive_gamma` 控制（默认 2.0）。几何一致性极高时（`ϕ_t → 1`），修正强度最高可放大至 `2^γ` 倍（默认 4 倍）。

非同步步的剪切力注入：

`p.grad.add_(v, alpha=α_t · ‖g‖ / ‖v‖)`

## 4. 代码锚点 (ARS2-Neo)

所有 A-GSAM 逻辑内聚于 [`ARS2Neo.step()`](optimizer/ars2_neo.py:110)，不依赖外部状态管理：

- 干涉因子计算：[`_calculate_global_phi()`](optimizer/ars2_neo.py:248)
- 方差跟踪：[`phi_var = beta · phi_var + (1-beta) · ϕ²`](optimizer/ars2_neo.py:140)
- 漂移判定阈值：[`threshold = - adaptive_lambda · std`](optimizer/ars2_neo.py:143)
- 同步决策：[`is_sync_step = is_drift or ...`](optimizer/ars2_neo.py:148)
- 自适应 α 计算：[`alpha_max · (1 + max(0, ϕ_t))^γ`](optimizer/ars2_neo.py:152)
- 非同步剪切力注入：[`p.grad.add_(v, alpha=α_t · g_norm / v_norm)`](optimizer/ars2_neo.py:214)
- 剪切力构造：[`state['shear_force'] = g_adv - proj_{g_base}(g_adv)`](optimizer/ars2_neo.py:202)
- 诊断输出：[`diagnostics`](optimizer/ars2_neo.py:276)

## 5. 超参数

| 参数 | 默认值 | 含义 |
|:---|:---:|:---|
| `adaptive_sync` | `False` | 启用 A-GSAM 模式 |
| `adaptive_beta` | 0.99 | 方差跟踪 EMA 衰减率 |
| `adaptive_lambda` | 2.0 | 漂移灵敏度（推荐 0.5~2.0） |
| `adaptive_gamma` | 2.0 | 自适应 α 放大指数 |
| `alpha` | 0.1 | 基础剪切力注入强度 |

## 6. 参考文献

- [1] J. Zhuang et al., "GSAM: Surrogate Gap Guided Sharpness-Aware Minimization," ICLR 2022.
- [2] P. Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization," ICLR 2021.
- [3] ARS 系列设计文档: [`1-ARS.md`](.roo/rules/1-ARS.md:1), [`1-ARS2.md`](.roo/rules/1-ARS2.md:1), [`2-AGAM.md`](.roo/rules/2-AGAM.md:1)
