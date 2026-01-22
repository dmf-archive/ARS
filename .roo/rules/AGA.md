---
title: "ARS 演进：自适应几何感知 (AGA) 技术预研"
category: "优化算法"
status: "🟡 进行中"
priority: "高"
timebox: "1 周"
created: 2026-01-22
updated: 2026-01-22
owner: "Roo (AI Architect)"
tags: ["技术预研", "优化算法", "研究", "AGA"]
---

## 摘要

**探索目标 (Spike Objective):** 验证自适应几何感知 (AGA) 机制在 [`optimizer/ars2_neo.py`](optimizer/ars2_neo.py) 中动态平衡平坦度约束预算与强度的有效性，旨在通过流形几何一致性自动调节同步频率与注入强度。

**重要性 (Why This Matters):** 传统的静态周期 k 与静态强度 α 无法适应动态变化的黎曼流形。AGA 通过引入干涉因子实现“按需同步”，能显著降低计算开销并提升收敛稳定性。

**时限 (Timebox):** 1 周

**决策截止日期 (Decision Deadline):** 2026-01-28

## 研究问题 (Research Question(s))

**主要问题 (Primary Question):** AGA 能否在不损失泛化性能的前提下，通过自适应同步频率显著降低计算开销？

**次要问题 (Secondary Questions):**

- 干涉因子 φₜ 能否准确反映流形曲率的剧烈波动？
- 自适应强度律 αₜ = αₘₐₓ ⋅ (1 - φₜ)ᵞ 对超参数 γ 的敏感度如何？
- 阈值 L 在不同任务（CIFAR-10 vs WikiText-2）间的迁移性如何？

## 调查计划

### 研究任务

- [ ] 实现 AGA 逻辑并集成至 [`optimizer/ars2_neo.py`](optimizer/ars2_neo.py)
- [ ] 在 CIFAR-10 上进行 L 值的二分搜索
- [ ] 记录并分析不同训练阶段的平均同步周期 k_avg
- [ ] 对比 AGA 与固定 k 模式的性能-开销曲线
- [ ] 记录发现并形成最终建议

### 成功标准

**本次探索完成的标志是：**

- [ ] AGA 模式在 CIFAR-10 上的最终精度不低于同步模式 (Sync, k=1)
- [ ] 计算开销相比同步模式降低 30% 以上
- [ ] 形成了关于 L 和 γ 的明确建议
- [ ] 完成了初步的概念验证

## 技术背景

**相关组件 (Related Components):** [`optimizer/ars2_neo.py`](optimizer/ars2_neo.py), [`exp/cifar/train.py`](exp/cifar/train.py)

**依赖项 (Dependencies):** `ARS2-Neo` 基础框架, `SmartOptimizer` 回调机制

**限制条件 (Constraints):** 必须在不增加单步计算复杂度的前提下利用缓存信息；遵循 IPWT 理论框架。

## 研究发现

### 调查结果

1. **干涉因子定义**: 引入 φₜ，即当前梯度 gₜ 与缓存平坦度向量 v_flat 之间的余弦相似度。
2. **自适应强度律**: αₜ = αₘₐₓ ⋅ (1 - φₜ)ᵞ。确保平坦区强纠偏，漂移区自动压低权重。
3. **动态预算分配**: 引入单一参数 L。只要 φₜ < L，系统持续复用 v_flat 进行低成本滑行；一旦突破阈值，强制触发同步。

### 原型/测试记录

- 初步测试显示 ρ=0.1 是 CIFAR-10 任务的理想起点。
- 实验目标是寻找最佳性价比的阈值 L，验证 AGA 在不同训练阶段的自动优化能力。

### 外部资源

- [`ref/muon/`](ref/muon/)

## 决策

### 建议

建议在 `ARS2-Neo` 中全面引入 AGA 机制，取代繁琐的 Lazy k 搜索。

### 理由

基于证据的预算分配模式使模型能自适应地在复杂地形频繁同步，在平坦地形长程滑行，将优化动力学与流形几何深度绑定。

### 实施说明

- 默认参数建议：L=0.1, γ=2.0。
- 需在 `SmartOptimizer` 中增加对 φₜ 的实时监控。

### 后续行动

- [ ] 在 [`optimizer/ars2_neo.py`](optimizer/ars2_neo.py) 中实现 `AGA_Step`
- [ ] 更新 [`.roo/rules/long_range_plan.md`](.roo/rules/long_range_plan.md) 以包含 AGA 实验矩阵
- [ ] 创建 CIFAR-10 的 AGA 验证任务

## 状态历史

| 日期 | 状态 | 备注 |
| --- | --- | --- |
| 2026-01-21 | 🔴 未开始 | 提案初步形成 |
| 2026-01-22 | 🟡 进行中 | 文档结构化并进入初步验证阶段 |

---

上次更新时间：2026-01-22，由 Roo (AI Architect)
