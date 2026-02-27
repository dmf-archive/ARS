# ARS：AdaRMSuon

## 0. 原理概述

ARS 把 Adam 风格统计预条件与矩阵级几何约束组合为一个可计算近似：

- 统计侧：二阶矩 `v_t` 提供局部尺度信息，用于预白化更新。
- 几何侧：Newton-Schulz 正交化在矩阵空间抑制共线冗余，使更新方向更接近低冗余子空间。

在该组合下，ARS 可被理解为一种高效的结构化自然梯度近似。

## 1. 定义

本文档中的 ARS 指的是以 [`AdaRMSuon`](optimizer/ada_rmsuon.py:57) 为前身的基础能量-几何解耦更新律，不包含 SAM 平坦度扰动。

## 2. 更新链路

给定梯度 `g_t`：

`m_t = β1·m_{t-1} + (1-β1)·g_t`

`v_t = β2·v_{t-1} + (1-β2)·g_t²`

`m̂_t = m_t/(1-β1^t)`

`v̂_t = v_t/(1-β2^t)`

`g_nat = m̂_t / (√v̂_t + ε)`

`E_t = ‖g_nat‖`

`S_t = P_st(g_nat)`

`Δθ_t = -η · E_t · S_t`

其中 `P_st` 由 Newton-Schulz 正交化实现。

## 3. 代码锚点

- 预白化：[`m_scaled = m_hat / (v_hat.sqrt() + eps)`](optimizer/ada_rmsuon.py:118)
- 能量提取：[`energy = m_scaled.norm()`](optimizer/ada_rmsuon.py:121)
- 正交化：[`zeropower_via_newtonschulz5(...)`](optimizer/ada_rmsuon.py:127)
- 注能更新：[`update = energy * s_ortho`](optimizer/ada_rmsuon.py:135)
- 参数更新：[`p.add_(update, alpha=-lr)`](optimizer/ada_rmsuon.py:140)

## 4. 边界

- ARS 关注“沿局部测地线高效滑行”。
- ARS 不单独解决尖锐极小值泛化风险。
- 与平坦度约束相关内容见 [`ARS2.md`](.roo/rules/ARS2.md:1)。
