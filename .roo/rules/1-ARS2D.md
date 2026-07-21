---
created: 2026-05-23
landed: 2026-05-23
status: experimental
---

# ARS2D: Bidirectional Orthogonalization

## 做了什么

ARS2D 在 ARS2 的单边 Newton-Schulz 正交化之后，对结果再执行一次 NS：

`U = NS(G_nat)` — 行正交（`UUᵀ ≈ Iₘ`）
`W = NS(Uᵀ)ᵀ` — 列正交（`WᵀW ≈ Iₙ`）
`Δθ = -η · ‖G_nat‖ · W`

两次 NS 模拟 K-FAC 的 `F⁻¹g ≈ A⁻¹ᐟ² G B⁻¹ᐟ²`，但无需显式计算或求逆协方差矩阵。

## 只对方阵有意义

双边正交化只在 *m=n* 时能同时达到行和列的等距。当 m ≠ n 时，二次 NS 无法给出双等距解——应使用 ARS2-GSAM（单边正交 + 平坦度约束）。

1D 参数（bias、LayerNorm）用固定 β 的 AdamW。

## 开销

相比 ARS2 的 1× NS，ARS2D 需要 2× NS。

*关联文档*：[`1-ARS.md`](.roo/rules/1-ARS.md:1), [`1-ARS2.md`](.roo/rules/1-ARS2.md:1), [`1-ARS2C.md`](.roo/rules/1-ARS2C.md:1)
