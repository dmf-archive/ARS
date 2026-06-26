# 对数压缩激活函数 (Log-compressed Activation Functions) 研究备选

`归档时间：`2026-06-27
`来源：`CRA-1-SAN 方向探索 → ReSRLU (sqrt) → ReLoLU (log) → 确认 NLReLU (2019) 已先验覆盖
`状态：`方向备选，当前不投入工程资源

## 方向背景

从 sqrt 压缩激活函数 (ReSRLU) 转向 log 压缩 (ReLoLU) 的动因：log 在信息论中对应 -log P(x) 即编码长度，比 sqrt 更深刻锚定 MDL/FEP。但搜索后发现已有充分先验工作。

## 已有先验

### NLReLU (Natural Logarithm ReLU) — 2019

`arXiv:1908.03682` — Liu et al., IEEE ICCC 2019

```
NLReLU(x) = ln(β · max(0, x) + 1.0)
```

- β 为可学习参数
- 在 shallow/deep CNNs 上 MNIST/CIFAR-10 验证：比 ReLU 高 0.16–2.04% 准确率
- 与 ELU/SELU 持平

### STL (Signed Truncated Logarithm) — 2023

`arXiv:2307.16389` — Yuanhao Gong, SZU

```
STL(x) = αx if |x| ≤ 1, α·sign(x)·(log(|x|)+1) if |x| > 1
```

- 六条理想数学性质全部满足（奇函数、单调、可微、无界、连续非零梯度、计算高效）
- CIFAR-10/100 跨 9 种架构达到 SOTA

### LogLU (Logarithmic Linear Unit) — 2024-2025

多个 OpenReview 版本:

```
LogLU(x) = x if x > 0, -log(-x+1) if x ≤ 0
```

- 正半轴线性、负半轴对数衰减
- Caltech 101 / Imagenette / ICLR 2026 submission

### Soft Exponential — 2016

`arXiv:1602.01321` — 参数 α 在 log/linear/exponential 间连续插值

## 差异化空间（已关闭）

若未来重启此方向，可能的差异化点：

1. MPS 绑定：τ_min = ε·√d 物理截断（AGAM 框架）
2. BiLoLU：正负半轴对称 log(1+|x|) 信息对称扩展
3. MDL/FEP/IPWT 理论锚定：log 压缩的变分自由能解释

## 结论

ReLoLU (log(1+max(0,x))) 形式与 2019 年 NLReLU 等效，不具原创性。该方向已封存，当前不投入工程资源。

---

## 调查结果：内置对数压缩的激活函数（2025-2026）

**不修改现有文档，仅汇报发现。**

### 现有文档已覆盖

`ref/activation-functions-log.md` 已正确覆盖显式 log 方案——NLReLU (2019)、STL (2023)、LogLU (2024-2025)、Soft Exponential (2016)。这些是设计上**显式包含 `log(x)` 或 `-log(-x+1)`** 的函数。

### "内置对数压缩"的候选者（非显式 log、但存在对数行为）

| 候选 | 内置对数的形式 | LLM-scale 验证 | 备注 |
|:---|:---|:---:|:---|
| **Softplus** — [`log(1 + exp(x))`](https://arxiv.org/abs/2501.13428) | 字面意义上的 log | 仅 attention 层已验证（ICML 2026 LSSA），FFN 层未大规模验证 | 凸共轭是负二元熵，直接锚定 MDL |
| **Mish** — [`x · tanh(log(1 + exp(x)))`](https://arxiv.org/abs/1908.08681) | softplus 嵌在 tanh 门内 | 无 LLM-scale 验证 | 预条件器效应使梯度更平滑 |
| **Serf** — [`x · erf(log(1 + exp(x)))`](https://arxiv.org/abs/2108.09598) | 同上，erf 替代 tanh | 无 LLM-scale 验证 | WACV 2023，CNN 上优于 Swish/Mish |
| **GoLU** — [`x · exp(-exp(-x))`](https://github.com/automl/GoLU) | Gumbel 分布隐式含 exp(-exp(-x)) 结构 | 无大规模验证 | 2025，声称方差压缩优于 GELU/SiLU |
| **PowLU** — 亚二次 ~x^0.5 | 不是 log，但实现 log 类压缩效果 | **124B Ling 架构已验证** | 唯一 100B+ 验证的非 SwiGLU 激活函数 |

### 关键结论

1. **Softplus 理论上最纯粹**——`log(1 + exp(x))` 就是字面对数，且凸共轭 = 负二元熵直接锚定信息论的编码长度。但缺乏 FFN 层的大规模验证。

2. **PowLU 工程上最有潜力**——亚二次增长（~x^0.5 而非 SwiGLU 的 x²），抑制 outlier，在 124B 参数规模验证过。与 ARS 的 SAM/MDL 平坦度目标一致。

3. **Mish/Serf/GoLU 不值得当前投入**——无 LLM-scale 验证，已被 SwiGLU 在工程上淹没。

4. **2025-2026 的最大趋势不是激活函数本身，而是用饱和函数替代 LayerNorm**（DyT/DyISRU/BHyT/Derf/SeeDNorm 谱系），但这是另一条技术路线。

5. 如果要快速验证 Softplus 作为 SwiGLU 替代在 FFN 中的表现，可以在 ARS 的 Grokking 基准上跑一次对比实验——这是最有增量信息的方向。
