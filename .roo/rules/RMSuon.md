# RMSuon 家族：在黎曼流形上滑行

状态: 生产就绪 (2025-12-31)
核心贡献: 确立了“能量-几何解耦” (Energy-Geometry Decoupling) 的算子复合范式，并为探索 Geodesic SAM 提供了理论与工程基础。

## 1. 理论演进：从下山到滑行

### 1.1 优化的本质：在测地线上滑行

在信息几何视角下，优化不仅是损失函数 `L(θ)` 的梯度下降，更是概率分布流形上的测地线运动。RMSuon 旨在通过解耦“步伐的大小”（能量统计）与“迈步的方向”（流形几何）来逼近这一理想状态。

### 1.2 问题：不同的优化器，对地形的假设不同

- SGD: 假设欧几里得平直空间。它是“盲人登山者”，仅凭局部坡度 `∇L` 迈步，在病态曲率下极易震荡。
- Adam/RMSProp: 引入二阶矩 `vₜ` 修正尺度。它能感知地形的“颠簸程度”（元不确定性），实现元素级自适应。但其逐元素 (element-wise) 的视角忽略了参数间的相关性，本质上是在做平行的标量优化。

### 1.3 Muon

[`Muon`](optimizer/muon.py) 引入了严格的几何约束：要求更新量必须是“正交”的。

- Stiefel 流形: 更新量 `Δθ` 被投影至 Stiefel 流形（满足 `UᵀU = I` 的矩阵集合）。
- 纯粹旋转: 投影通过 Newton-Schulz 迭代 `𝒫ₛₜ(X)` 实现。这保证了每一步都在改变特征空间的“基向量方向”，而非“模长强度”，从而从根本上消除了内部协变量偏移。

### 1.4 RMSuon

[`RMSuon`](optimizer/rmsuon.py) 提出了第一个解耦方案：

- 几何 (Geometry): 信任 Muon 的正交化动量 `𝒫ₛₜ(mₜ)` 提供的方向稳定性。
- 能量 (Energy): 信任 Adam 的宏观统计。从 Adam 更新量中提取 Frobenius 范数作为标量能量：
  `E = ‖m̂ₜ / (√(v̂ₜ) + ε)‖_F`
- 算子复合: 让正交化的“芭蕾舞步”根据 Adam 观测到的总体“环境能量”进行缩放。

### 1.5 AdaRMSuon

[`AdaRMSuon`](optimizer/ada_rmsuon.py) 进一步揭示了：原始梯度在弯曲流形上存在“几何畸变”。

- 预白化 (Pre-whitening): 并非直接投影动量，而是先用 `vₜ` 对梯度进行白化，获得近似的自然梯度 (Natural Gradient) `gₙₐₜ ≈ mₜ / √(vₜ)`。
- 投影映射: 在预白化后的空间（更接近黎曼平直切空间）执行正交化投影 `𝒫ₛₜ(gₙₐₜ)`。
- 形式化表达:
  `Δθₜ = η ⋅ ‖gₙₐₜ‖_F ⋅ 𝒫ₛₜ(gₙₐₜ)`
- 结论: 这使得模型能够沿着真正的测地线 (Geodesic) 滑行，在 Wikitext-2 实验中表现出断层级的收敛效率。

## 2. 实验对比：Wikitext-2

实验设置: Qwen3 (RoPE), Context 255

### 2.1 5 Epoch 快速测试

> 标准 wikitext2 line mode 实验

| 优化器 | 核心机制 | Best PPL | Final PPL | Grad Norm (End) |
| :--- | :--- | :--- | :--- | :--- |
| AdaRMSuon | Pre-white + NS + Energy | 83.88 | 87.61 | 0.92 |
| RMSuon (v1) | AdamW + NS + Energy | 99.07 | 134.11 | ~3.7 |
| AdaMuon | Sign + NS + Element-wise | 125.46 | 147.60 | ~5.6 |
| Muon | SGD + NS | 161.09 | 161.09 | ~1.1 |
| AdamW | Standard | 104.68 | 250.82 | ~4.5 |

结论:

1. AdaRMSuon 性能断层领先。
2. AdaMuon 不如初版的 RMSuon，证明其 sign(m) 的信息损失和元素级自适应的流形破坏是致命的。
3. 纯 Muon 缺乏自适应能力，收敛速度较慢。

### 2.2 30 Epoch 马拉松：过拟合的动力学

> 此实验使用已清理的 chunk mode wikitext2
> `outputs\wikitext2_rope_muon_epoch30`
> `outputs\wikitext2_rope_rmsuon_epoch30`

| 优化器 | Best PPL (Epoch) | Final PPL (Epoch 30) | 过拟合倍数 | 稳定性分析 |
| :--- | :--- | :--- | :--- | :--- |
| RMSuon | 190.63 (Ep 3) | ~54930 | ~288x | 极速收敛，灾难性过拟合。证明其寻找最小值的效率极高，但完全没有复杂度控制，容易陷入局部极小值。 |
| Muon | 329.99 (Ep 6) | ~587 | ~1.78x | 缓慢收敛，轻微过拟合。其内在的谱约束自带一种隐式正则化，但效率太低。 |

- RMSuon: 第 3 Epoch 即达到最优 PPL (190.6)，随后发生灾难性过拟合（30 Epoch 时 PPL > 50000），Muon后期缓慢过拟合。
- RMSuon 能以最高效的路径找到当前训练集的极小值，但也由于缺乏复杂度控制，容易陷入那些极其狭窄、泛化能力差的尖锐谷底 (Sharp Minima)。

## 3. 总结：我们需要更好的复杂度控制

AdaRMSuon 的成功与失败一体两面：它是一个极其高效的 Loss 优化器，但正因如此，它会毫不犹豫地将模型推向高复杂度的过拟合区域。

这证明，单纯沿着测地线滑行是不够的。我们还需要在滑行时，主动避开那些“狭窄而尖锐”的山谷，去寻找那些“宽阔平坦”的盆地。

下一阶段核心目标：引入复杂度约束 (Complexity Control)

- Geodesic SAM: 不再是在欧氏空间做球形扰动，而是在 AdaRMSuon 定义的测地线方向上进行流形扰动。
- 寻找“平坦盆地”: 在滑行的同时，通过扰动探测地形的二阶平坦度，主动避开尖锐谷底。
- 目标: 实现极速收敛与强泛化的最终统一。

## 4. 参考文献

- [1] L. Rui, "Integrated Predictive Workspace Theory," Zenodo, 2025.
- [2] Kingma & Ba, "Adam: A method for stochastic optimization," ICLR 2015.
- [3] Jordan et al., "Muon: An optimizer for hidden layers in neural networks," 2024.
- [4] Li et al., "ROOT: Robust orthogonalized optimizer," arXiv:2511.20626.
- [5] Si et al., "AdaMuon: Adaptive Muon optimizer," arXiv:2507.11005.
- [6] Li et al., "NorMuon: Making Muon more efficient and scalable," arXiv:2510.05491.
