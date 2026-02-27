# ARS 实验记录和杂项

## 有趣事实

在开发过程中，我们发现了一个命名上的有趣事实：

- AdaRMSuon 本身就可以缩写为 ARS
- 而 AdaRMSuon + SAM 本应称为 ARS2

这个混乱源于 RMSuon 是 RMS + Muon 的交错造词，AdaRMSuon 类似地延续了这一命名模式。为消除快速迭代中的识别歧义，现明确：

- ARS：*A*da*R*M*S*uon
- ARS2：*A*da*R*M*S*uon + *S*AM

## 实验对比：CIFAR-10 (LRP 验证)

实验设置: ResNet-18, 60-100 Epochs, Batch Size 256.

| 优化器 | Best Acc | Final Acc | Final Loss | Avg Time |
| :--- | :--- | :--- | :--- | :--- |
| **ARS2-Neo (Sync, ρ=0.1)** | **95.87%** | **95.73%** | **0.15** | ~104s |
| **ARS2-Neo (Base)** | 95.58% | 95.52% | 0.25 | ~71s |
| **ARS2-Neo (AGA, λ=2.0)** | 94.10% | 94.09% | 0.18 | ~90s |
| **AdamW** | 94.60% | 94.47% | 0.27 | ~58s |
| **Muon** | 93.76% | 93.69% | 0.29 | ~75s |

## 实验对比：Wikitext-2 (LRP 验证)

实验设置: Qwen3 (RoPE, 3-layer), Context 255.

| 优化器 | Best PPL | Last PPL | Avg Time |
| :--- | :--- | :--- | :--- |
| **AdamW** | 116.46 | 213.52 | ~300s |
| **Muon** | 111.35 | 475.65 | ~445s |
| **ARS2-Neo (Base)** | 96.10 | 3055.47 | ~425s |
| **ARS2-Neo (Sync)** | **90.69** | **330.85** | ~780s |
| **ARS2-Neo (AGA)** | 93.23 | 414.83 | ~545s |

## 流形感知扰动 (ASAM)

ARS2-Neo 不在欧氏空间做球形扰动，而是在由二阶矩 `v_hat` 定义的流形度量下计算对抗方向。

1. **流形度量估计**: 利用 Adam 的二阶矩 `v_hat` 近似局部曲率。
2. **自然梯度扰动**:
   `g_nat = ∇L / (√v_hat + ε)`
   `𝜀 = 𝜌 ⋅ g_nat / ‖g_nat‖`
   这相当于在黎曼流形上进行等距扰动。
3. **剪切力注入 (Shear Force Injection)**:
   在非同步步骤中，ARS2-Neo 复用并注入正交于基础梯度的“剪切力”向量 `v_flat`，从而在不增加计算量的前提下持续推动模型离开尖锐区域。

## Adaptive Geometric Awareness, AGA

传统的静态周期 $k$ 无法适应动态变化的黎曼流形。AGA 通过引入干涉因子实现“按需同步”，显著降低计算开销并提升收敛稳定性。**在未来的实验中，AGA 将作为首选模式，取代传统的 Sync Mode。**

### 1. 全局干涉因子 `ϕ_t`

为了确保跨层和跨设备的几何一致性，`ϕ_t` 定义为全局梯度的余弦相似度：
`ϕ_t = (∑_{p ∈ Θ} ⟨g_{t,p}, v_{flat,p}⟩) / (√(∑ ‖g_{t,p}‖²) ⋅ √(∑ ‖v_{flat,p}‖²))`
其中 $v_{flat,p}$ 是上次同步步存储的平坦度向量（剪切力）。

### 2. 正交基准与动态阈值

在病态曲率的高维流形中，梯度与缓存的剪切力更倾向于保持**正交**。系统采用 **0.0 基准模型**：

- **基准点**: `μ = 0.0` (Orthogonal Baseline)
- **噪声估计**: `ν_{ϕ, t} = β ⋅ ν_{ϕ, t-1} + (1-β) ⋅ (ϕ_t - 0.0)²`
- **判定准则**: 若 `ϕ_t < - λ ⋅ σ_{ϕ, t}`，判定为几何漂移 (Geometric Drift)，触发同步。
- **物理意义**: 只要梯度不显著地“反向”于平坦度向量，系统就认为当前流形是平滑的。

### 3. 自适应强度放大

在对齐良好（`ϕ_t > 0`）时“奖励”强度：
`α_t = α_{max} ⋅ (1 + max(0, ϕ_t))^γ`
该机制确保在几何一致性极高时，修正强度最高可放大至 `2^γ` 倍。

### 4. 核心超参数建议

- `aga_beta` ($\beta$): 建议 0.9。控制几何统计量的平滑度。
- `aga_lambda` ($\lambda$): 控制同步触发的灵敏度，间接影响算力开销。 建议 0.5 (Wikitext-2) 或 2.0 (CIFAR-10)，取决于预算。
- `aga_gamma` ($\gamma$): 建议 2.0。控制自适应强度律的非线性程度。

## 实验验证：Grokking 动力学 (Modular Addition)

实验设置: task/mod_addition.py (p=113, train_frac=0.3), 1-Layer Transformer (4 Heads, d_model=128, d_mlp=512).

| 优化器 | 拟合 (Epoch) | 顿悟 (Epoch) | 收敛 (Epoch) |
| :--- | :--- | :--- | :--- |
| **AdamW** | ~140 | 228 | 556 |
| **AdaRMSuon** | **28** | **54** | 300 |
| **ARS** | 17 | 100 | 290 |
| **Muon** | >156 | N/A | N/A |

## ARS2-Neo：重构和整合后的参考版本

ARS2-Neo 是 ARS 家族的集大成者，在统一的代码中实现了 AdaRMSuon 的几何优化与 SAM 的平坦度约束，通过参数配置灵活切换模式，旨在取代实验性的独立 `AdaRMSuon` 和 `ARS`。随着 ARS2-Neo 的成熟，我们将逐步移除旧的实验性优化器代码，以简化实验空间。

## 参考文献

- [1] L. Rui, "Integrated Predictive Workspace Theory," Zenodo, 2025.
- [2] Kingma & Ba, "Adam: A method for stochastic optimization," ICLR 2015.
- [3] Jordan et al., "Muon: An optimizer for hidden layers in neural networks," 2024.
- [4] Li et al., "ROOT: Robust orthogonalized optimizer," arXiv:2511.20626.
- [5] Si et al., "AdaMuon: Adaptive Muon optimizer," arXiv:2507.11005.
- [6] Li et al., "NorMuon: Making Muon more efficient and scalable," arXiv:2510.05491.
- [7] J. Zhuang et al., "GSAM: Surrogate Gap Guided Sharpness-Aware Minimization," in *Proc. 10th Int. Conf. Learn. Represent. (ICLR)*, 2022. [Official PyTorch Implementation](https://github.com/juntang-zhuang/GSAM)
