# ARS 实验记录和杂项

## 命名来源

在开发过程中，我们发现了一个命名上的有趣事实：

- AdaRMSuon 本身就可以缩写为 ARS
- 而 AdaRMSuon + SAM 本应称为 ARS2

这个混乱源于 RMSuon 是 RMS + Muon 的交错造词，AdaRMSuon 类似地延续了这一命名模式。为消除快速迭代中的识别歧义，现明确：

- ARS：`A`da`R`M`S`uon
- ARS2：`A`da`R`M`S`uon + `S`AM

| 名称 | 组成 | 控制 |
|:---|:---|:---|
| ARS | AdaRMSuon | 更新方向 |
| ARS2 | ARS + SAM | +平坦度（静态） |
| ARS2-Sync | ARS2 + k=1 | +每步同步 |
| ARS2-AGA | ARS2 + AGA | +自适应同步 |

## 实验对比：CIFAR-10 (LRP)

实验设置: ResNet-18, 60-100 Epochs, Batch Size 256.

| 优化器 | Best Acc | Final Acc | Final Train Loss | Best Eval Loss | Final Eval Loss | Avg Epoch Time | Gen Gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ARS2-Neo (Sync, ρ=0.1)** | **95.87%** | **95.73%** | 0.0347 | 0.1500 | 0.1500 | 104s | +0.14 |
| **ARS2-Neo (Base)** | 95.58% | 95.52% | 0.0181 | 0.2400 | 0.2500 | 71s | +0.06 |
| **ARS2-Neo (AGA, λ=2.0)** | 94.10% | 94.09% | 0.1251 | 0.1800 | 0.1800 | 90s | +0.01 |
| **AdamW** | 94.60% | 94.47% | 0.0451 | 0.2500 | 0.2700 | 58s | +0.13 |
| **Muon** | 93.76% | 93.69% | 0.0267 | 0.2900 | 0.2900 | 75s* | +0.07 |

## 实验对比：Wikitext-2 (LRP)

实验设置: Qwen3 (RoPE, 3-layer), Context 255.

| 优化器 | Best PPL | Final PPL | Best Eval Loss | Final Eval Loss | Final Train Loss | Avg Time | PPL Gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AdamW** | 116.46 | 213.52 | 4.76 | 5.36 | 2.9740 | 314s | +97.06 |
| **Muon** | 111.35 | 475.65 | 4.71 | 6.16 | 2.2938 | 445s | +364.30 |
| **ARS2-Neo (Base)** | 96.10 | 3055.47 | 4.57 | 8.02 | 0.9123 | 425s | +2959.37 |
| **ARS2-Neo (Sync)** | **90.69** | **330.85** | **4.51** | 5.80 | 1.6100 | 784s | +240.16 |
| **ARS2-Neo (AGA)** | 93.23 | 414.83 | 4.54 | 6.03 | 1.5906 | 546s | +321.60 |

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

### LRP 实验

| 优化器 | 拟合 (Epoch) | 顿悟 (Epoch) | 收敛 (Epoch) | 最佳 Eval Acc |
| :--- | :--- | :--- | :--- | :--- |
| **AdamW** | 113 | >600 | N/A | 15.65% |
| **Muon** | 22 | >347 | N/A | 36.83% |
| **ARS2-Neo (Base)** | 11 | 239 | 290 | 99.53% |
| **ARS2-Neo (AGA)** | 12 | **77** | **116** | **99.60%** |
| **ARS2C (AGA)** | 13 | 93 | 137 | 99.06% |
| **ARS2C (Scaler) (AGA)** | 13 | 75 | 172 | 99.03% |
| **ARS2D (Base)** | 11 | 237 | 264 | 99.05% |
| **ARS2D (AGA)** | 12 | **60** | **112** | **99.00%** |
| **ARS2C-SAGA** | 13 | 98 | **N/A** | **98.86%** |
| **ARS2DC-SAGA** | 13 | 79 | **204** | **99.16%** |

### SAGA 变体诊断分析

> 状态：SAGA（动态 ρ）在 Grokking 任务上表现不如预期。结合组合收敛定律，问题定位在 `cos_sim` 的**系统性偏低**导致的动态 ρ 过膨胀。

#### 关键诊断数据（ARS2C-SAGA Epoch 267）

| 指标 | 值 | 含义 |
|:---|---:|:---|
| `cos_sim_mean` | 0.178 | 曲率方向与更新方向几乎正交 |
| `rho_mean` | 0.038 | ρ 被压低（而非膨胀），与预期相反 |
| `alignment_mean` | 0.523 | β 调制在中间值 |
| `c_norm_mean` | 242.0 | 曲率强度很高 |

#### 根因分析

**ARS2C-SAGA（267ep 未达 99%）：**

其 `cos_sim_mean = 0.178` 远低于理论预期 —— Christoffel 方向 `c_ortho`（反映 Hessian-向量积）与 NGD 更新方向 `s_ortho`（反映预白化梯度）在 Grokking 的晚期记忆阶段呈近乎正交的关系。这导致：

- `f_a = 1 + (1 - cos_sim) = 1.822` 持续偏高
- `rho_target` 虽然经 `f_c` 的 log 压缩后仍被放大
- 但实际 `rho_mean = 0.038` 低于初始 `rho = 0.1` —— `f_c` 的 log 压缩配合 `nu` 的 EMA 跟踪使 c_norm 基线持续增长，导致 `f_c = 1 + log(1 + c_norm / nu)` 在后期趋近于 1，所以 ρ 实际上在下降

**真正的瓶颈在于：** SAGA 的 `cos_sim` 信号在 Grokking 的高精度微调阶段失去鉴别力 —— 当模型接近最优解时，梯度小而曲率变化大，两个方向虽然都蕴含几何信息但严重失配，导致 ρ 调制退化为随机游走。

**ARS2DC-SAGA（204ep 99.16%，2x 慢于 ARS2D AGA）：**

双正交化（`is_dual=True`）使 `s_ortho` 经过两次 NS 正交化（行+列），而 `c_ortho` 只做一次 NS 正交化 —— 两个方向的正交化深度不一致，系统性拉低 `cos_sim`。这导致：

1. `f_a` 系统性偏高 → SAM 扰动偏大 → 持续过度探索
2. 在高精度阶段（>98%）收敛极慢：95%→98% 用 50ep（vs ARS2D AGA 的 19ep）
3. 但双正交化的**更平坦诱导**最终帮助模型在 204ep 跨过 99% 阈值（vs ARS2C-SAGA 从未达到）

#### 待解决问题

- SAGA 的 `cos_sim` 信号需要重新设计：可能用行列对齐矩阵的均值替代全局标量，或引入 `alignment_mean` 作为 `cos_sim` 的替代信号
- 双正交化场景下，`c_ortho` 也应做双正交化以匹配 `s_ortho` 的几何深度
- New SAGA 的 `f_a = 1 + (1 - cos_sim)` 线性响应在 `cos_sim` 接近 0 时过于激进，可能需引入 sigmoid 门控

## 参考文献

- [1] L. Rui, "Integrated Predictive Workspace Theory," Zenodo, 2025.
- [2] Kingma & Ba, "Adam: A method for stochastic optimization," ICLR 2015.
- [3] Jordan et al., "Muon: An optimizer for hidden layers in neural networks," 2024.
- [4] Li et al., "ROOT: Robust orthogonalized optimizer," arXiv:2511.20626.
- [5] Si et al., "AdaMuon: Adaptive Muon optimizer," arXiv:2507.11005.
- [6] Li et al., "NorMuon: Making Muon more efficient and scalable," arXiv:2510.05491.
- [7] J. Zhuang et al., "GSAM: Surrogate Gap Guided Sharpness-Aware Minimization," in *Proc. 10th Int. Conf. Learn. Represent. (ICLR)*, 2022. [Official PyTorch Implementation](https://github.com/juntang-zhuang/GSAM)
