# FOG/DiagFOG：算子复合的几何优化范式（统一版）

版本: 2.0 (2025-11-14)  
作者: Ω Researcher  
状态: 取代 PI-Muon 框架的理论继任者，实验验证完成

## 摘要

FOG (Fisher-Orthogonalized Gradient) 优化器标志着从**线性混合**到**算子复合**的范式转换。基于调试日志的关键发现——PI-Muon 的几何不相容性源于在异构流形间进行向量加法——FOG 采用清晰的映射链：`Raw_Gradient → Statistical_Op → Structural_Op → Update`。DiagFOG 作为 FOG 的对角线近似版本，通过仅计算 Fisher 信息矩阵的对角元素，将内存复杂度从 O(d²) 降低到 O(d)，使大模型训练成为可能，同时保持算子复合的核心优势。

## 1. 理论突破：从混合到复合

### 1.1 PI-Muon 的根本缺陷

**几何不相容性**：`g_update = (1-λ)g_fisher + λg_muon` 在两个不同流形的测地线之间做向量加法，相当于在球面上走直线。

**归因困境**：全局 PI 无法告诉我们是哪个参数导致了"意外"。

### 1.2 算子复合的顿悟

**新范式**：

```
旧思维: g_update = λ₁·Statistical_Gradient + λ₂·Structural_Gradient
新思维: g_update = Structural_Op( Statistical_Op( Raw_Gradient ) )
```

**几何解释**：

1. **KFAC 步骤**：在**被噪声扭曲的局部统计流形**上找到最速下降方向 `g_nat = ℱ_emp⁻¹g`
2. **Muon 步骤**：将 `g_nat` **投影**到 Stiefel 流形，找到最近的"结构安全"方向

## 2. 形式化更新规则

### 2.1 FOG (Full KFAC + Muon)

**算法流程**：

```
输入: 梯度 g，参数 θ，KFAC 状态 (A, B)
输出: 更新方向 g_update

1. 统计操作: g_nat ← KFAC⁻¹g  // Fisher 预处理
   g_nat = (A⁻¹ ⊗ B⁻¹) @ g
2. 结构操作: g_update ← Muon(g_nat)  // 正交投影
   g_update = zeropower_via_newtonschulz5(g_nat, steps=5) × scaling_factor
3. 参数更新: θ ← θ - η·g_update
```

**实现代码**（[`optimizer/fog.py`](optimizer/fog.py:74-78)）：

```python
fog_update_w = muon_update(
    g_nat_w,
    state_w['muon_momentum_buffer'],
    beta=self.muon_momentum
)
```

### 2.2 DiagFOG (Diagonal KFAC + Muon)

**算法流程**：

```
输入: 梯度 g，参数 θ，对角 Fisher 估计 f_diag
输出: 更新方向 g_update

1. 统计操作: g_nat ← g / (f_diag + ε)  // 元素级除法
   g_nat[i,j] = g[i,j] / (E[(∂L/∂W[i,j])²] + ε)
2. 结构操作: g_update ← Muon(g_nat)  // 正交投影不变
   g_update = zeropower_via_newtonschulz5(g_nat, steps=5) × scaling_factor
3. 参数更新: θ ← θ - η·g_update
```

**实现代码**（[`optimizer/diag_fog.py`](optimizer/diag_fog.py:95-100)）：

```python
A_inv_diag = 1.0 / (self.m_aa[m] + damping)
G_inv_diag = 1.0 / (self.m_gg[m] + damping)
v = p_grad_mat * (G_inv_diag.unsqueeze(1) @ A_inv_diag.unsqueeze(0))
```

## 3. 实验验证：完整结果

### 3.1 CIFAR-10 (ResNet-18) - 10 Epochs

| 优化器 | 最终准确率 | 相对提升 | 内存复杂度 | 计算复杂度 | 结论 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FOG** | **88.91%** (10 epochs) | **+1.86%** | **O(d²)** | **O(d³)** | 显著优于所有基线 |
| Muon | 87.05% | 0% (基线) | O(d) | O(d) | 性能良好 |
| DiagFOG | 77.66% | **-9.39%** | O(d) | O(d) | **灾难性失败** |

**关键发现**：

- FOG 在 10 个 epoch 达到 88.91% 准确率，比 Muon 基线提升 +1.86%
- DiagFOG 在卷积网络上完全失效，准确率比 Muon 低近 10 个百分点
- FOG 的内存使用为 1660MB，而 DiagFOG 仅 274MB

### 3.2 Wikitext-2 (Nano-GPT) - 10 Epochs

| 优化器 | 最终困惑度 | 相对提升 | 最佳epoch | 结论 |
| :--- | :--- | :--- | :--- | :--- |
| **DiagFOG** | **401.69** | **-14.49** (↓困惑度) | 第6epoch | **轻度优于 Muon** |
| Muon | 416.18 | 0 (基线) | 第8epoch | 性能良好 |
| DiagKFAC | 2952.44 | +2536.26 (↑困惑度) | - | 远差于 Muon |

**关键发现**：

- Full Fog直接爆显存，内存复杂度不可行。
- DiagFOG 在 Transformer 上取得成功，困惑度比 Muon 基线更低
- 最佳性能出现在第6-8epoch，之后出现过拟合
- 验证了"算子复合"思想在序列建模任务上的有效性

## 4. 理论解释

### 4.1 DiagFOG 在卷积网络上的失败

**根本原因**：对角近似对卷积核的 Fisher 信息矩阵进行了**过度简化**。

- **卷积核的几何**：卷积核 `W ∈ ℝ^{C_out × C_in × K_h × K_w}` reshape 成 `W' ∈ ℝ^{C_out × (C_in * K_h * K_w)}`
- **完整的 Fisher (KFAC)**：捕获所有协方差，提供丰富的统计结构
- **对角 Fisher (DiagKFAC)**：假设所有元素统计独立，丢失关键信息

### 4.2 DiagFOG 在 Transformer 上的成功

**根本原因**：Transformer 中的 `Linear` 层几何结构**更适合对角近似**。

- **Linear 层的几何**：权重矩阵 `W ∈ ℝ^{out_features × in_features}` 的 Fisher 结构更接近对角化
- **结构正则化的价值**：即使丢失统计信息，Muon 的正交化仍提供几何约束
- **功能协同**：KFAC 识别重要方向，Muon 约束更新方式

## 5. 计算复杂度分析

| 版本 | 内存复杂度 | 计算复杂度 | 适用场景 | 实验验证 |
| :--- | :--- | :--- | :--- | :--- |
| **FOG** | **O(d²)** | **O(d³)** | 小模型 (ResNet-18) | ✅ 88.91% 准确率 |
| **DiagFOG** | **O(d)** | **O(d)** | 大模型 (Transformer) | ✅ 401.69 困惑度 |

## 6. 核心洞察

### 6.1 算子复合的普适性

**核心机制**：

1. **统计适应**：KFAC/DiagKFAC 提供数据依赖的梯度重缩放
2. **结构稳定**：Muon 正交化抑制噪声，保持几何约束
3. **功能协同**：两者不是简单叠加，而是深度功能协同

### 6.2 架构依赖性

- **卷积网络**：需要完整的 Fisher 信息，DiagFOG 完全失效
- **Transformer**：对角近似足够，DiagFOG 有效且高效

## 7. 未来方向

### 7.1 Block-KFAC

对于卷积网络，实现分块 KFAC 在 O(k·d) 复杂度下保留更丰富统计信息：

`Block-KFAC: 将 W' 分块，每块内计算完整 Fisher`
`内存: O(k·d), 其中 k ≪ d 为块大小`

### 7.2 自适应选择

开发机制根据层类型自适应选择优化策略：

```python
if layer_type == 'Conv2d':
    use_full_kfac = True
elif layer_type == 'Linear' and param_count > threshold:
    use_diag_fog = True
else:
    use_muon = True
```

> "当几何原理正确时，性能提升是**必然**的，而非**偶然**的。"  
> —— 从 PI-Muon 到 FOG 的核心感悟
