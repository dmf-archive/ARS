# ARS2-Neo: Long-Range Dynamics & Spectral Collapse Verification Plan

> **Status**: Draft
> **Target**: NeurIPS 2026 / ICLR 2027
> **Focus**: Empirical Verification of Geodesic Optimization & Manifold Flatness

本计划旨在为 **ARS2-Neo** 的论文手稿提供决定性的实验证据。不同于早期的快速验证（Quick Check），本阶段实验关注长周期训练下的动力学行为，旨在验证 **IPWT (整合预测工作空间理论)** 对优化过程的核心预测：**有效的泛化对应于参数流形上的谱熵坍缩与自由能最小化。**

## 1. 理论动机与验证目标

根据 `ARS2_Neo_Paper_Draft_CN.md` 中的理论框架，我们需要通过以下实验回答审稿人可能提出的核心质疑：

1. **收敛极限 (The Limit of Convergence)**:
   - _Hypothesis_: ARS2-Neo 在短周期内的爆发力是否会在长周期中衰减？
   - _Target_: 证明 ARS2-Neo 不仅是“起步快”，而且能达到比 SGD/AdamW 更高的收敛精度（CIFAR-10 > 97%）。

2. **谱熵坍缩 (Spectral Entropy Collapse)**:
   - _Hypothesis_: 优化器诱导的稀疏性（Optimizer-induced Sparsity）是真实存在的。
   - _Target_: 观测到 ARS2-Neo 的更新矩阵谱熵 $H(S)$ 随训练进行显著下降，证明其自动发现了低维流形。

3. **Lazy Mode 的有效性 (Efficiency of Manifold Sliding)**:
   - _Hypothesis_: 流形曲率的变化率远低于参数更新率，因此剪切力 $v_{flat}$ 可以被复用。
   - _Target_: 证明 $k=5$ 的 Lazy Mode 在计算开销仅增加 ~25% 的情况下，能保持 Sync Mode 95% 的性能。

## 2. 实验矩阵 (Experiment Matrix)

所有配置文件遵循命名规范：`ars2_neo_<mode>_<epochs>e_<rho>_<k>_<alpha>.toml`。

### 2.1 E1: 视觉任务的收敛极限 (CIFAR-10)

> **对应论文章节**: 4.2 Convergence Analysis & 4.5 Scalability

| ID          | Config File                                 | Mode | Epochs | Params (ρ/k/α) | 科学目标                                               |
| :---------- | :------------------------------------------ | :--- | :----- | :------------- | :----------------------------------------------------- |
| **C-Base**  | `cifar10_ars2_neo_base_200e.toml`           | Base | 200    | - / 0 / -      | **Baseline**: 测定纯几何优化的过拟合边界。             |
| **C-Sync**  | `cifar10_ars2_neo_sync_200e_010.toml`       | Sync | 200    | 0.1 / 1 / -    | **SOTA**: 冲击 93% 精度，对标 AdaFisher/SAM。          |
| **C-Lazy5** | `cifar10_ars2_neo_lazy_200e_010_5_010.toml` | Lazy | 200    | 0.1 / 5 / 0.1  | **Efficiency**: 验证 $k=5$ 是最佳甜点位 (Sweet Spot)。 |
| **C-Lazy3** | `cifar10_ars2_neo_lazy_200e_010_3_010.toml` | Lazy | 200    | 0.1 / 3 / 0.1  | **Ablation**: 验证剪切力注入频率对精度的影响。         |

### 2.2 E2: 语言建模的深度泛化 (WikiText-2)

> **对应论文章节**: 4.3 Generalization Dynamics & 4.4 Spectral Analysis

| ID         | Config File                                            | Mode | Epochs | Params (ρ/k/α) | 科学目标                                        |
| :--------- | :----------------------------------------------------- | :--- | :----- | :------------- | :---------------------------------------------- |
| **W-Base** | `wikitext2_line_rope_ars2_neo_base_50e.toml`           | Base | 50     | - / 0 / -      | **Control**: 记录无平坦度约束下的过拟合曲线。   |
| **W-Sync** | `wikitext2_line_rope_ars2_neo_sync_50e_010.toml`       | Sync | 50     | 0.1 / 1 / -    | **Dynamics**: 捕捉谱熵坍缩与 PPL 下降的相关性。 |
| **W-Lazy** | `wikitext2_line_rope_ars2_neo_lazy_50e_010_5_010.toml` | Lazy | 50     | 0.1 / 5 / 0.1  | **Stability**: 验证长周期下的数值稳定性。       |

## 3. 关键监控指标 (Key Metrics)

为了支持论文中的 "Dynamics Analysis" 章节，我们需要收集以下高分辨率数据：

### 3.1 性能指标 (Performance)

- **Test Accuracy / PPL**: 标准评估指标。

- **Train-Test Gap**: 量化泛化能力，验证 SAM 机制的有效性。

### 3.2 动力学探针 (Dynamics Probes)

- **Spectral Entropy $H(S)$**:
  $$ H(S) = -\sum p_i \log p_i, \quad p_i = \sigma_i / \sum \sigma_k $$
  _预期_: ARS2-Neo 的 $H(S)$ 应比 AdamW 下降得更快、更低。

- **Update Norm $\| \Delta \theta \|_F$**: 监控能量注入的稳定性。
- **Effective Rank**: 更新矩阵的有效秩。

## 4. 执行路线图 (Execution Roadmap)

### Phase 1: Baseline Establishment (T+2 Days)

_目标_: 确立 Base 模式的性能下限和过拟合点。

- [ ] Run `C-Base` (CIFAR-10 200e)
- [ ] Run `W-Base` (WikiText-2 50e)

### Phase 2: SOTA Pushing (T+4 Days)

_目标_: 获取论文所需的 "Bold" 数据，证明 ARS2-Neo 的优越性。

- [ ] Run `C-Sync` (CIFAR-10 200e) -> 目标 Acc > 92.5%
- [ ] Run `W-Sync` (WikiText-2 50e) -> 目标 PPL < 75

### Phase 3: Efficiency Verification (T+6 Days)

_目标_: 验证 Lazy Mode 的工程价值。

- [ ] Run `C-Lazy5` & `C-Lazy3`
- [ ] Run `W-Lazy`

### Phase 4: Data Synthesis & Plotting (T+7 Days)

_目标_: 生成论文插图。

- [ ] 绘制 "Accuracy vs. Wall-clock Time" 曲线 (证明 Lazy Mode 优势)。
- [ ] 绘制 "Spectral Entropy Evolution" 曲线 (证明理论假设)。
- [ ] 整理 "Ablation Study" 表格。

## 5. 风险与应对 (Risk Management)

1. **数值不稳定性 (Numerical Instability)**:
   - _Risk_: 长周期训练中，$\|g_{nat}\|$ 可能因方差累积而爆炸。
   - _Mitigation_: 监控 `grad_norm`，必要时在 `config` 中启用 `grad_clip` 或调整 `trust_region_clip`。

2. **过拟合 (Overfitting)**:
   - _Risk_: 即便有 SAM，WikiText-2 在 50e 后仍可能过拟合。
   - _Mitigation_: 实施 Early Stopping，论文中报告 "Best Epoch" 性能，并分析过拟合发生的临界点（这也是有价值的实验结果）。

3. **资源争用**:
   - _Risk_: 长周期实验占用大量 GPU 时间。
   - _Mitigation_: 优先运行 `C-Sync` 和 `W-Sync`，这是论文的核心论据。Lazy Mode 实验优先级次之。
