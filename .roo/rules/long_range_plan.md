# ARS2-Neo: Long-Range Dynamics & Spectral Collapse Verification Plan

> **Status**: Active (2026-01-21)
> **Target**: NeurIPS 2026 / ICLR 2027
> **Focus**: Empirical Verification of Geodesic Optimization & Manifold Flatness
> **Note**: 本计划已根据 [`training_refactor_plan.md`](.roo/rules/training_refactor_plan.md:1) 完成原子化重构，正式进入长周期实验阶段。

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

## 2. 实验设计哲学 (DFS-Binary Search)

为了在有限算力下最大化信息熵，我们采用 **深度优先二分搜索 (DFS-Binary Search)** 策略：

1. **Baseline Skip**: 已知 Muon 优于 AdamW，故直接以 Muon 为基准线。
2. **ρ-DFS (Rho Search)**: 在 Sync 模式下寻找最佳扰动半径 ρ。由于几何流形曲率对 ρ 敏感，我们将从锚点出发向两端探测。
3. **k-DFS (Lazy Efficiency)**: 在确定 ρ_opt 后，固定该参数，通过二分搜索扫描 k 值与注入强度 α 的组合。
4. **AGA 自动化**: 通过搜索几何一致性阈值 $L$ 实现全自动效率平衡，验证 $k_{eff}$ 是否随训练阶段呈现对数增长。

## 3. 实验矩阵 (Experiment Matrix)

所有配置文件遵循命名规范：`lrp_<task>_ars2_neo_<mode>_<epochs>e_<params>.toml`。

### 3.1 E1: 视觉任务的收敛极限 (CIFAR-10)

> **对应论文章节**: 4.2 Convergence Analysis & 4.5 Scalability
> **配置基准**: ResNet-18, Batch Size 256.

| ID | Config File | Mode | Epochs | Params (ρ/k/α) | 科学目标 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C-Muon** | `lrp_cifar10_muon_100e.toml` | Muon | 200 | Standard | **Baseline**: 纯几何优化的基准线。 |
| **C-Base** | `lrp_cifar10_ars2_neo_base_100e.toml` | Base | 200 | k=0 | **Ablation**: 验证能量解耦架构的独立有效性。 |
| **C-Sync** | `lrp_cifar10_ars2_neo_sync_100e_010.toml` | Sync | 200 | 0.1 / 1 / - | **SOTA**: 冲击 93%+ 精度，对标 AdaFisher/SAM。 |
| **C-Lazy5** | `lrp_cifar10_ars2_neo_lazy_100e_010_5_010.toml` | Lazy | 200 | 0.1 / 5 / 0.1 | **Efficiency**: 验证 $k=5$ 是最佳甜点位。 |
| **C-AGA** | `lrp_cifar10_ars2_neo_aga_100e_L10.toml` | AGA | 200 | L=0.10 | **Dynamics**: 验证几何一致性驱动的自动步长。 |

### 3.2 E2: 语言建模的深度泛化 (WikiText-2)

> **对应论文章节**: 4.3 Generalization Dynamics & 4.4 Spectral Analysis
> **配置基准**: Qwen3 (RoPE), Context 255.

| ID | Config File | Mode | Epochs | Params (ρ/k/α) | 科学目标 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **W-Base** | `lrp_wikitext2_ars2_neo_base_50e.toml` | Base | 50 | - / 0 / - | **Control**: 记录无平坦度约束下的过拟合曲线。 |
| **W-Sync** | `lrp_wikitext2_ars2_neo_sync_50e_010.toml` | Sync | 50 | 0.1 / 1 / - | **Dynamics**: 捕捉谱熵坍缩与 PPL 下降的相关性。 |
| **W-Lazy** | `lrp_wikitext2_ars2_neo_lazy_50e_010_5_010.toml` | Lazy | 50 | 0.1 / 5 / 0.1 | **Stability**: 验证长周期下的数值稳定性。 |

### 3.3 E3: Grokking 动力学 (Modular Addition)

> **对应论文章节**: 4.6 Phase Transition Analysis

| ID | Config File | Mode | Epochs | Params | 科学目标 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **G-Sync** | `lrp_grok_ars2_neo_sync_1000e.toml` | Sync | 1000 | ρ=0.1 | 验证平坦度约束对泛化相变的加速效应。 |

## 4. 关键监控指标 (Key Metrics)

### 4.1 性能指标 (Performance)

- **Test Accuracy / PPL**: 标准评估指标。
- **Train-Test Gap**: 量化泛化能力，验证 SAM 机制的有效性。

### 4.2 动力学探针 (Dynamics Probes)

- **Spectral Entropy $H(S)$**:
  $$ H(S) = -\sum p_i \log p_i, \quad p_i = \sigma_i / \sum \sigma_k $$
  _预期_: ARS2-Neo 的 $H(S)$ 应比 AdamW 下降得更快、更低。
- **Update Norm $\| \Delta \theta \|_F$**: 监控能量注入的稳定性。
- **Effective Rank**: 更新矩阵的有效秩。
- **Average k (AGA Mode)**: 监控几何一致性随训练阶段的变化。

## 5. 执行路线图 (Execution Roadmap)

### Phase 1: Baseline & Anchor (T+2 Days)

- [ ] Run `C-Muon`, `C-Base`, `C-Sync` (100e)
- [ ] Run `W-Base`, `W-Sync` (50e)
- _Decision_: 若 `Sync > Base`，说明 SAM 有效，进入 ρ-DFS。

### Phase 2: ρ-DFS & AGA Dynamics (T+4 Days)

- [ ] 执行 ρ 探测：`ρ=0.05`, `ρ=0.2`, `ρ=0.5`
- [ ] Run `C-AGA` 采集 `avg_k` 曲线。

### Phase 3: Efficiency & Grokking (T+6 Days)

- [ ] Run `C-Lazy5`, `W-Lazy`
- [ ] Run `G-Sync`

### Phase 4: Data Synthesis & Plotting (T+7 Days)

- [ ] 绘制 "Accuracy vs. Wall-clock Time" 曲线。
- [ ] 绘制 "Spectral Entropy Evolution" 曲线。
- [ ] 整理 "Ablation Study" 表格。

## 6. 风险与应对 (Risk Management)

1. **数值不稳定性**: 监控 `grad_norm`，必要时启用 `trust_region_clip`。
2. **过拟合**: 实施 Early Stopping，报告 "Best Epoch" 性能并分析过拟合临界点。
3. **资源争用**: 优先运行 `C-Sync` 和 `W-Sync`，这是论文的核心论据。
