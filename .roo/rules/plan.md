# 实验计划（修正版）

## PI-Muon：预测完整性驱动的二阶优化器

PI-Muon 是一个用「预测完整性」动态调控「统计适应性」与「结构稳定性」权重的二阶优化器。它将常规梯度下降分解为两种自然梯度：

- 统计自然梯度（KFAC Fisher 或其对角近似）：用于快速适应新分布；
- 结构自然梯度（Muon 正交更新）：用于最小扰动地保持既有知识。

融合权重 λₜ 由实时预测完整性 PI 计算得出，无需手动调参，形成自由能原理的工程闭环。

### 第一阶段：收敛速度验证（核心目标）

> 核心假设：基于 Fisher 的二阶信息应比一阶方法更有效。即使加入和  Muon 的正交投影，PI-Muon 的收敛速度也应显著优于 Muon 和 AdamW（p < 0.05）。

#### 阶段 1：Fisher NGD 模块实现（1.5 天）

- [ ] 版本 A (AdaFisher-PI-Muon)：实现 对角近似 Fisher 自然梯度 (`g_fisher_diag`)。
  - 复用 `ref/Adafisher/optimizers/AdaFisher.py` 的对角 KFAC 逻辑。
  - 在 CIFAR-10 和 Wikitext-2 上，单次 `step()` 耗时 < 1.5× AdamW，内存占用 < 1.2×。
- [ ] 版本 B (KFAC-PI-Muon)：实现 完整 KFAC Fisher 自然梯度 (`g_fisher_kfac`)。
  - 复用 `ref/Adafisher/optimizers/kfac.py` 的完整 KFAC 逻辑。
  - 在 CIFAR-10 和 Wikitext-2 上，评估其计算开销（预计 > 3× AdamW）。
- [ ] 验证目标：任一版本的 Fisher NGD 在 50 个 epoch 内达到 比 AdamW 更低的验证损失。

#### 阶段 2：Muon 结构梯度模块（0.5 天）

- [ ] 集成现有 Muon 代码，实现 结构自然梯度 (`g_muon`)，确保与 Fisher NGD 模块作用于同一参数子集。
- [ ] 在 CIFAR-10 和 Wikitext-2 上，Muon 更新与 `g_fisher` 更新维度匹配，无运行时错误。

#### 阶段 3：PI 实时计算（0.5 天）

- [ ] 实现 预测完整性 (`PI`) 的零成本计算：`PI = exp(-α * (H(y|x) + γ * ||g||²))`。
- [ ] `PI` 值在 `[0, 1]` 区间，计算耗时 < 0.1× 反向传播。

#### 阶段 4：动态正交融合（1 天）

- [ ] 实现 PI 驱动的正交投影融合逻辑：
  - 高 PI (`> 0.5`)：统计主导，`g_update = g_fisher + α(PI) * Ortho(g_muon, g_fisher)`
  - 低 PI (`≤ 0.5`)：结构主导，`g_update = g_muon + β(PI) * Ortho(g_fisher, g_muon)`
- [ ] 融合后的 `g_update` 与任一原始梯度夹角 < 45°，避免方向突变。

#### 阶段 5：收敛速度对比（1 天）

- [ ] 在 CIFAR-10 和 Wikitext-2 上，对比 两个 PI-Muon 版本 与最强基线（Muon、AdamW）的：
  - 达到目标验证损失所需的 迭代次数（越少越好）
  - 最终验证损失（越低越好）
- [ ] 退出条件（Kill Criteria）：若 两个 PI-Muon 版本 在 任一任务 上：
  - 收敛速度 均未显著优于 Muon（p < 0.05）
  - 则终止 PI-Muon 路线，转向 PI-ZPD 或其他方向。

> 总预期工时：4.5 天（含调试与可视化）
> 成功标准：至少一个 PI-Muon 版本 在 至少一个任务 上，同时实现：
>
> - 收敛速度 显著优于 Muon（p < 0.05）
> - 最终验证损失 不劣于 AdamW（p > 0.05）
>
> 则进入下一阶段：持续学习验证。

### 第二阶段：持续学习验证（独立任务）

> 目标：在 MNIST→FashionMNIST 混合学习 场景下，验证 PI-Muon 是否比基线更有效地缓解灾难性遗忘。

#### 阶段 6：MNIST-FashionMNIST 混合学习（2 天）

- [ ] 实现 MNIST→FashionMNIST 混合学习任务：先训练 MNIST 10 epoch，再训练 FashionMNIST 10 epoch，记录 MNIST 遗忘率。
- [ ] 对比 PI-Muon 与最强基线（PIWD、AdamW）的：
  - MNIST 最终准确率（越高越好）
  - FashionMNIST 最终准确率（越高越好）
  - MNIST 遗忘率（越低越好）
- [ ] 退出条件（Kill Criteria）：若 PI-Muon 在 任一指标 上：
  - 未显著优于 PIWD（p < 0.05）
  - 则终止 PI-Muon 路线，转向 PI-ZPD 或其他方向。

> 成功标准：PI-Muon 在 至少一个指标 上，显著优于 PIWD（p < 0.05），则进入下一阶段：工程化与分布式实现。

## PI-ZPD: 预测完整性引导的最近发展区优化

PI-ZPD 把被动梯度下降升维成可微分课程设计：用实时预测完整性 PI 作为模型认知坐标，动态软化交叉熵损失 `L = w(PI)·CE`。高 PI 时 w→1，全力学习；低 PI 时 w→0，抑制梯度，实现维果茨基「最近发展区」的可计算脚手架。

此时「教师」由外部真理 y_true 与内部信念 y_pred 动态合成，PI 权重在每一步仲裁应更多服从世界还是保留偏见，使巩固与探索保持自适应平衡。样本若落在 ZPD 内则全力更新，在外则策略性等待，防止灾难性冲击。

该变体待 baseline 套件完成后实现，用于持续学习场景。
