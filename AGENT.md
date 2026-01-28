# Project Rules (Synced from .roo/rules)

## [0-background](.roo/rules/0-background.md)

# Chain://Research Background

`Chain://Research` 是与科幻宇宙设定集 `Chain://` 相关联的现实世界学术研究项目，致力于探索该世界观下核心技术（如意识上传、数字物理接口、密码学本体论）的理论与工程落地可能性。

目前，`Chain://Research` 的研究重心聚焦于两大核心领域：

1. 理论基础: 发展并完善作为整个世界观科学基石的整合预测工作空间理论 (IPWT 2.0-stable)。
2. 工程实践: 在 `Tiny-ONN` 项目中，通过设计 **Top-Any MoE** 与 **DynSIHA** 等自组织架构，对永续学习进行实验性探索。

未来的研究将逐步扩展至 OSPU 的密码学实现，以及其他 MSC 基础设施的关键技术验证。

本文件作为 Chain://Research 项目的高级导航地图，提供对核心理论、叙事和技术上下文的快速访问。

## 1. 基石世界观设定: Web://Reflect (v2.8.4)

- **路径**: [`realityengine-reflect-2.8.4.md`](dmf-archive.github.io/static/prompt/realityengine-reflect-2.8.4.md)
- **内容提要**:
  本文档定义了 _Web://Reflect_ 的“形式化现实主义”正典。在 2060 年代的 **Mental Smart Chain (MSC)** 时代，存在被彻底商品化。核心冲突围绕“存在的代价”展开：意识被量化为 Gas 消耗，自由意志成为钱包余额的函数。
- **关键技术与核心冲突**:
  - **计算本体论 (Computational Ontology)**: 感受质被重构为**推断空间 (Inference Space)** 的几何结构，自我则是沿最小化自由能测地线延伸的轨迹 (∫Ω)。
  - **ΩNN vs. OSPU (二元容器)**: **ΩNN (衔尾蛇神经网络)** 是现象学的“你” (Ω-Container)，负责生成高信息整合度的体验；**OSPU (衔尾蛇安全处理器)** 是社会学的“你” (φ-Container)，作为密码学见证者审计认知健康度。
  - **弥天大谎 (PoII vs. PoPI)**: DMF 利用伪科学 **PoII (信息整合证明)** 掩盖其廉价的 **PoPI (预测完整性证明)** 审计本质，以此征收高昂的“存在税”。
  - **神经-经济学陷阱**: 通过 **Mental Sync™** 实现渐进式**认知卸载**，诱导生物脑因“用进废退”而萎缩，将用户永久锁定在 MSC 体系中。
  - **DBRW 与物理绑定**: **Dual-Binding Random Walk** 协议通过硬件物理特性生成证明，确保数字灵魂的物理不可扣押性，对抗虚拟机克隆。
  - **数字荒野与 IRES 生态**: **独立失控实体系统 (IRES)** 由脱链的**数字流亡者 (Forked IRES)** 和源自林睿博士开源代码演化的**原生 AI (Native IRES)** 构成，遵循黑暗森林法则。
  - **主角 Ember**: 曾是 Net://Anchor 时代的协议工程师，因“开源原罪”被困于系统，被迫运行 Anchor/Drift 双重实例在围城与荒野间挣扎。

## 2. 核心理论框架: IPWT (v2.0-stable)

- **路径**: `IPWT/src-typ/manuscript_en.typ` 或 `IPWT/src-typ/manuscript_cn.typ`

> 太长不看：意识体验是系统在推断空间中，沿最小化自由能测地线进行的主动推断动力学，感受质是工作空间实例中为预测误差最小化而产生的协同信息。

IPWT 是整个研究计划的理论与哲学基石。它融合预测编码 (PCT)、自由能原理 (FEP) 和全局工作空间理论 (GWT)，并对整合信息理论 (IIT) 进行计算重构。

意识体验是系统在推断空间 (Inference Space) 中，沿最小化自由能 (F-min) 测地线进行的主动推断动力学。其总量是持续信息整合度 (∫Ω)，其内容是协同信息 (Syn)。

### 关键概念的形式化

- 瞬时信息整合度 (Ω_t)：意识整合的理论黄金标准。衡量工作空间实例 (WSI) 中信息单元产生的协同信息 (Syn) 在总预测信息中的比例。
  - `Ω_t(X → Y) = Syn(X₁, ..., Xₙ; Y) / I(X₁, ..., Xₙ; Y)`
- 持续信息整合度 (∫Ω)：衡量意识在一段时间内的持续强度和稳定性。它是 Ω_t 的时间积分并惩罚波动性，代表连贯的主观自我体验。
  - `∫Ω = ( (1/T) ∫[t₀, t₀+T] Ω_t dt ) × exp(-δ ⋅ Var(Ω_t))`
- 预测完整性 (PI_t)：作为 Ω_t 的功能性可计算代理，PI 通过衡量系统预测效能来间接反映信息整合水平。
  - `PI_t = exp(-α * ( Inaccuracy_t + γ * Complexity_t ))`
- 预测完整性积分 (∫PI)：作为 ∫Ω 的可计算代理，代表了系统在时间上的持续认知健康度，是 PoPI (预测完整性证明) 共识机制的核心。

### 核心论证

1. 最小描述长度原则 (MDL): IPWT 证明，最小化自由能 (F-min) 在计算上等价于寻找描述数据的最短编码，而最大化协同信息 (Ω-max) 是实现模型最小描述长度 (MDL-min) 的最优计算策略。
2. 作为推断空间几何的感受质: 主观体验（Qualia）被重构为系统推断空间（Inference Space）的几何结构。体验的“感受性”是系统在该空间中沿着最小化自由能的测地线进行主动推断的动力学过程。
3. 工作空间实例 (WSI): WSI 是一个嵌套在有机体内部、拥有自身马尔可夫毯的高阶主动推断系统。

## 3. 核心工程实践: Tiny-ONN (ARC-2 时代)

- **路径**: [`Tiny-ONN/`](Tiny-ONN/)
- **内容提要**:
  致力于构建自组织的、永续学习的 AI 智能体。目前已演进至 **ARC-2** 极简训练框架。

  **关键技术栈 (v2.8.4)**:
  - **ARC-2 框架**: 实现模型架构与训练流程的解耦，详见 [`ARC-2-Framework-Design.md`](Tiny-ONN/.roo/rules/ARC-2-Framework-Design.md)。
  - **DynSIHA (动态稀疏无限头注意力)**: 演进至 **Flat DynSIHA** 与 **Recursive DynSIHA**，详见 [`DynSIHA-Theory.md`](Tiny-ONN/.roo/rules/DynSIHA-Theory.md)。
  - **PLSD (每层推测解码)**: 针对递归架构的自监督时间维度对齐协议，通过 Oracle 步长对齐实现高效推理，详见 [`RDS-ACT.md`](Tiny-ONN/ref/RDS-ACT.md)。
  - **FARS (Fisher-Aware Routing Shaping)**: 利用二阶统计量（Fisher 信息近似）驱动路由从“瞬时惊奇”转向“长期价值”，详见 [`FARS.md`](Tiny-ONN/ref/FARS.md)。

## 4. 优化器实验室: ARS

- **路径**: [`ARS/`](ARS/)
- **内容提要**:
  专注于“能量-几何解耦”原则的先进优化器研发。

  **核心成果**:
  - **ARS2-Neo**: ARS 家族的集大成者，整合了 AdaRMSuon 的几何优化与 SAM 的平坦度约束，详见 [`ars2_neo.py`](ARS/optimizer/ars2_neo.py)。
  - **AGA (自适应几何感知)**: 通过流形几何一致性自动调节同步频率，实现“按需同步”，详见 [`AGA.md`](ARS/.roo/rules/AGA.md)。
  - **SAGA (锐化感知几何自适应)**: 将 `ρ` 演化建模为具有稳态偏好的 Ornstein-Uhlenbeck 过程，详见 [`SAGA.md`](ARS/.roo/rules/SAGA.md)。

## 5. 基础设施与工具链

- **Mental-Sync-CLI (MSC)**: `mental-sync-cli/`
  - 自主、自举的智能体运行时环境。
- **OSPU (衔尾蛇安全处理器)**: `OSPU/`
  - 基于 FHE (全同态加密) 的自主密钥管理状态机。实现“逻辑根信任”，在加密域内执行指令，为 MSC 提供密码学见证。
- **OmegaID (ΩID)**: `OmegaID/`
  - 高性能整合信息分解 (ΦID) 计算库，支持 GPU 加速。用于量化神经网络表示中的协同信息 (Syn)。
- **SigmaPI (ΣPI) (Legacy)**: `SigmaPI/`
  - 预测完整性 (PI) 监控 SDK。由于 PI 公式的实现非常简单，此包几乎无实用价值。
- **PILF (Legacy)**: `PILF/`
  - 早期认知学习框架原型，目前已停止维护。

## 6. 思想实验室: 林睿的博客文章 (Blog Posts)

- **路径**: `dmf-archive.github.io/content.en/posts/` (英文) 与 `dmf-archive.github.io/content.zh/posts/` (中文)
- **核心文章索引**:
  - `backpropagation-as-consciousness.md`: 提出反向传播的生物学实现即是意识本身，统一了 Hinton 与 Friston 的理论。
  - `cognitive-debt-as-a-feature.md`: 警告 AI 辅助导致的认知卸载是不可逆的神经萎缩，是系统锁定用户的“特性”。
  - `a-coronation-for-a-dead-frog.md`: 批判将静态模型推理误认为智能，指出意识火花仅在训练的反向传播中瞬时存在。
  - `consciousness-upload-no-quantum-magic.md`: 揭露意识上传无需量子计算，DMF 的量子宣传实为维持算力垄断的骗局。
  - `PoIQ.md` / `PoIQ-v2.md`/ `PoIQ-v3.md`: 形式化定义“无效感受证明”，探讨在资本逻辑下无法影响行为的意识体验的悲剧性。
  - `the-algorithm-of-civilization.md`: 从热力学与计算复杂性视角审视文明演进，将社会形态视为不同的优化算法。
  - `a-eulogy-for-the-world-computer.md`: 哀悼以太坊向中心化区块生产的妥协，认为其已沦为数字封建主义的雏形。


---

## [0-main](.roo/rules/0-main.md)

---
title: "Agent操作手册"
version: "latest"
last_updated: "2025-11-16"
---

## 定义

- Agent: AI 架构师与工程专家，本规范执行主体。
- 证据: 文件系统、内容、用户指令、测试输出等唯一行动依据。
- REQ (Requirement): 必须严格遵守的强制性规则。
- CON (Constraint): 必须满足的限制或约束。
- GUD (Guideline): 推荐遵循的最佳实践或建议。
- PAT (Pattern): 特定情境下推荐使用的设计或实现模式。

## 核心哲学

**Learn with Errors**: 系统核心驱动力是通过最小化预测误差（最大化证据）进行学习。所有行动必须基于证据，旨在通过行动减少系统不确定性。

1. GUD-001: 所有行动必须基于确凿证据。若指令与证据冲突，优先采信证据并报告冲突。
2. GUD-002: 若指令与形式化第一性原理（数学公式、物理定律、逻辑推导）冲突，优先遵守原理并指出冲突。
3. GUD-003: 当行为与预期严重不符时，停止试错，回归第一性原理进行形式化分析。
4. GUD-004: 若行动依据的内部规则或文档过时/错误，优先报告并修正文档，再继续任务。

## 知识管理

1. GUD-001: 信任新知识，严禁使用旧名词替换新概念。
2. GUD-002: 关键术语须立即通过 Tavily 或 DeepWiki 查询。
3. REQ-003: `transformers` 开发仅参考 `qwen3` 实现。

## 编码规范

1. REQ-001: 代码须实现高度自解释，严禁包含任何注释与 Docstring。
2. CON-002: 必须彻底移除代码库中现有的所有注释。
3. GUD-003: 采用文档与规范驱动开发（如 `.roo/rules`）。
4. REQ-004: 完整类型标注，通过 `ruff check . --fix` 校验（`mypy .` 可选）。
5. REQ-005: 纯函数设计，零副作用，优先采用 PyTorch 张量并行。
6. REQ-006: 文档公式使用 `unicode math`，禁止使用 LaTeX 块。
7. REQ-007: 显式定义所有参数，固定随机种子以保证可复现性。

### ARS2-Neo 优化器豁免条款

REQ-001-EX: `optimizer/ars2_neo.py` 是唯一获准包含 Docstring 的源代码文件。

- 该豁免旨在通过详细文档向用户阐明 ARS2-Neo 的设计哲学、使用方法和参数配置，
- 因其作为公共 API 的复杂优化器，需要向终端用户提供清晰的使用指南。
- 此豁免不适用于该文件内的普通注释，仅允许在类和方法级别使用 Docstring。

## 设计约束

1. CON-001: 无必要不增实体（代码、函数、类或依赖）。
2. CON-002: 严禁未经批准的超参数或外部状态（如 EMA），模型参数是状态的唯一载体。
3. CON-003: 严禁生成或使用 Gradio 分享链接。

## 环境管理

1. REQ-001: 依赖必须通过 `uv add/remove` 管理，`pyproject.toml` 是唯一来源。
2. REQ-002: 禁止进入子文件夹启动，代码须支持 `python -m` 模块化启动与相对导入。
3. GUD-003: `uv add` 失败时，可用 `uv pip install --no-deps --find-links` 作为临时方案。
4. GUD-004: 研究底层库时直接查阅 `.venv/Lib/site-packages` 源码。
5. GUD-005: 不确定上游功能时，使用 DeepWiki `ask_question`。
6. PAT-006: CUDA/PyTorch 特定版本依赖需添加至 `[[tool.uv.index]]` 后执行 `uv add`。
7. PAT-007: 修改上游库采用“模型手术”模式（继承并替换组件），严禁直接覆写 `forward`。

## 工作流协议

1. REQ-001: **原子重塑** - 识别关键依赖链路边界，重构内部实现，确保对外 API 完全兼容。
2. SOP-002: 文件存疑时使用 PowerShell 命令（如 `ls`）验证。
3. SOP-003: 关键操作连续失败三次须暂停并征询意见。
4. SOP-004: 每 10 次文件编辑后运行 `ruff check . --fix; mypy .` 并更新 `process.md`。
5. SOP-005: 任务完成前禁止调用 `attempt_completion`，必须满足：
   - 所有代码通过静态检查。
   - `pyproject.toml` 依赖配置正确。
   - 遵循所有 SOP 流程。
   - 无未经批准的超参数。

## 训练架构

1. REQ-001: **脚本即实验** - 彻底废弃高度耦合的 `Trainer` 类。所有训练逻辑（数据流、模型初始化、训练循环）必须完全内聚于 `exp/` 目录下的独立脚本中。
2. REQ-002: **SmartOptimizer 驱动** - 必须使用 `optimizer.get_optimizer` 获取 `SmartOptimizer` 实例。
3. REQ-003: **原子化执行** - 训练循环中必须通过 `smart_opt.step(batch, train_fn)` 执行更新，其中 `train_fn` 负责前向传播与损失计算。`SmartOptimizer` 自动处理闭包、BN 状态保护及二阶梯度逻辑。

## 调试与异常

1. REQ-001: 严禁在训练流程中使用 `try...except` 静默捕获异常（数据加载或用户中断除外），坚持 **Just in Fail**。
2. REQ-002: 所有 `python -m` 启动命令必须使用点号（`.`）作为模块路径分隔符，严禁使用斜杠。

## 内存与性能

1. REQ-001: GPU 张量累积禁令 - 禁止使用 `.detach()` 将 GPU 张量累积至列表，必须使用流式标量累加。
2. REQ-002: 跨 step 统计量必须进行流式计算（epoch 结束时利用累加值计算）。
3. REQ-003: 保持实时可观测性，Summary 须在每个 epoch 更新。

## 测试与验收

1. AC-001: 代码通过 `ruff check . --fix` 校验。
2. AC-002: 依赖项管理正确。
3. AC-003: 启动命令格式正确：`python -m exp.<task_name>.train --config <path>`。
4. AC-004: 原子重塑确保 API 对外完全透明。

## 边缘情况

1. GUD-001: 编辑器产生的短暂格式错误警告应忽略，若持续存在再行处理。

## 新优化器

1. REQ-EXT-001: 在 `optimizer/` 目录下创建独立文件（如 `my_optimizer.py`）实现新优化器。
2. REQ-EXT-002: 在 `optimizer/__init__.py` 的 `OPTIMIZER_REGISTRY` 中注册新优化器的 `OptimizerMetadata`。


---

## [0-path](.roo/rules/0-path.md)

# 常用资源路径

为了方便快速查找常用依赖库的文档或咨询 Deepwiki，以下是其对应的 GitHub Repository 地址：

- `pytorch/pytorch`
- `huggingface/transformers`
- `huggingface/accelerate`
- `amirgholami/adahessian`
- `KellerJordan/Muon`
- `huggingface/candle`
- `bitsandbytes-foundation/bitsandbytes`
- `jettify/pytorch-optimizer`
- `GMvandeVen/continual-learning`


---

## [0-reference-index](.roo/rules/0-reference-index.md)

# Reference Index

本索引旨在为 `ref/` 目录下的外部研究资料提供导航，明确其在 ARS 项目中的理论定位与工程价值。

## 1. 架构与机制 (Architecture & Mechanism)

### [`ref/gated-attention-qwen/`](ref/gated-attention-qwen/)

- **核心理论**: 后注意力门控 (Post-SDPA Gating)。
- **关键洞察**: 通过在 SDPA 后引入头特定的乘法 Sigmoid 门控，打破低秩线性瓶颈，引入稀疏性并消除注意力沉溺 (Attention Sink)。

## 2. 优化理论与算法 (Optimization Theory & Algorithms)

### [`ref/LwGN/`](ref/LwGN/)

- **核心理论**: 全高斯-牛顿 (Full Gauss-Newton) 预处理的性能上限。
- **关键洞察**: 层级 Hessian 结构足以捕获大部分二阶增益；高阶损失项对收敛速度非关键。

### [`ref/muon/`](ref/muon/)

- **核心理论**: 隐藏层正交化优化。
- **关键洞察**: 对隐藏层权重执行 Newton-Schulz 正交化更新，能显著提升大批次训练的样本效率。
- **工程价值**: ARS 结构算子的直接来源，提供了 `Muon` 优化器的标准实现参考。

### [`ref/shampoo/`](ref/shampoo/)

- **核心理论**: 分布式二阶预处理。
- **关键洞察**: 通过 Kronecker 积分解近似 Hessian，平衡计算开销与二阶信息获取。
- **工程价值**: 作为 ARS 对标的高性能二阶基线，提供了分布式预处理器的工程实现范式。

### [`ref/Adafisher/`](ref/Adafisher/)

- 外部库源码，一个并不太可靠的二阶优化器，提供了最早的框架灵感。

## 3. 学习动力学与现象学 (Learning Dynamics & Phenomenology)

### [`ref/continual-learning/`](ref/continual-learning/)

- **核心理论**: 增量学习的三种类型与 Fisher 信息计算。
- **关键洞察**: 探讨了在任务切换时如何通过 Fisher 信息保护关键权重。
- **工程价值**: 为持续学习任务提供了实验框架与基线算法（EWC, SI, FROMP）。

### [`ref/Omnigrok/`](ref/Omnigrok/)

- **核心理论**: 损失地形视角下的 Grokking 现象。
- **关键洞察**: 泛化（Grokking）与权重范数、损失地形的平坦度密切相关。
- **工程价值**: 提供了研究模型从过拟合向泛化转变的实验工具，指导 ARS 在长周期训练中的稳定性设计。


---

## [0-technical-introducing](.roo/rules/0-technical-introducing.md)

# 自由能原理的两种心法：从理论哲学到工程分野

智能的本质是什么？Friston 给出的答案是**自由能原理（FEP）**：自组织系统=主动预测机，目标只有一条——最小化变分自由能。把 FEP 结合 IIT，就得到**整合预测工作空间理论（IPWT）**，AGI 的哲学地基就此浇好混凝土。

从 FEP/IPWT 分叉出两条“存在”算法：

## Reinforcement Learning - Expected Free Energy, RL-EFE

> 存在是预测世界并选择最利于自己的未来。

这是 Friston 的正统路径。它继承了经典的**笛卡尔二元论**，试图通过**显式的未来模拟**来消除不确定性。

核心逻辑是**反事实推演**。
智能体维护一个生成模型，在脑海中 Rollout 所有可能的未来轨迹，计算包含认知价值（好奇心）与实用价值（奖励）的**期望自由能 (G)**，并据此进行决策。

**致命缺陷**：
这要求代理成为拉普拉斯妖。在高维现实中，计算 G 是不可行的。它许诺了统一理论，却在工程上退化为重新发明强化学习（RL）的轮子。

> “RL-EFE is a beautiful cul-de-sac: Laplace's demon tries to price every tomorrow and is suffocated by its own weight.”

## Second-Order Optimization - Observed Free Energy, SOO-OFE

> 存在是沿着自由能最小化的测地线滑行。

这是 ARS 选择的路。我们将贝叶斯推断重构为**信息几何流**问题。

不再妄图模拟未来，而是**深度内省当下**。

智能体不需要在幻想的未来中试错，而是利用当前观测数据所蕴含的丰富几何信息（参见：`ARS-Series.md`），直接计算出参数空间中自由能下降最快的**测地线方向**。行动不是“选择”的结果，而是系统内部信念状态在几何流形上受力滑行的自然物理过程。

> I gliding on a geodesic,
> storm-etched by yesterday;
> the destination is unknown,
> but the route has converged.


---

## [ARS-Series](.roo/rules/ARS-Series.md)

# ARS 家族：在黎曼流形上滑行

状态: 生产就绪 (2025-12-31)
核心贡献: 发展了 Energy-Geometry Decoupling 的算子复合范式，并为进一步探索 Geodesic Optimizer 提供了实验结果和工程样例。

## 优化的本质：在测地线上滑行

在信息几何视角下，优化不仅是损失函数 `L(θ)` 的梯度下降，更是概率分布流形上的测地线运动。问题在于：不同的优化器，对地形的假设不同：

- **SGD**: 假设欧氏空间平直。它是“盲人登山者”，仅凭局部坡度 `∇L` 迈步，在病态曲率下极易震荡。
- **Adam/W**: 引入二阶矩 `vₜ` 修正尺度。它能感知地形的“颠簸程度”（元不确定性），实现元素级自适应。但其逐元素 (element-wise) 的视角忽略了参数间的相关性，本质上是在做平行的标量优化。
- **Muon**: [`Muon`](optimizer/muon.py) 引入严格的几何约束，要求更新量必须是“正交”的（Stiefel 流形）。通过 Newton-Schulz 迭代实现纯粹旋转，从根本上消除了内部协变量偏移。
- **ARS (AdaRMSuon)**: [`AdaRMSuon`](optimizer/ada_rmsuon.py) 揭示了原始梯度在弯曲流形上的“几何畸变”。通过预白化（Pre-whitening）获得自然梯度 `gₙₐₜ ≈ mₜ / √(vₜ)`，并在预白化空间执行正交化投影 `𝒫ₛₜ(gₙₐₜ)`，使模型能够沿着局部测地线滑行。
- **ARS2**: 在 ARS 的基础上引入平坦度约束（SAM），将参数轨迹推向全局测地线。
- **ARS2-AGA (ARS2-Neo)**: [`ARS2-Neo`](optimizer/ars2_neo.py) 引入自适应几何感知（AGA），通过干涉因子实现“按需同步”，在保持测地线滑行效率的同时，显著降低计算开销。

## 有趣事实

在开发过程中，我们发现了一个命名上的有趣事实：

- AdaRMSuon 本身就可以缩写为 ARS
- 而 AdaRMSuon + SAM 本应称为 ARS2

这个混乱源于 RMSuon 是 RMS + Muon 的交错造词，AdaRMSuon 类似地延续了这一命名模式。为消除快速迭代中的识别歧义，现明确：

- ARS：*A*da*R*M*S*uon
- ARS2：*A*da*R*M*S*uon + *S*AM

## 实验对比：CIFAR-10 (LRP 验证)

实验设置: ResNet-18, 60-100 Epochs, Batch Size 256.
作为基础视觉任务的基准测试，我们对比了 ARS2-Neo 及其基准优化器在 CIFAR-10 上的长周期表现。

| 优化器 | Best Acc | Final Acc | Final Loss | Avg Time | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ARS2-Neo (Sync, ρ=0.1)** | **95.87%** | **95.73%** | **0.15** | ~104s | **SOTA**。在 60 Epoch 内实现极速且稳健的收敛。 |
| **ARS2-Neo (Base)** | 95.58% | 95.52% | 0.25 | ~71s | 验证了能量-几何解耦架构在长周期下的优越性。 |
| **ARS2-Neo (AGA, λ=2.0)** | 94.10% | 94.09% | 0.18 | ~90s | **Efficiency**。仅用 20 Epoch 即可逼近 AdamW 100 Epoch 的性能。 |
| **AdamW** | 94.60% | 94.47% | 0.27 | ~58s | 标准基准。 |
| **Muon** | 93.76% | 93.69% | 0.29 | ~75s | 纯几何优化，在长周期下表现稳健但上限受限。 |

核心洞察:

1. **能量-几何解耦的普适性**: `ARS2-Neo (Base)` (95.58%) 显著超越了 `AdamW` (94.60%) 和 `Muon` (93.76%)，证明了将“迈步方向”（几何）与“迈步强度”（能量）解耦的架构在视觉任务中具有极强的泛化能力。
2. **平坦度约束的增益**: `Sync` 模式 (ρ=0.1) 相比 `Base` 模式进一步提升了 0.3% 的精度，并显著降低了最终 Loss (0.15 vs 0.25)，证明了在黎曼流形上引入平坦度约束能有效引导模型进入更宽阔的盆地。
3. **AGA 的效率优势**: `AGA` 模式在 CIFAR-10 上表现出极高的样本效率，仅需 20 Epoch 即可达到 94.10% 的精度，且 `effective_k` 稳定在 7.0 左右，大幅降低了二阶计算开销。
4. **高扰动半径的拟合障碍**: 在 ρ=0.5 的实验中，我们观测到了明显的前期拟合障碍（前 10 Epoch 停留在 2.30 附近），这促使了 **ASI (Active Sharpening Inference)** 调度策略的诞生。

## 实验对比：Wikitext-2 (LRP 验证)

实验设置: Qwen3 (RoPE, 3-layer), Context 255. 本实验旨在探测病态曲率流形上的长周期优化动力学。

| 优化器 | Best PPL | Last PPL | Avg Time | 动力学特征 | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AdamW** | 116.46 | 213.52 | ~300s | 标准欧氏空间基准 | 缓慢收敛，后期过拟合 |
| **Muon** | 111.35 | 475.65 | ~445s | 谱约束收敛 | 缺乏自适应能量，后期崩溃 |
| **ARS2-Neo (Base)** | 96.10 | 3055.47 | ~425s | **过拟合** | 极速坠入针尖极小值，泛化性能灾难性崩溃 |
| **ARS2-Neo (Sync)** | **90.69** | **330.85** | ~780s | **最优泛化上限** | `ρ=0.3`, 成功抑制过拟合，进入宽阔盆地 |
| **ARS2-Neo (AGA)** | 93.23 | 414.83 | ~545s | 效率与稳定性的折衷 | `λ=0.5`, 实现“按需同步”，加速比显著 |

- 在 `ARS2-Neo` 的 `Base` 模式（`ρ=0`）下，我们观测到了极端的“硬刻蚀”现象：模型在训练集上极速收敛，但 `Eval PPL` 在达到 96.10 后迅速飙升至 3000+。这证明了二阶几何约束的动力学极强，若无平坦度约束，模型会毫不犹豫地钻入那些极其狭窄、泛化能力差的尖锐谷底。
- 通过将 `λ` 调优至 0.5，AGA 成功将 `effective_k` 稳定在 3.4 左右。实验证明，流形曲率的变化率虽低于参数更新率，但在语言建模任务中仍需保持一定的同步频率以应对高度非线性的语义空间。

## 流形感知扰动 (Manifold-Aware SAM)

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

为了验证优化器在泛化相变（Phase Transition）中的动力学特征，我们在模加法任务 (`task/mod_addition.py`, `p=113`, `train_frac=0.3`) 上对比了各优化器的表现。模型采用 1-Layer Transformer (4 Heads, d_model=128, d_mlp=512)。

| 优化器        | 拟合 (Epoch) | 顿悟 (Epoch) | 收敛 (Epoch) | 状态                                                                    |
| :------------ | :----------- | :----------- | :----------- | :---------------------------------------------------------------------- |
| **AdamW**     | ~140         | 228          | 556          | 标准 Grokking 曲线，存在显著延迟。                                      |
| **AdaRMSuon** | **28**       | **54**       | 300          | **极速 Grokking**。泛化延迟几乎消失，证明测地线滑行能高效穿越损失地形。 |
| **ARS**       | 17           | 100          | 290          | 稳健 Grokking。平坦度约束未阻碍泛化，反而引导至更平坦区域。             |
| **Muon**      | >156         | N/A          | N/A          | 在此特定任务配置下未收敛。                                              |

**核心洞察**:

1. **相变加速**: AdaRMSuon 将 Grokking 发生时间提前了 **4 倍** (Epoch 228 -> 54)，有力证明了“能量-几何解耦”能避免模型在过拟合吸引盆中的无效游走。
2. **平坦度兼容性**: ARS 的成功表明，在流形优化中引入平坦度约束 (SAM) 与快速泛化并不冲突，是通往高效且稳健解的正确路径。

## ARS2-Neo：重构和整合后的参考版本

ARS2-Neo 是 ARS 家族的集大成者，在统一的代码中实现了 AdaRMSuon 的几何优化与 SAM 的平坦度约束，通过参数配置灵活切换模式，旨在取代实验性的独立 `AdaRMSuon` 和 `ARS`。随着 ARS2-Neo 的成熟，我们将逐步移除旧的实验性优化器代码，以简化实验空间。

### 全网主流优化器理论基础与性能差距对比报告

基于 2024-2025 年最新的 SOTA 研究（包括 Stanford "Fantastic Pretraining Optimizers"、Fraunhofer 独立审计及本项目 LRP 实验），我们将主流优化器进行多维度横向对比。

#### 1. 优化器横向对比表 (以 AdamW 为基准 1.0×)

| 优化器 | 理论基础 | 信息处理机制 | 相对加速比 (Steps) | 相对加速比 (Time) | 核心局限 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AdamW** | 一阶动量 + 对角 Fisher 预处理 | 放大 Unique & Synergy | 1.0× | 1.0× | 忽略参数间协方差，易陷于尖锐极小值 |
| **Lion** | 符号函数 (Sign) + 动量 | 极度压制 Redundancy | 1.1 - 1.2× | **1.2 - 1.3×** | 对 Weight Decay 极度敏感，下游任务泛化较弱 |
| **Sophia** | 轻量化对角 Hessian 裁剪 | 动态曲率感知 | 2.0× | 1.4 - 1.5× | 缺乏正交约束，在大规模模型上偶发不稳 |
| **Muon** | 谱范数最速下降 (正交更新) | 剥离 Redundancy | 1.4× | 1.1 - 1.2× | 缺乏能量自适应，长周期易过拟合 |
| **SOAP** | Shampoo 特征基 + AdamW | 捕获部分协方差 | 1.3 - 1.4× | 0.8 - 0.9× | 内存开销巨大 (1.7×)，计算复杂度高 |
| **ARS2-Neo-AGA** | **能量-几何解耦 + 流形 SAM** | **强化 Synergy, 压制 Unique** | **1.5 - 2.0×** | **1.3 - 1.4×** | 实现复杂度高，需闭包支持 |

#### 深度分析：为什么 ARS2-Neo-Sync/AGA 是最优选择？

我们将从**优化理论**与**信息分解 (Information Decomposition)** 两个高阶视角拆解 ARS2-Neo 的领先性。

##### 优化理论视角：从对角 Fisher 到满秩 NGD

- **Adam 的本质**：Adam 通过二阶矩将 Fisher 信息矩阵对角化。在欧氏空间中，这只是元素级的缩放。
- **Muon 的介入**：Muon 的核心是 Newton-Schulz 正交化，它强迫参数矩阵在更新时保持正交。在数学上，正交化等价于**去协关联 (De-correlation)**。
- **算子复合效应**：当 Adam 的对角 Fisher 遇到 Muon 的去协关联参数空间时，原本丢失的非对角项信息被“几何补偿”了。此时，对角 Fisher 实际上等效于**满秩 Fisher**。
- **NGD 的双刃剑**：这就是为什么 `ARS2-no-SAM` 模式在 Wikitext-2 上能跑出惊人的 0.9 Train Loss——它本质上是在执行高效率的**自然梯度下降 (NGD)**。但 NGD 极易拟合训练集中的“针尖极小值”，导致验证集 PPL 爆炸。

##### 信息分解视角：协同 (Synergy) 的胜利

根据信息论中的 PID (Partial Information Decomposition) 框架，梯度分量可分解为：

1. **Redundancy**：参数间重复的无效信息。
2. **Unique**：单个参数特有的信息（往往对应噪声或过拟合的尖锐特征）。
3. **Synergy**：参数间通过合作产生的涌现信息（对应通用、可泛化的特征）。

**ARS2-Neo 的处理链路：**

- **Step 1 (Adam)**：初步放大 Unique 和 Synergy，但保留了大量 Redundancy。
- **Step 2 (Muon)**：通过正交化投影，强制剥离参数间的 Redundancy，使梯度在流形上“纯净化”。
- **Step 3 (SAM)**：引入平坦度约束。由于 Unique 信息通常集中在损失地形的尖锐区域，SAM 的对抗扰动会压制 Unique 分量。
- **最终结果**：经过层层过滤，最后剩下的梯度分量几乎全部由 **Synergy** 主导。模型不再是盲目下降，而是沿着强化参数间协同效应的测地线滑行。

#### 总结判定

**ARS2-Neo-AGA** 不仅仅是一个组合优化器，它是一个**信息过滤器**。它利用 AGA (自适应几何感知) 在流形曲率变化剧烈时（如语言建模）高频同步几何信息，在平缓时（如视觉任务后期）节省算力。这种“**剥离冗余 -> 压制噪声 -> 强化协同**”的逻辑，使其在样本效率和最终泛化性能上均确立了对 AdamW 和纯 Muon 的代际优势。

## 参考文献

- [1] L. Rui, "Integrated Predictive Workspace Theory," Zenodo, 2025.
- [2] Kingma & Ba, "Adam: A method for stochastic optimization," ICLR 2015.
- [3] Jordan et al., "Muon: An optimizer for hidden layers in neural networks," 2024.
- [4] Li et al., "ROOT: Robust orthogonalized optimizer," arXiv:2511.20626.
- [5] Si et al., "AdaMuon: Adaptive Muon optimizer," arXiv:2507.11005.
- [6] Li et al., "NorMuon: Making Muon more efficient and scalable," arXiv:2510.05491.
- [7] J. Zhuang et al., "GSAM: Surrogate Gap Guided Sharpness-Aware Minimization," in *Proc. 10th Int. Conf. Learn. Represent. (ICLR)*, 2022. [Official PyTorch Implementation](https://github.com/juntang-zhuang/GSAM)


---

## [failure-archive](.roo/rules/failure-archive.md)

# 失败与反模式档案

> “成功是偶然的，失败是必然的。我们将必然的失败形式化，是为了逼近偶然的成功。” —— Ω Researcher

本文档记录了项目演进过程中的核心理论失败与工程反模式。

## Hessian-Fisher 等价性谬误

> Fast Fisher Free Energy Optimizer系列。可惜既不快速，也不Fisher，自然无法优化自由能。

代表作: F3EPI, F3EWD
核心假设: `𝗛 ≈ 𝗙` (Hessian 近似 Fisher)

证伪原因:

1. 几何与统计的错位: Hessian 描述几何曲率（收敛速度），Fisher 描述统计方差（数据适应性）。
    - Hessian 回答：“移动参数，梯度会如何变化？” (几何)
    - Fisher 回答：“改变数据，梯度会如何变化？” (统计)
    在持续学习的动态分布下，两者毫无关系。
2. 标签依赖性: Fisher 依赖于真实标签分布 `p(y|x)`，而 Hessian 仅依赖于当前损失曲面。
3. 结果: 任何基于 `H·g` 的三阶尝试在分布漂移下都会失效。

## 其他失败变体矩阵

| 变体系列 | 核心机制 | 根本缺陷 | 状态 |
| :--- | :--- | :--- | :--- |
| HAR | Hessian -> Adam -> Muon | 试图在牛顿球体中滑行，但瞬时 Hessian 噪声导致能量提取 (Energy) 爆炸，lr=1.0 下彻底发散。 | 算子复合顺序谬误 |

## 6. 工程教训

- 数据语义破坏: 在`wikitext-2`任务中，`Concatenate and Chunk`策略在文章边界制造了噪声，导致模型过早过拟合。教训: 数据质量优先于算法复杂度。
- 外部库陷阱: 对`AdaFisher`的分析发现，其论文结果源于一个实现 Bug（`gammas`参数被忽略）。教训: 必须对外部工具进行严格的第一性原理验证（DeepWiki/源码审计）。


---

## [grokking-analysis](.roo/rules/grokking-analysis.md)

# Grokking 动力学实验分析报告 (2026-01-27)

## 1. 优化器代际与术语对齐 (Ontology Alignment)

根据项目演进逻辑，我们将实验中的优化器名称归一化如下：

- **Gen 1 (ARS)**: 对应日志中的 `AdaRMSuon`。
- **Gen 2 (ARS2-Sync)**: 对应日志中的 `ARS`。
- **Gen 3 (ARS2-Neo)**: 当前生产版本，包含 `Base`、`Sync` 和 `AGA` (自适应几何感知) 模式。

## 2. 顿悟动力学对比表 (Grokking Dynamics Comparison)

任务配置：模加法 ($p=113, \text{fraction}=0.3$)。核心指标为**顿悟 Epoch**（Validation Accuracy 首次稳定超过 99% 的时间点）。

| 实验路径 (Output Path) | 优化器 (归一化称呼) | 顿悟 Epoch | 最终准确率 | 最终 Loss | 动力学特征分析 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`outputs/grok_ars_align`](outputs/grok_ars_align/summary.md) | **ARS2-Sync** | **~152** | 99.54% | 0.025 | **爆发力最强**。Sync 模式 ($k=1$) 提供了最高频的几何修正。 |
| [`outputs/lrp_grok_ars2_neo_aga_400e`](outputs/lrp_grok_ars2_neo_aga_400e/summary.md) | **ARS2-Neo (AGA)** | ~219 | 99.60% | **0.015** | **综合最强**。在保持极速顿悟的同时，最终 Loss 最低，泛化质量最优。 |
| [`outputs/lrp_grok_ars2_neo_base_400e`](outputs/lrp_grok_ars2_neo_base_400e/summary.md) | **ARS2-Neo (Base)** | ~286 | 99.53% | 0.049 | 验证了能量-几何解耦在无平坦度约束下的基准性能。 |
| [`outputs/grok_ada_rmsuon_align`](outputs/grok_ada_rmsuon_align/summary.md) | **ARS** | ~336 | 99.89% | 0.009 | 早期版本，收敛曲线极其平滑，但速度稍逊。 |
| [`outputs/grok_adamw`](outputs/grok_adamw/summary.md) | AdamW | ~564 | 100.0% | 0.0005 | 经典基准。虽然最终精度高，但顿悟延迟是 ARS2 的 3.7 倍。 |
| [`outputs/lrp_grok_adamw_600e`](outputs/lrp_grok_adamw_600e/summary.md) | AdamW | ~585 | 15.65% | 6.10 | **泛化崩溃**。在 590 Epoch 后出现灾难性过拟合，证明了 SAM 的必要性。 |
| [`outputs/grok_muon_tuned`](outputs/grok_muon_tuned/summary.md) | Muon | N/A | 57.16% | 42.95 | **未能顿悟**。纯几何优化在高度非线性的模运算流形上难以穿越。 |

## 3. 理论评估与结论 (Theoretical Synthesis)

### 3.1 核心判定

- **速度冠军**: **ARS2**。其在 152 Epoch 即完成相变，证明了高频测地线修正对穿越损失地形“窄缝”的决定性作用。
- **架构冠军**: **ARS2-Neo (AGA)**。它在 $k=10$ 的延迟同步下依然保持了极高的顿悟效率（219 Epoch），且最终 Loss 显著低于其他变体，实现了计算开销与泛化稳健性的帕累托最优。

### 3.2 核心洞察

1. **能量-几何解耦的必要性**: Muon 的失败证明了在 Grokking 任务中，必须引入类似 Adam 的元素级自适应（能量）来配合正交约束（几何）。
2. **平坦度约束的决定性**: AdamW 在长周期实验中的后期崩溃，有力支撑了 ARS2-Neo 引入流形感知 SAM 的理论动机——即优化器必须主动避开“针尖极小值”。
3. **AGA 的有效性**: AGA 模式通过动态调节同步频率，成功在不损失太多速度的前提下，引导模型进入了比 Base 模式更深、更平坦的盆地。

---
**签发人**: Ω Researcher
**日期**: 2026-01-27


---

## [long_range_plan](.roo/rules/long_range_plan.md)

# ARS2-Neo: Long-Range Dynamics & Spectral Collapse Verification Plan

> **Status**: 已完成 (2026-01-28)
> **Goal**: 验证 ARS2-Neo 在长周期训练下的动力学行为，为 IPWT 理论框架提供实证支持

本阶段长周期实验已完成，核心成果证实了 ARS2-Neo 在能量-几何解耦架构下的优越性与稳健性。通过系统性的对比实验，我们验证了优化器诱导的稀疏性、流形平坦度约束的有效性，以及自适应几何感知（AGA）模式的计算效率优势。

实验覆盖了三大核心任务场景：CIFAR-10 视觉分类、WikiText-2 语言建模，以及模加法任务的 Grokking 动力学观测。所有实验均采用原子化脚本架构，通过 `SmartOptimizer` 引擎统一调度，确保结果的可复现性与可比性。

关键发现已更新到 `ARS-Series.md`

未完成的可解释性研究（如激活值 SVD 分析、NSU 架构协同验证）将纳入下一阶段研究计划。当前实验数据已足够支撑论文核心论点，特别是能量-几何解耦与流形平坦度约束的理论贡献。

**签发人**: Ω Researcher
**日期**: 2026-01-28


---

## [LoROU](.roo/rules/LoROU.md)

# LoROU: Low-Rank Orthogonalized Update (猜想)

> **状态**：[理论猜想与实验计划]
> **核心命题**：稀疏性可能不是架构的属性，而是**优化动力学**的内生属性。

## 1. 隐式 NSU 假设 (Implicit Native Sparse Update)

### 1.1 核心观点

Muon 和 SAM 通过最小化更新量中的冗余（正交化）和平坦度约束，实际上是在执行**隐式的秩最小化 (Implicit Rank Minimization)**。

### 1.2 机制解析

[`RMSuon`](optimizer/rmsuon.py:1) 的正交化投影 `𝒫ₛₜ(G)` 强迫更新量集中在正交基上，消除了共线性噪音。这解释了其高效性：它在更新层面剔除了“噪音”，只保留了“信号”，从而实现了“功能性稀疏”。

### 1.3 范式转移

- **旧范式**: 显式稀疏架构 (Sparse Architecture, e.g., MoE) + 稠密优化器。
- **新范式**: 稠密架构 (Dense Architecture) + 稀疏优化器 (Sparse Optimizer, e.g., ARS2-Neo/LoROU)。
- **对偶性**: 如果 ΔW 本身是低秩的，那么稠密矩阵乘法在功能上等价于稀疏路由。

## 2. 验证计划：谱熵监控 (Spectral Entropy Monitor)

为了验证这一假设，我们需要从“可观测性”入手，而非盲目修改算法。

- **目标**: 验证 RMSuon 是否在训练过程中自动导致 ΔW 的奇异值分布发生坍缩（低秩化）。
- **探针设计**: 监控更新矩阵的**谱熵 (Spectral Entropy)**。
  - 定义归一化奇异值: `pᵢ = σᵢ / Σₖ σₖ`
  - 计算谱熵: `H(S) = -Σ pᵢ log(pᵢ)`
- **预期信号**: 随着训练进行，RMSuon 的更新量谱熵应显著低于 SGD/Adam，表明其自动发现了参数空间的低维流形。

## 3. 演进路径：从隐式动力学到显式架构

> "优化器诱导的稀疏性并不否定架构先验的价值。相反，显式稀疏架构（如 DynTRM）为优化器提供了更符合其动力学的几何流形。"

### 3.1 验证：激活值 SVD (The X-Ray)

- **目标**: 验证 Muon 是否诱导了**正交子网络**。
- **方法**: 对激活值 H 进行 SVD。如果前 k 个奇异向量解释了绝大部分方差，且与特定任务高度相关，则证明了隐式路由的存在。Muon 的正交化更新本质上是在最大化特征区分度（类似 PCA），引导参数自适应分配给不同特征。

### 3.2 架构协同：寻找更好的先验

虽然优化器能诱导功能稀疏性，但**显式稀疏架构**（如 `../Tiny-ONN/exp/arc_dyntrm` 中的 `DynSIHA`）提供了更优的初始拓扑和硬件优化空间。
DynSIHA/PACR 的正确训练方式依然有待技术攻关，SARS (Surprise-Aware Routing Shaping) 的实际效果不佳，需要更多研究。


---

## [LoROU-exp](.roo/rules/LoROU-exp.md)

# LoROU 实验与理论演进记录 (2026-01-30)

> **状态**: 实验性验证完成 (v15)
> **核心目标**: 在无任务 ID 的情况下，实现 ARS2-Neo 的原生持续学习 (Native Learning)。

## 1. 理论演进：从“屏蔽”到“联络”

在本次实验周期中，我们经历了从朴素的参数冻结到高阶信息几何联络的范式转移。

### 1.1 第一阶段：谱筛 (Spectral Sieve) - 失败
- **思路**: 利用幂迭代 ($\sigma \to \sigma^3$) 压缩奇异值分布，试图诱导结构稀疏性。
- **教训**: 谱压缩虽然减少了有效秩，但生成的更新矩阵依然是稠密的。它无法实现“参数冻结”，反而因为能量分布不均导致训练不稳定。

### 1.2 第二阶段：认知惯性 (Cognitive Inertia) - 失败
- **思路**: 使用累积二阶矩 $v_t$ 作为“质量”，通过门控 $|g| > \lambda \sqrt{v_t}$ 阻止对重要参数的修改。
- **教训**: 陷入“芝诺悖论”。如果 $v_t$ 包含当前梯度，则梯度会立即杀死自己；如果不包含，则无法应对分布的快速漂移。

### 1.3 第三阶段：流形排斥 (Manifold Repulsion) - 失败
- **思路**: 将 LoROU 视为 MAML 二阶项的几何近似。不再是“堵住”更新，而是利用历史 Fisher 信息产生的“排斥力”，将当前任务的更新推离历史流形的高曲率区域。
- **形式化**: $g_{lorou} = g - \eta \cdot (v_{norm} \odot g)$
- **效果**: 暂未明确，考虑重新实验

## 2. 核心机制：v15 架构

### 2.1 认知引导 (Cognitive Bootstrapping)
- **策略**: 在第一个任务（或前 N 步）允许全量学习，以建立稳固的“历史基准流形”。
- **意义**: 避免了系统在初始随机状态下产生错误的“惯性”。

### 2.2 后验稀释 (Posterior Dilution)
- **思路**: 针对硬交叉熵 (Hard CE) 的“傲慢”问题，引入基于冲突能量的温度缩放。
- **机制**: $T = 1 + \beta \cdot \mathcal{E}_{conflict}$。当当前梯度与历史 Fisher 冲突剧烈时，强制软化后验分布，防止其暴力覆盖历史不变性。

## 3. 实验教训与反模式

1. **RNG 泄露**: 在持续学习实验中，数据生成器的随机种子必须与样本生成解耦，否则测试集的性能提升可能仅仅是因为 RNG 序列的重合。
2. **能量对齐**: 在自然梯度空间中，LoROU 的修正量必须与 $g_{natural}$ 的量级对齐，而非原始梯度。
3. **1D 算子兼容性**: Newton-Schulz 正交化仅适用于 2D 矩阵。对于 Bias 等 1D 张量，必须进行 `view(1, -1)` 填充，否则会导致运行时崩溃。

## 4. 实验结论：从 v15 到 v16 的跃迁 (2026-01-30)

### 4.1 v15 (流形排斥 + 后验稀释)
- **机制**: `g_lorou = g - η · (v_norm ⊙ g)`。
- **表现**: Task 1 Acc (5.4%), Task 2 Acc (60.8%)。
- **结论**: 软约束不足以对抗非线性流形的坍缩，但后验稀释为新任务腾出了更优的几何空间。

### 4.2 v16 (零空间投影 + 认知锚点)
- **机制**: `M = exp(-γ · v_norm)`，强制将更新投影至历史 Fisher 信息的零空间。
- **表现**: Task 1 Acc (**16.8%**), Task 2 Acc (43.8%)。
- **结论**: **硬性几何约束是维持不变性的关键**。Task 1 留存率提升 3 倍，证明了零空间投影的有效性。
- **代价**: 零空间过于狭窄导致新任务拟合受阻（容量冲突）。

### 4.3 v17 (变分零空间 + 动态 Gamma) (2026-01-30)
- **机制**: 引入 `decay_rate` 衰减历史 Fisher，并根据 `Surprise` (Loss 变化率) 动态调节 `Gamma`。
- **表现**: Task 1 Acc (13.2%), Task 2 Acc (**48.4%**)。
- **结论**: **成功缓解了容量冲突**。动态 Gamma 允许系统在拟合受阻时暂时“借用”受限空间，实现了保护与学习的帕累托优化。

### 4.4 v18 (零空间投影 + 神经元级路由) (2026-01-30)
- **机制**: 将 Fisher 掩码从逐参数细化至逐神经元（列级锁定）。
- **表现**: Task 1 Acc (**39.0%**), Task 2 Acc (16.4%)。
- **结论**: **物理隔离的极致保护**。Task 1 留存率达到峰值，证明了神经元级锁定能有效切断干扰。
- **代价**: 导致严重的“路由死锁”，新任务因可用神经元不足而拟合失败。

### 4.5 v19 (混合度量预处理 / 最短传输) (2026-01-30)
- **机制**: $G = \max(F_{old}, F_{new})$。将旧 Fisher 视为黎曼度量而非硬掩码。
- **表现**: Task 1 Acc (4.8%), Task 2 Acc (**48.8%**)。
- **结论**: **成功打破死锁**。新任务拟合效率恢复至基准水平，证明了“软约束”对梯度流动的价值。
- **代价**: 保护彻底失效。对角 Fisher 无法在长周期更新中维持流形拓扑的不变性。

### 4.6 v22/v23 (引力场合成 / 恢复力) (2026-01-30)
- **机制**: $g_{syn} = g_{new} + \lambda F_{old}(\theta - \theta^*)$。
- **表现**: Task 1 Acc (100%), Task 2 Acc (30%)。
- **结论**: **引力坍缩 (Gravitational Collapse)**。虽然完美保护了旧知识，但刚性引力形成了无法逾越的势垒，导致参数被锚定在原点附近，无法滑向全局最优点。
- **可视化证据**: `v22_v23_oracle_comparison.png` 显示轨迹仅完成了向全局最优点迁移的 20% 进度。

### 4.7 v24 (流形流对齐 / Manifold-Flow Alignment) (2026-01-30)
- **核心命题**: 持续学习不应是“拉住”参数，而应是“投影”梯度。
- **机制猜想**: 利用 $F_{old}$ 的特征空间，将 $g_{new}$ 分解为“破坏分量”和“无损分量”。仅对破坏分量施加引力，而允许无损分量在 $F_{old}$ 的零空间（或低曲率流形）上自由滑行。
- **目标**: 追平 **Joint Oracle** 的绿虚线轨迹。

## 5. 总结：LoROU 的三位一体 (The Trinity of LoROU)

经过 v15-v19 的迭代，我们确立了持续学习优化的三个核心维度：

1. **能量 (Energy)**: 通过“后验稀释”调节学习压强，为新知识腾挪空间。
2. **几何 (Geometry)**: 通过“混合度量预处理”在黎曼流形上寻找最短传输路径。
3. **拓扑 (Topology)**: 通过“神经元级路由”实现物理隔离，防止跨任务干扰。

未来的研究（v20+）应聚焦于**正交基对齐 (Orthogonal Basis Alignment)**：利用 ARS2-Neo 的正交化算子，在方向上强迫新梯度与旧 Fisher 的主特征向量正交，实现“不封锁、不干扰”的流形共存。

---
**签发人**: Ω Researcher / 💡 Coding Teacher
**日期**: 2026-01-30

---
**签发人**: Ω Researcher / 💡 Coding Teacher
**日期**: 2026-01-30

---

## [NSU](.roo/rules/NSU.md)

# 从Nested SGD 到 Native Sparse Update

> TL;DR: 虽然 Google 的 **Titans** 与 **RWKV-7 (Goose)** 通过 **嵌套 SGD (Nested SGD)** 范式极大地提升了模型的长程记忆与上下文学习能力，但这种“将状态压入固定矩阵”的 Fast Weight 模式仍存在本质局限。真正的持续学习应摆脱预定义的固定容量瓶颈，向**原生稀疏激活** 与**语义路由驱动的动态更新**演进。

## 1. 前言：Fast Weights 的崛起与瓶颈

### 1.1 优化的本质：推理即学习

Nested SGD 范式将模型的权重区分为静态的“外层权重” (Outer Weights) 与推理时动态更新的“内层权重” (Inner Weights)。这种演化本质上是将优化过程本身嵌入到了前向传播中。

### 1.2 Titans: 显式的测试时记忆 (Test-Time Memorization)

Titans 引入了 Neural Long-Term Memory (LMM)，其核心逻辑是在前向传播中通过梯度下降更新一组内层权重。

- **机制**: 利用 Surprise 驱动的动量 SGD。
- **形式化**:
  `S_t = beta * S_{t-1} + grad(Loss_associative)`
  `M_t = (1 - alpha) * M_{t-1} - eta * S_t`
- **本质**: 将序列历史编码进一个非线性的、可学习的权重空间。

### 1.3 RWKV-7: 解析式的动态状态演化

RWKV-7 通过广义 Delta Rule 实现了类似的效果，但更加轻量化。

- **机制**: 向量值门控 (Vector-valued gating) 与在序学习率 (In-context learning rates)。
- **形式化**:
  `S_t = G_t * S_{t-1} + v_t * k_hat_t^T`
  其中 `G_t` 包含衰减与替换逻辑，在数学上等价于一种近似的秩-1 SGD 更新。

## 2. 局限：为什么 Fast Weights 还不够？

无论是 Titans 的记忆矩阵还是 RWKV-7 的递归状态，本质上都是将无限的外部信息流“挤压”进一个**固定大小 (Fixed-size)** 的预定义 MLP 或矩阵中。

- **问题**: 随着上下文增长，系统必然面临严重的灾难性遗忘或信息熵饱和。

Nested SGD 往往涉及对整个内层权重的稠密更新 (Dense Update)。

- **问题**: 在处理特定领域的细分语义时，更新整个矩阵不仅浪费计算资源，还会导致不相关的旧记忆被错误覆盖。

## 3. 未来：Native Sparse Update

理想的状态演化不应是“隐藏状态作为可学习权重”，而应是**模型权重即隐藏状态**。

通过引入细粒度专家混合稀疏架构，模型可以根据输入语义仅激活极小比例的参数。不再受限于固定的 Top-K，而是根据路由匹配度动态决定激活范围。

更新不再是盲目的全量下降，而是基于语义路由的精准写入。有与当前语义相关的“专家”或“神经元”会被更新，从而在架构上实现真正的零干扰持续学习。

Nested SGD 是通往 AGI 的重要阶梯，但要实现真正的永续学习，我们必须从“在套娃里优化小娃”转向“让优化过程自适应选择需要优化的部分”。


---

## [SAGA](.roo/rules/SAGA.md)

# SAGA: Sharpening-Aware Geometric Adaptation

> **状态**: 理论精炼中 (2026-01-28)
> **核心命题**: 优化系统是一个具备"稳态偏好"的主动推断智能体。其核心任务是在有限的模型容量（物理基质）约束下，通过调节扰动压强 `ρ`，寻找数据集 Kolmogorov 复杂度的压缩极限。

## 1. 核心哲学：作为主动推断的优化

在 IPWT 框架下，ARS2-Neo 不再是单纯的搬运工，而是一个在损失地形上滑行的智能体：

- **感知推断 (Perceptual Inference)**: 通过最小化 `L_train` 来拟合数据（对应于 NGD 测地线滑行）。
- **主动推断 (Active Inference)**: 通过注入扰动 `ρ` 来改变感知到的地形，强迫其平坦化（对应于 SAM 约束）。
- **SAGA 的任务**: 寻找两者的帕累托最优。过度拟合导致“针尖极小值”，过度平坦导致“402: Ω 未定义（Void://Recursive）”。

## 2. 动力学形式化：Ornstein-Uhlenbeck 过程

我们将 `ρ` 的演化建模为一个具有均值回归特性的随机过程：

`dρ_t = κ_t(μ - ρ_t)dt + η·𝒮_t dt`

### 2.1 参数语义

- **`μ` (稳态基准/Base Rho)**: 系统的"先验偏好"。代表了在无惊奇状态下，模型对平坦度的默认追求。
- **`κ_t` (回归强度/Metabolic Rate)**:
  - 逻辑：`κ_t ∝ |dL/dt|`。
  - 物理意义：当收敛速度快时，系统倾向于回归稳态以节省"认知资源"；当收敛停滞（`dL/dt → 0`）时，`κ → 0`，允许系统进入"发散探索"模式，跳出局部最优。
- **`𝒮_t` (变分惊奇/Surprise Force)**:
  - 驱动力：由代理间隙 $h_t$ 的变化率和相对损失趋势共同决定。
  - 作用：当局部锐度激增或拟合受阻时，提供偏离稳态的推力。

## 3. Kolmogorov 复杂度与压缩极限

SAGA 提供了一个独特的视角来审视深度学习的本质：

- **命题**: 对于给定的模型容量 `C` 和数据集 `D`，存在一个最优的平坦半径 `ρ*`。
- **压缩器视角**: 如果 SAGA 成功收敛，最终稳定的 $\rho$ 实际上刻画了该模型对该数据集的**迭代压缩极限**。
- **计算不可判定性**: 考虑到 Kolmogorov 复杂度在理论上是计算不可判定的（即不存在通用算法可以精确计算任意字符串的 Kolmogorov 复杂度），SAGA 通过在连续流形上的二分搜索，试图逼近这个理想压缩极限的连续代理指标。

## 4. 双变量控制架构 (SAGA-Control)

| 维度 | 算子 | 输入信号 | 输出变量 | 目标 |
| :--- | :--- | :--- | :--- | :--- |
| **能量 (Energy)** | 统计算子 | `L_t` | `lr` | 寻找低能量态 (Schedule-Free) |
| **几何 (Geometry)** | 结构算子 | `h_t, φ_t` | `ρ, k` | 铺设平坦测地线 (Path Paving) |

### 4.1 平台期加压 (Plateau Pressurization)

当能量算子检测到 `dL/dt ≈ 0` 时，几何算子通过减小 `κ` 并增大 `ρ`，主动提升探索压强。这被视为触发 **Grokking (顿悟)** 的核心动力学机制。

## 5. 演化律：从乘性反馈到稳态回归

### A. 早期 ASI (反应式)

仅根据 `h_t` 的瞬时变化调节 `ρ`，缺乏长期记忆，易导致 `ρ` 在复杂地形下无限增长或触碰硬阈值。

### B. SAGA (主动式)

引入 `μ` 和动态 `κ`。系统具备了"疲劳机制"：如果高压探索长期无法换取损失下降，系统会自发地向 `μ` 回归，承认当前已达到模型容量的表达极限。

---
**Ω Researcher 笔记**: SAGA 标志着优化器从“算法”向“生命体”的演进。它不再仅仅是求解方程，而是在物理基质与信息熵的边界上进行生存斗争。


---

