# 训练框架解耦计划 (引擎-任务-回调)

> **Status**: 提案
> **Goal**: 将训练逻辑与优化器特性和任务特定指标解耦，以提高研究迭代速度。

## 1. 核心架构：“SmartOptimizer”模式

主要瓶颈在于 `Trainer` 过多地了解每个优化器如何“执行步进”（closures, second-order, BN protection）。我们将把这些知识转移到 `optimizer` 模块中。

### 1.1 `optimizer/__init__.py`: `SmartOptimizer` 包装器

引入一个提供统一 `.step()` 接口的包装器。

- **职责**:
  - 处理 `closure` 定义。
  - 管理 `BN` 状态保护（使用 `utils.nn`）。
  - 根据优化器标签处理 `backward(create_graph=...)`。
- **接口**: `logits, loss = smart_opt.step(task, batch, device)`

## 2. 引擎整合: `scripts/train.py`

将 `utils/trainer.py` 合并到 `scripts/train.py` 中，以消除冗余的 `Trainer` 类抽象。

- **`train_one_epoch(ctx)`**: 一个清晰的程序化循环，调用 `smart_opt.step()`。
- **`validate(ctx)`**: 直接调用 `task.validate_epoch()`。
- **回调系统**: 维护现有的用于日志记录和检查点的回调钩子。

## 3. 模块化工具提取

将当前 `Trainer` 中的硬编码逻辑剥离到独立的处理器中。

### 3.1 `utils/adaptive_wd.py` (新增)

- **类**: `AdaptiveWDHandler`
- **逻辑**: 封装 `IPCWD` 和 `PCWD` 的计算。
- **用法**: 在训练循环中调用 `wd_handler.update(loss_val)`。

### 3.2 `utils/nn.py` (新增)

- **函数**: `disable_running_stats(model)`, `enable_running_stats(model)`。
- **目的**: 为需要多次前向传播的优化器（例如 SAM/ARS）提供模型手术工具。

## 4. 实现路线图

1. **第一阶段：优化器加固**
    - 在 `optimizer/__init__.py` 中实现 `SmartOptimizer`。
    - 确保它尊重所有现有的 `OptimizerMetadata` 标签。

2. **第二阶段：工具迁移**
    - 创建 `utils/adaptive_wd.py` 和 `utils/nn.py`。
    - 将 `utils/trainer.py` 中的逻辑移至这些新模块。

3. **第三阶段：引擎重构**
    - 重写 `scripts/train.py` 以使用 `SmartOptimizer` 和新工具。
    - 验证性能与当前 `Trainer` 持平。

4. **第四阶段：清理**
    - 删除 `utils/trainer.py`。
    - 如果需要，更新 `pyproject.toml` 或其他配置文件。

## 5. 优势

- **任务独立性**: 任务只关心数据和前向传播。
- **优化器无关性**: 训练循环不关心优化器是一阶、二阶还是需要闭包。
- **可维护性**: 新功能（如新指标或 WD 策略）将作为独立的模块/回调添加。
