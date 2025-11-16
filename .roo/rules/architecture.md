# F3EO-Bench: 轻量级三阶优化器评测框架

```ascii
f3eo-bench/
├── README.md      # 30 秒上手命令
├── pyproject.toml # 只留 torch rich tqdm
├── optimizer/
│   ├── __init__.py
│   ├── f3eo.py    # 三阶核心，<150 行
│   └── adahessian.py  # 二阶对照，直接抄官方
├── model/
│   ├── __init__.py
│   ├── resnet.py # ResNet-18
│   ├── vit.py   # Transformer
│   └── nano-gpt.py   # MNIST/Fashion 头
├── data/ # 自动创建的数据集缓存
├── config/
│   ├── cifar10.toml
│   ├── wikitext2.toml
│   └── cl_stream.toml
├── task/ # 不同任务具体的训练调度器
│   ├── cifar10.py
│   ├── wikitext2.py
│   └── cl_stream.py   # 持续学习 MNIST→Fashion
├── outputs/# gitignored，自动生成
│   ├── report/      # markdown report
│   └── checkpoints/   # 只存 best.pt
└──  scripts/
   ├── train.py   # 统一入口、rich log print
   └── notebook/
      └── loss_landscape.ipynb  # 损失地形可视化（参考 adafisher）
```

## 实验流水线（多种配置，一条命令）

```bash
python -m scripts/train.py --config config/cifar10.toml
```

## 挂载新优化器流程

为保证框架的可扩展性，添加一个新的优化器需要遵循以下三个步骤。这个流程确保了优化器能够被正确地实例化、配置，并与需要模型实例的二阶方法（如 KFAC）兼容。

1. **创建优化器实现**: 在 `optimizer/` 目录下创建一个新的 Python 文件（例如 `my_optimizer.py`），并在其中实现你的优化器类。

2. **在工厂函数中注册**: 打开 [`optimizer/__init__.py`](optimizer/__init__.py)，在 `get_optimizer` 工厂函数中，为你的新优化器添加一个 `elif` 分支，用于导入和实例化它。

---

[以下是重构计划书]

---

## 持续学习架构演进状态

**版本**: 2.1
**状态**: **部分实现完成**
**日期**: 2025-11-16

### 已完成的重构

#### 1. 多任务训练循环

- ✅ [`scripts/train.py`](scripts/train.py:53-57) 已支持多任务配置，通过 `config["experiment"]["tasks"]` 加载多个任务
- ✅ [`utils/trainer.py`](utils/trainer.py:54-57) 已实现多任务循环，遍历 `task_names` 依次执行每个任务
- ✅ 任务间共享单一模型和优化器实例，符合持续学习场景需求

#### 2. 持续学习评估任务

- ✅ [`task/mnist_cl.py`](task/mnist_cl.py:1-126) 已实现 MNIST→Fashion 持续学习评估
- ✅ [`task/fashion_cl.py`](task/fashion_cl.py:1-139) 已实现 Fashion→MNIST 持续学习评估
- ✅ 两个任务均在 `validate_epoch` 中评估当前任务性能和对先前任务的遗忘程度

#### 3. 参数分组与优化器兼容性

- ✅ 所有任务已实现 `get_param_groups()` 方法，支持 FOG/DiagFOG 的参数分组策略
- ✅ [`optimizer/__init__.py`](optimizer/__init__.py:56-67) 已注册 `Hadron` 和 `DiagHadron` 优化器

### 待实现的核心特性

#### 1. 课程表 (Curriculum) 配置系统

**状态**: 未实现

当前配置使用扁平化任务列表：

```toml
[experiment]
tasks = ["mnist_cl", "fashion_cl"]
```

目标架构需要课程表配置：

```toml
[[curriculum]]
task = "mnist_cl"
mode = "train"
epochs = 5

[[curriculum]]
task = "fashion_cl"
mode = "train"
epochs = 5
```

**实现路径**: 需要修改 [`scripts/train.py`](scripts/train.py:53-57) 和 [`utils/trainer.py`](utils/trainer.py:54-57) 以支持课程表解析和 `mode` 属性处理。

#### 2. 任务切换回调事件

**状态**: 未实现

需要在 [`utils/callbacks/base.py`](utils/callbacks/base.py:1-56) 中添加：

```python
@abstractmethod
def on_task_begin(self, task_name: str, task_mode: str, **kwargs):
    pass

@abstractmethod
def on_task_end(self, task_name: str, **kwargs):
    pass
```

**实现路径**: 修改基类并更新所有回调实现，在 [`utils/trainer.py`](utils/trainer.py:54-57) 的任务切换边界插入事件广播。

#### 3. CLMetricsLogger 回调

**状态**: 未实现

需要创建新的回调类，监听 `on_task_begin` 事件执行零样本评估。

**实现路径**: 在 [`utils/callbacks/`](utils/callbacks/) 目录下新建 `cl_metrics.py`，实现零样本遗忘测量和学习冲击记录。

### 架构设计总结

当前架构已通过**多任务循环**和**评估任务**实现了持续学习的核心能力，但缺乏**课程表配置**的灵活性和**任务切换观测**的精细度。建议根据研究需求优先级，逐步实施待实现特性。
