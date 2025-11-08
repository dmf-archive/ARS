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

## 2. 技术契约

1. **零手工调参**  
   所有实验使用同级配置文件，学习率、batch、epoch 固定；只换优化器类名。
2. **单卡 8 GB 上限**  
   默认 bf16 native，batch 自动回退到不 OOM。
3. **可复现**  
   全局 `seed=42`，PyTorch 确定性卷积；日志输出完整命令与环境哈希。
4. **终端可视化**  
   每 10 步刷新 Rich 表格：loss、acc、lr、GPU-Mem、F3EO-grad-norm。
5. **一键报告**  
   跑完自动生成 `outputs/summary.md`：
   - step级别print log markdownlized转录
   - 最终验证指标
   - 遗忘率（CL 任务）

## 4. 实验流水线（3 个脚本，一条命令）

```bash
python -m scripts/train.py --config config/cifar10.toml
```
