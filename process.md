# ARS2C Implementation Log

## 2026-05-16: ARS2C 初始实现

### 变更摘要

基于 [`.roo/rules/ARS2C.md`](.roo/rules/ARS2C.md) 和 [`ref/ARS2C-Research-Report.md`](ref/ARS2C-Research-Report.md) 的设计规范，实现 ARS2C (Christoffel-Aware Dynamic Beta Optimization) 优化器。

### 新增文件

| 文件 | 说明 |
|:---|:---|
| [`optimizer/ars2c.py`](optimizer/ars2c.py) | ARS2C 优化器核心实现，继承 ARS2Neo，添加 Christoffel 动态 β |
| [`config/lrp_wikitext2_ars2c_sync_10e.toml`](config/lrp_wikitext2_ars2c_sync_10e.toml) | WikiText-2 Sync 模式控制变量实验 |
| [`config/lrp_wikitext2_ars2c_aga_10e.toml`](config/lrp_wikitext2_ars2c_aga_10e.toml) | WikiText-2 AGA 模式控制变量实验 |
| [`config/lrp_cifar10_ars2c_sync_60e_rho01.toml`](config/lrp_cifar10_ars2c_sync_60e_rho01.toml) | CIFAR-10 Sync 模式控制变量实验 |
| [`config/lrp_cifar10_ars2c_aga_20e.toml`](config/lrp_cifar10_ars2c_aga_20e.toml) | CIFAR-10 AGA 模式控制变量实验 |

### 修改文件

| 文件 | 变更 |
|:---|:---|
| [`optimizer/__init__.py`](optimizer/__init__.py) | 注册 ARS2C 到 OPTIMIZER_REGISTRY |

### 核心设计

- **动态 β 机制**: 在 SAM sync step 中复用 HVP 采样 (g_base, g_adv)，计算结构化 Christoffel 矩阵 `c_ortho`
- **几何对齐**: `alignment = |⟨c_ortho, s_ortho⟩| / (‖c_ortho‖ · ‖s_ortho‖)` 驱动 β 线性插值
- **β 范围**: β₁ ∈ [0.5, 0.95], β₂ ∈ [0.9, 0.9995]
- **1D 参数**: 保持固定 β（走 AdamW track）
- **诊断输出**: 新增 `alignment`, `beta1_dynamic`, `beta2_dynamic` 指标

### 静态检查

- `ruff check . --fix`: ✅ 通过
- `ty check optimizer/ars2c.py`: ✅ 通过
- 全项目 ty check 报错均为预存问题（tokenizers 类型标注、optimizer/__init__.py 泛型签名），与本次变更无关

### 实验对照矩阵

| ARS2-Neo 基线 | ARS2C 对照 | 任务 | 模式 |
|:---|:---|:---|:---|
| `lrp_wikitext2_ars2_neo_sync_10e` | `lrp_wikitext2_ars2c_sync_10e` | WikiText-2 | Sync (k=1) |
| `lrp_wikitext2_ars2_neo_aga_10e` | `lrp_wikitext2_ars2c_aga_10e` | WikiText-2 | AGA |
| `lrp_cifar10_ars2_neo_sync_60e_rho01` | `lrp_cifar10_ars2c_sync_60e_rho01` | CIFAR-10 | Sync (k=1) |
| `lrp_cifar10_ars2_neo_aga_20e` | `lrp_cifar10_ars2c_aga_20e` | CIFAR-10 | AGA |

### 启动命令

```powershell
python -m exp.wikitext-2.train --config config/lrp_wikitext2_ars2c_sync_10e.toml
python -m exp.wikitext-2.train --config config/lrp_wikitext2_ars2c_aga_10e.toml
python -m exp.cifar.train --config config/lrp_cifar10_ars2c_sync_60e_rho01.toml
python -m exp.cifar.train --config config/lrp_cifar10_ars2c_aga_20e.toml
```
