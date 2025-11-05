# F3EL: Fast Fisher Free-Energy Optimizer with Loss-scaling

## 目的

解决 F3EO 训练后期因三阶协同梯度 `δ_meta` 持续过强导致的性能停滞。

## 核心思想

用当前批次主损失 `L` 对 `δ_meta` 做内生缩放：  
`g_eff = g − L · δ_meta`  

- 误差大 → 强化协同度更新，快速重塑结构  
- 误差小 → 自动削弱 `δ_meta`，平滑过渡到纯梯度微调  

## 实现要点

- `step()` 新增可选 `loss` 参数  
- 若 `loss_scaling=True` 且 `loss` 有效，则用 `loss.detach()` 缩放 `meta_grad`  
- 开关 `loss_scaling` 便于消融  

## 实验

CIFAR-10，配置与 F3EO 完全一致，仅开启 `loss_scaling`。  
监控：损失/准确率曲线、最终验证准确率、收敛步数。

## 预期

早期速度与 F3EO 持平；后期更平滑、不停滞；最终验证准确率更高或更稳定。

## 提要

**状态**: 实验已暂停。
**原因**: F3EO 主实验表现远超预期，在 CIFAR-10 上仅用 28 epoch 即达到 90.6% 验证准确率，按此趋势 50 epoch 即可逼近 95%。因此，当前优先级为全力推进 F3EO，F3EL 作为技术储备，暂时不急。
