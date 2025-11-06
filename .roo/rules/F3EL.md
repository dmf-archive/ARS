# F3EL: Fast Fisher Free-Energy Optimizer with Loss-scaling

F3EL 用主损失 `L` 缩放三阶复杂度梯度 `δ_meta`（`g_eff = g − L·δ_meta`），试图在训练后期抑制过拟合。该思路已被 **F3EPI** 取代，后者以预测完整性 `PI` 为反馈信号，实现更精细的协同平衡。F3EL 代码留档，仅作技术备份。
