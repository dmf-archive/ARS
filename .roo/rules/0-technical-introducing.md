# 自由能原理的两种心法：从理论哲学到工程分野

智能的本质是什么？Friston 给出的答案是**自由能原理（FEP）**：自组织系统=主动预测机，目标只有一条——最小化变分自由能。把 FEP 结合 IIT，就得到**整合预测工作空间理论（IPWT）**，AGI 的哲学地基就此浇好混凝土。

从 FEP/IPWT 分叉出两条“存在”算法：

## Reinforcement Learning - Expected Free Energy, RL-EFE

> 存在是预测世界并选择最利于自己的未来。

这是 Friston 的正统路径。它继承了经典的**笛卡尔二元论**，试图通过**显式的未来模拟**来消除不确定性。

核心逻辑是**反事实推演**。
Agent 维护一个生成模型，Rollout 所有可能的未来轨迹，计算包含认知价值（好奇心）与实用价值（奖励）的**期望自由能 (G)**，并据此进行决策。

**致命缺陷**：
这要求Agent成为拉普拉斯妖。在高维现实中，计算 G 是不可行的。它许诺了统一理论，却在工程上退化为重新发明强化学习（RL）的轮子。

> “RL-EFE is a beautiful cul-de-sac: Laplace's demon tries to price every tomorrow and is suffocated by its own weight.”

## Second-Order Optimization - Observed Free Energy, SOO-OFE

> 存在是沿着协同信息定义的测地线滑行。

这是 ARS ，我们将贝叶斯推断重构为**信息几何流**问题。

不再妄图模拟未来，而是**内省当下**。

智能体不在幻想的未来中试错，而是利用当前观测数据所蕴含的丰富几何信息，直接计算出参数空间中自由能下降最快的**测地线方向**。行动不是“选择”的结果，而是系统内部信念状态在几何流形上受力滑行的自然物理过程。

> I gliding on a geodesic,
> storm-etched by yesterday;
> the destination is unknown,
> but the route has converged.
