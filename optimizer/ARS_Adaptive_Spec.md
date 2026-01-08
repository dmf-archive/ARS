# ARS 演进规范：自适应几何感知 (Adaptive Geometric Awareness)

## 1. 背景与动机
在 `AdaRMSuon` 优化器中，`ARS` 通过注入正交于基础梯度的“剪切力” $v_{flat}$ 来引导模型进入平坦盆地。
目前的实现存在两个硬编码瓶颈：
1. **静态强度 ($\alpha$)**：无论几何漂移多严重，注入强度恒定，可能导致有害干涉。
2. **静态周期 ($k$)**：无论流形曲率如何，固定步数后才重新同步，可能在平坦区浪费计算量，或在剧烈变化区失效。

## 2. 方案一：自适应强度 (Adaptive $\alpha_t$)

### 2.1 核心逻辑
根据当前梯度 $g_t$ 与缓存平坦度向量 $v_{flat}$ 的**对齐程度**动态缩放注入强度。

### 2.2 数学形式
定义干涉因子 $\phi_t$：
$$\phi_t = \frac{|\langle g_t, v_{flat} \rangle|}{\|g_t\| \cdot \|v_{flat}\|}$$

自适应律：
$$\alpha_t = \alpha_{max} \cdot (1 - \phi_t)^\gamma$$

- **$\alpha_{max}$**：配置中的 `alpha` 上限。
- **$\gamma$ (Sensitivity)**：敏感度系数。$\gamma=1$ 为线性衰减，$\gamma=2$ 为更激进的平方抑制。
- **物理直觉**：当 $g_t$ 开始向 $v_{flat}$ 方向偏转时，说明 $v_{flat}$ 已不再纯粹是“剪切力”，应自动压低其权重以保护收敛稳定性。

## 3. 方案二：自适应预算 (Adaptive Budget $k_t$)

### 3.1 核心逻辑
将 $k$ 从“固定周期”转变为“最大容忍周期”。引入 **几何一致性阈值 (Cosine Limit)**。

### 3.2 判定机制
在每个 Lazy Step，监控 $\phi_t$：
- **若 $\phi_t \le \phi_{limit}$**：继续 Lazy 模式，复用 $v_{flat}$。
- **若 $\phi_t > \phi_{limit}$ 或 $step \ge k_{max}$**：强制触发 **Sync Step**。

### 3.3 优势
- **按需同步**：在流形平坦区域，系统会自动延长 Lazy 周期（最高至 $k_{max}$），提升训练吞吐量。
- **即时纠偏**：在进入高曲率区域时，系统会感知到 $\phi_t$ 激增并立即重新采样基础梯度和正交基。

## 4. 参数配置参考
```toml
[optimizer]
name = "ARS"
k = 10           # 最大 Lazy 步数
alpha = 0.1      # 最大注入强度
cos_limit = 0.1  # 几何漂移阈值 (phi_limit)
gamma = 2        # 强度衰减敏感度
```
