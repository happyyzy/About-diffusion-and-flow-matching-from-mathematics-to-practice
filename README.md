# About-diffusion-and-flow-matching-from-mathematics-to-practice
从数学的角度解释当今最流行的t2l框架：diffusion和rectified flow，从高层理论解释如何一眼看破在stable diffusion和flux各版本训练/推理中出现的trick。也可以看作[MIT S186 diff and flow matching](https://diffusion.csail.mit.edu/)的理论注解，毕竟这堂课就讲这俩模型。

## 什么是flow matching
flow matching的想法很简单，过去的生成算法都是encoder-decoder,训练数据要从encoder流入latent space,推理时再从latent space出发流到真实图像分布，那为啥不直接学习后者的向量场呢？

下面给出数学描述：

### 😎 概率路径与速度场
---
假设我们定义了一条路径 $p_t(x)$，对每个 $t$，有一个密度 $p_t(x)$。  
这条路径的演化被如下的ode的flow推动：

$\frac{\partial x}{\partial t} = u(x, t),$

其中 $u(x, t)$ 是我们要学习的 **速度场**。

良好定义的路径还必须满足概率守恒方程（即连续性方程）：

$\frac{\partial p_t(x)}{\partial t} + \nabla \cdot ( p_t(x) \, u(x, t) ) = 0.$

它其实就是flow matching版本的Fokker-Planck方程，由归一化条件/质量守恒得到。注意这与diffusion不同：diffusion的SDE自然对应随机变量的演化，所以归一化条件自动满足，其Fokker-Planck方程由SDE本身导出。
所以flow matching有个问题：最终流向的分布全测度未必是1，此时根据归一化分布采样就好。



### 💖 训练目标 / 损失函数
---
假设我们从数据分布采样到 $x_1 \sim p_1(x)$，  
然后在路径上给定中间时刻 $t$，我们可以定义一个条件分布（“从数据点向前／向后推”）  
$p(x_t \mid x_1, t)$，它有一个已知／可计算的速度场（或者说条件速度场）  
$u_{\text{cond}}(x_t, t; x_1)$。这个速度场通常比较简单／显式。

那么我们训练 $u_\theta$ 使得：

$$L(\theta) = \mathbb{E}_{t, x_1, x_t \sim p(\cdot \mid x_1, t)} \big[ \| u_\theta(x_t, t) - u_{\text{cond}}(x_t, t; x_1) \|^2 \big].$$

这个损失就是在逼近网络匹配真实的速度场 $u(x, t)$ ，至于为什么用平方损失，理论上的考量是是因为其有较好的数学结构，Hilber空间等泛函分析理论能够在此大展拳脚，我们很快会在后面看到。



### ✔ 采样
---
在训练好 $u_\theta$ 后，生成过程就简单许多：  
从初始简单分布（如标准高斯）在 $t = 0$ 取样 $x_0$，  
然后解下面的常微分方程（ODE）：

$\frac{d x}{d t} = u_\theta(x, t),$

将它从 $t = 0$ 移动到 $t = 1$，最终得到一个接近目标数据分布的样本 $x_1$。



### ✨ 路径设计
---
路径 $p_t$ 的设计是关键。  
常见的选择包括 **高斯插值路径（Gaussian paths）**，  
或者设计成与 **最优传输路径（optimal transport）** 更贴近，  
从而使得速度场更为合理、高效。

比如在 *Flow Matching* 的原始工作里，就提出了若干路径选项，  
其中就包括使用最优传输插值的路径。

## 什么是rectified flow
rectified flow认为：为了减少推理步数，速度场 $u_{\text{cond}}(x_t, t; x_1)$ 应该就是连接起点和终点的直线。

所以训练目标就变成最小化以下损失函数：

$$L = \mathbb{E}_{x_0, x_1, t} \big[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \big].$$

注意：我不喜欢写

$$\frac{d x}{d t} = x_1-x_0$$

这种数学上非常糟糕的表达式--这里面的变量没法解释：虽然对确定的采样条件向量场是直线，但是整个速度场并非直线。正确的思路是：先用rf的loss毕竟这种“局部直线”的速度场v，然后再用v做推理：

$$\frac{d x}{d t} = v(x, t),$$

问题来了，对两个分布 $p_0$ 和 $p_1$ ，真的能通过这种直线流把一个变成另一个吗？熟悉条件概率的人一眼就能看出来，这个 $L^2$ loss其实在目标函数取条件期望的时候取得最小值，该最小值是 $X_1-X_0$
在 $L^2$ 空间里相对于 $X_t$ 的条件期望:

$$v^\star(x, t) = \mathbb{E}[X_1 - X_0 \mid X_t = x],$$

其中随机变量满足：

$X_t = (1 - t) X_0 + t X_1, \quad (X_0, X_1) \sim \pi \in \Pi(p_0, p_1).$

这里 $\Pi(p_0, p_1)$ 表示所有边缘分布分别为 $p_0$ 与 $p_1$ 的联合分布.

所以这个MSE loss是有严格下界的，不小于Hilbert空间里 $X_1-X_0$ 到t时刻事件域对应的子空间的距离，换言之由于随机采样的引入，universal approximation在这里并不适用。

那啥时候能把loss降到0呢？至少理论上我们希望如此，不然理论上都降不到0实际训练就更糟糕了。熟悉应用数学的立马就能看出flow matching就是在做一种最优传输。

### 🧩 背景：两个分布之间的“耦合（Coupling）”

学调和分析的肯定熟悉Coupling，各种Interpolation定理都要用；学随机过程的必然也不陌生，Markov chain的极限分布定理就是这么来的。最优传输的coupling就是一回事：要把初始高斯分布和最终数据分布给配个对，才方便分布的传输。

给定两个概率分布 $p_0$ 和 $p_1$，  
我们想描述“如何把一个随机变量 $X_0 \sim p_0$ 变成另一个随机变量 $X_1 \sim p_1$”。

最一般的方式是考虑它们的联合分布 $\pi(x_0, x_1)$，满足边缘条件：

$\int \pi(x_0, x_1) \, d x_1 = p_0(x_0), \quad \int \pi(x_0, x_1) \, d x_0 = p_1(x_1).$

所有满足该条件的联合分布构成一个集合：

$\Pi(p_0, p_1) = \{ \pi \,:\, \text{marginals are } p_0, p_1 \}.$

这就是所谓的 **Kantorovich 耦合集合**，  
它包含了所有可能的“匹配方式”（couplings），即所有从 $p_0$ 到 $p_1$ 的随机对应关系。

---

### 🧮 Monge 耦合的定义

**Monge 耦合（Monge coupling）** 是其中一种“极端特殊”的耦合形式。  
它要求：对每个起点 $x_0$，都只对应唯一的终点 $x_1 = T(x_0)$。

换句话说，Monge 耦合由一个确定性映射 $T: \mathcal{X} \to \mathcal{X}$ 给出，  
即

$X_1=T(X_0).$

---

### 🧭 Brenier 定理（存在与唯一性）

当代价函数为平方欧氏距离：

$c(x_0, x_1) = \tfrac{1}{2} \| x_0 - x_1 \|^2,$

并且 $p_0$ 是绝对连续分布时，存在一个非常重要的结论：

**Brenier’s Theorem (1987)：**  
存在唯一的最优 Monge 映射 $T = \nabla \varphi$ ，其中 $\varphi$ 是凸函数，使得

$$T_{*} p_0 = p_1.$$

这个映射 $T$ 实现了最小传输代价，也定义了 **2-Wasserstein 空间上的 geodesic  
因此，Monge 映射 $T$ 给出了最短传输路径，也即最“高效”的从 $p_0$ 到 $p_1$ 的流。

---

### 📈 与 Rectified Flow 的关系
我们现在来证明：Brenier映射 $X_1=T(X_0)$ 使得rf loss的最优速度场

$$v^\star(x, t) = \mathbb{E}[X_1 - X_0 \mid X_t = x]$$

确实是起点指向终点的线段。

#### 🧮 第 1 步： $\Phi_t$ 的可逆性（决定性）

定义函数：

$\psi_t(x) := \tfrac{1 - t}{2} \|x\|^2 + t \, \varphi(x).$

因为 $(1 - t) > 0$ 且 $\varphi$ 为凸函数，所以 $\psi_t$ 是严格凸的。  
根据[数学知识](#注释)其梯度

$\nabla \psi_t(x) = (1 - t)x + t \nabla \varphi(x) = \Phi_t(x)$

是拓扑同胚。

由此可知，对于 $t \in [0, 1)$，存在连续的逆映射 $\Phi_t^{-1}$，使得：

$X_0 = \Phi_t^{-1}(X_t) $

---

### 📘 注释

这一步只用到“二次代价 $\Rightarrow T = \nabla \varphi \Rightarrow \Phi_t = \nabla \psi_t$ 为严格凸梯度”，  
从而保证 $\Phi_t$ 的可逆性。

所以二次代价的最优传输问题看起来似乎和rf的loss很像，但实际上是个单向充分条件的关系：前者只有唯一解，只是后者众多解中的一个。后面我们还要找出rf loss的更多最优解来为训练过程的time weighted策略做解释（类似diffusion的snr）。

另外数学知识是： $C^1$ 的严格凸映射的梯度是拓扑同胚，这一点只需要通过简单凸分析得到梯度映射是单射，再结合Brouwer区域不变性即可。

---

#### ✅ 第 2 步

令 $h(x_0) = X_1 - X_0 = T(x_0) - x_0$，则有：

$\mathbb{E}[X_1 - X_0 \mid X_t] = (T - \text{Id})(\Phi_t^{-1}(X_t)) = T(\Phi_t^{-1}(X_t)) - \Phi_t^{-1}(X_t).$

于是对任意 $x$：

$v^\star(x, t) = \mathbb{E}[X_1 - X_0 \mid X_t = x] = T(\Phi_t^{-1}(x)) - \Phi_t^{-1}(x).$

---

这表明在最优传输（$T = \nabla \varphi$）下，  rf的loss是0，最优速度场确实是直线。

### stable diffusion3中的两种训练策略

讲了一些理论，我们搞清楚了什么条件下rf的loss会更低，现在我们就将其应用于实践，看看stable diffusion3是如何做的。

#### Reflow
在直接用rf的loss训练模型后,轨迹的直线性没有保证,因为随机采样噪声和样本点对应一个随机的coupling,根据上面的理论我们需要一个一一对应的coupling:

$X_1=T(X_0).$

此时理论上loss能降到0,实际的训练也验证了这一点:只要用直接训练的rf模型生成噪声-图像对,并喂进去训练一轮,loss就能大幅下降;换句话说,这是属于rf的self instruct和one-epoch原则.

#### U-shaped time weight
先介绍一下动态形式的最优传输问题.

**Benamou–Brenier (2000)** 证明了如下的动态版本最优传输问题：

$W_2^2(p_0, p_1) = \inf_{(p_t, v_t)} \int_0^1 \! \int \|v_t(x)\|^2 \, p_t(x) \, d x \, d t,$

约束条件为连续性方程（概率守恒）：

$\partial_t p_t + \nabla \cdot (p_t v_t) = 0.$

---

#### 🚀 最优解的结构

最优解满足：

$v_t^\star(x) = \nabla \varphi_t(x),$

其中 $\varphi_t$ 是一个随时间变化的势函数（potential function）。

---

#### ✨ 结论

这意味着：  
在最优传输（Optimal Transport, OT）的动态形式下，  
**最优的速度场在每个时刻都是梯度场（irrotational field）**，  
结合我们之前对rf的loss的讨论,动态最优传输的解也是最优rf解;换言之,我们可以采取不同的时间步采样,来使得loss下降的同时提高模型的泛化性。这里和diffusion的snr还不太一样：snr是按照时间步均匀采样再对loss加权，这里直接在时间步采样时加权。

#### 实验结果
2024年stable diffusion组的实验《Improving the Training of Rectified Flows》。

## 📊 表 A：消融表——一步 / 两步 FID 的逐项改进

**列：** CIFAR-10、AFHQ-64、FFHQ-64；每列下面给出 NFE=1 或 NFE=2（步数）。  
**行：** 从旧 2-Rectified Flow 基线开始，逐项叠加训练策略。

| 方法 | CIFAR-10 (1/2步 FID) | AFHQ-64 (1/2步 FID) | FFHQ-64 (1/2步 FID) | 说明 |
|------|--------------------|--------------------|--------------------|------|
| Base [Liu et al., 2022] (A) | 12.21 / — | — | — | 旧版 2-RF 基线，一步表现较差 |
| (A) + EDM init + large batch (B) | 7.14 / 3.61 | ↓ (~8→5) | ↓ (~8→5) | 用 EDM 教师预热并扩大 batch |
| (B) + Our $p_t$ (C) | 5.17 / 3.37 | ↓ | ↓ | 改进路径分布，更贴近 OT 能量 |
| (C) + Huber (D) | 5.24 / 3.34 | ± | ± | 损失函数改为 Huber，变化小 |
| (C) + LPIPS-Huber (E) | 3.42 / 2.95 | ↓ | 5.21 / 4.26 | 感知度量 LPIPS + Huber，大幅提升 |
| (C) + LPIPS-Huber$_t$ (F) | 3.38 / 2.76 | 4.11 / 3.12 | 5.65 / 4.41 | 加入时间加权版本，主推方案 |
| (F) + Real data (G) | **3.07 / 2.40** | — | — | 混入真实对齐数据，最佳结果 |

**结论：**  
横向看，一步 / 两步 FID 一路下降；关键贡献点是：
- **EDM 预热**
- **感知加权路径分布 $p_t$**
- **LPIPS-Huber（带时间权重）**

仅一轮 ReFlow，就将一步 FID 从 12.21 降到 3.38（约 72% 改进）。

---

## 📈 表 B：CIFAR-10 无条件对比（Table 3）

**列：** NFE（步数，越少越难）、FID（↓越好）、IS（↑越好）。  
按方法分组：传统扩散、蒸馏扩散、score 蒸馏、consistency、Rectified Flow。

| 方法类别 | 一步 (NFE=1) | 两步 (NFE=2) | 说明 |
|-----------|--------------|--------------|------|
| Rectified Flows (ours) | 3.38 | 2.76 | 本文方法 |
| Consistency / iCT-deep | **2.51** | — | 一步最强 |
| Score Distillation (DMD) | 2.62 | — | 很强 |
| CTM + GAN | 1.98 | — | 属于 GAN 精炼路线，范式不同 |

**结论：**  
在不依赖 GAN 精炼的路线下，本文 1–2 步 FID 进入第一梯队：  
一步略逊 iCT-deep / DMD，但优于大多数经典蒸馏方法；两步性能极具竞争力。

---

## 🧠 表 C：ImageNet-64 类条件对比（Table 4）

**列：** NFE、FID（↓）、Precision（↑）、Recall（↑）。

| 方法 | 一步 (NFE=1) | 两步 (NFE=2) | Precision / Recall | 说明 |
|------|---------------|---------------|--------------------|------|
| Rectified Flows (ours) | 4.31 | 3.64 | ↑ | 本文方法 |
| Consistency (iCT) | **4.02** | **3.20** | ↑↑ | 略强 |
| CD / PD / DEIS 等蒸馏法 | 5–6 | 4–5 | — | 较弱 |

**结论：**  
在更困难的 ImageNet-64 上，本文超越多数蒸馏法，与 iCT 差距缩小，保持第一梯队竞争力。

---

## 🌀 图 D：采样器与“新更新规则”的影响（Figure 4）

三张子图分别为 CIFAR-10、AFHQ-64、FFHQ-64 的 FID–NFE 曲线。

- **Euler / Heun：** 常见一阶 / 二阶 ODE 求解器。  
- **+ new sampler：** 作者提出的新更新规则（含历史相关项）。

**观察：**
- Heun > Euler：二阶法少步时更稳定；
- “+ new sampler” 进一步降低少步 FID（1–4 步最明显）；
- 大步数时曲线趋近，说明“改进采样器”主要提升少步性能。

**结论：** 少步采样推荐 “Heun + 新更新规则”。

---

## 💰 表 E：训练总成本对比（Table 6）

| 方法 | 合成配对前向次数 (M) | 训练前向 (M) | 总成本相对值 |
|------|-------------------|-------------|--------------|
| ReFlow (本文) | 395 | 1,433.6 | ×1 |
| CD (Consistency Distillation) | — | 5,734.4 | ×3.1 |
| CT (Consistency Training) | — | 2,867.2 | ×1.5 |

**结论：**  
在达到相同少步指标下，ReFlow 的算力成本最低，比 CD 便宜约 3 倍，也低于 CT。

---

## 🔄 图 F：反演（Inversion）与重建（Figure 5）

**(a) Reconstruction Loss vs. NFE：**  
MSE 随 NFE 上升而下降；本文曲线明显低于 EDM，说明可逆性 / 重建性更好。

**(b) 分布高斯性检验：**  
用 $\|z\|_2^2$ 检查反演噪声是否近似高斯。本文在 NFE=8 时更接近理论卡方分布，EDM 偏差明显。

**(c) 视觉例子：**  
(8 + 8) 步 “数据 → 噪声 → 重建”：
- EDM 的噪声不自然、重建模糊；
- 本文噪声更“真”、重建更清晰。

**结论：**  
改良版 Rectified Flow 不仅少步采样优异，  
同时具备更好的 **可逆性与重建稳定性**，支持高质量反演与编辑。
所以SD3在模型更大、分辨率更高的同时采样步数还比之前少很多。

### 最后的注记
1.动态OT问题的物理味道很浓，熟悉场论的同学一定看出来它其实就是在说自由流体是无旋的，或者自由磁场是无旋的，而欧氏空间/单连通区域的无旋场就是梯度场；另外Arnold的MMCM里面也提到一些从几何/对称性角度出发分析这个二次泛函的方法。

2.从数学角度来看，动态OT其实类似于Chern-Simons functional的临界点曲率都是0，或者是概率版本的Morse flow（W2空间上的geodesics）。另外二次OT问题其实就是静态的二次代价强化学习，动态OT也就是一种强化学习,或者从应用数学角度来看，都统一到随机最优控制的框架下（ref Arnold stochastic odes，读了可以理解很多现象，比如为什么SGD非常roboust:不稳定的ode加上噪声项就能变稳定）。
## (1) 最优传输的动态形式（Benamou–Brenier）

**Benamou & Brenier (2000)** 给出的动态表述为：

$\displaystyle 
\min_{v_t, \rho_t} \int_0^1 \int \tfrac{1}{2} \|v_t(x)\|^2 \, \rho_t(x) \, dx \, dt
$

约束条件：

$\displaystyle 
\partial_t \rho_t + \nabla \cdot (\rho_t v_t) = 0, 
\quad \rho_0 = p_0, \; \rho_1 = p_1.
$

这里 $v_t$ 是速度场（控制变量），连续性方程为状态演化约束。  
这其实就是一个最优控制问题，只不过没有随机性，且 cost 是能量形式 $\|v\|^2$。

---

## (2) 强化学习的连续化（随机控制形式）

强化学习（Reinforcement Learning, RL）的连续时间极限（或随机控制形式）为：

$\displaystyle 
\max_{u_t} \; \mathbb{E}\Big[\int_0^T r(x_t, u_t) \, dt \Big]
$

约束条件：

$\displaystyle 
dx_t = f(x_t, u_t) \, dt + \sigma \, dW_t,
$

其中：
- $r$ 是即时奖励函数；
- $f$ 是系统动力学（即环境转移方程）；
- $\sigma \, dW_t$ 是随机扩散项。

这在形式上就是带**随机扩散项**的最优控制问题。

---

## 🧠 三、两者的统一语言：控制的“流形式”

OT 的连续性方程：

$\partial_t \rho_t + \nabla \cdot (\rho_t v_t) = 0$

与 RL 的 Fokker–Planck 方程：

$\partial_t \rho_t + \nabla \cdot (\rho_t f_t) - \tfrac{1}{2} \nabla^2 : (\rho_t \sigma \sigma^\top) = 0$

结构完全一致，只是强化学习多了一个“噪声项”。

---

### 💬 解释

- 在 OT 中，$v_t$ 控制密度流的确定性演化；
- 在 RL 中，$\pi(a|s)$（策略）控制带噪声的状态分布演化。

因此，**强化学习的策略 $\pi(a|s)$ 本质上就是对密度流的控制；**  
而 OT 是噪声为 0、单步完成的极限情形。

---

## 🔄 四、强化学习 ≈ 动态最优传输（Dynamic OT）

近年来出现了一个研究方向：**Dynamic Optimal Transport (DOT)** 或 **Stochastic OT (SOT)**，  
它直接建立了 RL 与 OT 的统一框架。

| 版本 | 数学形式 | 对应到 RL |
|------|-----------|------------|
| deterministic OT | $\displaystyle \min \int \|v_t\|^2 \, \rho_t$ | deterministic policy |
| stochastic OT | $\displaystyle \min \mathbb{E} \int c(x_t, a_t) \, dt$<br> s.t. $dx_t = f(x_t, a_t)\, dt + \sigma\, dW_t$ | stochastic policy / MDP |
| controlled OT | 加入奖励项或控制能量正则 | entropy-regularized RL |

---

### ✅ 结论

强化学习可以看作是**最优传输的动态随机推广**：

- 将“最小化传输代价”改为“最大化累积奖励”；  
- 将“固定初末分布”改为“受动力学约束的状态分布”；  
- 将“确定性路径”推广为“随机过程下的分布流动”。

这为 **Rectified Flow 与 RL** 提供了共同的数学框架：  
它们都在优化“路径分布 + 速度场”的能量泛函，只是目标函数（cost/reward）与随机性不同。

3.关于self instruct

Reflow可以从最优传输导出，那llm的sel instruct呢？其实早在二三十年前统计学习主宰nlp的时代人们就指出这种思想就是EM算法：

## 🧩 数学上可以看作「硬 EM + 自蒸馏」

x,z分别是训练数据，真实数据。

当 E 步直接取最大后验（hard assignment）时，有：

$z^{(t)} = \arg\max_z \, p_{\theta^{(t)}}(z \mid x),$

此时 M 步变为：

$\theta^{(t+1)} = \arg\max_\theta \, \log p_\theta(x, z^{(t)}),$

这等价于在“伪标签” $z^{(t)}$ 上重新训练模型。

---

### 🔁 与 Self-Instruct 的关系

在文本生成领域，**Self-Instruct** 正是这种“hard EM”机制的体现：

- **生成阶段（E 步）**：  
  相当于执行 $\arg\max_y \, p_\theta(y \mid x)$，即模型自己生成伪标签或答案；
  
- **微调阶段（M 步）**：  
  再在这些伪样本上重新拟合 $p_\theta$，更新模型参数。

---

### 🧠 直观理解

- EM 算法的“E 步”提供“软”或“硬”的隐变量推断；
- “M 步”则在这些隐变量（或伪标签）上重新训练；
- 自蒸馏（Self-Distillation）或 Self-Instruct 都是在**同一模型**上反复执行这种更新。

再回忆一下，其实早期的生成算法都是这么推出的，比如ae/vae/diffusion等。

4.关于one-epoch原则

## 1️⃣ 在传统推荐系统中的「One-Step / Shallow Update」原则

### 💡 概念

在早期的协同过滤（Collaborative Filtering）或矩阵分解（Matrix Factorization, MF）框架中，  
算法通常只执行 **单步（one-step）用户–物品更新**：

$\hat{r}_{ui} = p_u^\top q_i$

其中：
- $p_u$：用户向量；
- $q_i$：物品向量。

它们通过一次梯度更新或闭式解求得，不进行多轮传播。

---

### 📘 原则含义

- 保证训练简单、收敛快；  
- 避免多步传播导致的噪声放大；  
- 在 **在线推荐** 或 **流式更新** 场景中尤为重要，强调 “**一次更新即生效**”。

---

### 🧩 代表场景

- **ALS / SGD 矩阵分解**：每次仅对一对 (user, item) 做一次梯度下降；  
- **FM / FFM (Factorization Machines)**：交互特征仅做一次线性组合，无多层传播。

---

## 🧠 2️⃣ 图神经网络 (Graph-based RecSys) 中的 “One-Step Propagation”

### 💡 概念

现代推荐系统常基于 **用户–物品二分图（user–item graph）**。  
GNN 模型（如 GCN、LightGCN、NGCF）通常采用多层传播：

$h_u^{(k+1)} = \sum_{i \in \mathcal{N}(u)} w_{ui} \, h_i^{(k)}.$

---

### ⚠️ 问题发现

研究发现：
- 多层（$k > 2$）传播会出现 **过平滑（over-smoothing）**；  
- 噪声用户或冷启动节点的错误会被放大。

---

### 💡 One-Step / Shallow GNN 原则

在推荐图中，只传播 **一次或两次邻接信息** 已足够。  
深层传播不仅计算冗余，还会损害个性化表达。

---

### 🧾 代表论文

- 🔹 **LightGCN**（He et al., *SIGIR 2020*）  
  明确提出删除非线性与权重矩阵，只保留一次加权平均传播。

- 🔹 **SGL / SimGCL / UltraGCN**  
  等后续工作均遵循此原则，强调结构简化与高效传播。

---

### ✅ 实践效果

- **训练更快**：无需多层传播；  
- **泛化更强**：避免过平滑；  
- **效果更优**：性能接近或超越深层 GNN。

---

### 🔄 类比：与 Rectified Flow 的「一轮 ReFlow 就足够」

| 推荐系统 | 生成模型 |
|-----------|-----------|
| One-Step / Shallow Update | One-Round ReFlow |
| 近似一次传播，达到全局效果 | 近似一次流动，逼近最优传输路径 |
| 控制噪声放大与传播深度 | 控制轨迹误差与采样复杂度 |

➡️ 二者的核心思想一致：  
**“一次传播” ≈ “最优**

## 🧩 一、LightGCN 做了什么

### 💡 原始的 GCN 一层更新公式

传统图卷积网络（GCN）的单层更新为：

$h_i^{(k+1)} = \sigma \big( W^{(k)} \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{d_i d_j}} \, h_j^{(k)} \big)$

其中包括两部分计算：

- **线性变换**：$W^{(k)}$  
- **非线性激活**：$\sigma(\cdot)$

---

### ⚙️ LightGCN 的核心改动

LightGCN 直接把这两部分都删掉，仅保留**邻居平均传播**：

$h_i^{(k+1)} = \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{d_i d_j}} \, h_j^{(k)}.$

最终的用户表示是多层结果的加权平均：

$h_u = \sum_{k=0}^{K} \alpha_k \, h_u^{(k)}.$

也就是说，LightGCN **只在图结构上传播**，完全不引入额外的权重矩阵或非线性激活。

---

## 🧠 二、那它不就是线性传播吗？表达力从哪来？

### ✅ 1. 表达力来自「图结构」而非非线性

在推荐系统中，真正关键的是：
> 谁和谁相连，以及相连多远。

LightGCN 的传播可写作：

$H^{(K)} = A^K X,$

其中 $A$ 是邻接矩阵，$X$ 是初始嵌入（用户、物品向量）。

这相当于在用户–物品二分图上统计 **K 阶邻域共现模式**：

| 阶数 | 含义 |
|------|------|
| 一阶 | 直接交互（用户喜欢的物品） |
| 二阶 | 共同喜欢的物品（用户之间相似） |
| 三阶 | 间接相似用户 / 跨域兴趣传播 |

这些高阶邻接关系本身已隐式构成复杂的非线性依赖。

---

### ✅ 2. 嵌入向量仍是可学习参数

初始嵌入 $X$（即用户向量 $p_u$ 与物品向量 $q_i$）是可学习的。  
训练目标通常为 BPR（Bayesian Personalized Ranking）：

$\text{Loss} = -\log \sigma(h_u^\top h_i - h_u^\top h_j)$

这意味着：
- 虽然传播是线性的；
- 但评分函数 $h_u^\top h_i$ 是**双线性**的。

因此整体模型仍是**非线性的二次型结构**。  
它的表达力 ≈ **矩阵分解 + 多阶邻域正则化**。

---

### ✅ 3. 图卷积的谱特性本身非平凡

LightGCN 的传播矩阵是归一化邻接矩阵 $A_{\text{sym}}$ 的幂：

$H^{(K)} = P_K(\Lambda) \, U^\top X$

这意味着它在谱空间上执行了多项式滤波（Polynomial Filtering）：

- 对应图拉普拉斯谱的低频部分；
- 相当于在图上执行**低通滤波器**；
- 不同阶 $k$ 的聚合权重 $\alpha_k$ 可调节平滑度与局部性。

为什么是低通滤波器？因为归一化邻接矩阵特征值都在0和1之间，而小特征值对应大的Laplace二次型值，也就是高噪声，或者可以自己算一算简单图的频谱也能看出来。

---

### 🧩 小结

LightGCN 虽然看似删除了“复杂模块”，但：

| 模块 | 是否保留 | 作用 |
|------|------------|------|
| 权重矩阵 $W$ | ❌ 删除 | 减少过拟合与噪声传播 |
| 非线性激活 $\sigma$ | ❌ 删除 | 避免过平滑 |
| 邻接传播 $A$ | ✅ 保留 | 核心：捕捉图结构依赖 |
| 多层聚合 $\alpha_k$ | ✅ 保留 | 控制平滑度与多阶特征融合 |

➡️ 这正体现了 **“浅层传播即可捕获复杂模式”** 的思想，  
与 Rectified Flow 的“一步近似最优路径”具有惊人的相似性。

## 补充：关于推理时的ode求解器

这里讲一个基于经典数值分析的ode解法henu，再讲一个专门出现在生成模型里的ode解法DPM solver。

## 🧩 从 RK2 出发：基本思想

我们要求解常微分方程：

$\displaystyle \frac{dx}{dt} = f(x, t), \quad x(t_0) = x_0.$

---

### ⚙️ RK2（二阶显式 Runge–Kutta 方法）

RK2 是一类二阶显式 Runge–Kutta 方法，其一般形式为：

$\displaystyle
\begin{cases}
k_1 = f(x_n, t_n), \\
k_2 = f(x_n + a \, \Delta t \, k_1, \, t_n + a \, \Delta t), \\
x_{n+1} = x_n + \Delta t [\, b_1 k_1 + b_2 k_2 \,].
\end{cases}
$

其中系数满足二阶精度条件：

$b_1 + b_2 = 1, \quad b_2 a = \tfrac{1}{2}.$

不同参数 $(a, b_1, b_2)$ 对应不同算法，如 Midpoint、Ralston、Heun 等。

---

## 🧠 Heun 方法：RK2 的一个特例

Heun 法的参数取：

$a = 1, \quad b_1 = \tfrac{1}{2}, \quad b_2 = \tfrac{1}{2}.$

代入得：

$\displaystyle
\begin{cases}
k_1 = f(x_n, t_n), \\
k_2 = f(x_n + \Delta t \, k_1, \, t_n + \Delta t), \\
x_{n+1} = x_n + \frac{\Delta t}{2}(k_1 + k_2).
\end{cases}
$

---

### 📐 几何解释

Heun 方法可以理解为：

> **“先用 Euler 预测，再用梯形平均修正”**。

- 第一步：用 Euler 预测区间末点；
- 第二步：用起点与终点的平均斜率修正；
- 等价于**改进的欧拉法（Improved Euler）**或**梯形法（Trapezoidal Rule）**。

其几何含义是：  
用区间两端的斜率平均近似曲线的整体斜率，以减少截断误差。

---

## 🔬 Heun 方法在生成模型中的角色

Heun 方法在现代生成建模中有了“第二次生命”，  
尤其在 **扩散模型（Diffusion Models）** 与 **Rectified Flow（RF）** 中。

---

### (1) 在扩散模型（Diffusion）中的角色

在 **SD3 官方实现** 与 **Flux** 框架中，Heun 仍是核心推理求解器之一。

---

### (2) 在 Rectified Flow (RF) / ReFlow 中的角色


由于 RF 训练的轨迹几乎是**直线轨迹（straight-line trajectory）**，  
低阶 Heun 方法已能达到足够高的精度。

在 Stability AI 的 **NeurIPS 2024 ReFlow 论文** 中指出：

> “A single iteration of ReFlow plus Heun sampling achieves nearly straight-line trajectories with 4× faster sampling.”

---
## 🧠 DPM Solver

DPM说白了就是：DDIM的ode是一阶非齐次线性ode，可以直接写出显式解（这大概是任何ode课程必讲的部分），显示解有积分项，那就通过各种方式数值计算这个积分。注意DDIM的ode是一阶非齐次线性ode是个特性，对于一般的flow matching或diffusion ode并不成立。

## 🧩  从扩散 ODE 出发

扩散模型的反向过程可以写为（含噪声参数 $\beta(t)$）：

$\displaystyle 
\frac{d x_t}{d t} = -\frac{1}{2}\beta(t)\,x_t - \frac{1}{2}\beta(t)\,\epsilon_\theta(x_t, t),
$

其中：

- 第一项 $-\frac{1}{2}\beta(t)x_t$：**线性衰减项**（类似漂移项）；
- 第二项 $-\frac{1}{2}\beta(t)\epsilon_\theta(x_t, t)$：**模型校正项**，由网络预测的噪声引入。

---

## 🧮 2. 写成线性系统形式

将上式改写为**线性非齐次 ODE**形式：

$\displaystyle 
\frac{d x_t}{d t} = A(t)\,x_t + g(x_t, t),
$

其中：

$\displaystyle A(t) = -\frac{1}{2}\beta(t)\,I, \quad g(x_t, t) = -\frac{1}{2}\beta(t)\,\epsilon_\theta(x_t, t).$

---

## 🧠 3. 用积分因子（Integrating Factor）法

乘上积分因子 $\exp\!\Big(-\!\int_0^t A(s)\,ds\Big)$，得到：

$\displaystyle 
\frac{d}{dt}\!\left[e^{-\int_0^t A(s)\,ds} \, x_t \right]
= e^{-\int_0^t A(s)\,ds}\, g(x_t, t).
$

两边积分：

$\displaystyle 
e^{-\int_0^{t_{n+1}} A(s)\,ds} x_{t_{n+1}}
= e^{-\int_0^{t_n} A(s)\,ds} x_{t_n}
+ \int_{t_n}^{t_{n+1}} e^{-\int_0^{s} A(u)\,du} g(x_s, s) \, ds.
$

再两边乘以 $e^{\int_0^{t_{n+1}} A(s)\,ds}$：

$\displaystyle 
x_{t_{n+1}} = e^{\int_{t_n}^{t_{n+1}} A(s)\,ds} \, x_{t_n}
+ \int_{t_n}^{t_{n+1}} e^{\int_{s}^{t_{n+1}} A(u)\,du}\, g(x_s, s) \, ds.
$

这就是**指数积分器（Exponential Integrator）**的精确形式。

---

## ⚙️ 4. 在扩散 ODE 下的具体形式

代入 $A(t) = -\frac{1}{2}\beta(t)I$ 与 $g(x_t, t) = -\frac{1}{2}\beta(t)\epsilon_\theta(x_t, t)$，得：

$\displaystyle 
x_{t_{n+1}}
= e^{-\int_{t_n}^{t_{n+1}} \frac{\beta(s)}{2}ds} \, x_{t_n}
- \int_{t_n}^{t_{n+1}} e^{-\int_{s}^{t_{n+1}} \frac{\beta(u)}{2}du}
\, \frac{\beta(s)}{2}\, \epsilon_\theta(x_s, s) \, ds.
$

---

最最后提一嘴，从数学角度看，rf和diffusion的ode形式不同，应当采取不同的ode求解器，比如DPM就不应用于SD3/SD3.5，也有一些文章支持此观点：
>“For evaluations on Stable Diffusion 3 [10], we exclude DPM-Solver++ from comparison, as UniPC has consistently outperformed it.”
>from [arxiv]('https://arxiv.org/html/2506.21757v2?utm_source=chatgpt.com')
















