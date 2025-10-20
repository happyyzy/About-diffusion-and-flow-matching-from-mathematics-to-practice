# About-diffusion-and-flow-matching-from-mathematics-to-practice
从数学的角度解释当今最流行的t2l框架：diffusion和rectified flow，从高层理论解释如何一眼看破在stable diffusion和flux各版本训练/推理中出现的trick。也可以看作[MIT S186 diff and flow matching](https://diffusion.csail.mit.edu/)的理论注解，毕竟这堂课就讲这俩模型。

## 什么是flow matching
flow matching的想法很简单，过去的生成算法都是encoder-decoder,训练数据要从encoder流入latent space,推理时再从latent space出发流到真实图像分布，那为啥不直接学习后者的向量场呢？

下面给出数学描述：

---

### 概率路径与速度场

假设我们定义了一条路径 $p_t(x)$，对每个 $t$，有一个密度 $p_t(x)$。  
这条路径的演化被如下的ode的flow推动：

$\frac{\partial x}{\partial t} = u(x, t),$

其中 $u(x, t)$ 是我们要学习的 **速度场**。

良好定义的路径还必须满足概率守恒方程（即连续性方程）：

$\frac{\partial p_t(x)}{\partial t} + \nabla \cdot ( p_t(x) \, u(x, t) ) = 0.$

它其实就是flow matching版本的Fokker-Planck方程，由归一化条件/质量守恒得到。注意这与diffusion不同：diffusion的SDE自然对应随机变量的演化，所以归一化条件自动满足，其Fokker-Planck方程由SDE本身导出。
所以flow matching有个问题：最终流向的分布全测度未必是1，此时根据归一化分布采样就好。

---

### 训练目标 / 损失函数

假设我们从数据分布采样到 $x_1 \sim p_1(x)$，  
然后在路径上给定中间时刻 $t$，我们可以定义一个条件分布（“从数据点向前／向后推”）  
$p(x_t \mid x_1, t)$，它有一个已知／可计算的速度场（或者说条件速度场）  
$u_{\text{cond}}(x_t, t; x_1)$。这个速度场通常比较简单／显式。

那么我们训练 $u_\theta$ 使得：

$$L(\theta) = \mathbb{E}_{t, x_1, x_t \sim p(\cdot \mid x_1, t)} \big[ \| u_\theta(x_t, t) - u_{\text{cond}}(x_t, t; x_1) \|^2 \big].$$

这个损失就是在逼近网络匹配真实的速度场 $u(x, t)$ ，至于为什么用平方损失，理论上的考量是是因为其有较好的数学结构，Hilber空间等泛函分析理论能够在此大展拳脚，我们很快会在后面看到。

---

### 采样

在训练好 $u_\theta$ 后，生成过程就简单许多：  
从初始简单分布（如标准高斯）在 $t = 0$ 取样 $x_0$，  
然后解下面的常微分方程（ODE）：

$\frac{d x}{d t} = u_\theta(x, t),$

将它从 $t = 0$ 移动到 $t = 1$，最终得到一个接近目标数据分布的样本 $x_1$。

---

### 路径设计

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

问题来了，对两个分布 $p_0$ 和 $p_1$ ，真的能通过这种直线流把一个变成另一个吗？熟悉条件概率的人一眼就能看出来，这个 $L^2$ loss其实在目标函数取条件期望的时候取得最小值，该最小值是 $X_1-X_0$
在 $L^2$ 空间里相对于 $X_t$ 的条件期望:

$$v^\star(x, t) = \mathbb{E}[X_1 - X_0 \mid X_t = x],$$

其中随机变量满足：

$X_t = (1 - t) X_0 + t X_1, \quad (X_0, X_1) \sim \pi \in \Pi(p_0, p_1).$

这里 $\Pi(p_0, p_1)$ 表示所有边缘分布分别为 $p_0$ 与 $p_1$ 的联合分布.

所以这个MSE loss是有严格下界的，不小于Hilbert空间里 $X_1-X_0$ 到t时刻时间域对应的子空间的距离，换言之由于随机采样的引入，universal approximation在这里并不适用。

那啥时候能把loss降到0呢？至少理论上我们希望如此，不然理论上都降不到0实际训练就更糟糕了。熟悉应用数学的立马就能看出flow matching就是在做一种最优传输。

### 🧩 1. 背景：两个分布之间的“耦合（Coupling）”

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

### 🧮 2. Monge 耦合的定义

**Monge 耦合（Monge coupling）** 是其中一种“极端特殊”的耦合形式。  
它要求：对每个起点 $x_0$，都只对应唯一的终点 $x_1 = T(x_0)$。

换句话说，Monge 耦合由一个确定性映射 $T: \mathcal{X} \to \mathcal{X}$ 给出，  
即

$X_1=T(X_0).$

---

### 🧭 3. Brenier 定理（存在与唯一性）

当代价函数为平方欧氏距离：

$c(x_0, x_1) = \tfrac{1}{2} \| x_0 - x_1 \|^2,$

并且 $p_0$ 是绝对连续分布时，存在一个非常重要的结论：

**Brenier’s Theorem (1987)：**  
存在唯一的最优 Monge 映射 $T = \nabla \varphi$ ，其中 $\varphi$ 是凸函数，使得

$$T_{\#} p_0 = p_1.$$

这个映射 $T$ 实现了最小传输代价，也定义了 **2-Wasserstein 空间上的 geodesic  
因此，Monge 映射 $T$ 给出了最短传输路径，也即最“高效”的从 $p_0$ 到 $p_1$ 的流。

---

### 📈 4. 与 Rectified Flow 的关系
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

$X_0 = \Phi_t^{-1}(X_t) \quad \text{a.s.}$

---

### 📘 注释

这一步只用到“二次代价 $\Rightarrow T = \nabla \varphi \Rightarrow \Phi_t = \nabla \psi_t$ 为严格凸梯度”，  
从而保证 $\Phi_t$ 的可逆性。

所以二次代价的最优传输问题看起来似乎和rf的loss很像，但实际上是个单向充分条件的关系：前者只有唯一解，只是后者众多解中的一个。后面我们还要找出rf loss的更多最优解来为训练过程的time weighted策略做解释（类似diffusion的snr）。

另外数学知识是： $C^1$ 的严格凸映射的梯度是拓扑同胚，这一点只需要通过简单凸分析得到梯度映射是单射，再结合Brouwer区域不变性即可。

---

#### 第 2 步

令 $h(x_0) = X_1 - X_0 = T(x_0) - x_0$，则有：

$\mathbb{E}[X_1 - X_0 \mid X_t] = (T - \text{Id})(\Phi_t^{-1}(X_t)) = T(\Phi_t^{-1}(X_t)) - \Phi_t^{-1}(X_t).$

于是对任意 $x$：

$v^\star(x, t) = \mathbb{E}[X_1 - X_0 \mid X_t = x] = T(\Phi_t^{-1}(x)) - \Phi_t^{-1}(x).$

---

这表明在最优传输（$T = \nabla \varphi$）下，  rf的loss是0，最优速度场确实是直线。

### stable diffusion3中的两种训练策略

讲了一些理论，我们搞清楚了什么条件下rf的loss会更低，现在我们就将其应用于实践，看看stable diffusion3是如何做的。

#### Reflow






