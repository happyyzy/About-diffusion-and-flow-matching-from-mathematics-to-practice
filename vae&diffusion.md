# 这篇主要讲老派模型:ae,vae,diffusion

# Autoencoder
主要讲讲 loss 怎么来的：  
目标是最大化 $p_{\theta,\phi}(x)$ ，也即训练数据 $x$ 通过 encoder 隐空间 $z$ 流出 decoder 之后复原的概率：

$$
L_{\mathrm{AE}} = -\, \mathbb{E}_{q_{\phi}(z \mid x)} \left[ \log p_{\theta}(x \mid z) \right]
$$

我们这样解释这个 loss：极大似然拆分为  

$$
\max_{\theta}\, \log p_{\theta}(x)
= \max_{\theta}\, \log \int p_{\theta}(x \mid z)\, p(z)\, dz
$$

这里面只包含 decoder 的 $\theta$ 参数，然而隐空间的分布 $p(z)$ 不知道，所以引入 encoder 参数 $\phi$，用 $q_{\phi}(z)$ 来逼近。

---

**基本假设 1：** encoder 的条件输出 $q_{\phi}(z\mid x)$ 是条件高斯分布， $\phi$ 是 encoder 的参数； 

$$
q_{\phi}(z \mid x) = \mathcal{N}\big(z\, \mu_{\phi}(x), \mathrm{diag}(\sigma_{\phi}(x)^2)\big)
$$


注意这不算个很强的假设，它其实是混合高斯模型 GMM 的连续化，而熟悉泛函分析的都知道这样最后得到的无条件分布 $q_{\phi}(z)$ 在概率测度空间里是稠密的。换言之，该假设对隐空间分布没有任何限制。

---

**基本假设 2：** decoder 的条件输出 $p_{\theta}(x\mid z)$ 是条件高斯分布，$\theta$ 是 decoder 的分布  

$$
p_{\theta}(x \mid z) = \mathcal{N}\\big(\hat{x}_{\theta}(z),\, \sigma^{2} I\big)
$$

所以 AE 的 loss 引入 $\phi$ 对真正的极大似然又近似了一层，并不是完全的 MLE。  
现在可以推出具体表达式了：  

$$
L_{\mathrm{AE}} = \sum_{z}\, \frac{1}{2\sigma^{2}}\, \|\, x - \hat{x}^{\theta}(z) \,\|^{2} + \text{const.}
$$

怎么对 $z$ 采样呢？直接采样是不行的，因为直接对 $q_{\phi}(z \mid x)$ 相当于拿上一轮训练好的参数 $\phi$ 来计算，梯度无法反向传播到 encoder，可以做一个数学上等价的操作： 

$$
\varepsilon \sim \mathcal{N}(0, I)
\qquad\text{and}\qquad
z = \mu_{\phi}(x) + \sigma_{\phi}(x) \odot \varepsilon
$$

这就叫重参数化技巧。

至此我们完全搞明白了 AE 的 loss：  

$$
L_{\mathrm{AE}}
= \sum_{z} \frac{1}{2\sigma^{2}}
\left\|\, x - \hat{x}^{\theta}\big(z=\mu_{\phi}(x)+\sigma_{\phi}(x)\odot\varepsilon\big) \,\right\|^{2}
$$


# Variational Autoencoder
## 讲讲 VAE 的 loss 怎么来的。  

我们说了 AE 的 loss 其实是对似然函数做了二次近似，而 VAE 就要对似然函数本身取极大。熟悉统计学习的都知道，此时应该用 ELBO（可参见李航的《统计学习方法》，那里面叫 F-函数）来算 loss：

$$
L(x) = -\, \mathbb{E}_{z \sim q_{\phi}(z \mid x)} \big[ \log p_{\theta}(x \mid z) \big] + D_{\mathrm{KL}} \big( q_{\phi}(z \mid x) \,\|\, p(z) \big)
$$

第一项和 AE 一样，不说了。为了算第二项，我们需要比 AE 多做一个假设：隐变量分布 $p(z)$ 是高斯分布 $\mathcal{N}(0, I)$。  
这可以看成是 VAE 为了优化完整似然函数，对 AE 做出的让步。

这时候就能直接算出：

$$
D_{\mathrm{KL}} \big( q_{\phi}(z \mid x) \,\|\, p(z) \big)
= \tfrac{1}{2} \sum_i \big( \mu_i^2 + \sigma_i^2 - \log \sigma_i^2 - 1 \big)
$$

所以总的 loss 就是：

$$
L_{\mathrm{VAE}}
= \sum_{\varepsilon \sim \mathcal{N}(0, I)} \| x - \hat{x} \|^2
+ \sum_i \big( \mu_i^2 + \sigma_i^2 - \log \sigma_i^2 - 1 \big)
$$

## 实际的vae怎么做的
### SDXL的vae
Stable Diffusion XL使用了和之前Stable Diffusion系列一样的VAE结构（KL-f8），但在训练中选择了更大的Batch-Size（256 vs 9），并且对模型进行指数滑动平均操作（EMA，exponential moving average），EMA对模型的参数做平均，从而提高性能并增加模型鲁棒性。

SD 2.x VAE是基于SD 1.x VAE微调训练了Decoder部分，同时保持Encoder部分权重不变，使他们有相同的Latent特征分布，所以SD 1.x和SD 2.x的VAE模型是互相兼容的。而SDXL VAE是重新从头开始训练的，所以其Latent特征分布与之前的两者不同。

由于Latent特征分布产生了变化，SDXL VAE的缩放系数也产生了变化。VAE在将Latent特征送入U-Net之前，需要对Latent特征进行缩放让其标准差尽量为1，之前的Stable Diffusion系列采用的缩放系数为0.18215，由于Stable Diffusion XL的VAE进行了全面的重训练，所以缩放系数重新设置为0.13025。

SDXL 的 VAE 使用的是一种组合损失：

$$
L_{\text{total}}
= \lambda_{\text{rec}} \, \| x - \hat{x} \|_1
+ \lambda_{\text{perc}} \sum_j w_j \, \| \phi_j(x) - \phi_j(\hat{x}) \|_1
+ \lambda_{\text{KL}} \, D_{\mathrm{KL}} \big( q_{\phi}(z \mid x) \,\|\, p(z) \big)
$$

其中：

- 第一项：L1 重构损失（局部细节）
- 第二项：感知损失（全局语义）
- 第三项：KL 正则（潜空间规整）

实际上，KL 项的权重通常非常小（$10^{-6} \sim 10^{-3}$），  
因为 SDXL 的 VAE 更偏向 **“感知压缩器（perceptual compressor）”**，  
而非严格意义上的概率模型。

### SD3的vae
之前 SD 系列中使用的 VAE 模型是将一个 **$512 \times 512$** 的图像编码为 **$64 \times 64$** 的 latent 特征，  
在 8 倍下采样的同时设置 **4 个通道（channel = 4）**。  
这种情况存在一定的压缩损失，直接影响是对 latent 特征重建时容易产生小物体畸变（例如人眼崩溃、文字扭曲等）。

---

因此，**SD 3 模型通过提升 latent 通道数** 来增强 VAE 的重建能力，提高重建后的图像质量。  
SD 3 技术报告中对不同 **通道数（4、8、16）** 的对比实验显示：  
当设置为 **16 通道** 时，VAE 模型的整体性能  
（FID 指标降低、Perceptual Similarity 指标降低、SSIM 指标提升、PSNR 指标提升）  
相比 **4 通道** 时有显著提升。  
因此 SD 3 最终确定使用了 **16 通道的 VAE 模型**。

---

与此同时，随着 VAE 的通道数增加到 16，  
扩散模型部分（U-Net 或 DiT）的通道数也需要同步修改：  
- 修改扩散模型与 VAE Encoder 衔接的第一层；  
- 修改与 VAE Decoder 衔接的最后一层。  

虽然这不会显著增加整体参数量，  
但会提升任务整体的训练难度。  

---

当通道数从 4 增加到 16 时，SD 3 需要学习和拟合的内容量也随之增加了 4 倍，  
因此必须提升整体参数规模以增强模型容量（*model capacity*）。

SD 3 论文中的模型通道数与模型容量对比实验结果表明：  
- 当模型参数量较小时，16 通道 VAE 的重建效果并不优于 4 通道 VAE；  
- 随着模型参数量逐步增大，16 通道 VAE 的重建性能优势开始显现；  
- 当模型深度（*depth*）增加到 **22** 时，16 通道的 VAE 性能明显优于 4 通道的 VAE。


 ### FLUX.1的vae
FLUX.1系列中，FLUX.1 VAE架构依然继承了SD 3 VAE的8倍下采样和输入通道数（16）。在FLUX.1 VAE输出Latent特征，并在Latent特征输入扩散模型前，还进行了Pack_Latents操作，一下子将Latent特征通道数提高到64（16 -> 64），换句话说，FLUX.1系列的扩散模型部分输入通道数为64，是SD 3的四倍。
这也代表FLUX.1要学习拟合的内容比起SD 3也增加了4倍，所以官方大幅增加FLUX.1模型的参数量级来提升模型容量

SD 3和FLUX.1的Patch化方法的不同：

1. SD 3（下采样卷积）：想象我们有一个大蛋糕，SD 3的方法就像用一个方形模具，从蛋糕上切出一个
的小方块。在这个过程中，我们提取了蛋糕的部分信息，但是由于进行了压缩，Patch块的大小变小了，信息会有所丢失。

2. FLUX.1（通道堆叠）：FLUX.1 的方法更像是直接把蛋糕的
块堆叠起来，不进行任何压缩或者切割。我们仍然保留了蛋糕的所有部分，但是它们不再分布在平面上，而是被一层层堆叠起来，像是三明治的层次。这样一来，蛋糕块的大小没有改变，只是它们的空间位置被重新组织了。

总的来说，相比SD 3，FLUX.1将 
 特征Patch化操作应用于扩散模型之前。这也表明FLUX.1系列模型认可了SD 3做出的贡献，并进行了继承与优化。

目前发布的FLUX.1-dev和FLUX.1-schnell两个版本的VAE结构是完全一致的。同时与SD 3相比，FLUX.1 VAE并不是直接沿用SD 3的VAE，而是基于相同结构进行了重新训练，两者的参数权重是不一样的。

# Diffusion







