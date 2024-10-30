# 第十章 主成分分析中的降维  

直接处理高维数据，例如图像，会带来一些困难：分析起来困难，解释起来困难，可视化几乎是不可能的，从实用角度来看，存储数据向量可能会很昂贵。然而，高维数据通常具有我们可以利用的特性。例如，高维数据往往是冗余的，即许多维度是多余的，并且可以通过其他维度的组合来解释。此外，高维数据中的维度通常是相关的，因此数据具有固有的低维结构。降维利用了这种结构和相关性，使我们能够使用更紧凑的数据表示形式，理想情况下不会丢失信息。我们可以将降维视为一种压缩技术，类似于 jpeg 或 mp3，它们是图像和音乐的压缩算法。  

在本章中，我们将讨论主成分分析（PCA），一种线性降维算法。PCA 由 Pearson（1901 年）和 Hotelling（1933 年）提出，已有超过 100 年的历史，仍然是数据压缩和数据可视化最常用的技术之一。它还用于识别简单的模式、潜在因素和高维数据的结构。  

![](images/50489edf9bb50bc65d25f03965ce8ad7e32c692868603685ffdc8a455ccf306d.jpg)  
(a) 数据集，包含 $x_{1}$ 和 $x_{2}$ 坐标。  

![](images/8372e2d03b2d3e92f4fa22e68ed2f12341f7a0293eff26dedef4583d28df12e0.jpg)  
(b) 压缩数据集，仅包含 $x_{1}$ 坐标。  
一个 $640 \times 480$ 像素的彩色图像是一个在百万维空间中的数据点，其中每个像素对应三个维度，分别对应每个颜色通道（红、绿、蓝）。  

主成分分析 PCA 降维  

在信号处理领域，PCA 也被称为 Karhunen-Loève 变换。在本章中，我们将从第一原理出发推导 PCA，利用我们对基和基变换（第 2.6.1 节和第 2.7.2 节）、投影（第 3.8 节）、特征值（第 4.2 节）、高斯分布（第 6.5 节）和约束优化（第 7.2 节）的理解。  

降维通常利用高维数据（例如，图像）的一个特性，即它通常位于低维子空间中。图 10.1 给出了二维中的一个说明性示例。尽管图 10.1(a) 中的数据并不完全位于一条直线上，但数据在 $x_{2}$ 方向上的变化很小，因此我们可以将其视为一条直线上的数据——几乎没有任何损失；见图 10.1(b)。要描述图 10.1(b) 中的数据，仅需 $x_{1}$ 坐标即可，数据位于 $\mathbb{R}^{2}$ 的一个一维子空间中。  

图 10.1  
说明：降维。 (a) 原始数据集在 $x_{2}$ 方向上的变化不大。 (b) (a) 中的数据可以仅使用 $x_{1}$ 坐标来表示，几乎没有任何损失。  

Karhunen-Loève 变换 信号处理社区中 PCA 也被称为 Karhunen-Loève 变换。

## 10.1 问题设定  

在主成分分析（PCA）中，我们感兴趣的是找到数据点 $\pmb{x}_{n}$ 的投影 $\tilde{\pmb{x}}_{n}$，使其尽可能与原始数据点相似，但具有显著较低的固有维度。图 10.1 给出了这种可能性的示例。  

数据协方差矩阵  

更具体地，我们考虑一个独立同分布的数据集 $\mathcal{X}=\{\pmb{x}_{1},.\,.\,.\,,\pmb{x}_{N}\}$，$\pmb{x}_{n}\in\mathbb{R}^{D}$，均值为 0，且具有数据协方差矩阵（6.42）  

$$
\pmb{S}=\frac{1}{N}\sum_{n=1}^{N}\pmb{x}_{n}\pmb{x}_{n}^{\top}.
$$  

此外，我们假设存在一个低维度压缩表示（代码）  

$$
\pmb{z}_{n}=\pmb{B}^{\top}\pmb{x}_{n}\in\mathbb{R}^{M}
$$  

其中，我们定义投影矩阵  

$$
\pmb{B}:=[\pmb{b}_{1},.\,.\,.\,,\pmb{b}_{M}]\in\mathbb{R}^{D\times M}\,.
$$  

矩阵 $\pmb{B}$ 的列 $b_{1},\dots,b_{M}$ 形成 $M$ 维子空间的基，在该子空间中，投影数据 $\tilde{\pmb{x}}=\pmb{B}\pmb{B}^{\top}\pmb{x}\in\mathbb{R}^{D}$ 存在。  

我们假设 $\pmb{B}$ 的列是正交归一的（定义 3.7），即 $\pmb{b}_{i}^{\top}\pmb{b}_{j}=0$ 当且仅当 $i\neq j$，且 $\pmb{b}_{i}^{\top}\pmb{b}_{i}=1$。我们寻求一个 $M$ 维子空间 $U\subseteq\mathbb{R}^{D}$，$\mathrm{dim}(U)=M<D$，将数据投影到该子空间上。我们用 $\tilde{\pmb{x}}_{n}\in U$ 表示投影数据，用 $z_{n}$ 表示其在基向量 $b_{1},\dots,b_{M}$ 下的坐标。我们的目标是找到投影 $\tilde{\pmb{x}}_{n}\in\mathbb{R}^{D}$（或等价地，代码 $z_{n}$ 和基向量 $b_{1},\dots,b_{M}$），使得它们尽可能与原始数据 $\pmb{x}_{n}$ 相似，并且压缩导致的损失最小。

### 示例 10.1 (坐标表示/代码)  

考虑 $\mathbb{R}^{2}$ 与标准基 $\boldsymbol{e}_{1} = [1,0]^{\top}$, $\boldsymbol{e}_{2} = [0,1]^{\top}$。从

![](images/591bb1128102e4068ebcd25547580cf85ec109b692a26a034c3530d9ed68f018.jpg)  
图 10.2 主成分分析的图形说明。在主成分分析（PCA）中，我们找到原始数据 $\pmb{x}$ 的压缩版本 $_z$。压缩数据可以重构为 $\tilde{\pmb{x}}$，它生活在原始数据空间中，但具有比原始数据集中的这些基向量表示的内在更低维度的表示，例如

$$
\begin{array}{r}{{\binom{5}{3}}=5e_{1}+3e_{2}\,.}\end{array}
$$
然而，当我们考虑形式为

$$
{\tilde{\mathbf{x}}}={\binom{0}{z}}\in\mathbb{R}^{2}\,,\quad z\in\mathbb{R}\,,
$$
的向量时，它们总是可以写成 $0e_{1}+z e_{2}$。为了表示这些向量，只需记住/存储 $\tilde{\pmb{x}}$ 相对于 $\boldsymbol{e}_{2}$ 向量的坐标/代码 $z$ 即可。  

更精确地说，$\tilde{\pmb{x}}$ 向量集（在标准向量加法和标量乘法下）形成一个向量子空间 $U$（参见第 2.4 节），因为 $\mathrm{dim}(U)=1$，因为 $U=\operatorname{span}[e_{2}]$。  

向量空间的维数对应于其基向量的数量（参见第 2.6.1 节）。  

在第 10.2 节中，我们将找到低维表示，尽可能保留信息并最小化压缩损失。第 10.3 节中给出了 PCA 的另一种推导，其中我们将考虑最小化原始数据 $\pmb{x}_{n}$ 与其投影 $\tilde{\pmb{x}}_{n}$ 之间的平方重构误差 ${\|\pmb{x}_{n}-\tilde{\pmb{x}}_{n}\|}^{2}$。  

图 10.2 说明了我们在 PCA 中考虑的设置，其中 $_z$ 表示压缩数据 $\tilde{\pmb{x}}$ 的低维表示，并充当瓶颈，控制原始数据 $\pmb{x}$ 与 $\tilde{\pmb{x}}$ 之间可以流动的信息量。在 PCA 中，我们考虑原始数据 $\pmb{x}$ 与其低维代码 $\textit{\textbf{z s0}}$ 之间的线性关系，即 $\pmb{z}=\mathbf{\bar{\Sigma}}_{\pmb{B}^{\top}\pmb{x}}$ 和 ${\tilde{\textbf{x}}}=\textbf{ B z}$，对于合适的矩阵 $B$。基于将 PCA 视为数据压缩技术的动机，我们可以将图 10.2 中的箭头视为表示编码器和解码器的一对操作。表示 $B$ 的线性映射可以视为解码器，将低维代码 $\boldsymbol{z}\in\mathbb{R}^{M}$ 重新映射回原始数据空间 $\mathbb{R}^{D}$。同样，$B^{\top}$ 可以视为编码器，将原始数据 $\pmb{x}$ 编码为低维（压缩）代码 $z$。  

在本章中，我们将使用 MNIST 手写数字数据集作为反复出现的例子，该数据集包含 60,000 个 0 到 9 的手写数字样本。每个数字是一个大小为 $28\times28$ 的灰度图像，即它包含 784 个像素，因此我们可以将该数据集中的每个图像视为向量 $\pmb{x}\in\mathbb{R}^{784}$。这些数字的示例如图 10.3 所示。

## 10.2 最大方差视角  

图10.1给出了如何使用单个坐标表示二维数据集的例子。在图10.1(b)中，我们选择忽略数据的$x_{2}$坐标，因为这并没有增加太多信息，使得压缩后的数据与图10.1(a)中的原始数据相似。我们也可以选择忽略$x_{1}$坐标，但这样压缩后的数据与原始数据差异很大，数据中的许多信息将会丢失。

如果我们把数据中的信息内容解释为数据的“空间填充”程度，那么我们可以通过观察数据的分布来描述数据中的信息。从第6.4.1节我们知道，方差是数据分布的一个指标，我们可以将PCA视为一种降维算法，它在低维表示中最大化方差，以尽可能保留信息。图10.4说明了这一点。

考虑到第10.1节讨论的设置，我们的目标是找到一个矩阵$B$（见(10.3)），当将数据投影到由$B$的列$b_{1},\dots,b_{M}$张成的子空间上时，尽可能保留信息。压缩后的数据保留最多信息等价于在低维编码中捕获最大的方差（Hotelling, 1933）。

注释。 (中心化数据) 对于(10.1)中的数据协方差矩阵，我们假设数据已经中心化。我们可以不失一般性地做出这个假设：假设$\pmb{\mu}$是数据的均值。利用我们在第6.4.4节讨论的方差性质，我们得到

$$ \operatorname{V}_{z}[z] = \operatorname{V}_{x}[\boldsymbol{B}^{\top}(\boldsymbol{x} - \boldsymbol{\mu})] = \operatorname{V}_{x}[\boldsymbol{B}^{\top}\boldsymbol{x} - \boldsymbol{B}^{\top}\boldsymbol{\mu}] = \operatorname{V}_{x}[\boldsymbol{B}^{\top}\boldsymbol{x}] \,. $$

即低维编码的方差与数据的均值无关。因此，我们不失一般性地假设在本节余下的部分中数据的均值为0。在这种假设下，低维编码的均值也为0，因为$\mathbb{E}_{z}[z]=\mathbb{E}_{\pmb{x}}[\pmb{B}^{\top}\pmb{x}]=\pmb{B}^{\top}\mathbb{E}_{\pmb{x}}[\pmb{x}]=\mathbf{0}$。

![](images/5b32190f98b433b78ee77458b7d000bcb32da8e0f603ad71ac794198c0626b9a.jpg)  
图10.4  PCA找到一个低维子空间（直线），当数据（蓝色）投影到这个子空间（橙色）上时，尽可能保留数据的方差（数据的分布）。

### 10.2.2   $M$ -维子空间的最大方差  

假设我们已经找到了前 $m-1$ 个主成分，即与 $S$ 的前 $m-1$ 个最大特征值相关的 $S$ 的前 $m-1$ 个特征向量。由于 $S$ 是对称的，谱定理（定理 4.15）表明我们可以使用这些特征向量来构造 $\mathbb{R}^{D}$ 中一个 $(m-1)$ 维子空间的正交规范特征基。通常，第 $m$ 个主成分可以通过从数据中减去前 $m-1$ 个主成分 $b_{1}, \dots, b_{m-1}$ 的影响，从而试图找到能够压缩剩余信息的主成分。我们得到新的数据矩阵

$$
\hat{\pmb{X}}:=\pmb{X}-\sum_{i=1}^{m-1}\pmb{b_{i}}\pmb{b_{i}^{\top}}\pmb{X}=\pmb{X}-\pmb{B_{m-1}}\pmb{X}\,,
$$

其中 $\pmb{X}=\left[{\pmb x}_{1}, \dots, {\pmb x}_{N}\right] \in \mathbb{R}^{D \times N}$ 包含数据点作为列向量，且 $\pmb{B_{m-1}:=\sum_{i=1}^{m-1}b_{i}b_{i}^{\top}}$ 是一个投影矩阵，将数据投影到由 $b_{1}, \dots, b_{m-1}$ 张成的子空间上。

**注释（记号）**。在整个本章中，我们不遵循将数据 $\pmb{x}_{1}, \dots, \pmb{x}_{N}$ 作为数据矩阵的行来收集的习惯，而是定义它们为 $\pmb{X}$ 的列。这意味着我们的数据矩阵 $\pmb{X}$ 是一个 $D \times N$ 矩阵，而不是传统的 $N \times D$ 矩阵。我们选择这种形式的原因是，这样可以避免转置矩阵或重新定义作为左乘矩阵的行向量。$\diamondsuit$

矩阵 $\hat{\pmb{X}}:=\left[\hat{\pmb x}_{1}, \dots, \hat{\pmb x}_{N}\right] \in \mathbb{R}^{D \times N}$ 在 (10.17) 中包含尚未被压缩的数据信息。

为了找到第 $m$ 个主成分，我们最大化方差

$$
V_{m}=\mathbb{V}[z_{m}]=\frac{1}{N}\sum_{n=1}^{N}z_{m n}^{2}=\frac{1}{N}\sum_{n=1}^{N}(\pmb{b}_{m}^{\top}\pmb{\hat{x}_{n}})^{2}=\pmb{b}_{m}^{\top}\pmb{\hat{S}}\pmb{b_{m}}\,，
$$

其中 $\left|\left|b_{m}\right|\right|^{2}=1$，我们遵循与 (10.9b) 相同的步骤，并定义 $\pmb{\hat{S}}$ 为变换数据集 $\hat{\mathcal{X}}:=\{\hat{\mathbf{x}}_{1}, \dots, \hat{\mathbf{x}}_{N}\}$ 的数据协方差矩阵。正如之前，当我们单独考虑第一个主成分时，我们解决了一个约束优化问题，并发现最优解 $b_{m}$ 是 $\pmb{\hat{S}}$ 的特征向量，与 $\pmb{\hat{S}}$ 的最大特征值相关。

结果表明 $b_{m}$ 也是 $S$ 的特征向量。更一般地，$S$ 和 $\pmb{\hat{S}}$ 的特征向量集是相同的。由于 $S$ 和 $\pmb{\hat{S}}$ 都是对称的，我们可以找到它们的正交规范特征基（谱定理 4.15），即两者都存在 $D$ 个不同的特征向量。接下来，我们证明 $S$ 的每个特征向量也是 $\pmb{\hat{S}}$ 的特征向量。假设我们已经找到了 $\pmb{\hat{S}}$ 的特征向量 $b_{1}, \dots, b_{m-1}$。考虑 $S$ 的一个特征向量 $b_{i}$，即 $S\pmb{b}_{i}=\lambda_{i}\pmb{b}_{i}$。一般地，

$$
\begin{array}{c}
\pmb{\hat{S}}\pmb{b}_{i}=\frac{1}{N}\pmb{\hat{X}}\pmb{\hat{X}}^{\top}\pmb{b}_{i}=\frac{1}{N}(\pmb{X}-\pmb{B}_{m-1}\pmb{X})(\pmb{X}-\pmb{B}_{m-1}\pmb{X})^{\top}\pmb{b}_{i}\\
=(\pmb{S}-\pmb{S}\pmb{B}_{m-1}-\pmb{B}_{m-1}\pmb{S}+\pmb{B}_{m-1}\pmb{S}\pmb{B}_{m-1})\pmb{b}_{i}\,.
\end{array}
$$

我们区分两种情况。如果 $i \geqslant m$，即 $b_{i}$ 不是前 $m-1$ 个主成分的特征向量且 $\pmb{B}_{m-1}\pmb{b}_{i}=\pmb{0}$。如果 $i < m$，即 $b_{i}$ 是前 $m-1$ 个主成分中的一个，则 $b_{i}$ 是投影到 $\pmb{B}_{m-1}$ 所投影的主子空间上的基向量。由于 $b_{1}, \dots, b_{m-1}$ 是这个主子空间的正交规范基，我们得到 $\pmb{B}_{m-1}\pmb{b}_{i}=\pmb{b}_{i}$。两种情况可以总结为：

$$
\pmb{B}_{m-1}\pmb{b}_{i}=\pmb{b}_{i}\quad\text{if}\;i<m\,,\qquad \pmb{B}_{m-1}\pmb{b}_{i}=\pmb{0}\quad\text{if}\;i\geqslant m\,.
$$

在 $i \geqslant m$ 的情况下，通过使用 (10.20) 在 (10.19b) 中，我们得到 $\pmb{\hat{S}}\pmb{b}_{i}=(\pmb{S}-\pmb{B}_{m-1}\pmb{S})\pmb{b}_{i}=\pmb{S}\pmb{b}_{i}=\lambda_{i}\pmb{b}_{i}$，即 $\pmb{b}_{i}$ 也是 $\pmb{\hat{S}}$ 的特征向量，特征值为 $\lambda_{i}$。具体地，

$$
\pmb{\hat{B}}\pmb{b}_{m}=\pmb{S}\pmb{b}_{m}=\lambda_{m}\pmb{b}_{m}\,.
$$

方程 (10.21) 表明 $\pmb{b}_{m}$ 不仅是 $S$ 的特征向量，也是 $\pmb{\hat{S}}$ 的特征向量。具体地，$\lambda_{m}$ 是 $\pmb{\hat{S}}$ 的最大特征值，同时也是 $S$ 的第 $m$ 大特征值，两者都与特征向量 $\pmb{b}_{m}$ 相关。

在 $i < m$ 的情况下，通过使用 (10.20) 在 (10.19b) 中，我们得到

$$
\pmb{\hat{S}}\pmb{b}_{i}=(\pmb{S}-\pmb{S}\pmb{B}_{m-1}-\pmb{B}_{m-1}\pmb{S}+\pmb{B}_{m-1}\pmb{S}\pmb{B}_{m-1})\pmb{b}_{i}=\pmb{0}=\pmb{0}\pmb{b}_{i}
$$

这意味着 $\pmb{b}_{1}, \dots, \pmb{b}_{m-1}$ 也是 $\pmb{\hat{S}}$ 的特征向量，但它们与特征值 $0$ 相关，因此 $\pmb{b}_{1}, \dots, \pmb{b}_{m-1}$ 张成了 $\pmb{\hat{S}}$ 的零空间。

这一推导表明，$M$ 维子空间的最大方差与特征值分解之间存在密切联系。我们将在第 10.4 节中重新审视这种联系。

总体而言，每个 $S$ 的特征向量也是 $\pmb{\hat{S}}$ 的特征向量。然而，如果 $S$ 的特征向量属于 $(m-1)$ 维主子空间，则 $\pmb{\hat{S}}$ 相关的特征值为 $0$。

利用关系 (10.21) 和 $\pmb{b}_{m}^{\top}\pmb{b}_{m}=1$，数据投影到第 $m$ 个主成分上的方差为

$$
V_{m}=\pmb{b}_{m}^{\top}\pmb{S}\pmb{b}_{m}\stackrel{(10.21)}{=}\lambda_{m}\pmb{b}_{m}^{\top}\pmb{b}_{m}=\lambda_{m}\,。
$$

这意味着，当数据投影到 $M$ 维子空间时，数据的方差等于与数据协方差矩阵的相应特征向量相关的特征值之和。

![](images/d8acc77deca19f4c2be567022a04a07b21409eb0f944aac3744dd378936a0426.jpg)
图 10.5 MNIST “8” 训练数据的属性。 (a) 按降序排序的特征值；(b) 与最大特征值相关的主成分捕获的方差。

(a) 所有数字 $^{\alpha}8^{\circ}$ 在 MNIST 训练集中的数据协方差矩阵的特征值（按降序排序）。 (b) 与最大特征值相关的主成分捕获的方差。

![](images/6f3500287f1d18a92dfd74e2adc29666569dd4c4eecf282924067c790bcf2bad.jpg)
图 10.6 投影方法的示例：找到一个子空间（直线），使得投影（橙色）和原始（蓝色）数据之间的差向量长度最小。

取 MNIST 训练数据中的所有数字“8”，我们计算数据协方差矩阵的特征值。图 10.5(a) 显示了数据协方差矩阵的 200 个最大特征值。我们看到，只有少数特征值的值与 0 差异显著。因此，当将数据投影到由相应特征向量张成的子空间上时，大部分方差主要由少数几个主成分捕获，如图 10.5(b) 所示。

总体而言，为了找到一个 $\mathbb{R}^{D}$ 中的 $M$ 维子空间，该子空间保留尽可能多的信息，PCA 告诉我们选择 (10.3) 中矩阵 $\pmb{B}$ 的列作为与数据协方差矩阵 $\pmb{s}$ 的前 $M$ 个最大特征值相关的 $M$ 个特征向量。PCA 可以通过前 $M$ 个主成分捕获的最大方差为

$$
V_{M}=\sum_{m=1}^{M}\lambda_{m}\,，
$$

其中 $\lambda_{m}$ 是数据协方差矩阵 $\pmb{s}$ 的前 $M$ 个最大特征值。因此，通过 PCA 压缩数据所损失的方差为

$$
J_{M}:=\sum_{j=M+1}^{D}\lambda_{j}=V_{D}-V_{M}\,。
$$

我们还可以定义捕获的相对方差为 $\cfrac{V_{M}}{V_{D}}$，以及压缩所损失的相对方差为 $1-\cfrac{V_{M}}{V_{D}}$。

## 10.3 投影视角

在下面的内容中，我们将推导PCA作为直接最小化平均重构误差的算法。这种视角使我们能够将PCA解释为实现最优线性自编码器的算法。我们将大量参考第2章和第3章。  

在上一节中，我们通过最大化投影空间中的方差来推导PCA，以尽可能保留信息。在  

![](images/5b26f500808f590be2bb89e3b922affcc574515b93f1752ee0af3d18c1db8c60.jpg)  
图10.7 简化的投影设置。(a) 一个向量 $\pmb{x} \in \mathbb{R}^{2}$（红色十字）将被投影到由 $\pmb{b}$ 张成的一维子空间 $U \subseteq \mathbb{R}^{2}$ 上。(b) 显示了 $\pmb{x}$ 与一些候选向量 $\tilde{\pmb{x}}$ 之间的差向量。  

![](images/9e067caa5461b47d9bb5a2d62d39ffcf8c55d97165f5eb2d7731e5e9f2cd4cf2.jpg)  

接下来，我们将考察原始数据 ${\pmb x}_{n}$ 与其重构 $\tilde{\pmb{x}}_{n}$ 之间的差向量，并最小化这种距离，使得 ${\pmb x}_{n}$ 与 $\tilde{\pmb{x}}_{n}$ 尽可能接近。图10.6说明了这种设置。

### 10.3.1 设置与目标  

假设一个（有序）正交规范基（ONB）$B=\left(b_{1},.\,.\,.\,,b_{D}\right)$ 于 $\mathbb{R}^{D}$ ，即 $\pmb{b}_{i}^{\top}\pmb{b}_{j}=1$ 当且仅当 $i=j$，否则为 0。  

从第 2.5 节我们知道，对于 $\mathbb{R}^{D}$ 的一个基 $(\boldsymbol{b}_{1},.\,.\,.\,,\boldsymbol{b}_{D})$ ，向量 $\pmb{x}\in\mathbb{R}^{D}$ 可以表示为基向量的线性组合，即向量 $\tilde{\pmb{x}}\in U$ 可以是 $\mathbb{R}^{3}$ 中的一个平面的向量。平面的维数是 2，但这些向量相对于 $\mathbb{R}^{3}$ 的标准基仍然有三个坐标。  

$$
\pmb{x}=\sum_{d=1}^{D}\zeta_{d}\pmb{b}_{d}=\sum_{m=1}^{M}\zeta_{m}\pmb{b}_{m}+\sum_{j=M+1}^{D}\zeta_{j}\pmb{b}_{j}
$$

对于合适的坐标 $\zeta_{d}\in\mathbb{R}$。  

我们感兴趣的是找到向量 $\tilde{\textbf{\textit{x}}}\in\mathbb{R}^{D}$，它们位于低维子空间 $U\subseteq\mathbb{R}^{D}$ 中，$\mathrm{dim}(U)=M$，使得  

$$
\tilde{\pmb{x}}=\sum_{m=1}^{M}z_{m}\pmb{b}_{m}\in U\subseteq\mathbb{R}^{D}
$$  

尽可能接近 $\pmb{x}$。请注意，此时我们需要假设 $\tilde{\pmb{x}}$ 的坐标 $z_{m}$ 和 $\pmb{x}$ 的坐标 $\zeta_{m}$ 不相同。  

接下来，我们使用这种表示 $\tilde{\mathbfit{x}}$ 的方式来找到最优坐标 $z$ 和基向量 $b_{1},\dots,b_{M}$，使得 $\tilde{\pmb{x}}$ 尽可能接近原始数据点 $\pmb{x}$，即我们旨在最小化 (欧几里得) 距离 $\lVert\pmb{x}-\tilde{\pmb{x}}\rVert$。图 10.7 说明了这种设置。  

不失一般性，我们假设数据集 ${\mathcal X}=\{{\pmb x}_{1},.\,.\,.\,,{\pmb x}_{N}\},$ ，$\pmb{x}_{n}\in\mathbb{R}^{D}$，位于原点 0 处，即 $\mathbb{E}[{\mathcal{X}}]=\mathbf{0}$。不假设零均值的情况下，我们会得到完全相同的结果，但符号会更加复杂。  

我们感兴趣的是将数据集 $\mathcal{X}$ 投影到 $\mathbb{R}^{D}$ 中的低维子空间 $U$ 上，该子空间的维数为 $\mathrm{dim}(U)=M$，且具有正交规范基向量 $b_{1},\dots,b_{M}$。我们将称这个子空间 $U$ 为主子空间。主子空间的数据点投影表示为  

$$
\tilde{\pmb{x}}_{n}:=\sum_{m=1}^{M}z_{mn}\pmb{b}_{m}=\pmb{B}\pmb{z}_{n}\in\mathbb{R}^{D}\,，
$$  

其中 $\boldsymbol{z}_{n}:=\,[z_{1n},.\,.\,.\,,z_{Mn}]^{\intercal}\,\in\,\mathbb{R}^{M}$ 是 $\tilde{\pmb{x}}_{n}$ 相对于基 $(b_{1},.\,.\,.\,,b_{M})$ 的坐标向量。更具体地说，我们希望 $\tilde{\pmb{x}}_{n}$ 尽可能接近 ${\pmb x}_{n}$。  

我们在接下来使用的相似度量是向量 $\pmb{x}$ 与 $\tilde{\pmb{x}}$ 之间的平方距离（欧几里得范数）$\lVert\pmb{x}-\tilde{\pmb{x}}\rVert^{2}$。因此，我们将目标定义为最小化平均平方欧几里得距离（重构误差）(Pearson, 1901)  

$$
J_{M}:=\frac{1}{N}\sum_{n=1}^{N}\|\pmb{x}_{n}-\tilde{\pmb{x}}_{n}\|^{2}\,，
$$  

明确指出我们投影数据的子空间的维数为 $M$。为了找到这种最优线性投影，我们需要找到主子空间的正交规范基和相对于此基的投影坐标 $\pmb{z}_{n}\in\mathbb{R}^{M}$。  

为了找到坐标 $z_{n}$ 和主子空间的正交规范基，我们采用两步方法。首先，我们优化给定的正交规范基 $(b_{1},.\,.\,.\,,b_{M})$ 下的坐标 $z_{n}$；其次，我们找到最优的正交规范基。

### 10.3.2 寻找最优坐标  

让我们从寻找投影 $\tilde{\pmb{x}}_{n}$ 的最优坐标 $z_{1n},.\,.\,.\,,z_{M n}$ 开始，其中 $n=1,\cdot\cdot\cdot,N$。考虑图 10.7(b)，其中主子空间由单一向量 $^{b}$ 张成。从几何意义上讲，寻找最优坐标 $z$ 相当于找到相对于 $^{b}$ 的线性投影 $\tilde{\pmb{x}}$，使得 $\tilde{\textbf{\textit{x}}}-\textbf{\textit{x}}$ 的距离最小化。从图 10.7(b) 可以看出，这将是正交投影，接下来我们将证明这一点。  

我们假设 $\mathbb{R}^{D}$ 子空间 $U$ 的正交规范基 $(b_{1},.\,.\,.\,,b_{M})$。为了找到相对于此基的最优坐标 $z_{m}$，我们需要计算偏导数

$$
\begin{array}{l}
\displaystyle\frac{\partial J_{M}}{\partial z_{i n}}=\frac{\partial J_{M}}{\partial\tilde{\pmb{x}}_{n}}\frac{\partial\tilde{\pmb{x}}_{n}}{\partial z_{i n}}\,,\\
\displaystyle\frac{\partial J_{M}}{\partial\tilde{\pmb{x}}_{n}}=-\frac{2}{N}(\pmb{x}_{n}-\tilde{\pmb{x}}_{n})^{\top}\in\mathbb{R}^{1\times D}\,,
\end{array}
$$

(b) 使图 (a) 中距离最小化的向量 $\tilde{\pmb{x}}$ 是其在 $U$ 上的正交投影。$\tilde{\pmb{x}}$ 相对于张成 $U$ 的基向量 $^{b}$ 的坐标是我们需要缩放 $^{b}$ 以“到达” $\tilde{\pmb{x}}$ 的因子。

$$
\frac{\partial\tilde{\mathbf{x}}_{n}}{\partial z_{i n}}\overset{(10.28)}{=}\frac{\partial}{\partial z_{i n}}\left(\sum_{m=1}^{M}z_{m n}\mathbf{b}_{m}\right)=\mathbf{b}_{i}
$$

对于 $i=1,\dots,M$，因此我们得到

$$ \frac{\partial J_{M}}{\partial z_{i n}} \overset{\substack{(10.30b)\\(10.30c)}}{=} -\frac{2}{N} (\pmb{x}_{n} - \tilde{\pmb{x}}_{n})^{\top} \pmb{b}_{i} \overset{(10.28)}{=} -\frac{2}{N} \left( \pmb{x}_{n} - \sum_{m=1}^{M} z_{m n} \pmb{b}_{m} \right)^{\top} \pmb{b}_{i} $$

$$
\stackrel{\mathrm{ONR}}{=}-\frac{2}{N}(\pmb{x}_{n}^{\top}\pmb{b}_{i}-z_{i n}\pmb{b}_{i}^{\top}\pmb{b}_{i})=-\frac{2}{N}(\pmb{x}_{n}^{\top}\pmb{b}_{i}-z_{i n})\,.
$$

${\pmb x}_{n}$ 相对于基向量 $b_{1},\dots,b_{M}$ 的最优投影的坐标是 ${\pmb x}_{n}$ 在主子空间上的正交投影的坐标，因为 $\pmb{b}_{i}^{\top}\pmb{b}_{i}=1$。将这个偏导数设为 0，立即得到最优坐标

$$
\boldsymbol{z}_{i n}=\boldsymbol{\mathbf{\mathit{x}}}_{n}^{\top}\boldsymbol{\mathbf{\mathit{b}}}_{i}=\boldsymbol{\mathbf{\mathit{b}}}_{i}^{\top}\boldsymbol{\mathbf{\mathit{x}}}_{n}
$$

对于 $i\;=\;1,.\,.\,.\,,M$ 和 $n\:=\:1,\cdot\cdot\cdot,N$。这意味着投影 $\tilde{\pmb{x}}_{n}$ 的最优坐标 $z_{i n}$ 是 ${\bf x}_{n}$ 在由 $b_{i}$ 张成的一维子空间上的正交投影的坐标（参见第 3.8 节）。因此：

最优线性投影 $\tilde{\pmb{x}}_{n}$ 是正交投影。$\tilde{\pmb{x}}_{n}$ 相对于基 $(b_{1},.\,.\,.\,,b_{M})$ 的坐标是 ${\pmb x}_{n}$ 在主子空间上的正交投影的坐标。正交投影是给定目标 (10.29) 的最佳线性映射。在 (10.26) 中 ${\pmb x}$ 的坐标 $\zeta_{m}$ 和 (10.27) 中 $\tilde{\pmb{x}}$ 的坐标 $z_{m}$ 对于 $m=1,\cdot\cdot\cdot,M$ 必须相同，因为 $U^{\perp}=\operatorname{span}[b_{M+1},.\,.\,.\,,b_{D}]$ 是 $U=\operatorname{span}[b_{1},.\,.\,.\,,b_{M}]$ 的正交补（参见第 3.6 节）。  

注释 (正交投影与正交规范基向量)。让我们简要回顾第 3.8 节中的正交投影。如果 $\left(\boldsymbol{b}_{1},.\,.\,.\,,\boldsymbol{b}_{D}\right)$ 是 $\mathbb{R}^{D}$ 的正交规范基，则

$$
\tilde{\pmb{x}}=\pmb{b}_{j}(\pmb{b}_{j}^{\top}\pmb{b}_{j})^{-1}\pmb{b}_{j}^{\top}\pmb{x}=\pmb{b}_{j}\pmb{b}_{j}^{\top}\pmb{x}\in\mathbb{R}^{D}
$$

是 $\pmb{x}$ 在由第 $j$ 个基向量张成的子空间上的正交投影，且 $\bar{z_{j}}=\bar{\pmb{b}_{j}^{\top}\pmb{x}}$ 是这个投影相对于张成该子空间的基向量 $b_{j}$ 的坐标，因为 $z_{j}{\pmb b}_{j}=\tilde{\pmb{x}}$。图 10.8(b) 说明了这种情况。  

${\pmb b}_{j}^{\top}{\pmb x}$ 是 $\mathbf{\nabla}_{\mathbf{x}}$ 在由 $b_{j}$ 张成的子空间上的正交投影的坐标。  

更一般地，如果我们希望投影到 $\mathbb{R}^{D}$ 的 $M$ 维子空间上，我们可以通过正交规范基向量 $b_{1},\dots,b_{M}$ 得到 $\pmb{x}$ 在 $M$ 维子空间上的正交投影，即

$$
{\tilde{\pmb{x}}}=B({\underbrace{\pmb{B}^{\top}\pmb{B}}_{=\pmb{I}}})^{-1}\pmb{B}^{\top}\pmb{x}=B\pmb{B}^{\top}\pmb{x}\,，
$$

其中我们定义 $B\,:=\,[b_{1},.\,.\,.\,,b_{M}]\,\in\,\mathbb{R}^{D\times M}$。相对于有序基 $(b_{1},.\,.\,.\,,b_{M})$ 的正交投影是 $\boldsymbol{z}:=\boldsymbol{B}^{\intercal}\boldsymbol{x}$，如第 3.8 节所述。  

我们可以将坐标视为在由 $(b_{1},.\,.\,.\,,b_{M})$ 定义的新坐标系中的投影向量的表示。注意尽管 $\tilde{\textbf{\textit{x}}}\in\mathbb{R}^{D}$，但我们仍需要 $M$ 个坐标 $z_{1},\dots,z_{M}$ 来表示这个向量；相对于基向量 $(b_{M+1},.\,.\,.\,,b_{D})$ 的其他 $D-M$ 个坐标总是 0。  

到目前为止，我们已经证明，对于给定的正交规范基，可以通过正交投影到主子空间上来找到 $\tilde{\pmb{x}}$ 的最优坐标。接下来，我们将确定最佳基是什么。

### 10.3.3 寻找主子空间的基础向量  

为了确定主子空间的基础向量 $b_{1},\dots,b_{M}$，我们重新表述损失函数（10.29），利用我们目前的结果，这将使我们更容易找到基础向量。为了重新表述损失函数，我们利用之前的结果，得到

$$
\tilde{\pmb{x}}_{n}=\sum_{m=1}^{M}z_{m n}\pmb{b}_{m}\overset{(10.32)}{=}\sum_{m=1}^{M}(\pmb{x}_{n}^{\top}\pmb{b}_{m})\pmb{b}_{m}\,.
$$

我们利用点积的对称性，得到

$$
\tilde{\pmb{x}}_{n}=\left(\sum_{m=1}^{M}\pmb{b}_{m}\pmb{b}_{m}^{\top}\right)\pmb{x}_{n}\,.
$$


![](images/f02d16f841b59d191e517fc0975874a8928666c4e7cdab1e1b720bedbddb823c.jpg)
图10.9 正交投影和位移向量。当将数据点 $\pmb{x}_{n}$（蓝色）投影到子空间 $U_{1}$ 时，我们得到 $\tilde{\pmb{x}}_{n}$（橙色）。位移向量 $\tilde{\pmb{x}}_{n}-\pmb{x}_{n}$ 完全位于 $U_{1}$ 的正交补子空间 $U_{2}$ 中。

由于我们通常可以将原始数据点 $\pmb{x}_{n}$ 写成所有基础向量的线性组合，因此有

$$
\begin{array}{l}
\pmb{x}_{n}=\sum_{d=1}^{D}z_{d n}\pmb{b}_{d}\stackrel{\mathrm{(10.32)}}{=}\sum_{d=1}^{D}(\pmb{x}_{n}^{\top}\pmb{b}_{d})\pmb{b}_{d}=\left(\sum_{d=1}^{D}\pmb{b}_{d}\pmb{b}_{d}^{\top}\right)\pmb{x}_{n}\\
=\left(\sum_{m=1}^{M}\pmb{b}_{m}\pmb{b}_{m}^{\top}\right)\pmb{x}_{n}+\left(\sum_{j=M+1}^{D}\pmb{b}_{j}\pmb{b}_{j}^{\top}\right)\pmb{x}_{n}\,,
\end{array}
$$

其中我们将 $D$ 项的和分为 $M$ 项和 $D-M$ 项的和。利用这个结果，我们发现原始数据点 $\pmb{x}_{n}$ 与它的投影之间的位移向量 $\pmb{x}_{n}-\tilde{\pmb{x}}_{n}$，即差向量，是

$$
\begin{array}{r l r}
\pmb{x}_{n}-\tilde{\pmb{x}}_{n}=\left(\sum_{j=M+1}^{D}\pmb{b}_{j}\pmb{b}_{j}^{\top}\right)\pmb{x}_{n}\\
=\sum_{j=M+1}^{D}(\pmb{x}_{n}^{\top}\pmb{b}_{j})\pmb{b}_{j}\,.
\end{array}
$$

这意味着差是数据点在主子空间的正交补子空间上的投影：我们识别公式（10.38a）中的矩阵 $\sum_{j=M+1}^{D}\pmb{b}_{j}\pmb{b}_{j}^{\top}$ 为执行这种投影的投影矩阵。因此，位移向量 $\pmb{x}_{n}-\tilde{\pmb{x}}_{n}$ 位于如图10.9所示的正交补子空间中。

注释（低秩近似）：在公式（10.38a）中，我们看到将 $\pmb{x}_{n}$ 投影到 $\tilde{\pmb{x}}_{n}$ 的投影矩阵为

$$
\sum_{m=1}^{M}\pmb{b}_{m}\pmb{b}_{m}^{\top}=\pmb{B}\pmb{B}^{\top}\,.
$$

由于它是秩为 $M$ 的矩阵，我们看到 $\pmb{B}\pmb{B}^{\top}$ 是对称的。因此，平均平方重构误差可以写为

$$
\begin{array}{l}
\frac{1}{N}\sum_{n=1}^{N}\|\pmb{x}_{n}-\tilde{\pmb{x}}_{n}\|^{2}=\frac{1}{N}\sum_{n=1}^{N}\|\pmb{x}_{n}-\pmb{B}\pmb{B}^{\top}\pmb{x}_{n}\|^{2}\\
=\frac{1}{N}\sum_{n=1}^{N}\|(\pmb{I}-\pmb{B}\pmb{B}^{\top})\pmb{x}_{n}\|^{2}\,.
\end{array}
$$

寻找使原始数据 $\pmb{x}_{n}$ 与其投影 $\tilde{\pmb{x}}_{n}$ 之间的差异最小的正交规范基础向量 $b_{1},\dots,b_{M}$，等价于找到最佳秩 $M$ 近似 $\bar{\pmb{B}}\pmb{B}^{\dagger}$ 的单位矩阵 $\pmb{I}$（见第4.6节）。$\diamondsuit$ 主成分分析（PCA）找到最佳秩 $M$ 近似单位矩阵。

现在我们有了重新表述损失函数（10.29）的所有工具。

$$
J_{M}=\frac{1}{N}\sum_{n=1}^{N}\|\pmb{x}_{n}-\tilde{\pmb{x}}_{n}\|^{2}\overset{(10.38b)}{=}\frac{1}{N}\sum_{n=1}^{N}\left\|\sum_{j=M+1}^{D}(\pmb{b}_{j}^{\top}\pmb{x}_{n})\pmb{b}_{j}\right\|^{2}\,.
$$

我们现在显式地计算平方范数，并利用 $b_{j}$ 形成正交规范基的事实，得到

$$
\begin{array}{l}
J_{M}=\frac{1}{N}\sum_{n=1}^{N}\sum_{j=M+1}^{D}(\pmb{b}_{j}^{\top}\pmb{x}_{n})^{2}=\frac{1}{N}\sum_{n=1}^{N}\sum_{j=M+1}^{D}\pmb{b}_{j}^{\top}\pmb{x}_{n}\pmb{b}_{j}^{\top}\pmb{x}_{n}\\
=\frac{1}{N}\sum_{n=1}^{N}\sum_{j=M+1}^{D}\pmb{b}_{j}^{\top}\pmb{x}_{n}\pmb{x}_{n}^{\top}\pmb{b}_{j}\,,
\end{array}
$$

其中我们在最后一步利用了点积的对称性，写成 $\pmb{b}_{j}^{\top}\pmb{x}_{n}\equiv\pmb{x}_{n}^{\top}\pmb{b}_{j}$。我们现在交换求和符号，得到

$$
\begin{array}{c}
J_{M}=\sum_{j=M+1}^{D}\pmb{b}_{j}^{\top}\left(\frac{1}{N}\sum_{n=1}^{N}\pmb{x}_{n}\pmb{x}_{n}^{\top}\right)\pmb{b}_{j}=\sum_{j=M+1}^{D}\pmb{b}_{j}^{\top}S\pmb{b}_{j}\eqno{(10.43)}\\
=\sum_{j=M+1}^{D}\mathrm{tr}(\pmb{b}_{j}^{\top}S\pmb{b}_{j})=\sum_{j=M+1}^{D}\mathrm{tr}(S\pmb{b}_{j}\pmb{b}_{j}^{\top})=\mathrm{tr}\left(\left(\sum_{j=M+1}^{D}\pmb{b}_{j}\pmb{b}_{j}^{\top}\right)S\right)\,,
\end{array}
$$

其中我们利用迹算子 $\operatorname{tr}(\cdot)$（见（4.18））是线性的且对它的参数进行循环排列不变的性质。由于我们假设数据集是中心化的，即 $\mathbb{E}[\mathcal{X}]=\mathbf{0}$，我们识别 $S$ 为数据的协方差矩阵。由于（10.43b）中的投影矩阵是秩为 $D-M$ 的矩阵，它是秩为 $D-M$ 的矩阵。

公式（10.43a）表明，我们可以将平均平方重构误差等价地表述为数据的协方差矩阵。最小化平均平方重构误差等价于最小化数据协方差矩阵在主子空间正交补子空间上的投影。最小化平均平方重构误差等价于最大化投影数据的方差。

投影到主子空间的正交补子空间。最小化平均平方重构误差因此等价于最小化我们忽略的子空间（即主子空间的正交补子空间）上的数据方差。等价地，我们最大化我们保留在主子空间中的投影的方差，这将投影损失立即与第10.2节讨论的最大方差形式的主成分分析（PCA）联系起来。但这同时也意味着我们将获得与最大方差视角相同的结果。因此，我们省略与第10.2节中给出的推导相同的推导，并在投影视角的背景下总结早期的结果。

当投影到 $M$ 维主子空间时，平均平方重构误差为

$$
J_{M}=\sum_{j=M+1}^{D}\lambda_{j}\,,
$$

其中 $\lambda_{j}$ 是数据协方差矩阵的特征值。因此，为了最小化（10.44），我们需要选择最小的 $D-M$ 个特征值，这表明它们对应的特征向量是主子空间正交补子空间的基础。因此，这意味着主子空间的基础由与数据协方差矩阵最大 $M$ 个特征值对应的特征向量 $b_{1},\dots,b_{M}$ 组成。

## 10.4 特征向量计算与低秩逼近  

在之前的章节中，我们通过数据协方差矩阵的特征值最大的特征向量获得了主子空间的基底  

$$
\begin{array}{c}{\displaystyle S=\frac{1}{N}\sum_{n=1}^{N}\pmb{x}_{n}\pmb{x}_{n}^{\top}=\frac{1}{N}\pmb{X}\pmb{X}^{\top}\,,}\\ {\pmb{X}=[\pmb{x}_{1},.\,.\,.\,,\pmb{x}_{N}]\in\mathbb{R}^{D\times N}\,.}\end{array}
$$
注意 $\boldsymbol{X}$ 是一个 $D\times N$ 矩阵，即它是“典型”数据矩阵的转置（Bishop, 2006; Murphy, 2012）。为了获得 $s$ 的特征值（以及对应的特征向量），我们可以采用两种方法：  

我们进行特征分解（参见第 4.2 节），直接计算 $s$ 的特征值和特征向量。我们使用奇异值分解（参见第 4.5 节）。由于 $s$ 是对称的，并且可以分解为 $\bar{\mathbf{\mathit{X}X^{\top}}}$（忽略因子 $\frac{1}{N}$），$s$ 的特征值是 $\boldsymbol{X}$ 的奇异值的平方。  

使用特征分解或奇异值分解来计算特征向量。  

具体来说，$\boldsymbol{X}$ 的奇异值分解表示为  

$$
\underbrace{X}_{D\times N}=\underbrace{U}_{D\times D}\underbrace{\Sigma}_{D\times N}\underbrace{V^{\top}}_{N\times N},
$$
其中 $U\in\mathbb{R}^{D\times D}$ 和 $V\boldsymbol{V}^{\top}\in\mathbb{R}^{N\times N}$ 是正交矩阵，$\Sigma\in\mathbb{R}^{D\times N}$ 是一个只有奇异值 $\sigma_{ii}\geqslant 0$ 为非零元素的矩阵。因此有  

$$
\boldsymbol{S}=\frac{1}{N}\boldsymbol{X}\boldsymbol{X}^{\top}=\frac{1}{N}\boldsymbol{U}\boldsymbol{\Sigma}\underbrace{\boldsymbol{V}^{\top}\boldsymbol{V}}_{=\boldsymbol{I}_{N}}\boldsymbol{\Sigma}^{\top}\boldsymbol{U}^{\top}=\frac{1}{N}\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{\top}\boldsymbol{U}^{\top}\,.
$$
根据第 4.5 节的结果，$\boldsymbol{U}$ 的列是 $X X^{\top}$（因此也是 $S$）的特征向量。此外，$s$ 的特征值 $\lambda_{d}$ 与 $\boldsymbol{X}$ 的奇异值之间的关系为  

$\boldsymbol{U}$ 的列是 $s$ 的特征向量。  

$$
\lambda_{d}=\frac{\sigma_{d}^{2}}{N}\,.
$$
这种 $s$ 的特征值与 $\boldsymbol{X}$ 的奇异值之间的关系提供了最大方差观点（第 10.2 节）与奇异值分解之间的联系。

### 10.4.1 使用低秩矩阵近似的PCA 

埃克特-杨定理 

为了最大化投影数据的方差（或最小化平均平方重构误差），主成分分析（PCA）选择（10.48）中的 $U$ 的列作为与数据协方差矩阵 $s$ 的最大 $M$ 个特征值相关的特征向量，从而使我们识别 $U$ 为（10.3）中的投影矩阵 $B$，该矩阵将原始数据投影到维度为 $M$ 的低维子空间上。埃克特-杨定理（第4.6节定理4.25）提供了一种直接估计低维表示的方法。考虑最佳秩-$M$逼近

$$
\tilde{\pmb{X}}_{M}:=\operatorname{argmin}_{\operatorname{rk}(\pmb{A})\leqslant M}\|\pmb{X}-\pmb{A}\|_{2}\in\mathbb{R}^{D\times N}
$$

其中 $\left\|\cdot\right\|_{2}$ 是在（4.93）中定义的谱范数。埃克特-杨定理表明 $\tilde{\boldsymbol{X}}_{M}$ 由截断奇异值分解（SVD）得到的前 $M$ 个奇异值给出。换句话说，我们得到

$$
\tilde{\pmb{X}}_{M}=\underbrace{\pmb{U}_{M}}_{D\times M}\underbrace{\pmb{\Sigma}_{M}}_{M\times M}\underbrace{\pmb{V}_{M}^{\top}}_{M\times N}\in\mathbb{R}^{D\times N}
$$

其中 $\pmb{U}_{M}\ :=\ [\pmb{u}_{1},.\,.\,.\,,\pmb{u}_{M}]\ \in\ \mathbb{R}^{D\times M}$ 和 $V_{\scriptscriptstyle M}:=$ $[\pmb{v}_{1},.\,.\,.\,,\pmb{v}_{M}]\in\mathbb{R}^{N\times M}$ 以及一个对角矩阵 $\pmb{\Sigma}_{M}\in\mathbb{R}^{M\times M}$，其对角元素是 $X$ 的最大 $M$ 个奇异值。

### 10.4.2 Practical Aspects  

阿贝尔-鲁菲尼定理 np.linalg.eigh 或 np.linalg.svd  

幂迭代  

在其他需要矩阵分解的基本机器学习方法中，寻找特征值和特征向量也很重要。理论上，如我们在第4.2节中讨论的，我们可以将特征值作为特征多项式的根来求解。然而，对于大于 $4\times4$ 的矩阵，这在实践中是不可能的，因为我们需要找到五次或更高次多项式的根。然而，阿贝尔-鲁菲尼定理（鲁菲尼，1799；阿贝尔，1826）表明，对于五次或更高次的多项式，不存在代数解。因此，在实践中，我们使用迭代方法来求解特征值或奇异值，这些方法在所有现代线性代数软件包中都有实现。

在许多应用中（如本章中介绍的PCA），我们只需要几个特征向量。完全计算分解然后丢弃所有特征值超出前几个的特征向量是浪费资源的。事实上，如果我们只对前几个特征向量（具有最大特征值）感兴趣，那么直接优化这些特征向量的迭代过程在计算上比完整的特征分解（或SVD）更高效。在只需要第一个特征向量的极端情况下，一种简单的方法称为幂迭代非常高效。幂迭代选择一个不在 $s$ 的零空间中的随机向量 $\pmb{x}_{0}$，并遵循迭代

$$
{\pmb x}_{k+1}=\frac{{\pmb S}{\pmb x}_{k}}{\|{\pmb S}{\pmb x}_{k}\|}\,,\quad k=0,1,.\,.\,.\,.
$$

这意味着向量 $\mathbf{\Delta}\mathbf{x}_{k}$ 在每次迭代中乘以 $s$，然后进行归一化，即我们始终有 $\|\pmb{x}_{k}\|=1$。这个向量序列收敛到 $S$ 的最大特征值对应的特征向量。原始的Google PageRank算法（Page等人，1999）使用了这种算法，根据网页的超链接对其进行排名。

## 10.5 高维空间中的主成分分析（PCA）

为了进行PCA，我们需要计算数据的协方差矩阵。在$D$维空间中，数据的协方差矩阵是一个$D \times D$的矩阵。计算这个矩阵的特征值和特征向量在$D$较大时是计算密集型的，因为其计算复杂度与$D$的三次方成正比。因此，如我们之前讨论的，当维度非常高时，PCA将不可行。例如，如果我们的${\pmb x}_{n}$是具有10,000个像素的图像（例如，$100 \times 100$），我们需要计算一个$10{,}000 \times 10{,}000$的协方差矩阵的特征分解。在接下来的部分中，我们将为数据点远少于维度的情况提供一个解决方案，即$N \ll D$。

假设我们有一个中心化数据集$\pmb{x}_{1}, \dots, \pmb{x}_{N}$，$\pmb{x}_{n} \in \mathbb{R}^{D}$。那么数据的协方差矩阵可以表示为

$$
\boldsymbol{S} = \frac{1}{N} \boldsymbol{X} \boldsymbol{X}^{\intercal} \in \mathbb{R}^{D \times D}\,,
$$

其中$\boldsymbol{X} = [\pmb{x}_{1}, \dots, \pmb{x}_{N}]$是一个$D \times N$的矩阵，其列是数据点。

现在假设$N \ll D$，即数据点的数量小于数据的维度。如果没有重复的数据点，协方差矩阵$\boldsymbol{S}$的秩为$N$，因此它有$D - N + 1$个特征值为0。直观上，这意味着存在一些冗余。在接下来的部分中，我们将利用这一点，将$D \times D$的协方差矩阵转换为一个$N \times N$的协方差矩阵，其特征值均为正。

在PCA中，我们得到了特征向量方程

$$
\boldsymbol{S} \pmb{b}_{m} = \lambda_{m} \pmb{b}_{m}\,, \quad m = 1, \dots, M\,,
$$

其中$\pmb{b}_{m}$是主子空间的基向量。让我们稍微重写这个方程：根据(10.53)中的定义，我们得到

$$
\pmb{S} \pmb{b}_{m} = \frac{1}{N} \pmb{X} \pmb{X}^{\top} \pmb{b}_{m} = \lambda_{m} \pmb{b}_{m}\,.
$$

我们现在从左侧乘以$\pmb{X}^{\top} \in \mathbb{R}^{N \times D}$，得到

$$
\frac{1}{N} \underbrace{\pmb{X}^{\top} \pmb{X}}_{N \times N} \underbrace{\pmb{X}^{\top} \pmb{b}_{m}}_{=: \pmb{c}_{m}} = \lambda_{m} \pmb{X}^{\top} \pmb{b}_{m} \iff \frac{1}{N} \pmb{X}^{\top} \pmb{X} \pmb{c}_{m} = \lambda_{m} \pmb{c}_{m}\,,
$$

我们得到一个新的特征向量/特征值方程：$\lambda_{m}$仍然是特征值，这证实了我们在第4.5.3节中的结果，即$\pmb{X} \pmb{X}^{\top}$的非零特征值等于$\pmb{X}^{\top} \pmb{X}$的非零特征值。我们得到矩阵$\mathbf{\Omega}_{\frac{1}{N}} \pmb{X}^{\top} \pmb{X} \in \mathbf{\bar{R}}^{N \times N}$与$\lambda_{m}$相关的特征向量为$\pmb{c}_{m} := \pmb{X}^{\top} \pmb{b}_{m}$。假设没有重复的数据点，这个矩阵的秩为$N$且可逆。这也意味着$\frac{1}{N} \bar{\pmb{X}}^{\top} \pmb{X}$的非零特征值与数据的协方差矩阵$\pmb{S}$相同。但这是一个$N \times N$的矩阵，因此我们可以比原始$D \times D$的协方差矩阵更高效地计算特征值和特征向量。

现在我们已经得到了$\frac{1}{N} \pmb{X}^{\top} \pmb{X}$的特征向量，接下来我们要恢复原始的特征向量，因为PCA仍然需要它们。目前，我们知道$\frac{1}{N} \pmb{X}^{\top} \pmb{X}$的特征向量。如果我们左乘我们的特征值/特征向量方程$\pmb{X}$，我们得到

$$
\frac{1}{N} \pmb{X} \pmb{X}^{\top} \pmb{c}_{m} = \lambda_{m} \pmb{X} \pmb{c}_{m}
$$

并再次恢复数据的协方差矩阵。这现在也意味着我们恢复了$\pmb{X} \pmb{c}_{m}$作为$\pmb{S}$的特征向量。

注释。如果我们想应用我们在第10.6节中讨论的PCA算法，我们需要将$\pmb{S}$的特征向量$\pmb{X} \pmb{c}_{m}$归一化，使其范数为1。$\diamondsuit$

## 10.6 PCA 实践中的关键步骤  

在下面的内容中，我们将通过一个运行示例来逐一介绍 PCA 的步骤，该示例总结在图 10.11 中。我们给定一个二维数据集（图 10.11(a)），并希望通过 PCA 将其投影到一维子空间中。  

1. 均值减去 我们首先通过计算数据集的均值 $\pmb{\mu}$ 并从每个数据点中减去它来中心化数据。这确保了数据集的均值为 0（图 10.11(b)）。均值减去不是严格必要的，但可以减少数值问题的风险。

2. 标准化 将数据点除以数据集的每个维度 $d=1,\cdot\cdot\cdot,D$ 的标准差 $\sigma_{d}$。现在数据是无量纲的，并且在每个轴上的方差为 1，这在图 10.11(c) 中由两个箭头表示。这一步完成了数据的标准化。

3. 协方差矩阵的特征分解 计算数据的协方差矩阵及其特征值和对应的特征向量。由于协方差矩阵是对称的，谱定理（定理 4.15）表明我们可以找到一组特征向量的正交规范基。在图 10.11(d) 中，特征向量按对应的特征值的大小进行了缩放。较长的向量覆盖了主子空间，我们将其表示为 $U$。数据的协方差矩阵由椭圆表示。

4. 投影 我们可以将任意数据点 $\pmb{x}_{*}\in\mathbb{R}^{D}$ 投影到主子空间中：为了正确地进行投影，我们需要使用第 $d$ 维训练数据的均值 $\mu_{d}$ 和标准差 $\sigma_{d}$ 对 $\mathbf{x}_{\ast}$ 进行标准化，使得

$$
x_{*}^{(d)}\gets\frac{x_{*}^{(d)}-\mu_{d}}{\sigma_{d}}\,,\quad d=1,.\,.\,.\,,D\,,
$$
其中 $x_{*}^{(d)}$ 是 $\pmb{x}_{*}$ 的第 $d$ 个分量。我们得到的投影为

$$
\tilde{\pmb{x}}_{*}=\pmb{B B}^{\top}\pmb{x}_{*}
$$
在主子空间基下的坐标为

$$
\boldsymbol{z}_{\ast}=\boldsymbol{B}^{\intercal}\boldsymbol{x}_{\ast}
$$
这里，$B$ 是一个矩阵，其列包含数据协方差矩阵最大特征值对应的特征向量。PCA 返回的是坐标 (10.60)，而不是投影 $\pmb{x}_{*}$。标准化数据集后，(10.59) 只给出了标准化数据集中的投影。为了在原始数据空间中获得投影（即在标准化之前），我们需要撤销标准化 (10.58)，乘以标准差并加上均值，以便我们得到

$$
\tilde{x}_{*}^{(d)}\leftarrow\tilde{x}_{*}^{(d)}\sigma_{d}+\mu_{d}\,,\quad d=1,.\,.\,.\,,D\,.
$$
图 10.11(f) 说明了在原始数据空间中的投影。![](images/b13696e7f794dd2d9427cb1b355201dbc6fa89c99fa250874320c417edda5254.jpg)

### 示例 10.4 (MNIST 数字：重构)  

在下面的内容中，我们将应用主成分分析（PCA）到包含 60,000 个手写数字 0 到 9 的 MNIST 数字数据集上。每个数字是一个大小为 $28 \times 28$ 的图像，即它包含 784 个像素，因此我们可以将数据集中的每个图像视为一个向量 $\pmb{x} \in \mathbb{R}^{784}$。图 10.3 显示了这些数字的一些示例。  

![](images/834ef79ff1aa577e686271ed716ddc1e31afabb498342ea2748f37e5811e8445.jpg)  
图 10.12 增加主成分数量对重构效果的影响。  

为了说明目的，我们对 MNIST 数字数据集的一个子集应用 PCA，并专注于数字“8”。我们使用了 5,389 张数字“8”的训练图像，并按照本章所述确定了主子空间。然后，我们使用学到的投影矩阵重构了一组测试图像，如图 10.12 所示。图 10.12 的第一行显示了测试集中的四张原始数字。接下来的行分别显示了使用 1 维、10 维、100 维和 500 维主子空间重构的这些数字。我们看到，即使使用 1 维主子空间，我们也能得到相当不错的原始数字重构，但这些重构是模糊且通用的。随着主成分数量的增加（PCs），重构变得越来越清晰，更多的细节被考虑进去。使用 500 个主成分时，我们几乎可以完全重构训练数据，其中包含数字 8（一些边界像素在整个数据集中没有变化，因为它们总是黑色的）。  

图 10.13 显示了平均平方重构误差，它是  
$$
\frac{1}{N}\sum_{n=1}^{N}\left\|\pmb{x}_{n}-\tilde{\pmb{x}}_{n}\right\|^{2}=\sum_{i=M+1}^{D}\lambda_{i}\,，
$$  
作为主成分数量 $M$ 的函数。我们可以看到，主成分的重要性迅速下降，增加更多的主成分只能获得微小的改进。这与图 10.5 中的观察结果完全一致，在该图中我们发现，投影数据的大部分方差仅由少数几个主成分捕获。大约使用 550 个主成分时，我们可以几乎完全重构包含数字 8 的训练数据（一些边界像素在整个数据集中没有变化，因为它们总是黑色的）。  

![](images/9d7ed24e047c2b50cf828545fa93a0232faf935e6f16de0f9d389ebe88ae8a2c.jpg)  
图 10.13 平均平方重构误差作为主成分数量的函数。平均平方重构误差是主子空间正交补中的特征值之和。

## 10.7 隐变量视角  

在之前的章节中，我们没有使用概率模型的概念，而是通过最大方差和投影视角推导了PCA。一方面，这种方法可能令人满意，因为它让我们可以避开概率论带来的所有数学难题，但另一方面，概率模型会给我们更多的灵活性和有用的见解。更具体地说，概率模型会  

提供一个似然函数，我们可以明确地处理噪声观测（我们甚至没有讨论过这一点） 允许我们通过边缘似然进行贝叶斯模型比较，如第8.6节所述 将PCA视为生成模型，从而可以模拟新的数据 允许我们与相关算法建立直接联系 处理随机缺失的数据维度，通过应用贝叶斯定理 处理新数据点的新颖性 提供一个扩展模型的原理性方法，例如混合PCA模型 使我们之前推导的PCA作为特殊情况 通过边缘化模型参数实现完全贝叶斯处理 通过引入一个连续值的隐变量 $\boldsymbol{z}\in\mathbb{R}^{M}$，可以将PCA表述为概率隐变量模型。Tipping和Bishop（1999）提出了这种隐变量模型，称为概率PCA（PPCA）。PPCA解决了上述大部分问题，我们通过最大化投影空间中的方差或最小化重构误差所获得的PCA解，是无噪声情况下最大似然估计的特殊情况。

### 10.7.1 生成过程与概率模型  

在 PPCA 中，我们明确地写下了线性降维的概率模型。为此，我们假设一个连续的潜在变量 $\boldsymbol{z} \in \mathbb{R}^{M}$ 有一个标准正态先验 $p(z) = \mathcal{N}(\mathbf{0}, I)$，并且潜在变量与观测数据 $\pmb{x}$ 之间存在线性关系，其中

$$
\pmb{x} = B z + \pmb{\mu} + \pmb{\epsilon} \in \mathbb{R}^{D}\,,
$$

其中 $\pmb{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^{2} \pmb{I})$ 是高斯观测噪声，$B \in \mathbb{R}^{D \times M}$ 和 $\pmb{\mu} \in \mathbb{R}^{D}$ 描述了从潜在变量到观测变量的线性/仿射映射。因此，PPCA 通过以下方式将潜在变量和观测变量联系起来：

$$
p(\pmb{x} | z, B, \pmb{\mu}, \sigma^{2}) = \mathcal{N}\big(\pmb{x} | B z + \pmb{\mu}, \sigma^{2} \pmb{I}\big)\,.
$$

总体而言，PPCA 引入了以下生成过程：

$$
\begin{array}{c}
z_{n} \sim \mathcal{N}\big(z | \mathbf{0}, I\big) \\
\pmb{x}_{n} | z_{n} \sim \mathcal{N}\big(\pmb{x} | B z_{n} + \pmb{\mu}, \sigma^{2} \pmb{I}\big)
\end{array}
$$

祖先采样 为了生成给定模型参数的典型数据点，我们遵循祖先采样方案：首先从 $p(z)$ 中采样一个潜在变量 $z_{n}$。然后，使用 $z_{n}$ 在 (10.64) 中来条件化采样数据点，即 $\pmb{x}_{n} \sim p(\pmb{x} | z_{n}, B, \pmb{\mu}, \sigma^{2})$。

这个生成过程允许我们写出概率模型（即所有随机变量的联合分布；参见第 8.4 节）：

$$
p(\pmb{x}, z | \pmb{B}, \pmb{\mu}, \sigma^{2}) = p(\pmb{x} | z, \pmb{B}, \pmb{\mu}, \sigma^{2}) p(z)\,,
$$

这立即导致了图 10.14 中的图形模型，使用第 8.5 节的结果。

![](images/23b3e5f7743acd733f0dbde336a2dd2d7a78fbf14f9b1ea942792e7ae3e5ea5b.jpg)

注释。注意连接潜在变量 $\boldsymbol{z}$ 和观测数据 $\pmb{x}$ 的箭头方向：箭头指向 $\pmb{x}$，这意味着 PPCA 模型假设低维潜在原因 $\boldsymbol{z}$ 用于高维观测 $\pmb{x}$。最终，我们显然对给定观测结果的 $\boldsymbol{z}$ 有所了解感兴趣。为了达到这个目的，我们将应用贝叶斯推断来“反转”箭头隐式地从观测结果到潜在变量。$\diamondsuit$

示例 10.5 (使用潜在变量生成新数据)

![](images/107f0604c4020e0cbf92434a367480a50aa0d9f91a871576a75f8b9aa1371850.jpg)

图 10.15 生成新的 MNIST 数字。潜在变量 $\mathscr{Z}$ 可以用来生成新数据 $\tilde{\pmb{x}} = \pmb{B z}$。如果我们保持接近训练数据，生成的数据将更加逼真。

图 10.15 显示了使用二维主子空间（蓝色点）通过 PCA 找到的 MNIST 数字 $\alpha 8^\circ$ 的潜在坐标。我们可以查询这个潜在空间中的任意向量 $z_{\ast}$ 并生成一个类似于数字 $\alpha 8^\circ$ 的图像 $\tilde{\pmb{x}}_{\ast} = \pmb{B z}_{\ast}$。我们展示了八个这样的生成图像及其相应的潜在空间表示。根据我们在潜在空间中查询的位置，生成的图像看起来不同（形状、旋转、大小等）。如果我们远离训练数据查询，我们会看到越来越多的伪影，例如左上角和右上角的数字。请注意，这些生成图像的固有维度仅为两个。

### 10.7.2 似然与联合分布  

利用第6章的结果，通过积分消除潜在变量 $_z$ （参见第8.4.3节），我们得到该概率模型的似然为  

$$
\begin{align}
p(\boldsymbol{x} \mid \boldsymbol{B}, \mu, \sigma^2) &= \int p(\boldsymbol{x} \mid \boldsymbol{z}, \boldsymbol{B}, \mu, \sigma^2) p(\boldsymbol{z}) \, \mathrm{d}\boldsymbol{z} \\
&= \int \mathcal{N}\big(\boldsymbol{x} \mid \boldsymbol{B}\boldsymbol{z} + \mu, \sigma^2 \boldsymbol{I}\big) \mathcal{N}\big(\boldsymbol{z} \mid \mathbf{0}, \boldsymbol{I}\big) \, \mathrm{d}\boldsymbol{z}
\end{align}
$$

根据第6.5节，上述积分的解是一个均值为  

$$
\mathbb{E}_{x}[x]=\mathbb{E}_{z}[B z+\pmb{\mu}]+\mathbb{E}_{\pmb{\epsilon}}[\pmb{\epsilon}]=\pmb{\mu}
$$  

的高斯分布，并且协方差矩阵为  

$$
\begin{array}{r l}
&{\mathbb{V}[{\pmb x}]=\mathbb{V}_{z}[{\pmb B}{\pmb z}+{\pmb\mu}]+\mathbb{V}_{{\pmb\epsilon}}[{\pmb\epsilon}]=\mathbb{V}_{z}[{\pmb B}{\pmb z}]+\sigma^{2}{\pmb I}}\\
&{\quad\quad={\pmb B}\mathbb{V}_{z}[{\pmb z}]{\pmb B}^{\top}+\sigma^{2}{\pmb I}={\pmb B}{\pmb B}^{\top}+\sigma^{2}{\pmb I}\,.}
\end{array}
$$  

式（10.68b）中的似然可以用于模型参数的最大似然或MAP估计。  

注释。我们不能使用式（10.64）中的条件分布进行最大似然估计，因为它仍然依赖于潜在变量。我们所需的用于最大似然（或MAP）估计的似然函数应仅依赖于数据 $\pmb{x}$ 和模型参数，而不应依赖于潜在变量。$\diamondsuit$  

根据第6.5节，一个高斯随机变量 $_z$ 和它的线性/仿射变换 $\pmb{x}=\pmb{B z}$ 具有联合高斯分布。我们已经知道边缘分布 $p(z)=\mathcal{N}\big(z\,|\,\mathbf{0},\,I\big)$ 和 $p(\pmb{x})=\mathcal{N}(\mathbf{\boldsymbol{x}}\,|\,\mathbf{\boldsymbol{\mu}},\,{\boldsymbol{B}}{\boldsymbol{B}}^{\top}+\sigma^{2}{\boldsymbol{I}})$。缺失的协方差矩阵为  

$$
\operatorname{Cov}[\pmb{x},\pmb{z}]=\operatorname{Cov}_{z}[\pmb{B}z+\pmb{\mu}]=\pmb{B}\operatorname{Cov}_{z}[z,\pmb{z}]=\pmb{B}\,.
$$  

因此，PPCA的概率模型，即潜在变量和观测变量的联合分布，明确给出为  

$$
\begin{array}{c c}
{{p(\pmb{x},z\,|\,\pmb{B},\pmb{\mu},\sigma^{2})=\mathcal{N}\left(\left[\pmb{x}\right]\,\left|\!\begin{array}{c}{{\pmb{\mu}}}\\ {{\bf{0}}}\end{array}\!\right|,\,\left[\begin{array}{c c}{{\pmb{B}B^{\top}+\sigma^{2}{\pmb{I}}}}&{{\pmb{B}}}\\ {{\pmb{B}^{\top}}}&{{\pmb{I}}}\end{array}\right]\right)\,,}}
\end{array}
$$  

其中均值向量长度为 $D+M$，协方差矩阵大小为 $(D+M)\times(D+M)$。

### 10.7.3 后验分布  

在公式 (10.72) 中的联合高斯分布 $p(\pmb{x},z\,|\,B,\pmb{\mu},\sigma^{2})$ 允许我们通过应用第 6.5.1 节中的高斯条件化规则立即确定后验分布 $p(z\mid x)$。给定观测值 $\pmb{x}$ 的潜在变量的后验分布为  

$$
\begin{array}{r}{p(\boldsymbol{z}\,|\,\boldsymbol{x})=\mathcal{N}\big(\boldsymbol{z}\,|\,\boldsymbol{m},\,\boldsymbol{C}\big)\,,\qquad\qquad}\\ {\boldsymbol{m}=\boldsymbol{B}^{\top}(\boldsymbol{B}\boldsymbol{B}^{\top}+\sigma^{2}\boldsymbol{I})^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\,,}\\ {\boldsymbol{C}=\boldsymbol{I}-\boldsymbol{B}^{\top}(\boldsymbol{B}\boldsymbol{B}^{\top}+\sigma^{2}\boldsymbol{I})^{-1}\boldsymbol{B}\,.}\end{array}
$$

注意，后验协方差不依赖于观测数据 $\pmb{x}$。对于数据空间中的新观测值 $\pmb{x}_{*}$，我们使用 (10.73) 来确定对应的潜在变量 $z_{*}$ 的后验分布。协方差矩阵 $\boldsymbol{C}$ 允许我们评估嵌入的置信度。协方差矩阵 $\boldsymbol{C}$ 的行列式较小（衡量体积）时，说明潜在嵌入 $z_{\ast}$ 较为确定。如果我们得到一个后验分布 $p(z_{*}\,|\,\pmb{x}_{*})$ 有较大方差，我们可能会遇到异常值。然而，我们可以通过探索这个后验分布来理解在该后验下哪些其他数据点 $\pmb{x}$ 是合理的。为此，我们利用 PPCA 的生成过程，通过生成在该后验下合理的新的数据来探索潜在变量的后验分布：  

1. 从潜在变量的后验分布 (10.73) 中采样一个潜在变量 $z_{*}\sim p(z\mid\pmb{x}_{*})$。

 2. 从 (10.64) 中采样一个重构向量 $\tilde{\pmb{x}}_{\ast}\sim p(\pmb{x}\,|\,z_{\ast},B,\pmb{\mu},\sigma^{2})$。

如果我们多次重复这个过程，我们可以探索潜在变量 $z_{*}$ 的后验分布 (10.73) 及其对观测数据的影响。采样过程实际上假设了在后验分布下合理的数据。

# 10.8 进一步阅读  

我们从两个角度推导了主成分分析（PCA）：（a）最大化投影空间中的方差；（b）最小化平均重构误差。然而，PCA还可以从不同的角度进行解释。让我们回顾一下我们所做的事情：我们使用高维数据 $\boldsymbol{x} \in \mathbb{R}^{D}$ 并使用矩阵 $B^{\top}$ 找到一个低维表示 $\boldsymbol{z} \in \mathbb{R}^{M}$。矩阵 $B$ 的列是与数据协方差矩阵 $S$ 中最大特征值相关的特征向量。一旦我们有了低维表示 $\boldsymbol{z}$，我们可以在原始数据空间中得到它的高维版本（$\boldsymbol{x} \approx \bar{\boldsymbol{x}} = \bar{\boldsymbol{B}} \boldsymbol{z} = \boldsymbol{B} \boldsymbol{B}^{\top} \boldsymbol{x} \in \mathbb{R}^{D}$），其中 $\boldsymbol{B} \boldsymbol{B}^{\top}$ 是一个投影矩阵。

我们也可以将PCA视为一个线性自编码器，如图10.16所示。自编码器将数据 $\boldsymbol{x}_{n} \in \mathbb{R}^{D}$ 编码为一个代码 $\boldsymbol{z}_{n} \in \mathbb{R}^{M}$，然后再将其解码为一个与 $\boldsymbol{x}_{n}$ 相似的 $\tilde{\boldsymbol{x}}_{n}$。从数据到代码的映射称为编码器，从代码到原始数据空间的映射称为解码器。如果我们考虑线性映射，解码器

![](images/48739c26737f7e7e818ec2004cc2faacdeb511a7c6a8608d418d45768824517e.jpg)  
图10.16 PCA可以被视为一个线性自编码器。它将高维数据 $\pmb{x}$ 编码为一个低维表示（代码）$\boldsymbol{z} \in \mathbb{R}^{M}$，并通过解码器将其解码。解码后的向量 $\tilde{\boldsymbol{x}}$ 是原始数据 $\pmb{x}$ 在 $M$ 维空间上的正交投影。

通过最小化数据 $\boldsymbol{x}_{n}$ 与其重构 $\tilde{\boldsymbol{x}}_{n} = \mathcal{B} \boldsymbol{z}_{n}$ 之间的平均平方误差，我们得到

$$
\frac{1}{N}\sum_{n=1}^{N}\|\boldsymbol{x}_{n}-\tilde{\boldsymbol{x}}_{n}\|^{2}=\frac{1}{N}\sum_{n=1}^{N}\left\|\boldsymbol{x}_{n}-\boldsymbol{B B}^{\top}\boldsymbol{x}_{n}\right\|^{2}\,.
$$
编码器 解码器 生成器

代码是原始数据的压缩版本。

这意味着我们最终得到与（10.29）中讨论的目标函数相同，因此当我们最小化自编码损失时可以得到PCA的解。如果我们用非线性映射替换PCA的线性映射，我们将得到一个非线性自编码器。一个典型的例子是深度自编码器，其中线性函数被深度神经网络所取代。在这个背景下，编码器也被称为识别网络或推理网络，而解码器也被称为生成器。

PCA的另一种解释与信息论有关。我们可以将代码视为原始数据点的一个较小或压缩版本。当我们使用代码重构原始数据时，我们不会得到原始数据点的精确副本，而是一个稍微失真或有噪声的版本。这意味着我们的压缩是“有损”的。直观上，我们希望最大化原始数据和低维代码之间的相关性。更正式地说，这与互信息有关。通过最大化互信息，我们可以在信息论的核心概念（MacKay, 2003）中得到与（10.3）节讨论的PCA相同的解。

在我们的讨论中，假设模型的参数，即 $B, \mu_{.}$ 和似然参数 $\sigma^{2}$ 是已知的。Tipping 和 Bishop (1999) 描述了如何在 PPCA 情况下推导这些参数的最大似然估计（注意我们在本章中使用了不同的符号）。当将 $D$ 维数据投影到 $M$ 维子空间时，最大似然参数为

$$
\begin{array}{l}{\displaystyle\boldsymbol{\mu}_{\mathrm{ML}}=\frac{1}{N}\sum_{n=1}^{N}\boldsymbol{x}_{n}\,,}\\ {\displaystyle\boldsymbol{B}_{\mathrm{ML}}=\boldsymbol{T}(\boldsymbol{\Lambda}-\sigma^{2}\boldsymbol{I})^{\frac{1}{2}}\boldsymbol{R}\,,}\\ {\displaystyle\sigma_{\mathrm{ML}}^{2}=\frac{1}{D-M}\sum_{j=M+1}^{D}\lambda_{j}\,,}\end{array}
$$
其中 $\boldsymbol{T} \in \mathbb{R}^{D \times M}$ 是数据协方差矩阵的主成分向量，$\boldsymbol{\Lambda}=\operatorname{diag}(\lambda_{1},.\,.\,.\,,\lambda_{M}) \in \mathbb{R}^{M \times M}$ 是一个对角矩阵，其对角线上的元素是与主轴相关的特征值，$\boldsymbol{R} \in \mathbb{R}^{M \times M}$ 是一个任意的正交矩阵。最大似然解 $\boldsymbol{B}_{\mathrm{ML}}$ 在任意正交变换下是唯一的，例如，我们可以将 $\boldsymbol{B}_{\mathrm{ML}}$ 右乘任何旋转矩阵 $R$，使得（10.78）实际上是奇异值分解（见第4.5节）。Tipping 和 Bishop (1999) 给出了证明的大纲。

给定（10.77）的最大似然估计 $\boldsymbol{\mu}$ 是数据的样本均值。给定（10.79）的最大似然估计 $\sigma^{2}$ 是在主子空间正交补中的平均方差，即我们无法用前 $M$ 个主成分捕捉到的平均剩余方差被视为观测噪声。

（10.78）中的矩阵 $\boldsymbol{\Lambda}-\sigma^{2}\boldsymbol{I}$ 保证是半正定的，因为数据协方差矩阵的最小特征值被噪声方差 $\sigma^{2}$ 从下界限制。

在噪声极限 $\sigma \rightarrow 0$ 下，PPCA 和 PCA 提供相同的解：由于数据协方差矩阵 $S$ 是对称的，它可以对角化（见第4.4节），即存在一个矩阵 $\boldsymbol{T}$ 使得

$$
\boldsymbol{S}=\boldsymbol{T}\boldsymbol{\Lambda}\boldsymbol{T}^{-1}\,.
$$
在 PPCA 模型中，数据协方差矩阵是似然函数 $p(\boldsymbol{x} \lrcorner \boldsymbol{B}, \boldsymbol{\mu}, \sigma^{2})$ 的协方差矩阵，即 $\boldsymbol{B}\boldsymbol{B}^{\intercal} + \sigma^{2}\boldsymbol{I}$，见（10.70b）。对于 $\sigma \rightarrow 0$，我们得到 $\boldsymbol{B B}^{\top}$，因此数据协方差必须等于 PCA 数据协方差（及其在（10.80）中的因子分解），即

$$
\mathrm{Cov}[\mathcal{X}]=\boldsymbol{T}\boldsymbol{\Lambda}\boldsymbol{T}^{-1}=\boldsymbol{B}\boldsymbol{B}^{\top} \iff \boldsymbol{B}=\boldsymbol{T}\boldsymbol{\Lambda}^{\frac{1}{2}}\boldsymbol{R}\,,
$$
即我们得到（10.78）中的最大似然估计 $\sigma = 0$。从（10.78）和（10.80）可以看出，（P）PCA 对数据协方差矩阵进行了分解。

在数据按顺序到达的流式设置中，建议使用迭代期望最大化（EM）算法进行最大似然估计（Roweis, 1998）。

为了确定潜在变量的维度（代码的长度，我们投影数据到的低维子空间的维度），Gavish 和 Donoho (2014) 建议，如果我们可以估计数据的噪声方差 $\sigma^{2}$，则应丢弃所有小于 $\frac{4\sigma \sqrt{D}}{\sqrt{3}}$ 的奇异值。或者，我们可以使用嵌套交叉验证（第8.6.1节）或贝叶斯模型选择准则（第8.6.2节）来确定数据的固有维度的良好估计（Minka, 2001b）。

贝叶斯 PCA

因子分析

过于灵活的似然性能够解释的不仅仅是噪声。

独立成分分析 ICA

盲源分离

类似于我们在第9章中对线性回归的讨论，我们可以对模型参数放置先验分布并将其积分出去。通过这样做，我们（a）避免了参数的点估计及其带来的问题（见第8.6节），并且（b）允许自动选择适当的潜在空间维度 $M$。在贝叶斯 PCA 中，Bishop (1999) 提出了对模型参数放置先验 $p(\mu, B, \sigma^{2})$。生成过程允许我们积分模型参数而不是条件化于它们，从而解决了过拟合问题。由于这种积分是不可解析的，Bishop (1999) 建议使用近似推理方法，如MCMC或变分推理。我们参考Gilks等人（1996）和Blei等人（2017）的工作以获取这些近似推理技术的更多细节。

因子分析（FA）（Spearman, 1904；Bartholomew等人, 2011）允许每个观测维度 $\sigma_{d}^{2}$ 有不同的方差。这意味着FA比PPCA提供了更多的灵活性，但仍强制数据由模型参数 $B, \boldsymbol{\mu}$ 解释。然而，FA不再允许闭式最大似然解，因此我们需要使用迭代方案，如期望最大化算法来估计模型参数。与PPCA相比，FA在对数据进行缩放时不会改变，但在对数据进行旋转时会返回不同的解。

与PCA密切相关的一个算法是独立成分分析（ICA）（Hyvarinen等人, 2001）。我们再次从潜在变量视角出发，即 $p(\boldsymbol{x}_{n} \lrcorner \boldsymbol{z}_{n}) = \mathcal{N}(\boldsymbol{x}_{n} \lrcorner B \boldsymbol{z}_{n} + \boldsymbol{\mu}, \sigma^{2} \boldsymbol{I})$，我们现在将 $z_{n}$ 的先验分布改为非高斯分布。ICA可以用于盲源分离。想象你在繁忙的火车站有很多人在说话。你的耳朵扮演着麦克风的角色，它们线性地混合了火车站中的不同语音信号。盲源分离的目标是识别混合信号的组成部分。如我们在最大似然估计PPCA的上下文中讨论的那样，原始PCA解对任何旋转都是不变的。因此，PCA可以识别信号所在的最佳低维子空间，但不能识别信号本身（Murphy, 2012）。ICA通过将潜在源的先验分布 $p(z)$ 修改为非高斯先验 $p(z)$ 来解决这个问题。我们参考Hyvarinen等人（2001）和Murphy（2012）的书籍以获取更多关于ICA的细节。

PCA、因子分析和ICA是三种使用线性模型进行降维的例子。Cunningham和Ghahramani (2015) 提供了线性降维的更广泛综述。

我们讨论的（P）PCA模型允许几种重要的扩展。在第10.5节中，我们解释了当输入维度 $D$ 显著大于数据点数 $N$ 时如何进行PCA。通过利用PCA可以通过计算（许多）内积来执行的洞察，这一想法可以推向极端，考虑无限维特征。核技巧是核PCA的基础，允许我们隐式地计算无限维特征之间的内积（Schölkopf等人, 1998；Schölkopf和Smola, 2002）。

核技巧 核PCA

存在从PCA派生的非线性降维技术（Burges (2010) 提供了很好的综述）。我们之前在本节中讨论的PCA的自编码器视角可以用于将PCA视为深度自编码器的一个特例。在深度自编码器中，编码器和解码器都由多层前馈神经网络表示，它们本身是非线性映射。如果我们设置这些神经网络中的激活函数为恒等函数，模型将等同于PCA。另一种非线性降维方法是Lawrence (2005) 提出的高斯过程潜在变量模型（GP-LVM）。GP-LVM 从我们用于推导PPCA的潜在变量视角开始，将潜在变量 $\boldsymbol{z}$ 和观测值 $\boldsymbol{x}$ 之间的线性关系替换为高斯过程（GP）。与PPCA不同，GP-LVM 不估计映射的参数，而是通过积分出模型参数并为潜在变量 $\boldsymbol{z}$ 作出点估计来处理。类似于贝叶斯PCA，Titsias和Lawrence (2010) 提出的贝叶斯GP-LVM 维持潜在变量 $\boldsymbol{z}$ 的分布，并使用近似推理将其积分出去。