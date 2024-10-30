# 第九章 线性回归

在接下来的部分中，我们将应用第2章、第5章、第6章和第7章中的数学概念来解决线性回归（曲线拟合）问题。在回归问题中，我们的目标是找到一个映射输入 $\pmb{x}\in\mathbb{R}^{D}$ 到对应函数值 $f(\pmb{x})\in\mathbb{R}$ 的函数 $f$。我们假设给定一组训练输入 ${\pmb x}_{n}$ 和相应的噪声观测值 $y_{n}=f(\pmb{x}_{n}){+}{\epsilon}_{n}$，其中 $\epsilon$ 是一个独立同分布的随机变量，描述了测量/观察噪声以及潜在未建模的过程（在本章中我们将不再进一步考虑这些过程）。在整章中，我们假设噪声是零均值高斯噪声。我们的任务是找到一个不仅能够拟合训练数据，而且在预测非训练数据输入处的函数值时也表现良好的函数（见第8章）。图9.1给出了这种回归问题的一个示例。典型的回归设置如图9.1(a)所示：对于某些输入值 $x_{n}$，我们观察到（有噪声的）函数值 $y_{n}=f(x_{n})+\epsilon$。任务是推断生成数据并很好地推广到新输入位置的函数 $f$。可能的解决方案如图9.1(b)所示，在图中我们也展示了集中在函数值 $f(x)$ 处的三个分布，这些分布代表了数据中的噪声。

回归问题是机器学习中的基本问题，回归问题出现在广泛的研究领域和应用中，包括时间序列分析（例如系统识别）、控制和机器人技术（例如强化学习、前馈/逆模型学习）、优化（例如线搜索、全局优化）以及深度学习应用（例如计算机游戏、语音转文字翻译、图像识别、自动视频标注）。回归也是分类算法的关键组成部分。寻找回归函数需要解决各种问题，包括以下问题：

通常情况下，噪声类型也可以是一个“模型选择”，但我们在本章中将噪声固定为高斯噪声。

回归函数的模型（类型）和参数化选择。给定一个数据集，哪些函数类（例如多项式）是建模数据的良好候选者，我们应该选择哪种特定参数化（例如多项式的次数）？模型选择，如第8.6节讨论的那样，使我们能够比较不同的模型，找到合理解释训练数据的最简单模型。寻找好的参数。选定回归函数的模型后，如何找到好的模型参数？在这里，我们需要研究不同的损失/目标函数（它们决定了什么是“好的”拟合），以及允许我们最小化这些损失的优化算法。过拟合和模型选择。当回归函数过度拟合训练数据而不能推广到未见过的测试数据时，就会出现过拟合问题。过拟合通常发生在基础模型（或其参数化）过于灵活和表达能力强的情况下；详见第8.6节。我们将探讨底层原因，并讨论在线性回归背景下减轻过拟合影响的方法。损失函数与参数先验之间的关系。损失函数（优化目标）通常由概率模型启发和诱导。我们将探讨损失函数与产生这些损失的基础先验假设之间的联系。不确定性建模。在任何实际场景中，我们只能获得有限的（可能是大量的）训练数据来选择模型类别和相应的参数。鉴于这有限的训练数据并不能覆盖所有可能的情况，我们可能希望描述剩余的参数不确定性，以便在测试时获得模型预测的置信度度量；训练集越小，不确定性建模就越重要。一致的不确定性建模为模型预测提供了置信区间。

在接下来的部分中，我们将使用第3章、第5章、第6章和第7章中的数学工具来解决线性回归问题。我们将讨论最大似然估计和最大后验估计（MAP）以找到最优的模型参数。利用这些参数估计，我们将简要讨论泛化误差和过拟合问题。在本章的最后，我们将讨论贝叶斯线性回归，它允许我们从更高的层次上推理模型参数，从而解决了最大似然估计和最大后验估计中遇到的一些问题。

![](images/d5aaee68d4cb8990a2b715a2edc93fe202b7cbfab8e1f0569a2469f521f50aef.jpg)  
(a) 回归问题：观察到的带噪声的函数值，我们希望从中推断出生成这些数据的底层函数。  

![](images/1e51d709cb222f6f64d2da3409ce1152ed248aedc9bcb11a5d812cab91cd89d3.jpg)  
(b) 回归解：可能生成数据的函数（蓝色），以及在相应输入处的函数值的测量噪声（橙色分布）。

## 9.1 问题表述

由于存在观测噪声，我们将采用概率方法，并使用似然函数显式地建模噪声。更具体地说，在本章中，我们考虑一个回归问题，其似然函数为

$$
p(y\,|\,\pmb{x})=\mathcal{N}\big(y\,|\,f(\pmb{x}),\,\sigma^{2}\big)\,.
$$

这里，$\pmb{x}\in\mathbb{R}^{D}$ 是输入，$y\in\mathbb{R}$ 是带噪声的目标值（函数值）。根据（9.1），输入 $\pmb{x}$ 和目标值 $y$ 之间的函数关系可以表示为

$$
y=f({\pmb x})+\epsilon\,,
$$

其中 $\epsilon\sim\mathcal{N}(0,\,\sigma^{2})$ 是独立且同分布（i.i.d.）的高斯测量噪声，均值为 0，方差为 $\sigma^{2}$。我们的目标是找到一个接近（相似于）生成数据的未知函数 $f$ 并且具有较好泛化能力的函数。

在本章中，我们将关注参数模型，即选择一个参数化的函数并找到参数 $\pmb{\theta}$ 以“很好地”拟合数据。目前，我们假设噪声方差 $\sigma^{2}$ 已知，并专注于学习模型参数 $\pmb{\theta}$。在线性回归中，我们考虑参数 $\pmb{\theta}$ 在模型中线性出现的特殊情况。线性回归的一个例子如下：

$$
\begin{array}{r l}
&{p(y\,|\,\pmb{x},\pmb{\theta})=\mathcal{N}\big(y\,|\,\pmb{x}^{\top}\pmb{\theta},\,\sigma^{2}\big)}\\ 
&{\Longleftrightarrow y=\pmb{x}^{\top}\pmb{\theta}+\epsilon\,,\quad\epsilon\sim\mathcal{N}\big(0,\,\sigma^{2}\big)\,,}
\end{array}
$$

其中 $\pmb{\theta}\ \in\ \mathbb{R}^{D}$ 是我们要寻找的参数。由（9.4）描述的函数类是通过原点的直线。在（9.4）中，我们选择了参数化 $f(\pmb{x})=\pmb{x}^{\top}\pmb{\theta}$。

（9.3）中的似然是 $y$ 在 $\pmb{x}^{\top}\pmb{\theta}$ 处的概率密度函数。注意，唯一的不确定性来源来自观测噪声（因为 $_{\pmb{x}}$ 和 $\pmb{\theta}$ 在（9.3）中被视为已知）。如果没有观测噪声，$_{\pmb{x}}$ 和 $y$ 之间的关系将是确定性的，并且（9.3）将是一个狄拉克δ函数。

狄拉克δ函数（delta函数）在除了一个点外处处为零，且其积分等于 1。它可以被认为是当 $\sigma^{2}\to0$ 时的高斯分布。似然函数是指概率密度函数在给定点的值。
### 示例 9.1

对于 $x,\theta\in\mathbb{R}$，（9.4）中的线性回归模型描述了直线（线性函数），参数 $\theta$ 是直线的斜率。图 9.2(a) 展示了不同 $\theta$ 值下的几个示例函数。

（9.3）-（9.4）中的线性回归模型不仅对参数是线性的，而且对输入 $x$ 也是线性的。图 9.2(a) 展示了这类函数的例子。稍后我们将看到，$\bar{\boldsymbol{y}}=\bar{\boldsymbol{\phi}}^{\intercal}(\boldsymbol{x})\boldsymbol{\theta}$ 对于非线性变换 $\phi$ 也是一种线性回归模型，因为“线性回归”指的是参数线性组合的模型。

![](images/bd5c0f7b32f6abbcad06ad310e5e72f3bd83519aaff5cd6a1ce9f5d02e9dbe4f.jpg)

这里，“特征”是输入 $\pmb{x}$ 的表示 $\phi(\pmb{x})$。

在接下来的部分中，我们将详细讨论如何找到好的参数 $\pmb{\theta}$ 以及如何评估一组参数是否“工作良好”。目前，我们假设噪声方差 $\sigma^{2}$ 已知。

## 9.2 参数估计  

训练集 Figure 9.3 线性回归的概率图形模型。观测随机变量用阴影表示，确定性/已知值不带圆圈。

![](images/4cbde6acea0d6048712c5c27558586b80d53a8e9064fbb7d80241e6315fd1373.jpg)

考虑线性回归模型（9.4），假设我们给定一个训练集 $\mathcal{D}\;:=\;\{({\pmb x}_{1},y_{1}),\dots,({\pmb x}_{N},y_{N})\}$，其中 $N$ 个输入 $\pmb{x}_{n}\ \in\ \mathbb{R}^{D}$ 和相应的观测值/目标值 $y_{n}\in\mathbb{R}$，$n=1,\dots,N$。相应的图形模型如图9.3所示。注意 $y_{i}$ 和 $y_{j}$ 在给定各自的输入 $\mathbf{\Delta}x_{i},\mathbf{\Delta}x_{j}$ 的情况下是条件独立的，因此似然函数可以分解为

$$
\begin{array}{l l}{p(\boldsymbol{\mathcal{Y}}\,|\,\boldsymbol{\mathcal{X}},\pmb{\theta})=p(y_{1},\dots,y_{N}\,|\,\pmb{x}_{1},\dots,\pmb{x}_{N},\pmb{\theta})}\\ {\quad\quad\quad\quad=\displaystyle\prod_{n=1}^{N}p(y_{n}\,|\,\pmb{x}_{n},\pmb{\theta})=\displaystyle\prod_{n=1}^{N}\mathcal{N}\big(y_{n}\,|\,\pmb{x}_{n}^{\top}\pmb{\theta},\,\sigma^{2}\big)\,,}\end{array}
$$

其中我们定义 $\mathcal{X}:=\{\pmb{x}_{1},\dots,\pmb{x}_{N}\}$ 和 $\mathcal{Y}:=\{y_{1},\dots,y_{N}\}$ 分别为训练输入和对应的观测值集合。似然函数和因子 $p(y_{n}\,|\,\pmb{x}_{n},\pmb{\theta})$ 是高斯分布，因为噪声分布是高斯的；参见（9.3）。

接下来，我们将讨论如何找到线性回归模型（9.4）的最优参数 $\pmb{\theta}^{*}\in\mathbb{R}^{D}$。一旦找到了参数 $\pmb{\theta}^{*}$，我们可以通过在（9.4）中使用这个参数估计来预测函数值，因此在任意测试输入 $\pmb{x}_{*}$ 处对应的目标值 $y_{*}$ 的分布为

$$
p(y_{*}\,|\,\pmb{x}_{*},\pmb{\theta}^{*})=\mathcal{N}\big(y_{*}\,|\,\pmb{x}_{*}^{\top}\pmb{\theta}^{*},\,\sigma^{2}\big)\,.
$$

接下来，我们将探讨通过最大化似然函数来进行参数估计的问题，这在第8.3节中我们已经部分讨论过。
### 9.2.1 最大似然估计

寻找所需的参数 $\pmb{\theta}_{\mathrm{ML}}$ 的一种广泛使用的方法是最大似然估计，我们通过这种方法找到使似然性最大的参数 $\pmb{\theta}_{\mathrm{ML}}$（见公式 9.5b）。直观地说，最大化似然性意味着在给定模型参数的情况下最大化训练数据的预测分布。我们得到的最大似然参数为

$$
\pmb{\theta}_{\mathrm{ML}}\in\arg\operatorname*{max}_{\pmb{\theta}}p(\mathcal{Y}\,|\,\mathcal{X},\pmb{\theta})\,.
$$

**备注** 似然 $p(\pmb{y}\mid\pmb{x},\pmb{\theta})$ 并不是参数 $\pmb{\theta}$ 的概率分布：它只是参数 $\theta$ 的一个函数，但并不积分到 1（即，它是未归一化的），甚至可能不与 $\pmb{\theta}$ 相关。然而，公式 (9.7) 中的似然是 $\pmb{y}$ 的归一化概率分布。$\diamondsuit$

为了找到使似然性最大的所需参数 $\pmb{\theta}_{\mathrm{ML}}$，我们通常执行梯度上升（或负似然的梯度下降）。然而，在我们考虑的线性回归情况下，存在闭式解，这使得迭代梯度下降变得没有必要。实际上，我们通常不是直接最大化似然性，而是对似然函数应用对数变换，并最小化负对数似然。

**备注**（对数变换）由于似然性（9.5b）是 $N$ 个高斯分布的乘积，对数变换是有用的，因为（a）它不会遭受数值下溢的影响，（b）微分规则会变得更加简单。具体来说，当我们将 $N$ 个概率相乘时，可能会遇到数值下溢的问题，因为 $N$ 是数据点的数量，我们无法表示非常小的数字，如 $10^{-256}$。此外，对数变换会将乘积变成对数概率的和，这样对应的梯度将是各个梯度的和，而不是重复应用乘法规则（5.46）来计算 $N$ 项乘积的梯度。$\diamondsuit$

最大似然估计：最大化似然性意味着在给定参数的情况下最大化训练数据的预测分布。似然性并不是参数的概率分布。

由于对数函数是一个（严格地）单调递增函数，函数 $f$ 的最优值与 $\log f$ 的最优值相同。

为了找到线性回归问题的最优参数 $\pmb{\theta}_{\mathrm{ML}}$，我们最小化负对数似然：

$$
-\log p(\mathcal{Y}\,|\,\mathcal{X},\pmb{\theta})=-\log\prod_{n=1}^{N}p(y_{n}\,|\,\pmb{x}_{n},\pmb{\theta})=-\sum_{n=1}^{N}\log p(y_{n}\,|\,\pmb{x}_{n},\pmb{\theta})\,,
$$

在这里我们利用了由于训练集上的独立假设，似然性（9.5b）在数据点数量上分解的事实。

在线性回归模型（9.4）中，似然是高斯分布（由于高斯加性噪声项），因此我们得到：

$$
\log p(y_{n}\,|\,\pmb{x}_{n},\pmb{\theta})=-\frac{1}{2\sigma^{2}}(y_{n}-\pmb{x}_{n}^{\top}\pmb{\theta})^{2}+\mathrm{const}\,,
$$

其中常数包括所有与 $\pmb{\theta}$ 无关的项。使用（9.9）在负对数似然（9.8）中，我们得到（忽略常数项）：

$$
\begin{array}{l}{\displaystyle\mathcal{L}(\pmb{\theta}):=\frac{1}{2\sigma^{2}}\!\sum_{n=1}^{N}(y_{n}-\pmb{x}_{n}^{\top}\pmb{\theta})^{2}}\\ {\displaystyle\qquad=\frac{1}{2\sigma^{2}}(\pmb{y}-\pmb{X}\pmb{\theta})^{\top}(\pmb{y}-\pmb{X}\pmb{\theta})=\frac{1}{2\sigma^{2}}\|\pmb{y}-\pmb{X}\pmb{\theta}\|^{2}\,,}\end{array}
$$

负对数似然函数也称为误差函数。设计矩阵平方误差通常用作距离的度量。回想第 3.1 节，如果我们选择点积作为内积，则 $\|\pmb{x}\|^{2}=\pmb{x}^{\top}\pmb{x}$。

其中我们定义设计矩阵 $\pmb{X}\,:=\,[\pmb{x}_{1},\,.\,.\,,.\,,\pmb{x}_{N}]^{\top}\,\in\,\mathbb{R}^{N\times D}$ 为训练输入的集合，$\pmb{y}:=[y_{1},.\,.\,.\,,y_{N}]^{\top}\in\mathbb{R}^{N}$ 为收集所有训练目标的向量。注意，设计矩阵 $\pmb{X}$ 的第 $n$ 行对应于训练输入 $\pmb{x}_{n}$。在（9.10b）中，我们使用了以下事实：观测值 $y_{n}$ 和相应的模型预测 $\mathbf{\Delta}\mathbf{\mathcal{X}}_{n}^{\top}\mathbf{\theta}$ 之间的平方误差之和等于 $\pmb{y}$ 和 $X\theta$ 之间的平方距离。

有了（9.10b），我们现在有了需要优化的负对数似然函数的具体形式。我们立即看到（9.10b）是 $\pmb{\theta}$ 的二次函数。这意味着我们可以找到一个唯一的全局解 $\pmb{\theta}_{\mathrm{ML}}$ 来最小化负对数似然 $\mathcal{L}$。我们可以通过计算梯度并将其设置为 0 来找到全局最优解。

使用第 5 章的结果，我们计算 $\mathcal{L}$ 关于参数的梯度为

$$
\begin{array}{l}{\displaystyle\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\theta}=\frac{\mathrm{d}}{\mathrm{d}\theta}\left(\frac{1}{2\sigma^{2}}(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\theta})^{\top}(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\theta})\right)}\\ {\displaystyle\qquad=\frac{1}{2\sigma^{2}}\frac{\mathrm{d}}{\mathrm{d}\theta}\left(\boldsymbol{y}^{\top}\boldsymbol{y}-2\boldsymbol{y}^{\top}\boldsymbol{X}\boldsymbol{\theta}+\boldsymbol{\theta}^{\top}\boldsymbol{X}^{\top}\boldsymbol{X}\boldsymbol{\theta}\right)}\\ {\displaystyle\qquad=\frac{1}{\sigma^{2}}(-\boldsymbol{y}^{\top}\boldsymbol{X}+\boldsymbol{\theta}^{\top}\boldsymbol{X}^{\top}\boldsymbol{X})\in\mathbb{R}^{1\times D}\,.}\end{array}
$$

忽略重复数据点的可能性，$\operatorname{rk}(X)=D$ 如果 $N\geqslant D$，即我们没有比数据点多的参数。

最大似然估计器 $\pmb{\theta}_{\mathrm{ML}}$ 满足 $\begin{array}{r}{\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\pmb\theta}=\mathbf{0}^{\top}}\end{array}$（必要最优条件），我们得到

$$
\begin{array}{r l}&{\displaystyle\frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\pmb{\theta}}=\mathbf{0}^{\top}\,\,\stackrel{(9.11\mathrm{c})}{\longrightarrow}\,\pmb{\theta}_{\mathrm{ML}}^{\top}\pmb{X}^{\top}\pmb{X}=\pmb{y}^{\top}\pmb{X}}\\ &{\quad\quad\quad\quad\iff\pmb{\theta}_{\mathrm{ML}}^{\top}=\pmb{y}^{\top}\pmb{X}(\pmb{X}^{\top}\pmb{X})^{-1}}\\ &{\quad\quad\quad\iff\pmb{\theta}_{\mathrm{ML}}=(\pmb{X}^{\top}\pmb{X})^{-1}\pmb{X}^{\top}\pmb{y}\,.}\end{array}
$$

我们可以右乘第一个方程两边 $(\pmb{X}^{\top}\pmb{X})^{-1}$ 因为如果 $\operatorname{rk}(X)=D$，则 $\mathbf{\boldsymbol{X}}^{\top}\mathbf{\boldsymbol{X}}$ 是正定的，在这里 $\operatorname{rk}(X)$ 表示 $\boldsymbol{X}$ 的秩。

**备注** 将梯度设置为 $\mathbf{0}^{\top}$ 是一个必要且充分条件，我们得到一个全局最小值，因为海森矩阵 $\nabla_{\theta}^{2}{\mathcal{L}}(\mathbf{\theta})=X^{\top}X\in \mathbb{R}^{D\times D}$ 是正定的。$\diamondsuit$

**备注** （9.12c）中的最大似然解要求我们解决形如 $A\pmb\theta=\pmb b$ 的线性方程组，其中 $\bar{\pmb{A}}=(\pmb{X}^{\top}\pmb{X})$ 和 $\pmb{b}\doteq\pmb{X}^{\top}\pmb{y}$。$\diamondsuit$

### 示例 9.2（拟合直线）

让我们看一下图 9.2，在这里我们希望通过最大似然估计拟合一条直线 $f(x)=\theta x$，其中 $\theta$ 是未知斜率，到一个数据集。该模型类中的函数示例（直线）如图 9.2(a) 所示。对于图 9.2(b) 中显示的数据集，我们使用（9.12c）找到斜率参数 $\theta$ 的最大似然估计，并在图 9.2(c) 中获得最大似然线性函数。

到目前为止，我们考虑了（9.4）节中描述的线性回归设置，这使我们能够使用最大似然估计来拟合数据的直线。然而，当涉及到拟合更有趣的数据时，直线并不足够表达。幸运的是，线性回归为我们提供了一种在线性回归框架内拟合非线性函数的方法：“线性回归”仅指“参数线性”，因此我们可以对输入进行任意的非线性变换 $\phi({\pmb x})$，然后线性组合这一变换的各个组成部分。相应的线性回归模型如下：

线性回归指的是“参数线性”的回归模型，但输入可以经过任何非线性变换。

$$
\begin{array}{r l}
&{\quad p(\boldsymbol{y}\,|\,\pmb{x},\pmb{\theta})=\mathcal{N}\big(\boldsymbol{y}\,|\,\boldsymbol{\phi}^{\top}(\pmb{x})\pmb{\theta},\,\sigma^{2}\big)}\\ 
&{\quad\iff\boldsymbol{y}=\boldsymbol{\phi}^{\top}(\pmb{x})\pmb{\theta}+\epsilon=\displaystyle\sum_{k=0}^{K-1}\theta_{k}\phi_{k}(\pmb{x})+\epsilon\,,}
\end{array}
$$

$\phi:\mathbb{R}^{D}\rightarrow\mathbb{R}^{K}$ 是输入 $\pmb{x}$ 的（非线性）变换，而 $\phi_{k}:\mathbb{R}^{D}\rightarrow\mathbb{R}$ 是特征向量 $\phi$ 的第 $k$ 个分量。请注意，特征向量的模型参数 $\theta$ 仍然只以线性方式出现。

### 示例 9.3（多项式回归）

我们面对一个回归问题 $\boldsymbol{y}=\boldsymbol{\phi}^{\intercal}(\boldsymbol{x})\mathbf{\theta}{+}\boldsymbol{\epsilon}$，其中 $x\in\mathbb{R}$ 和 $\pmb{\theta}\in\mathbb{R}^{K}$。在此背景下经常使用的变换是

$$
\boldsymbol{\phi}(\boldsymbol{x})=\left[\begin{array}{c}{\phi_{0}(\boldsymbol{x})}\\ {\phi_{1}(\boldsymbol{x})}\\ {\vdots}\\ {\phi_{K-1}(\boldsymbol{x})}\end{array}\right]=\left[\begin{array}{c}{1}\\ {\boldsymbol{x}}\\ {\boldsymbol{x}^{2}}\\ {\boldsymbol{x}^{3}}\\ {\vdots}\\ {\boldsymbol{x}^{K-1}}\end{array}\right]\in\mathbb{R}^{K}\,.
$$

这意味着我们将原始的一维输入空间“提升”到一个 $K$ 维特征空间中，该空间包含所有单项式 $x^{k}$，其中 $k=0,\ldots,K-1$。利用这些特征，我们可以在线性回归框架内建模最高次数为 $K-1$ 的多项式：最高次数为 $K-1$ 的多项式是

$$
\boldsymbol{f}(\boldsymbol{x})=\sum_{k=0}^{K-1}\boldsymbol{\theta}_{k}\boldsymbol{x}^{k}=\boldsymbol{\phi}^{\intercal}(\boldsymbol{x})\boldsymbol{\theta}\,,
$$

其中 $\phi$ 如定义 (9.14)，且 $\pmb\theta=[\theta_{0},\ldots,\theta_{K-1}]^{\top}\in\mathbb R^{K}$ 包含了线性参数 $\theta_{k}$。

#### 特征矩阵（设计矩阵）

现在让我们来看看线性回归模型（9.13）中参数 $\pmb{\theta}$ 的最大似然估计。我们考虑训练输入 $\pmb{x}_{n}\in\mathbb{R}^{D}$ 和目标值 $y_{n}\in\mathbb{R}$，$n=1,\ldots,N$，并定义特征矩阵（设计矩阵）如下：

$$
\boldsymbol\Phi:=\left[\begin{array}{c}{\boldsymbol\phi^{\top}(\pmb{x}_{1})}\\ {\vdots}\\ {\boldsymbol\phi^{\top}(\pmb{x}_{N})}\end{array}\right]=\left[\begin{array}{ccc}{\phi_{0}(\pmb{x}_{1})}&{\cdots}&{\phi_{K-1}(\pmb{x}_{1})}\\ {\phi_{0}(\pmb{x}_{2})}&{\cdots}&{\phi_{K-1}(\pmb{x}_{2})}\\ {\vdots}&{}&{\vdots}\\ {\phi_{0}(\pmb{x}_{N})}&{\cdots}&{\phi_{K-1}(\pmb{x}_{N})}\end{array}\right]\in\mathbb{R}^{N\times K}\,,
$$

其中 $\Phi_{i j}=\phi_{j}(\pmb{x}_{i})$ 且 $\phi_{j}:\mathbb{R}^{D}\rightarrow\mathbb{R}$。

### 示例 9.4（二阶多项式特征矩阵）

对于二阶多项式和 $N$ 个训练点 $x_{n}\in\mathbb{R}$，$n=1,\ldots,N$，特征矩阵是

$$
\Phi=\left[\begin{array}{ccc}{1}&{x_{1}}&{x_{1}^{2}}\\ {1}&{x_{2}}&{x_{2}^{2}}\\ {\vdots}&{\vdots}&{\vdots}\\ {1}&{x_{N}}&{x_{N}^{2}}\end{array}\right]\,.
$$

利用定义在 (9.16) 中的特征矩阵 $\Phi$，线性回归模型 (9.13) 的负对数似然可以写成

$$
-\log p(\mathcal{Y}\,|\,\mathcal{X},\pmb{\theta})=\frac{1}{2\sigma^{2}}(\pmb{y}-\Phi\pmb{\theta})^{\top}(\pmb{y}-\Phi\pmb{\theta})+\mathrm{const}\,.
$$


将 (9.18) 与“无特征”模型的负对数似然 (9.10b) 进行比较，我们可以立即看到只需要将 $\boldsymbol{X}$ 替换为 $\Phi$。由于 $\boldsymbol{X}$ 和 $\Phi$ 都独立于我们希望优化的参数 $\pmb{\theta}$，我们直接得到线性回归问题的最大似然估计

$$
\pmb{\theta}_{\mathrm{ML}}=(\pmb{\Phi}^{\top}\pmb{\Phi})^{-1}\pmb{\Phi}^{\top}\pmb{y}
$$

其中定义了非线性特征 (9.13)。备注：当我们没有使用特征时，我们需要 $\mathbf{X}^{\top}\mathbf{X}$ 是可逆的，这当 $\operatorname{rk}(X)=D_{r}$ 成立时成立，即 $\boldsymbol{X}$ 的列是线性独立的。因此，在 (9.19) 中，我们需要 $\boldsymbol{\Phi}^{\intercal}\boldsymbol{\Phi}\in\mathbb{R}^{K\times K}$ 是可逆的。这当且仅当 $\mathrm{rk}(\Phi)=K$ 时成立。$\diamondsuit$

### 示例 9.5（最大似然多项式拟合）

![](images/e52ff1b6216019339a216be4c98bd9b35aed472573180ee8276cf001f4f7ec21.jpg)

图 9.4 多项式回归：(a) 数据集由 $(x_{n},y_{n})$ 对组成，$n=1,\ldots,10$；(b) 四次多项式的最大似然拟合。

考虑图 9.4(a) 中的数据集。数据集由 $N=10$ 对 $(x_{n},y_{n})$ 组成，其中 $x_{n}\sim\mathcal{U}[-5,5]$ 和 $y_{n}=-\sin(x_{n}/5)+\cos(x_{n})+\epsilon$，其中 $\epsilon\sim\mathcal{N}\big(0,\,0.2^{2}\big)$。

我们使用最大似然估计拟合一个四次多项式，即参数 $\pmb{\theta}_{\mathrm{ML}}$ 如 (9.19) 所示。最大似然估计在任意测试位置 $x_{*}$ 上给出函数值 $\bar{\phi}^{\top}(x_{*})\pmb{\theta}_{\mathrm{ML}}$。结果如图 9.4(b) 所示。

 估计噪声方差

到目前为止，我们假设噪声方差 $\sigma^{2}$ 是已知的。然而，我们也可以使用最大似然估计的原则来获得噪声方差的最大似然估计 $\sigma_{\mathrm{ML}}^{2}$。为此，我们遵循标准程序：写出对数似然函数，计算其关于 $\sigma^{2}\;>\;0$ 的导数，将其设为 0 并求解。对数似然函数由以下公式给出：

$$
\begin{array}{l}{\log p(\mathcal{Y}\,|\,\mathcal{X},\pmb{\theta},\sigma^{2})=\sum_{n=1}^{N}\log\mathcal{N}\bigl(y_{n}\,|\,\phi^{\top}(\pmb{x}_{n})\pmb{\theta},\,\sigma^{2}\bigr)}\\ {\qquad=\sum_{n=1}^{N}\biggl(-\frac{1}{2}\log(2\pi)-\frac{1}{2}\log\sigma^{2}-\frac{1}{2\sigma^{2}}(y_{n}-\phi^{\top}(\pmb{x}_{n})\pmb{\theta})^{2}\biggr)}\\ {\qquad=-\frac{N}{2}\log\sigma^{2}-\frac{1}{2\sigma^{2}}\underbrace{\sum_{n=1}^{N}(y_{n}-\phi^{\top}(\pmb{x}_{n})\pmb{\theta})^{2}}_{=:s}+\,\mathrm{const}\,.}\end{array}
$$
对数似然函数关于 $\sigma^{2}$ 的偏导数为：

$$
\begin{array}{c}{{\frac{\partial\log p(\mathcal{Y}\,\vert\,\mathcal{X},\pmb{\theta},\sigma^{2})}{\partial\sigma^{2}}=-\frac{N}{2\sigma^{2}}+\frac{1}{2\sigma^{4}}s=0}}\\ {{\Longleftrightarrow\frac{N}{2\sigma^{2}}=\frac{s}{2\sigma^{4}}}}\end{array}
$$
因此，我们得到：

$$
\sigma_{\mathrm{ML}}^{2}=\frac{s}{N}=\frac{1}{N}\sum_{n=1}^{N}(y_{n}-\phi^{\intercal}(\boldsymbol{x}_{n})\pmb{\theta})^{2}\,.
$$
因此，噪声方差的最大似然估计是噪声自由函数值 $\phi^{\top}(\pmb{x}_{n})\pmb{\theta}$ 和相应噪声观测值 $y_{n}$ 之间平方距离的样本均值。

### 9.2.2 线性回归中的过拟合

均方根误差（RMSE）

RMSE 是归一化的。

负对数似然没有单位。

我们刚刚讨论了如何使用最大似然估计来拟合线性模型（例如多项式）到数据中。我们可以通过计算误差/损失来评估模型的质量。一种方法是计算负对数似然（9.10b），我们最小化它来确定最大似然估计器。或者，由于噪声参数 $\sigma^{2}$ 不是自由模型参数，我们可以忽略缩放因子 $q/\sigma^{2}$，这样我们最终得到一个平方误差损失函数 $\lVert\pmb{y}-\Phi\pmb{\theta}\rVert^{2}$。我们通常不使用这个平方损失，而是使用均方根误差（RMSE）。

$$
\sqrt{\frac{1}{N}\left\|\pmb{y}-\boldsymbol{\Phi}\pmb{\theta}\right\|^{2}}=\sqrt{\frac{1}{N}\sum_{n=1}^{N}(y_{n}-\boldsymbol{\phi}^{\intercal}(\pmb{x}_{n})\pmb{\theta})^{2}}\,,
$$

这（a）允许我们比较不同大小的数据集的误差，并且（b）具有与观察到的函数值 $y_{n}$ 相同的比例和单位。例如，如果我们拟合一个将邮政编码（$\cdot_{x}$ 以纬度、经度给出）映射到房屋价格（$\cdot_{y}$ 值为欧元）的模型，则 RMSE 也以欧元为单位，而平方误差以欧元的平方为单位。如果我们选择包括原始负对数似然（9.10b）中的因子 $\sigma^{2}$，则我们最终得到一个无单位的目标，即在前面的例子中，我们的目标将不再以欧元或欧元的平方为单位。

对于模型选择（见第 8.6 节），我们可以使用 RMSE（或负对数似然）来确定多项式的最佳次数，通过找到最小化目标的多项式次数 $M$。由于多项式次数是自然数，我们可以进行暴力搜索并枚举所有（合理的）$M$ 值。对于大小为 $N$ 的训练集，测试 $0 \leqslant M \leqslant N-1$ 是足够的。对于 $M < N$，最大似然估计器是唯一的。对于 $M \geqslant N$，我们有比数据点更多的参数

![](images/4d86c8981746298eb89be6e20fd108da38f4333a26c706c838716a5c4df23669.jpg)

图 9.5 不同多项式次数 $M$ 的最大似然拟合。

比数据点多，需要求解欠定线性方程组（$\left[\Phi^{\top}\Phi\right.$ 在（9.19）中也将不再可逆），因此有无限多个可能的最大似然估计器。

图 9.5 显示了由最大似然确定的多项式拟合，这些拟合基于图 9.4(a) 中的数据集，其中 $N=10$ 观测值。我们注意到低次多项式（例如常数 $\mathit{\Omega}_{,M} = 0$）或线性（$M=1$）拟合数据较差，因此是真实潜在函数的差表示。对于次数 $M = 3, \ldots, 6$，拟合看起来合理且平滑地插值数据。当我们转向更高次的多项式时，我们注意到它们更好地拟合数据。在极端情况下，$M = N-1 = 9$，函数将通过每个数据点。然而，这些高次多项式剧烈振荡，是生成数据的潜在函数的差表示，因此我们遭受过拟合。

记住目标是通过准确预测新（未见）数据来实现良好的泛化。我们通过考虑一个由 200 个数据点组成的独立测试集来获得对泛化性能依赖于多项式次数 $M$ 的一些定量见解，这些数据点使用与生成训练集完全相同的程序生成。作为测试输入，我们选择了区间 $[-5,5]$ 中的 200 个线性网格点。对于每个 $M$ 的选择，我们评估训练数据和测试数据的 RMSE（9.23）。

现在来看测试误差，这是相应多项式泛化属性的定性度量，我们注意到初始时测试误差下降；见图 9.6（橙色）。对于四次多项式，测试误差相对较低，并且在五次之前保持相对恒定。然而，从六次开始，测试误差显著增加，高阶多项式具有非常差的泛化属性。在这个特定的例子中，这也从相应的 $M = N-1$ 情况中显而易见。这种情况在极端意义上是唯一的，否则相应的线性方程组的零空间将是非平凡的，我们将有无限多个线性回归问题的最优解。过拟合 注意噪声方差 $\sigma^{2} > 0$。

![](images/0d867439e74d332ba042c11b7085e5d39f75604a736726734efd25b3af4b1f0d.jpg)  

训练误差  

测试误差  

最大似然拟合在图9.5中。注意，训练误差（图9.6中的蓝色曲线）在多项式的次数增加时永远不会增加。在我们的例子中，最佳泛化（测试误差最小的点）是在多项式次数为 $M=4$ 时获得的。

### 9.2.3 最大后验估计

我们刚刚看到，最大似然估计容易导致过拟合。通常观察到，如果出现过拟合，参数值的大小会变得相对较大（Bishop, 2006）。

为了减轻过大参数值的影响，我们可以在参数上放置一个先验分布 $p(\pmb\theta)$。先验分布明确地编码了哪些参数值是合理的（在观察任何数据之前）。例如，单个参数 $\theta$ 的高斯先验 $p(\theta) = \mathcal{N}(0, 1)$ 表示参数值预期位于区间 $[-2, 2]$ 内（均值周围两个标准差范围内）。

一旦数据集 $\mathcal{X}, \mathcal{Y}$ 可用，我们不再最大化似然函数，而是寻找最大化后验分布 $p(\pmb\theta\,|\,\mathcal{X}, \mathcal{y})$ 的参数。这一过程称为最大后验估计（MAP）。

给定训练数据 $\mathcal{X}, \mathcal{Y}$ 的参数 $\pmb{\theta}$ 的后验分布是通过应用贝叶斯定理（第6.3节）得到的：

$$
p(\pmb\theta\,|\,\mathcal{X}, \mathcal{y}) = \frac{p(\mathcal{y}\,|\,\mathcal{X}, \pmb\theta)p(\pmb\theta)}{p(\mathcal{y}\,|\,\mathcal{X})}\,.
$$

由于后验分布明确依赖于参数先验 $p(\pmb\theta)$，先验将对作为后验最大值的参数向量产生影响。我们将在以下内容中更明确地看到这一点。最大化后验分布（9.24）的参数向量 $\pmb{\theta}_{\mathrm{MAP}}$ 是MAP估计。

为了找到MAP估计 $\pmb{\theta}_{\mathrm{MAP}}$，我们遵循与最大似然估计相似的步骤。我们从对数变换开始，并计算对数后验分布：

$$
\log p(\pmb\theta\,|\,\mathcal{X}, \mathcal{y}) = \log p(\mathcal{y}\,|\,\mathcal{X}, \pmb\theta) + \log p(\pmb\theta) + \mathrm{const}\,,
$$

其中常数项包括与 $\pmb{\theta}$ 无关的项。我们看到，（9.25）中的对数后验分布是似然函数 $p(\mathcal{Y}\,|\,\mathcal{X}, \pmb{\theta})$ 的对数和先验 $p(\pmb\theta)$ 的对数之和，因此MAP估计将是先验（我们对数据观察之前合理参数值的建议）和数据依赖似然函数之间的“折衷”。

为了找到MAP估计 $\pmb{\theta}_{\mathrm{MAP}}$，我们相对于 $\pmb{\theta}$ 最小化负对数后验分布，即我们求解：

$$
\pmb{\theta}_{\mathtt{M A P}} \in \arg\operatorname*{min}_{\pmb{\theta}} \left\{-\log p(\mathcal{Y}\,|\,\mathcal{X}, \pmb{\theta}) - \log p(\pmb{\theta})\right\}\,.
$$

负对数后验分布相对于 $\pmb{\theta}$ 的梯度是：

$$
-\frac{\mathrm{d}\log p(\pmb{\theta}\,|\,\mathcal{X}, \mathcal{Y})}{\mathrm{d}\pmb{\theta}} = -\frac{\mathrm{d}\log p(\mathcal{Y}\,|\,\mathcal{X}, \pmb{\theta})}{\mathrm{d}\pmb{\theta}} - \frac{\mathrm{d}\log p(\pmb{\theta})}{\mathrm{d}\pmb{\theta}}\,,
$$

其中我们识别出右侧第一项为（9.11c）中的负对数似然梯度。

对于参数 $\pmb{\theta}$ 的（共轭）高斯先验 $p(\pmb\theta) = \mathcal{N}(\mathbf{0}, b^2 \pmb{I})$，在线性回归设置（9.13）中，我们得到负对数后验分布：

$$
-\log p(\pmb\theta\,|\,\mathcal{X}, \mathcal{y}) = \frac{1}{2\sigma^2}(\pmb{y} - \Phi \pmb{\theta})^\top (\pmb{y} - \Phi \pmb{\theta}) + \frac{1}{2b^2} \pmb{\theta}^\top \pmb{\theta} + \mathrm{const}\,.
$$

这里，第一项对应于对数似然的贡献，第二项来源于对数先验。相对于参数 $\pmb{\theta}$ 的对数后验分布的梯度是：

$$
-\frac{\mathrm{d}\log p(\pmb{\theta}\,|\,\mathcal{X}, \mathcal{Y})}{\mathrm{d}\pmb{\theta}} = \frac{1}{\sigma^2} (\pmb{\theta}^\top \pmb{\Phi}^\top \pmb{\Phi} - \pmb{y}^\top \pmb{\Phi}) + \frac{1}{b^2} \pmb{\theta}^\top\,.
$$

我们通过将梯度设置为 $\mathbf{0}^\top$ 并求解 $\pmb{\theta}_{\mathrm{MAP}}$ 来找到MAP估计。我们得到：

$$
\begin{array}{c}
{\displaystyle \frac{1}{\sigma^2} (\pmb{\theta}^\top \pmb{\Phi} - \pmb{y}^\top \pmb{\Phi}) + \frac{1}{b^2} \pmb{\theta}^\top = \mathbf{0}^\top} \\
{\displaystyle \Longleftrightarrow \pmb{\theta}^\top \left( \frac{1}{\sigma^2} \pmb{\Phi}^\top \pmb{\Phi} + \frac{1}{b^2} \pmb{I} \right) - \frac{1}{\sigma^2} \pmb{y}^\top \pmb{\Phi} = \mathbf{0}^\top} \\
{\displaystyle \Longleftrightarrow \pmb{\theta}^\top \left( \pmb{\Phi}^\top \pmb{\Phi} + \frac{\sigma^2}{b^2} \pmb{I} \right) = \pmb{y}^\top \pmb{\Phi}} \\
{\displaystyle \Longleftrightarrow \pmb{\theta}^\top = \pmb{y}^\top \pmb{\Phi} \left( \pmb{\Phi}^\top \pmb{\Phi} + \frac{\sigma^2}{b^2} \pmb{I} \right)^{-1}}
\end{array}
$$

因此，MAP估计是（通过转置等式两边）：

$$
\pmb{\theta}_{\mathrm{MAP}} = \left( \pmb{\Phi}^\top \pmb{\Phi} + \frac{\sigma^2}{b^2} \pmb{I} \right)^{-1} \pmb{\Phi}^\top \pmb{y}\,.
$$

将（9.31）中的MAP估计与（9.19）中的最大似然估计进行比较，我们看到这两个解之间的唯一区别是逆矩阵中的附加项 $\frac{\sigma^2}{b^2} \pmb{I}$。这一项确保 $\pmb{\Phi}^\top \pmb{\Phi}$ 是对称的、半正定的。在（9.31）中的附加项是严格正定的，因此逆矩阵存在。

$\pmb{\Phi}^\top \pmb{\Phi} + \cfrac{\sigma^2}{b^2} \pmb{I}$ 是对称且严格正定的（即其逆矩阵存在，MAP估计是线性方程组的唯一解）。此外，它反映了正则化器的影响。

### 示例 9.6 (多项式回归的 MAP 估计)

在第 9.2.1 节中的多项式回归示例中，我们在参数 $\pmb{\theta}$ 上放置了一个高斯先验 $p(\pmb\theta)=\mathcal{N}(\mathbf0,\,I)$，并根据公式 (9.31) 确定 MAP 估计值。在图 9.7 中，我们展示了多项式次数为 6（左）和次数为 8（右）的最大似然估计和 MAP 估计。先验（正则化器）对于低次多项式不起主要作用，但对于高次多项式则保持函数相对平滑。尽管 MAP 估计可以防止过拟合，但它并不是解决此问题的一般方法，因此我们需要一种更原则的方法来处理过拟合问题。

![](images/9c1a3cce28f655253939755f38d6c60da23b92dd425d953bf58a4d38a2a08058.jpg)

图 9.7 多项式回归：最大似然估计和 MAP 估计。(a) 多项式次数为 6；(b) 多项式次数为 8。

### 9.2.4 MAP估计作为正则化

正则化 正则化最小二乘

数据拟合项 残差项 正则化项 正则化参数

与其在参数 $\pmb{\theta}$ 上放置先验分布，也可以通过正则化来减轻过拟合的影响，即通过正则化来惩罚参数的幅度。在正则化最小二乘中，我们考虑损失函数

$$
\left\|\pmb{y}-\Phi\pmb{\theta}\right\|^{2}+\lambda\left\|\pmb{\theta}\right\|_{2}^{2}\,,
$$
我们针对 $\pmb{\theta}$ 进行最小化（见第8.2.3节）。这里，第一项是数据拟合项（也称为残差项），它与负对数似然成正比；见（9.10b）。第二项称为正则化项，正则化参数 $\lambda\geqslant0$ 控制正则化的“严格性”。

注释。 代替欧几里得范数 $\left\|\cdot\right\|_{2}$，我们可以在（9.32）中选择任何 $p$-范数 $\left\lVert\cdot\right\rVert_{p}$。在实践中，较小的 $p$ 值会导致更稀疏的解。这里，“稀疏”意味着许多参数值 $\theta_{d}\,=\,0$，这对变量选择也是有用的。对于 $p\,=\,1$，正则化项称为 LASSO（最小绝对收缩和选择算子），由 Tibshirani（1996）提出。$\diamondsuit$

（9.32）中的正则化项 $\lambda\left\|\pmb{\theta}\right\|_{2}^{2}$ 可以解释为一个负对数高斯先验，我们在 MAP 估计中使用它；见（9.26）。更具体地说，对于高斯先验 $p(\pmb\theta)=\mathcal{N}\big(\mathbf0,\,b^{2}\pmb I\big)$，我们得到负对数高斯先验

$$
-\log p(\pmb\theta)=\frac{1}{2b^{2}}\left\lVert\pmb{\theta}\right\rVert_{2}^{2}+\mathrm{const}
$$
因此，对于 $\textstyle\lambda={\cfrac{1}{2b^{2}}}$，正则化项和负对数高斯先验是相同的。

鉴于（9.32）中的正则化最小二乘损失函数由与负对数似然和负对数先验密切相关的项组成，当我们最小化此损失时，我们得到一个与（9.31）中的 MAP 估计非常相似的解并不奇怪。更具体地说，最小化正则化最小二乘损失函数得到

$$
\pmb{\theta}_{\mathrm{RLS}}=(\pmb{\Phi}^{\top}\pmb{\Phi}+\lambda\pmb{I})^{-1}\pmb{\Phi}^{\top}\pmb{y}\,,
$$
对于 $\begin{array}{r}{\lambda\,=\,\cfrac{\sigma^{2}}{b^{2}}}\end{array}$，它与（9.31）中的 MAP 估计相同，其中 $\sigma^{2}$ 是噪声方差，$b^{2}$ 是高斯先验 $p(\pmb\theta)=\mathcal{N}\big(\mathbf0,\,b^{2}\pmb I\big)$ 的方差。

到目前为止，我们已经涵盖了使用最大似然和 MAP 估计的参数估计，其中我们找到使目标函数（似然或后验）优化的点估计 $\pmb{\theta}^{*}$。我们看到，最大似然和 MAP 估计都可能导致过拟合。在下一节中，我们将讨论贝叶斯线性回归，其中我们将使用贝叶斯推理（第8.4节）来找到未知参数的后验分布，然后使用它来进行预测。更具体地说，对于预测，我们将对所有合理的参数集进行平均，而不是专注于点估计。

## 9.3 贝叶斯线性回归

之前，我们研究了线性回归模型，其中我们估计模型参数 $\pmb{\theta}$，例如通过最大似然或最大后验估计。我们发现最大似然估计可能会导致严重的过拟合，特别是在小数据集的情况下。最大后验估计通过在参数上放置先验分布来解决这个问题，这个先验分布起到了正则化的作用。

贝叶斯线性回归将参数先验的概念推进了一步，并且不试图计算参数的点估计，而是考虑参数的完整后验分布，当进行预测时，使用这个后验分布。这意味着我们不会拟合任何参数，而是计算所有合理参数设置的平均值（根据后验分布）。

点估计是一个特定的参数值，而不是一个合理参数设置的分布。

贝叶斯线性回归

### 9.3.1 模型

在贝叶斯线性回归中，我们考虑模型

$$
\begin{array}{r l}
&{\text{先验}\,\,\,\,\,\,\,\,\,\,\,\,p(\pmb\theta)=\mathcal{N}\big(\pmb m_{0},\,\pmb S_{0}\big)\,,}\\
&{\text{似然}\,\,\,\,\,\,\,\,\,\,\,p(\boldsymbol{y}\,|\,\pmb x,\pmb\theta)=\mathcal{N}\big(\boldsymbol{y}\,|\,\boldsymbol\phi^{\top}(\pmb x)\pmb\theta,\,\sigma^{2}\big)\,,}
\end{array}
$$

图 9.8 贝叶斯线性回归的图形模型。

![](images/4af83e3010fb63a4945a827f34fa55278620c11e0a25b139e2034ad63ab75b80.jpg)

其中我们现在明确地在 $\pmb{\theta}$ 上放置一个高斯先验 $p(\pmb\theta)=\mathcal{N}\big(\pmb m_{0},\,\pmb S_{0}\big)$，这将参数向量转换为一个随机变量。这允许我们写出图 9.8 中对应的图形模型，其中我们明确地指出了高斯先验的参数。完整的概率模型，即观察到的和未观察到的随机变量的联合分布，$y$ 和 $\pmb{\theta}$ 分别是

$$
p(y,\pmb\theta\,|\,\pmb x)=p(y\,|\,\pmb x,\pmb\theta)p(\pmb\theta)\,.
$$
### 9.3.2 先验预测

在实践中，我们通常并不特别关注参数值 $\pmb{\theta}$ 本身。相反，我们更关注使用这些参数值所做出的预测。在贝叶斯框架中，我们利用参数分布，并在进行预测时对所有合理的参数设置进行平均。更具体地说，为了在输入 $\pmb{x}_{*}$ 处进行预测，我们对 $\pmb{\theta}$ 进行积分，得到

$$
p(y_{*}\,|\,\pmb{x}_{*})=\int p(y_{*}\,|\,\pmb{x}_{*},\pmb{\theta})p(\pmb{\theta})\mathrm{d}\pmb{\theta}=\mathbb{E}_{\pmb{\theta}}[p(y_{*}\,|\,\pmb{x}_{*},\pmb{\theta})]\,,
$$
这可以解释为根据先验分布 $p(\pmb{\theta})$ 对所有合理的参数 $\theta$ 进行平均预测 $y_{*}\,|\,\pmb{x}_{*},\pmb{\theta}$。需要注意的是，使用先验分布进行预测只需要指定输入 $\pmb{x}_{*}$，而不需要训练数据。

在我们的模型（9.35）中，我们选择了一个共轭（高斯）先验分布 $p(\pmb{\theta})=\mathcal{N}\big(\pmb m_{0},\,S_{0}\big)$，使得预测分布也是高斯分布（并且可以以封闭形式计算）：通过先验分布 $p(\pmb{\theta})=\mathcal{N}\big(\pmb m_{0},\,S_{0}\big)$，我们得到预测分布为

$$
\begin{array}{r}{p(y_{*}\,|\,\pmb{x}_{*})=\mathcal{N}\big(\pmb{\phi}^{\top}(\pmb{x}_{*})\pmb{m}_{0},\,\pmb{\phi}^{\top}(\pmb{x}_{*})S_{0}\phi(\pmb{x}_{*})+\sigma^{2}\big)\,,}\end{array}
$$
其中我们利用了以下事实：(i) 由于共轭性（见第6.6节）和高斯分布的边缘化性质（见第6.5节），预测是高斯分布；(ii) 高斯噪声是独立的，因此

$$
\mathbb{V}[y_{*}]=\mathbb{V}_{\pmb\theta}[\pmb\phi^{\top}(\pmb{x}_{*})\pmb\theta]+\mathbb{V}_{\epsilon}[\epsilon]\,,
$$
以及(iii) $y_{*}$ 是 $\pmb{\theta}$ 的线性变换，因此我们可以应用第6.50节和第6.51节中的规则来计算预测的均值和协方差。在（9.38）中，预测方差中的项 $\phi^{\top}(\bar{\pmb{x}_{*}})\bar{\pmb{S}_{0}}\bar{\phi(\pmb{x}_{*})}$ 明确地考虑了参数 $\pmb{\theta}$ 的不确定性，而 $\sigma^{2}$ 是由于测量噪声引起的不确定性贡献。

如果我们对预测无噪声函数值 $f(\pmb{x}_{*})\ =\ {\boldsymbol{\phi}}^{\top}({\pmb{x}}_{*}){\pmb{\theta}}$ 感兴趣，而不是噪声污染的目标 $y_{*}$，我们得到

$$
p(f(\pmb{x}_{*}))=\mathcal{N}\big(\pmb{\phi}^{\top}(\pmb{x}_{*})\pmb{m}_{0},\,\pmb{\phi}^{\top}(\pmb{x}_{*})\pmb{S}_{0}\phi(\pmb{x}_{*})\big)\,,
$$
这仅与（9.38）在预测方差中省略了噪声方差 $\sigma^{2}$ 有所不同。

**注释**（函数分布）：由于我们可以使用一组样本 $\pmb{\theta}_{i}$ 来表示分布 $p(\pmb{\theta})$，并且每个样本 $\pmb{\theta}_{i}$ 产生一个函数 $\check{f}_{i}\check{(\cdot)}\,=\,\overline{{\pmb{\theta}}}_{i}^{\top}\phi(\cdot)$，因此参数分布 $p(\pmb{\theta})$ 引入了一个函数分布 $p(f(\cdot))$。这里我们使用符号 $(\cdot)$ 来明确表示一个函数关系。$\blacklozenge$ 参数分布 $p(\pmb{\theta})$ 引入了一个函数分布。

![先验函数分布](images/57f257ab9853b845d8622072cd70baa5fac7bd6eb5517decfec47f77f52bc7a2.jpg)

图9.9 先验函数分布。 (a) 函数分布由均值函数（黑色线）和边缘不确定性（阴影部分）表示，分别代表67%和95%的置信区间；(b) 由先验分布产生的函数样本，该先验分布包括一些来自该先验的函数样本。

该图可视化了由参数先验引入的函数先验分布（阴影区域：深灰色：67%置信区间；浅灰色：95%置信区间），包括一些来自该先验的函数样本。

一个函数样本是通过首先从参数分布 $p(\pmb{\theta})$ 中采样一个参数向量 $\pmb{\theta}_{i}\,\sim\,p(\pmb{\theta})$ 得到的，$\bar{f_{i}(\cdot)}\,=\,\theta_{i}^{\top}\bar{\phi(\cdot)}$。我们使用了200个输入位置 $x_{*}\ \in\ [-5,5]$，并应用特征函数 $\phi(\cdot)$。图9.9中的不确定性（由阴影区域表示）仅由于参数不确定性，因为我们考虑了无噪声的预测分布（9.40）。

到目前为止，我们研究了使用参数先验 $p(\pmb{\theta})$ 进行预测的计算。然而，当我们有一个参数后验（给定一些训练数据 $\mathcal{X},~\mathcal{Y})$ 时，预测和推断的原则与（9.37）相同——我们只需要用后验 $p(\pmb{\theta}\,|\,\mathcal X,\mathcal y)$ 替换先验 $p(\pmb{\theta})$。在接下来的部分中，我们将详细推导后验分布，然后使用它来进行预测。

### 9.3.3 后验分布  

给定输入 $\pmb{x}_{n} \in \mathbb{R}^{D}$ 和相应的观测值 $y_{n} \in \mathbb{R}$，$n = 1, \ldots, N$，我们使用贝叶斯定理计算参数的后验分布为  

$$
p(\pmb{\theta} \mid \mathcal{X}, \mathcal{y}) = \frac{p(\mathcal{y} \mid \mathcal{X}, \pmb{\theta}) p(\pmb{\theta})}{p(\mathcal{y} \mid \mathcal{X})}\,,
$$  

其中 $\mathcal{X}$ 是训练输入的集合，$\mathcal{y}$ 是相应的训练目标的集合。此外，$p(\mathcal{Y} \mid \mathcal{X}, \pmb{\theta})$ 是似然函数，$p(\pmb{\theta})$ 是参数先验，且  

$$
p(\mathcal{Y} \mid \mathcal{X}) = \int p(\mathcal{Y} \mid \mathcal{X}, \pmb{\theta}) p(\pmb{\theta}) \mathrm{d} \pmb{\theta} = \mathbb{E}_{\pmb{\theta}}[p(\mathcal{Y} \mid \mathcal{X}, \pmb{\theta})]\,,
$$  

称为边缘似然证据。边缘似然证据是独立于参数 $\pmb{\theta}$ 的，并确保后验分布归一化，即它积分等于 1。我们可以将边缘似然证据视为在先验分布 $p(\pmb{\theta})$ 下的似然函数的平均值。

定理 9.1 (参数后验)。在我们的模型 (9.35) 中，参数后验 (9.41) 可以在闭式形式中计算为  

$$
\begin{array}{r l r}
&{}&{p(\pmb{\theta} \mid \mathcal{X}, \mathcal{Y}) = \mathcal{N}(\pmb{\theta} \mid \pmb{m}_{N}, \pmb{S}_{N})\,,} \\
&{}&{\pmb{S}_{N} = (\pmb{S}_{0}^{-1} + \sigma^{-2} \pmb{\Phi}^{\top} \pmb{\Phi})^{-1}\,,} \\
&{}&{\pmb{m}_{N} = \pmb{S}_{N} (\pmb{S}_{0}^{-1} \pmb{m}_{0} + \sigma^{-2} \pmb{\Phi}^{\top} \pmb{y})\,,}
\end{array}
$$  

其中下标 $N$ 表示训练集的大小。

证明 贝叶斯定理告诉我们，后验 $p(\pmb{\theta} \mid \mathcal{X}, \mathcal{Y})$ 与似然函数 $p(\mathcal{Y} \mid \mathcal{X}, \pmb{\theta})$ 和先验 $p(\pmb{\theta})$ 的乘积成比例：  

$$
\begin{array}{r l}
&{\mathrm{后验}\qquad p(\pmb{\theta} \mid \mathcal{X}, \mathcal{Y}) = \frac{p(\mathcal{Y} \mid \mathcal{X}, \pmb{\theta}) p(\pmb{\theta})}{p(\mathcal{Y} \mid \mathcal{X})}} \\
&{\mathrm{似然}\qquad p(\mathcal{Y} \mid \mathcal{X}, \pmb{\theta}) = \mathcal{N}(\pmb{y} \mid \pmb{\Phi} \pmb{\theta}, \sigma^{2} \pmb{I})} \\
&{\mathrm{先验}\qquad p(\pmb{\theta}) = \mathcal{N}(\pmb{\theta} \mid \pmb{m}_{0}, \pmb{S}_{0})\,.}
\end{array}
$$  

我们不需要直接考虑先验和似然函数的乘积，而是将其转换到对数空间，并通过完成平方来求解后验的均值和协方差。

似然函数和先验函数的和为  

$$
\begin{array}{r l r}
&{\log \mathcal{N}(\pmb{y} \mid \pmb{\Phi} \pmb{\theta}, \sigma^{2} \pmb{I}) + \log \mathcal{N}(\pmb{\theta} \mid \pmb{m}_{0}, \pmb{S}_{0})} &{(9.45a)} \\
&{= -\frac{1}{2} \left( \sigma^{-2} (\pmb{y} - \pmb{\Phi} \pmb{\theta})^{\top} (\pmb{y} - \pmb{\Phi} \pmb{\theta}) + (\pmb{\theta} - \pmb{m}_{0})^{\top} \pmb{S}_{0}^{-1} (\pmb{\theta} - \pmb{m}_{0}) \right) + \mathrm{const}}
\end{array}
$$  

其中常数项不依赖于 $\pmb{\theta}$。我们忽略这个常数项。现在我们分解 (9.45b)，得到  
$$
\begin{align}
&-\frac{1}{2} \left( \sigma^{-2} \pmb{y}^{\top} \pmb{y} - 2 \sigma^{-2} \pmb{y}^{\top} \pmb{\Phi} \pmb{\theta} + \pmb{\theta}^{\top} \sigma^{-2} \pmb{\Phi}^{\top} \pmb{\Phi} \pmb{\theta} + \pmb{\theta}^{\top} \pmb{S}_{0}^{-1} \pmb{\theta} \right.\\
&\left.- m_{0}^{\top} \pmb{S}_{0}^{-1} \pmb{\theta} + \pmb{m}_{0}^{\top} \pmb{S}_{0}^{-1} \pmb{m}_{0} \right) \\
= &-\frac{1}{2} \left( \pmb{\theta}^{\top} \left( \sigma^{-2} \pmb{\Phi}^{\top} \pmb{\Phi} + \pmb{S}_{0}^{-1} \right) \pmb{\theta} - 2 \left( \sigma^{-2} \pmb{\Phi}^{\top} \pmb{y} + \pmb{S}_{0}^{-1} \pmb{m}_{0} \right)^{\top} \pmb{\theta} \right) + \mathrm{const}\,.
\end{align}
$$

其中常数项包含 (9.46a) 中的黑项，这些项不依赖于 $\pmb{\theta}$。橙色项是线性项，蓝色项是二次项。通过检查 (9.46b)，我们发现这个方程是关于 $\pmb{\theta}$ 的二次方程。由于未归一化的对数后验分布是负的二次形式，这意味着后验是高斯分布，即  

$$
p(\pmb{\theta} \mid \mathcal{X}, \mathcal{Y}) = \exp(\log p(\pmb{\theta} \mid \mathcal{X}, \mathcal{Y})) \propto \exp(\log p(\mathcal{Y} \mid \mathcal{X}, \pmb{\theta}) + \log p(\pmb{\theta})) \tag{9.47a}$$
$$
\propto \exp \left( -\frac{1}{2} \left( \pmb{\theta}^{\top} \left( \sigma^{-2} \pmb{\Phi}^{\top} \pmb{\Phi} + \pmb{S}_{0}^{-1} \right) \pmb{\theta} - 2 \left( \sigma^{-2} \pmb{\Phi}^{\top} \pmb{y} + \pmb{S}_{0}^{-1} \pmb{m}_{0} \right)^{\top} \pmb{\theta} \right) \right)\,,\tag{9.47b}
$$  

其中我们使用了 (9.46b) 的结果。

接下来的任务是将这个未归一化的高斯分布转换为 $\mathcal{N}(\pmb{\theta} \mid \pmb{m}_{N}, \pmb{S}_{N})$ 的形式，即我们需要识别均值 $\pmb{m}_{N}$ 和协方差矩阵 $\pmb{S}_{N}$。为此，我们使用完成平方的概念。所需的对数后验分布为  

$$
\begin{array}{l}
\log \mathcal{N}(\pmb{\theta} \mid \pmb{m}_{N}, \pmb{S}_{N}) = -\frac{1}{2} \left( \pmb{\theta} - \pmb{m}_{N} \right)^{\top} \pmb{S}_{N}^{-1} \left( \pmb{\theta} - \pmb{m}_{N} \right) + \mathrm{const} \\
= -\frac{1}{2} \left( \pmb{\theta}^{\top} \pmb{S}_{N}^{-1} \pmb{\theta} - 2 \pmb{m}_{N}^{\top} \pmb{S}_{N}^{-1} \pmb{\theta} + \pmb{m}_{N}^{\top} \pmb{S}_{N}^{-1} \pmb{m}_{N} \right)\,.
\end{array}
$$  

这里，我们分解了二次形式 $(\pmb{\theta} - \pmb{m}_{N})^{\top} \pmb{S}_{N}^{-1} (\pmb{\theta} - \pmb{m}_{N})$，将其分解为仅关于 $\pmb{\theta}$ 的二次项（蓝色），线性项（橙色），和常数项（黑色）。这使得我们可以通过匹配 (9.46b) 和 (9.48b) 中的彩色表达式来找到 $\pmb{S}_{N}$ 和 $\pmb{m}_{N}$，从而得到  

完成平方  

由于 $p(\pmb{\theta} \mid \mathcal{X}, \mathcal{Y}) = \mathcal{N}(\pmb{m}_{N}, \pmb{S}_{N})$，因此 $\pmb{\theta}_{\mathrm{MAP}} = \pmb{m}_{N}$。  

$$
\begin{array}{c}
\pmb{S}_{N}^{-1} = \pmb{\Phi}^{\top} \sigma^{-2} \pmb{I} \pmb{\Phi} + \pmb{S}_{0}^{-1} \\
\Longleftrightarrow \pmb{S}_{N} = (\sigma^{-2} \pmb{\Phi}^{\top} \pmb{\Phi} + \pmb{S}_{0}^{-1})^{-1}
\end{array}
$$  

且  

$$
\begin{array}{r}
\pmb{m}_{N}^{\top} \pmb{S}_{N}^{-1} = (\sigma^{-2} \pmb{\Phi}^{\top} \pmb{y} + \pmb{S}_{0}^{-1} \pmb{m}_{0})^{\top} \\
\Longleftrightarrow \pmb{m}_{N} = \pmb{S}_{N} (\sigma^{-2} \pmb{\Phi}^{\top} \pmb{y} + \pmb{S}_{0}^{-1} \pmb{m}_{0})\,.
\end{array}
$$  

注释 (完成平方的一般方法)。如果给定方程  

$$
\pmb{x}^{\top} \pmb{A} \pmb{x} - 2 \pmb{a}^{\top} \pmb{x} + \mathrm{const}_{1}\,,
$$  

其中 $\pmb{A}$ 是对称且正定的，我们希望将其转换为形式  

$$
(\pmb{x} - \pmb{\mu})^{\top} \pmb{\Sigma} (\pmb{x} - \pmb{\mu}) + \mathrm{const}_{2}\,,
$$  

我们可以通过设置  

$$
\begin{array}{l}
\pmb{\Sigma} := \pmb{A}\,, \\
\pmb{\mu} := \pmb{\Sigma}^{-1} \pmb{a}
\end{array}
$$  

和 $\mathrm{const}_{2} = \mathrm{const}_{1} - \pmb{\mu}^{\top} \pmb{\Sigma} \pmb{\mu}$ 来实现。

我们看到 (9.47b) 中指数内的项可以表示为 (9.51) 的形式，其中  

$$
\begin{array}{r l}
&\pmb{A} := \sigma^{-2} \pmb{\Phi}^{\top} \pmb{\Phi} + \pmb{S}_{0}^{-1}\,, \\
&\pmb{a} := \sigma^{-2} \pmb{\Phi}^{\top} \pmb{y} + \pmb{S}_{0}^{-1} \pmb{m}_{0}\,.
\end{array}
$$  

由于在方程 (9.46a) 中识别 $\pmb{A}, \pmb{a}$ 可能比较困难，将这些方程转换为形式 (9.51) 有助于将二次项、线性项和常数项分离，从而简化找到所需解的过程。

### 9.3.4 后验预测  

在（9.37）中，我们使用参数先验 $p(\pmb\theta)$ 计算了测试输入 $\pmb{x}_{*}$ 的预测分布 $y_{*}$。原则上，使用参数后验 $p(\pmb\theta\,|\,\mathcal X,\mathcal y)$ 进行预测并不本质不同，因为在我们的共轭模型中，先验和后验都是高斯分布（参数不同）。因此，通过遵循与第9.3.2节相同的推理，我们得到后验预测分布

$$
\begin{array}{l}
p(y_{*}\,|\,\mathcal{X},\mathcal{Y},\pmb{x}_{*})=\int p(y_{*}\,|\,\pmb{x}_{*},\pmb{\theta})p(\pmb{\theta}\,|\,\mathcal{X},\mathcal{Y})\mathrm{d}\pmb{\theta} \\
\qquad\qquad=\int\mathcal{N}\big(y_{*}\,|\,\phi^{\top}(\pmb{x}_{*})\pmb{\theta},\,\sigma^{2}\big)\mathcal{N}\big(\pmb{\theta}\,|\,\pmb{m}_{N},\,\pmb{S}_{N}\big)\mathrm{d}\pmb{\theta} \\
\qquad\qquad=\mathcal{N}\big(y_{*}\,|\,\phi^{\top}(\pmb{x}_{*})\pmb{m}_{N},\,\phi^{\top}(\pmb{x}_{*})\pmb{S}_{N}\phi(\pmb{x}_{*})+\sigma^{2}\big)\,.
\end{array}
$$

$$
\begin{array}{r l}
\mathbb{E}[y_{*}\mid\mathcal{X},\mathcal{Y},\pmb{x}_{*}] &= \\
\boldsymbol{\phi}^{\top}(\pmb{x}_{*})\pmb{m}_{N} &= \\
\boldsymbol{\phi}^{\top}(\pmb{x}_{*})\pmb{\theta}_{\mathrm{MAP}}.
\end{array}
$$

项 $\boldsymbol{\phi}^{\intercal}(\pmb{x}_{*})S_{N}\boldsymbol{\phi}(\pmb{x}_{*})$ 反映了参数 $\pmb{\theta}$ 的后验不确定性。注意 $S_{N}$ 依赖于训练输入通过 $\Phi$；见（9.43b）。预测均值 $\boldsymbol{\phi}^{\top}(\mathbf{x}_{*})\mathbf{m}_{N}$ 与使用 MAP 估计 $\pmb{\theta}_{\mathrm{MAP}}$ 的预测一致。

注释（边缘似然与后验预测分布）. 通过替换（9.57a）中的积分，预测分布可以等价地写为期望 $\mathbb{E}_{\pmb{\theta}\,|\,\mathcal{X},\mathcal{Y}}[p(y_{*}\,|\,\pmb{x}_{*},\pmb{\theta})]$，其中期望是相对于参数后验 $p(\pmb\theta\,|\,\mathcal X,\mathcal y)$ 取的。

以这种方式写出后验预测分布突显了其与边缘似然（9.42）的密切相似性。边缘似然与后验预测分布之间的关键区别在于（i）边缘似然可以视为预测训练目标 $\pmb{y}$ 而不是测试目标 $y_{*}$，（ii）边缘似然相对于参数先验而不是参数后验进行平均。$\diamondsuit$

注释（无噪声函数值的均值和方差）. 在许多情况下，我们不关心（有噪声的）观测 $y_{*}$ 的预测分布 $p(y_{*}\,|\,\mathcal{X},\mathcal{Y},\pmb{x}_{*})$。相反，我们希望获得无噪声函数值 $f(\pmb{x}_{*})=\boldsymbol{\phi}^{\intercal}(\pmb{x}_{*})\pmb{\theta}$ 的分布。我们通过利用均值和方差的性质来确定相应的矩，得到

$$
\begin{array}{r l}
\mathbb{E}[f(\pmb{x}_{*})\,|\,\mathcal{X},\mathcal{Y}] &= \mathbb{E}_{\pmb{\theta}}[\phi^{\top}(\pmb{x}_{*})\pmb{\theta}\,|\,\mathcal{X},\mathcal{Y}] = \phi^{\top}(\pmb{x}_{*})\mathbb{E}_{\pmb{\theta}}[\pmb{\theta}\,|\,\mathcal{X},\mathcal{Y}] \\
&= \phi^{\top}(\pmb{x}_{*})\pmb{m}_{N} = \pmb{m}_{N}^{\top}\phi(\pmb{x}_{*})\,, \\
\mathbb{V}_{\pmb{\theta}}[f(\pmb{x}_{*})\,|\,\mathcal{X},\mathcal{Y}] &= \mathbb{V}_{\pmb{\theta}}[\phi^{\top}(\pmb{x}_{*})\pmb{\theta}\,|\,\mathcal{X},\mathcal{Y}] \\
&= \phi^{\top}(\pmb{x}_{*})\mathbb{V}_{\pmb{\theta}}[\pmb{\theta}\,|\,\mathcal{X},\mathcal{Y}]\phi(\pmb{x}_{*}) \\
&= \phi^{\top}(\pmb{x}_{*})S_{N}\phi(\pmb{x}_{*})\,.
\end{array}
$$

我们看到，预测均值与有噪声观测的预测均值相同，因为噪声的均值为0，而预测方差仅相差 $\sigma^{2}$，这是测量噪声的方差：当我们预测有噪声的函数值时，需要将 $\sigma^{2}$ 作为不确定性来源，但在无噪声预测中不需要这个项。这里唯一的剩余不确定性来自参数后验。$\diamondsuit$

注释（函数分布）. 我们积分掉 $\pmb{\theta}$ 诱导了一个函数分布：如果我们从参数后验 $p(\pmb\theta\,|\,\mathcal X,\mathcal y)$ 中采样 $\mathbf{\boldsymbol{\theta}}_{i}\,\sim$，我们得到一个函数实现 $\pmb\theta_{i}^{\top}\phi(\cdot)$。这个函数分布的均值函数，即所有期望值 $\mathbb{E}_{\theta}[f(\cdot)\,|\,\theta,\mathcal{X},\mathcal{Y}]$ 的集合，是 $\mathbf{\mathit{m}}_{N}^{\top}\mathbf{\phi}(\cdot)$。函数的方差，即函数 $f(\cdot)$ 的方差，是 $\boldsymbol{\phi}^{\intercal}(\cdot)\boldsymbol{S}_{\scriptscriptstyle N}\bar{\boldsymbol{\phi}}(\cdot)$。$\diamondsuit$

### 示例 9.8 (后验函数分布)  

让我们重新审视五次多项式的贝叶斯线性回归问题。我们选择参数先验为 $p(\pmb\theta)=\mathcal{N}\big(\mathbf0,\,\frac{1}{4}\pmb I\big)$ 。图 9.9 展示了由参数先验诱导的函数先验以及从该先验中抽取的样本函数。  

图 9.10 显示了通过贝叶斯线性回归得到的函数后验。训练数据集如 (a) 所示；(b) 展示了函数的后验分布，包括通过最大似然估计和 MAP 估计得到的函数。使用 MAP 估计得到的函数也对应于贝叶斯线性回归中的后验均值函数。图 (c) 展示了在该函数后验下的一些可能的函数实现（样本）。  

![](images/3903193385c85c2611dfeddc4eb53f36482caba625c940127ddc7f8e351174ec.jpg)  

图 9.11 展示了由参数后验诱导的函数后验分布。对于不同的多项式阶数 $M$ ，左图展示了最大似然函数 $\pmb\theta_{\mathrm{ML}}^{\top}\phi(\cdot)$ ，MAP 函数 $\pmb\theta_{\mathrm{MAP}}^{\top}\phi(\cdot)$ （这与后验均值函数相同），以及通过贝叶斯线性回归得到的 67% 和 95% 预测置信区间，用阴影区域表示。  

右图展示了函数后验的样本：我们从参数后验中抽取参数 $\pmb{\theta}_{i}$ ，并计算函数 $\bar{\phi}^{\top}({\pmb x}_{*}){\pmb\theta}_{i}$ ，这是在函数后验分布下的一个实现。对于低阶多项式，参数后验不允许参数变化太多：抽取的函数几乎相同。当我们通过增加参数使模型更加灵活（即最终得到高阶多项式），这些参数将不会被后验充分约束，抽取的函数可以很容易地通过视觉区分。我们还可以在左图中看到不确定性如何增加，尤其是在边界处。  

尽管对于七次多项式，MAP 估计提供了合理的拟合，但贝叶斯线性回归模型还告诉我们  

![](images/a142586eb6c91cf1c769aafdeb8e9c4eef788f60d09e33ca8e165582b1f8f90c.jpg)  
9.3 贝叶斯线性回归  

![](images/087afa32291b07ced4d623ad41ebb89fd90b82cd6e5a32b51a94fe1c994cffbf.jpg)  
(b) 多项式阶数 $M=5$ 的函数后验分布（左）和函数后验的样本（右）。  

图 9.11 贝叶斯线性回归。左图：阴影区域表示 67%（深灰色）和 95%（浅灰色）的预测置信区间。贝叶斯线性回归模型的均值与 MAP 估计一致。预测不确定性是噪声项和后验参数不确定性之和，这取决于测试输入的位置。右图：函数后验的样本。  

![](images/2f06cd0aea0d7e77324b76a2cd56cd53c22dc6f60b86a647d94b58627b970ce2.jpg)  
(c) 多项式阶数 $M=7$ 的函数后验分布（左）和函数后验的样本（右）。  

后验不确定性非常大。当我们在决策系统中使用这些预测时，这些信息可能是关键的，因为不良决策可能会产生重大后果（例如，在强化学习或机器人技术中）。

### 9.3.5 计算边缘似然  

在第8.6.2节中，我们强调了边缘似然对于贝叶斯模型选择的重要性。接下来，我们将计算具有共轭高斯先验参数的贝叶斯线性回归的边缘似然，即我们在本章讨论的设置。  

为了回顾，我们考虑以下生成过程：  

$$
\begin{array}{r}{\pmb{\theta}\sim\mathcal{N}\big(\pmb{m}_{0},\,\pmb{S}_{0}\big)\qquad\qquad}\\ {\quad\quad\quad\quad\quad\quad\quad\quad\quad\pmb{y}_{n}\,|\,\pmb{x}_{n},\pmb{\theta}\sim\mathcal{N}\big(\pmb{x}_{n}^{\top}\pmb{\theta},\,\sigma^{2}\big)\,,}\end{array}
$$

边缘似然可以解释为在先验下的期望似然，即$\mathbb{E}_{\pmb{\theta}}[p(\mathcal{Y}\mid\mathcal{X},\pmb{\theta})]$。对于$n=1,\cdot\cdot\cdot,N$，边缘似然由下式给出：  

$$
\begin{array}{l}{{\displaystyle p(\mathcal{Y}\,|\,\mathcal{X})=\int p(\mathcal{Y}\,|\,\mathcal{X},\pmb{\theta})p(\pmb{\theta})\mathrm{d}\pmb{\theta}}}\\ {{\displaystyle\qquad=\int\mathcal{N}\big(\pmb{y}\,|\,\pmb{X}\pmb{\theta},\,\sigma^{2}\pmb{I}\big)\mathcal{N}\big(\pmb{\theta}\,|\,\pmb{m}_{0},\,S_{0}\big)\mathrm{d}\pmb{\theta}\,,}}\end{array}
$$  

其中我们积分出模型参数$\pmb{\theta}$。我们分两步计算边缘似然：首先，我们证明边缘似然是高斯分布（作为$\pmb{y}$的分布）；其次，我们计算该高斯分布的均值和协方差。  

1. 边缘似然是高斯分布：根据第6.5.2节，我们知道（i）两个高斯随机变量的乘积是一个（未归一化的）高斯分布，（ii）高斯随机变量的线性变换仍然是高斯分布。在（9.61b）中，我们要求一个线性变换将$\mathcal{N}(\pmb{y}\,|\,\pmb{X\theta},\,\sigma^{2}\pmb{I})$转换为$\mathcal{N}(\pmb{\theta}\,|\,\pmb{\mu},\,\pmb{\Sigma})$的形式，其中$\pmb{\mu},\pmb{\Sigma}$是某些值。一旦完成这一转换，积分可以以闭式形式求解。结果是两个高斯分布乘积的归一化常数。归一化常数本身具有高斯形状；见（6.76）。  

2. 均值和协方差。我们通过利用随机变量的仿射变换的均值和协方差的标准结果来计算边缘似然的均值和协方差矩阵；见第6.4.4节。边缘似然的均值计算为  

$$ \begin{aligned} \operatorname{\mathbb{E}}[\operatorname{\mathcal{Y}}|\operatorname{\mathcal{X}}] &= \operatorname{\mathbb{E}}_{\theta, \epsilon}[\boldsymbol{X}\pmb{\theta} + \epsilon] \\ &= \boldsymbol{X} \operatorname{\mathbb{E}}_{\theta}[\pmb{\theta}] \\ &= \boldsymbol{X} \pmb{m}_{0}. \end{aligned} $$

注意$\mathbf{\boldsymbol{\epsilon}}\,\sim\mathcal{N}\mathbf{\left(0,\,\sigma^{2}I\right)}$是一个独立同分布的随机变量向量。协方差矩阵给出为  

$$
\begin{array}{r l}&{\operatorname{Cov}[\mathcal{Y}|\mathcal{X}]=\operatorname{Cov}_{\pmb{\theta},\pmb{\epsilon}}[\pmb{X}\pmb{\theta}+\pmb{\epsilon}]=\operatorname{Cov}_{\pmb{\theta}}[\pmb{X}\pmb{\theta}]+\sigma^{2}\pmb{I}}\\ &{\qquad\qquad=\pmb{X}\operatorname{Cov}_{\pmb{\theta}}[\pmb{\theta}]\pmb{X}^{\top}+\sigma^{2}\pmb{I}=\pmb{X}\pmb{S}_{0}\pmb{X}^{\top}+\sigma^{2}\pmb{I}\,.}\end{array}
$$

因此，边缘似然为  

$$
\begin{array}{l r}{{p(\mathcal{Y}\,|\,\mathcal{X})=(2\pi)^{-\frac{N}{2}}\operatorname*{det}(\pmb{X}\pmb{S}_{0}\pmb{X}^{\top}+\sigma^{2}\pmb{I})^{-\frac{1}{2}}}}&{{(9.64\pmb{x})}}\\ {{\quad\quad\quad\quad\cdot\exp\big(-\frac{1}{2}(\pmb{y}-\pmb{X}\pmb{m}_{0})^{\top}(\pmb{X}\pmb{S}_{0}\pmb{X}^{\top}+\sigma^{2}\pmb{I})^{-1}(\pmb{y}-\pmb{X}\pmb{m}_{0})\big)}}&{{\pmb{x}\in\mathbb{Z}.}}\end{array}
$$  

![](images/f532ac28bcd9959e98e2828bf1ac8a911414ec39503e685af6def192f9e50eb5.jpg)  

(a) 回归数据集，包含输入位置$x_{n}$处函数值$f(x_{n})$的噪声观测$y_{n}$（蓝色）。  

![](images/3a5a6bdfbc5af2123c26e71c343b00cd0b644c35c8401eb3c49d3dfcd35ca950.jpg)  
(b) 橙色点是噪声观测（蓝色点）在直线$\theta_{\mathrm{ML}}x$上的投影。线性回归问题的最大似然解找到一个子空间（直线），使得观测的整体投影误差（橙色线）最小化。  

$$
\mathbf{\Psi}=\mathcal{N}\big(\mathbf{\boldsymbol{y}}\mid\mathbf{\boldsymbol{X}}\mathbf{\mathit{m}}_{0},\,\mathbf{\boldsymbol{X}}\mathbf{\mathit{S}}_{0}\mathbf{\boldsymbol{X}}^{\top}+\sigma^{2}\mathbf{\boldsymbol{I}}\big)\,.
$$

由于边缘似然与后验预测分布之间有密切联系（见本节早些时候关于边缘似然和后验预测分布的备注），边缘似然的形式不应感到意外。

## 9.4 最大似然估计的正交投影

经过大量的代数运算，我们得到了最大似然估计和MAP估计。现在我们将提供最大似然估计的几何解释。考虑一个简单的线性回归设置

$$
y = x\theta + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

其中我们考虑通过原点的线性函数 $f : \mathbb{R} \rightarrow \mathbb{R}$（为了清晰，我们省略了特征）。参数 $\theta$ 确定直线的斜率。图9.12(a)显示了一个一维数据集。

对于训练数据集 $\{(x_1, y_1), \dots, (x_N, y_N)\}$，我们回顾了第9.2.1节的结果，并得到了斜率参数的最大似然估计为

$$
\theta_{\mathrm{ML}} = (X^\top X)^{-1} X^\top \pmb{y} = \frac{X^\top \pmb{y}}{X^\top X} \in \mathbb{R}
$$

其中 $X = [x_1, \dots, x_N]^\top \in \mathbb{R}^N$，$\pmb{y} = [y_1, \dots, y_N]^\top \in \mathbb{R}^N$

这意味着对于训练输入 $X$，我们得到了训练目标的最佳（最大似然）重构为

$$
\boldsymbol{X}\theta_{\mathrm{ML}} = \boldsymbol{X}\frac{\boldsymbol{X}^\top \boldsymbol{y}}{\boldsymbol{X}^\top \boldsymbol{X}} = \frac{\boldsymbol{X}\boldsymbol{X}^\top}{\boldsymbol{X}^\top \boldsymbol{X}} \boldsymbol{y}
$$

即，我们得到了 $\textbf{y}$ 和 $X\theta$ 之间最小平方误差的近似。

线性回归可以被视为求解线性方程组的一种方法。

最大似然线性回归执行正交投影。

当我们寻找 $\pmb{y} = \pmb{X}\theta$ 的解时，我们可以将线性回归视为求解线性方程组的问题。因此，我们可以联系到我们在第2章和第3章讨论的线性代数和解析几何的概念。特别是，仔细观察（9.67），我们看到最大似然估计 $\theta_{\mathrm{ML}}$ 在我们的例子中（9.65）实际上是对 $\textbf{y}$ 在由 $\boldsymbol{X}$ 张成的一维子空间上的正交投影。回忆正交投影的结果（第3.8节），我们识别 $\cfrac{X X^\top}{X^\top X}$ 作为投影矩阵，$\theta_{\mathrm{ML}}$ 作为投影到 $\mathbb{R}^N$ 中由 $\boldsymbol{X}$ 张成的一维子空间上的坐标，$\boldsymbol{X}\boldsymbol{\theta}_{\mathrm{ML}}$ 作为 $\textbf{y}$ 在这个子空间上的正交投影。

因此，最大似然解也提供了几何上最优的解，通过找到在由 $\boldsymbol{X}$ 张成的子空间中“最接近”观测值 $\textbf{y}$ 的向量，其中“最接近”意味着函数值 $y_n$ 到 $x_n\theta$ 的最小（平方）距离。这是通过正交投影实现的。图9.12(b)显示了将有噪声的观测值投影到最小化原始数据集与其投影之间平方距离的子空间上（注意 $x$ 坐标是固定的），这对应于最大似然解。

在一般的线性回归情况下，其中

$$
y = \boldsymbol{\phi}^\intercal(\mathbf{x})\boldsymbol{\theta} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

具有向量值特征 $\phi(\pmb{x}) \in \mathbb{R}^K$，我们再次可以解释最大似然结果

$$
\begin{array}{r l}
&{\pmb{y} \approx \pmb{\Phi}\pmb{\theta}_{\mathrm{ML}}}, \\
&{\pmb{\theta}_{\mathrm{ML}} = (\pmb{\Phi}^\top \pmb{\Phi})^{-1} \pmb{\Phi}^\top \pmb{y}}
\end{array}
$$

作为投影到 $\mathbb{R}^N$ 中由特征矩阵 $\Phi$ 的列张成的 $K$ 维子空间上，见第3.8.2节。

如果我们在构造特征矩阵 $\Phi$ 时使用的特征函数 $\phi_k$ 是正交归一化的（见第3.7节），我们得到一个特殊情况，其中 $\Phi$ 的列形成一个正交归一化基（见第3.5节），使得 $\boldsymbol{\Phi}^\intercal \boldsymbol{\Phi} = \boldsymbol{I}$。这将导致投影

$$
\pmb{\Phi}(\pmb{\Phi}^\top \pmb{\Phi})^{-1} \pmb{\Phi}^\top \pmb{y} = \pmb{\Phi} \pmb{\Phi}^\top \pmb{y} = \left(\sum_{k=1}^{K} \phi_k \phi_k^\top \right) \pmb{y}
$$

因此，最大似然投影只是 $\textbf{y}$ 在各个基向量 $\phi_k$ 上的投影之和，即 $\Phi$ 的列。此外，由于基的正交性，不同特征之间的耦合消失了。许多信号处理中的流行基函数，如小波和傅里叶基，都是正交基函数。

当基不是正交时，可以通过使用Gram-Schmidt过程将一组线性无关的基函数转换为正交基；见第3.8.3节和（Strang, 2003）。
## 9.5 进一步阅读  

在本章中，我们讨论了高斯似然和模型参数的共轭高斯先验下的线性回归，这允许闭式贝叶斯推理。然而，在某些应用中，我们可能希望选择不同的似然函数。例如，在二元分类设置中，我们只观察到两种可能的（分类）结果，高斯似然在这种情况下是不合适的。相反，我们可以选择伯努利似然，它将返回预测标签为 1（或 0）的概率。我们参考 Barber (2012)、Bishop (2006) 和 Murphy (2012) 的书籍，以获得分类问题的深入介绍。非高斯似然在计数数据中非常重要。计数是非负整数，此时二项式或泊松似然比高斯似然更适合。所有这些例子都属于广义线性模型（GLM）的范畴，这是一种灵活的线性回归泛化，允许响应变量具有除高斯分布以外的误差分布。GLM 通过允许线性模型与观测值通过光滑的函数 $\sigma(\cdot)$ 相关联，从而泛化了线性回归，其中 $\sigma(\cdot)$ 可能是非线性的，使得 $y = \sigma(f(\pmb{x}))$，而 $\boldsymbol{f}(\pmb{x}) = \pmb{\theta}^{\top}\boldsymbol{\phi}(\pmb{x})$ 是从 (9.13) 式得到的线性回归模型。因此，我们可以将广义线性模型视为函数复合 $y = \sigma \circ f$，其中 $f$ 是线性回归模型，$\sigma$ 是激活函数。尽管我们讨论的是“广义线性模型”，但输出 $y$ 已不再是参数 $\pmb{\theta}$ 的线性函数。在逻辑回归中，我们选择逻辑 sigmoid 函数
$$
\sigma(f) = \frac{1}{1 + \exp(-f)} \in [0, 1]
$$
，它可以解释伯努利随机变量 $y \in \{0, 1\}$ 观测到 $y=1$ 的概率。函数 $\sigma(\cdot)$ 称为转移函数或激活函数，其逆称为典范链函数。从这个角度来看，广义线性模型也是（深度）前馈神经网络的构建块：如果我们考虑广义线性模型 $\pmb{y} = \sigma(\pmb{A}\pmb{x} + \pmb{b})$，其中 $\pmb{A}$ 是权重矩阵，$\pmb{b}$ 是偏置向量，我们可以将这个广义线性模型视为具有激活函数 $\sigma(\cdot)$ 的单层神经网络。我们现在可以通过递归复合这些函数
$$
\begin{array}{c}
\pmb{x}_{k+1} = \pmb{f}_{k}(\pmb{x}_{k}) \\
\pmb{f}_{k}(\pmb{x}_{k}) = \sigma_{k}(\pmb{A}_{k}\pmb{x}_{k} + \pmb{b}_{k})
\end{array}
$$
其中 $k = 0, \ldots, K - 1$，$\pmb{x}_{0}$ 是输入特征，$\pmb{x}_{K} = \pmb{y}$ 是观测输出，使得 $\pmb{f}_{K-1} \circ \cdots \circ \pmb{f}_{0}$ 是 $K$ 层的深度神经网络。因此，这种深度神经网络的构建块是分类广义线性模型。广义线性模型是深度神经网络的构建块。  
逻辑回归 逻辑 sigmoid  
转移函数 激活函数 典范链函数  
对于普通的线性回归，激活函数只是恒等函数。有关 GLM 和深度网络之间关系的精彩文章，请参阅 https://tinyurl.com/glm-dnn。  
广义线性模型 (9.72) 中定义的。神经网络 (Bishop, 1995; Goodfellow et al., 2016) 比线性回归模型更具有表达力和灵活性。然而，最大似然参数估计是一个非凸优化问题，而在完全贝叶斯设置中参数的边际化是分析上不可解的。  
高斯过程  
核技巧  
变量选择  
LASSO  
我们简要提到了参数分布诱导回归函数分布的事实。高斯过程 (Rasmussen and Williams, 2006) 是一种回归模型，其中函数分布的概念是核心。与其将分布置于参数上，高斯过程直接在函数空间上放置分布，而无需通过参数的“迂回”。为此，高斯过程利用了核技巧 (Schölkopf and Smola, 2002)，这允许我们仅通过查看相应的输入 $\mathbf{\Delta}\mathbf{x}_{i}, \mathbf{\Delta}\mathbf{x}_{j}$ 来计算两个函数值 $f(\pmb{x}_{i}), f(\pmb{x}_{j})$ 之间的内积。高斯过程与贝叶斯线性回归和支持向量回归密切相关，也可以解释为具有无限隐藏单元的单隐藏层的贝叶斯神经网络 (Neal, 1996; Williams, 1997)。关于高斯过程的优秀介绍可以在 MacKay (1998) 和 Rasmussen and Williams (2006) 中找到。  
在本章讨论中，我们专注于高斯参数先验，因为它们允许线性回归模型的闭式推理。然而，即使在具有高斯似然的回归设置中，我们也可以选择非高斯先验。考虑一个输入 $\pmb{x} \in \mathbb{R}^{D}$ 的情况，以及一个较小的训练集，其大小为 $N \ll D$。这意味着回归问题是欠定的。在这种情况下，我们可以选择一个参数先验，使其强制稀疏性，即一个尝试将尽可能多的参数设置为 0 的先验（变量选择）。这种先验比高斯先验提供了更强的正则化，这通常会导致模型预测精度和可解释性的提高。拉普拉斯先验是用于此目的的一个常见例子。具有拉普拉斯先验的线性回归模型等同于具有 L1 正则化的线性回归 (LASSO) (Tibshirani, 1996)。拉普拉斯分布在零处尖锐地集中 (其一阶导数是不连续的)，并且比高斯分布更集中于零，这鼓励参数为 0。因此，非零参数对于回归问题至关重要，这也是我们称之为“变量选择”的原因。