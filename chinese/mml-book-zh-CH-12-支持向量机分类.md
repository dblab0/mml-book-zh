# 第十二章 支持向量机分类

一个例子是如果结果是有序的，比如小号、中号和大号的T恤。二元分类  

输入示例 ${\pmb x}_{n}$ 也可以被称为输入、数据点、特征或实例。类别 对于概率模型，使用 $\{0,1\}$ 作为二元表示是数学上方便的；参见例6.12后的注释。  

在许多情况下，我们希望我们的机器学习算法预测多个（离散）结果之一。例如，电子邮件客户端将邮件分为私人邮件和垃圾邮件，这有两种结果。另一个例子是望远镜识别夜空中是否为星系、恒星或行星。通常有少量的结果，更重要的是，这些结果通常没有额外的结构。在本章中，我们考虑输出为二元值的预测器，即只有两种可能的结果。这个机器学习任务称为 二元分类 。这与第9章中考虑的具有连续值输出的预测问题不同。  

对于二元分类，标签/输出可以取的值集是二元的，对于本章，我们用 $\{+1,-1\}$ 表示它们。换句话说，我们考虑的形式为

$$
f:\mathbb{R}^{D}\rightarrow\left\{+1,-1\right\}.
$$
从第8章中回忆，我们将每个示例（数据点） ${\pmb x}_{n}$ 表示为 $D$ 个实数的特征向量。标签通常分别称为正类和负类。应注意不要推断出正类 $+1$ 的直观属性。例如，在癌症检测任务中，患有癌症的患者通常被标记为 $+1$。原则上，任何两个不同的值都可以使用，例如 $\{\mathrm{True},\mathrm{False}\}$, $\{0,1\}$ 或 $\{\mathrm{red,blue}\}$。二元分类问题已经得到了充分的研究，我们将在第12.6节中推迟对其他方法的综述。  

我们介绍了一种称为支持向量机（SVM）的方法，它解决了二元分类任务。与回归类似，我们有一个监督学习任务，其中我们有一个 ${\pmb x}_{n}\in\mathbb{R}^{D}$ 以及它们对应的（二元）标签 $y_{n}\in\{+1,-1\}$，我们希望估计模型参数，以最小化分类误差。类似于第9章，我们考虑一个线性模型，并将非线性隐藏在示例的变换 $\phi$ 中（9.13）。我们将在第12.4节中再次回顾 $\phi$。  

SVM 在许多应用中提供了最先进的结果，并且具有坚实的理论保证（Steinwart 和 Christmann, 2008）。我们选择使用 SVM 来说明二元分类的两个主要原因。首先，SVM 允许以几何方式思考监督机器学习。在第9章中，我们从概率模型的角度考虑了机器学习问题，并使用最大似然估计和贝叶斯推断来解决它，而在这里我们将考虑一个几何推理的方法来解决机器学习任务。这依赖于我们在第3章中讨论的概念，如内积和投影。第二个使我们发现 SVM 有启发性的原因是，与第9章不同，SVM 的优化问题没有解析解，因此我们需要使用第7章中介绍的各种优化工具。  

SVM 对机器学习的看法与第9章的最大似然观点略有不同。最大似然观点基于对数据分布的概率模型，从而导出一个优化问题。相比之下，SVM 观点首先设计一个在训练期间要优化的特定函数，基于几何直觉。我们在第10章中已经看到了类似的情况，我们从几何原理出发推导出了主成分分析（PCA）。在 SVM 情况下，我们首先设计一个损失函数，该函数在训练数据上要最小化，遵循经验风险最小化的原则（第8.2节）。  

让我们推导出训练 SVM 对应的优化问题。直观上，我们想象可以由超平面分离的二元分类数据，如图12.1所示。这里，每个示例 ${\pmb x}_{n}$（一个二维向量）是一个二维位置 $(x_{n}^{(1)}, x_{n}^{(2)})$，而对应的二元标签 $y_{n}$ 是两种不同符号之一（橙色十字或蓝色圆盘）。“超平面”是机器学习中常用的术语，我们在第2.8节中已经遇到过。超平面是对应向量空间维度为 $D$ 的 $D-1$ 维仿射子空间。示例包括两个类别（有两种可能的标签），它们的特征（表示示例的向量的分量）排列成可以通过画一条直线来分离/分类它们的方式。  

接下来，我们形式化找到两个类别的线性分隔器的想法。我们引入了间隔的概念，然后扩展线性分隔器以允许示例落在“错误”的一侧，导致分类错误。我们提出了两种等效的SVM形式化方法：几何观点（第12.2.4节）和损失函数观点（第12.2.5节）。我们使用拉格朗日乘数推导出SVM的对偶形式（第7.2节）。对偶SVM使我们能够观察到SVM的第三种形式化方法：以每个类的示例的凸包形式（第12.3.2节）。最后，我们简要描述核函数以及如何数值求解非线性核-SVM优化问题。  

![](images/1e0267f0cf75387ffc0b8a18956a0fec1b13102fc773eef5715e630a61f2aa7b.jpg)  
图12.1 二维数据示例，说明了可以找到一个线性分类器将橙色十字与蓝色圆盘分开的直观数据。

## 12.1 分离超平面  

给定两个用向量表示的例子 $\mathbf{\Delta}\mathbf{x}_{i}$ 和 $\pmb{x}_{j}$，计算它们之间相似性的方法之一是使用内积 $\langle\pmb{x}_{i},\pmb{x}_{j}\rangle$。从第 3.2 节回顾可知，内积与两个向量之间的夹角密切相关。两个向量的内积值取决于每个向量的长度（范数）。此外，内积允许我们严格定义几何概念，如正交性和投影。  

许多分类算法的核心思想是将数据表示在 $\mathbb{R}^{D}$ 中，然后将这个空间分割，理想情况下，使得具有相同标签（且仅此标签）的例子位于同一个分割中。在二分类的情况下，空间会被分为两部分，分别对应正类和负类。我们考虑一种特别方便的分割方式，即使用超平面（线性分割）将空间分为两半。设数据空间中的一个例子 $\pmb{x}\in\mathbb{R}^{D}$，考虑一个函数  

$$
\begin{array}{r l}
&{f:\mathbb{R}^{D}\to\mathbb{R}}\\
&{\pmb{x}\mapsto f(\pmb{x}):=\left\langle\pmb{w},\pmb{x}\right\rangle+b\,,}
\end{array}
$$  

由参数 $\pmb{w}\,\in\,\mathbb{R}^{D}$ 和 $b\,\in\,\mathbb{R}$ 定义。从第 2.8 节回顾可知，超平面是仿射子空间。因此，我们定义用于分离二分类问题中两个类别的超平面为  

$$
\left\{{\pmb x}\in\mathbb{R}^{D}:f({\pmb x})=0\right\}\,.
$$  

图 12.2 显示了超平面的示意图，其中向量 $\mathbfit{w}$ 是超平面的法向量，$b$ 是截距。我们可以通过选择超平面上任意两个例子 $\mathbf{\Delta}\mathbf{x}_{a}$ 和 $\mathbf{\Delta}\mathbf{x}_{b}$，并证明它们之间的向量与 $\mathbfit{w}$ 正交来推导出 $\mathbfit{w}$ 是（12.3）中超平面的法向量。以方程形式表示，  

$$
\begin{array}{r l r}
&{}&{f(\pmb{x}_{a})-f(\pmb{x}_{b})=\left\langle\pmb{w},\pmb{x}_{a}\right\rangle+b-\left(\left\langle\pmb{w},\pmb{x}_{b}\right\rangle+b\right)}\\
&{}&{\quad=\left\langle\pmb{w},\pmb{x}_{a}-\pmb{x}_{b}\right\rangle\,,}
\end{array}
$$  

![](images/94a46bf3dbb616188dfaf758bb961e9c036325a1a5310b91a7f5d64a6b6a269e.jpg)  
图 12.2 分离超平面（12.3）的方程。 (a) 三维中的标准表示方式。 (b) 为了便于绘制，我们从侧面观察超平面。  

其中第二行通过内积的线性性质（第 3.2 节）获得。由于我们选择了 $\mathbf{\Delta}\mathbf{x}_{a}$ 和 $\mathbf{\Delta}\mathbf{x}_{b}$ 在超平面上，这意味着 $f(\pmb{x}_{a})=0$ 和 $f({\pmb x}_{b})=0$，从而 $\langle{\pmb w},{\pmb x}_{a}-{\pmb x}_{b}\rangle=0$。回忆第 2 章，我们知道当两个向量的内积为零时，它们是正交的。因此，我们得出结论 $\mathbfit{w}$ 与超平面上的任意向量正交。注释。从第 2 章回顾可知，我们可以用不同的方式思考向量。在本章中，我们将参数向量 $\mathbfit{w}$ 看作指示方向的箭头，即我们认为 $\mathbfit{w}$ 是一个几何向量。相反，我们将例子向量 $\pmb{x}$ 看作数据点（由其坐标表示），即我们认为 $\pmb{x}$ 是相对于标准基的向量坐标。$\diamondsuit$ $\mathbfit{w}$ 与超平面上的任意向量正交。  

当面对测试例子时，我们根据它位于超平面的哪一侧将其分类为正类或负类。注意（12.3）不仅定义了一个超平面，还定义了一个方向。换句话说，它定义了超平面的正侧和负侧。因此，为了分类测试例子 $\pmb{x}_{\mathrm{test}}$，我们计算函数 $f(\pmb{x}_{\mathrm{test}})$ 的值，如果 $f(\pmb{x}_{\mathrm{test}})\,\geqslant\,0$，则将其分类为 $+1$，否则分类为 $-1$。从几何上看，正类例子位于“超平面之上”，负类例子位于“超平面之下”。  

在训练分类器时，我们希望确保具有正标签的例子位于超平面的正侧，即  

$$
\langle{\pmb w},{\pmb x}_{n}\rangle+b\geqslant0\quad\mathrm{当}\quad y_{n}=+1
$$  

而具有负标签的例子位于负侧，即  

$$
\left\langle w,x_{n}\right\rangle+b<0\quad\mathrm{当}\quad y_{n}=-1\,.
$$  

参见图 12.2 以获得正类和负类例子的几何直觉。这两个条件通常以一个方程的形式呈现  

$$
y_{n}(\langle{\pmb w},{\pmb x}_{n}\rangle+b)\geqslant0\,.
$$  

方程（12.7）在将（12.5）和（12.6）两边分别乘以 $y_{n}=1$ 和 $y_{n}=-1$ 时等价于（12.5）和（12.6）。  

![](images/a6eaac9eeec5ee81eeb3f4d406b5f16ee9caf442a6b51b84559e309c02e34d5a.jpg)  
图 12.3 可能的分离超平面。有许多线性分类器（绿色线）可以将橙色十字与蓝色圆圈分开。

## 12.2 原始支持向量机

一个具有大间隔的分类器通常能很好地泛化（Steinwart 和 Christmann, 2008）。

间隔 在超平面附近可能存在两个或多个最近的示例。

基于点到超平面的距离概念，我们现在可以讨论支持向量机。对于一个线性可分的数据集 $\{({\pmb x}_{1},y_{1}),.\,.\,.\,,({\pmb x}_{N},y_{N})\}$，我们有无数个候选超平面（参见图 12.3），因此存在无数个分类器可以解决分类问题且无任何（训练）误差。为了找到一个独特的解，一个想法是选择一个最大化正负示例之间间隔的分离超平面。换句话说，我们希望正负示例之间有一个大的间隔（第 12.2.1 节）。接下来，我们将计算一个示例到超平面的距离以推导间隔。回想一下，超平面上给定点的最近点可以通过正交投影（第 3.8 节）获得。

### 12.2.1 间隔的概念

间隔的概念直观上很简单：它是分离超平面到数据集中最近示例的距离，假设数据集是线性可分的。然而，当我们试图形式化这个距离时，可能会遇到一个技术上的问题，这可能会让人困惑。这个问题在于我们需要定义一个测量距离的尺度。一个潜在的尺度是考虑数据的尺度，即 ${\pmb x}_{n}$ 的原始值。这存在一些问题，因为我们可以改变 ${\pmb x}_{n}$ 的单位和值，从而改变到超平面的距离。正如我们即将看到的，我们将基于超平面方程（12.3）本身来定义这个尺度。

考虑一个超平面 $\langle{\pmb w},{\pmb x}\rangle+b$，以及一个示例 $\mathbf{\delta}\mathbf{x}_{a}$ 如图 12.4 所示。不失一般性，我们可以假设示例 $\mathbf{\delta}\mathbf{x}_{a}$ 在超平面的正侧，即 $\langle{\pmb w},{\pmb x}_{a}\rangle+b>0$。我们希望计算 $\mathbf{\delta}\mathbf{x}_{a}$ 到超平面的距离 $r>0$。我们通过考虑 $\mathbf{\delta}\mathbf{x}_{a}$ 在超平面上的正交投影（第 3.8 节）来实现这一点，我们将这个投影表示为 $\pmb{x}_{a}^{\prime}$。由于 $\mathbfit{w}$ 与超平面正交，

![](images/42630d2186e1a716da7ed2447b8e641e26032a011bfaeefdf4bd555f7b569a6b.jpg)
图 12.4 通过向量加法表示到超平面的距离：$\pmb{x}_{a}=\pmb{x}_{a}^{\prime}+r\cfrac{\pmb{w}}{\|\pmb{w}\|}$。

超平面正交，我们知道距离 $r$ 只是这个向量 $\mathbfit{w}$ 的缩放。如果 $\mathbfit{w}$ 的长度已知，我们可以通过这个缩放因子 $r$ 来计算 $\mathbf{\delta}\mathbf{x}_{a}$ 和 $\pmb{x}_{a}^{\prime}$ 之间的绝对距离。为了方便，我们选择使用单位长度的向量（其范数为 1），通过将 $\mathbfit{w}$ 除以其范数 $\frac{\textbf{\textit{w}}}{\|\textbf{\textit{w}}\|}$ 来获得。使用向量加法（第 2.4 节），我们得到

$$
{\pmb x}_{a}={\pmb x}_{a}^{\prime}+r\frac{\pmb w}{\lVert\pmb w\rVert}\,.
$$
另一种思考 $r$ 的方式是它在由 $\pmb{w}/\|\pmb{w}\|$ 张成的子空间中的坐标。我们现在将 $\mathbf{\delta}\mathbf{x}_{a}$ 到超平面的距离表示为 $r$，如果选择 $\mathbf{\delta}\mathbf{x}_{a}$ 为距离超平面最近的点，这个距离 $r$ 就是间隔。

回想一下，我们希望正示例距离超平面超过 $r$，负示例距离超平面超过距离 $r$（负方向）。类似于 (12.5) 和 (12.6) 的组合成 (12.7)，我们将这个目标形式化为

$$
y_{n}(\langle{\pmb w},{\pmb x}_{n}\rangle+b)\geqslant r\,.
$$
换句话说，我们将要求示例至少距离超平面 $r$（正方向和负方向）合并为一个不等式。

由于我们只关心方向，我们对模型添加一个假设，即参数向量 $\mathbfit{w}$ 的长度为 1，即 $\|w\|=1$，我们使用欧几里得范数 $\|\pmb{w}\|=\sqrt{\pmb{w}^{\top}\pmb{w}}$（第 3.1 节）。这个假设也使得距离 $r$（12.8）的解释更加直观，因为它是一个长度为 1 的向量的缩放因子。

注释。 对于熟悉其他间隔定义的读者来说，我们的 $\|\pmb{w}\| = 1$ 定义与 SVM（例如 Schölkopf 和 Smola, 2002 提供的）的标准定义不同。在第 12.2.3 节中，我们将展示这两种方法的等价性。$\diamondsuit$ 我们将在第 12.4 节中看到其他内积的选择（第 3.2 节）。

将三个要求合并为一个约束优化问题

![](images/3da43516bee61f35c5ea9a93a6a60b51c8fcf222d1f68b47b943f23cc01786ca.jpg)
图 12.5 间隔的推导：$\textstyle r={\cfrac{1}{\|{\pmb{w}}\|}}$。

我们得到目标

$$
\begin{array}{r l}{\underset{\pmb{w},b,r}{\mathrm{max}}}&{\underbrace{r}_{\mathrm{marginal}}}\\ {\mathrm{subject~to}}&{\underbrace{y_{n}(\langle\pmb{w},\pmb{x}_{n}\rangle+b)\geqslant r}_{\mathrm{data~fiting}},\underbrace{\|\pmb{w}\|=1}_{\mathrm{normalization}},\quad r>0\,,}\end{array}
$$
这意味着我们希望最大化间隔 $r$，同时确保数据位于超平面的正确一侧。

注释。 间隔的概念在机器学习中被证明是非常普遍的。Vladimir Vapnik 和 Alexey Chervonenkis 使用它来证明当间隔较大时，函数类的“复杂性”较低，因此学习是可能的（Vapnik, 2000）。实际上，这个概念对于各种不同的理论分析泛化误差的方法都是有用的（Steinwart 和 Christmann, 2008；Shalev-Shwartz 和 Ben-David, 2014）。$\diamondsuit$

### 12.2.2 传统的边缘推导  

回顾一下，我们目前考虑的是线性可分的数据。  

在上一节中，我们通过观察我们只关心向量 $\mathbfit{w}$ 的方向而不是其长度，得出假设 $\|w\|=1$ 。在本节中，我们将通过不同的假设来推导边缘最大化问题。我们不选择将参数向量归一化，而是选择数据的尺度。我们选择这种尺度，使得预测器 $\langle{\pmb w},{\pmb x}\rangle+b$ 在最近的样本处的值为 1。我们也将数据集中最近于超平面的样本记为 $\mathbf{\delta}\mathbf{x}_{a}$ 。  

图 12.5 与图 12.4 相同，只是现在我们重新缩放了数据，使得样本 $\mathbf{\delta}\mathbf{x}_{a}$ 正好位于边缘上，即 $\langle{\pmb w},{\pmb x}_{a}\rangle+b=1$ 。由于 $\pmb{x}_{a}^{\prime}$ 是 $\mathbf{\delta}\mathbf{x}_{a}$ 在超平面上的正交投影，因此根据定义，它必须位于超平面上，即  

$$
\langle{\pmb w},{\pmb x}_{a}^{\prime}\rangle+b=0\,.
$$
通过将 (12.8) 代入 (12.11)，我们得到  

$$
\left\langle w,\pmb{x}_{a}-r\frac{\pmb{w}}{\|\pmb{w}\|}\right\rangle+b=0\,.
$$
利用内积的双线性性质（见第 3.2 节），我们得到  

$$
\langle{\pmb w},{\pmb x}_{a}\rangle+b-r\frac{\langle{\pmb w},{\pmb w}\rangle}{\|{\pmb w}\|}=0\,.
$$
根据我们对尺度的假设，第一项为 1，即 $\langle{\pmb w},{\pmb x}_{a}\rangle+b=1$ 。根据第 3.1 节中的 (3.16)，我们知道 $\langle{\pmb w},{\pmb w}\rangle=\|{\pmb w}\|^{2}$ 。因此，第二项简化为 $r\|w\|$ 。使用这些简化，我们得到  

$$
r={\frac{1}{\|w\|}}\,.
$$
这意味着我们用超平面法向量 $\mathbfit{w}$ 表达了距离 $r$ 。乍一看，这个方程似乎有些反直觉，因为我们似乎用向量 $\mathbfit{w}$ 的长度来表示到超平面的距离，但我们还不知道这个向量。一种思考方式是将距离 $r$ 视为一个临时变量，仅用于此推导。因此，在本节剩余部分，我们将距离到超平面表示为 $\frac{1}{||\pmb{w}||}$ 。在第 12.2.3 节中，我们将看到边缘等于 1 的选择等价于第 12.2.1 节中我们之前的假设 $\|w\|=1$ 。  

我们也可以将距离视为投影误差，即当将 ${\pmb x}_{a}$ 投影到超平面上时产生的误差。  

类似于获得 (12.9) 的论证，我们希望正负样本至少与超平面相距 1，这给出了条件  

$$
y_{n}(\langle\pmb{w},x_{n}\rangle+b)\geqslant1\,.
$$
将边缘最大化与样本需要位于超平面正确一侧（基于其标签）的事实结合起来，我们得到  

$$
\begin{array}{r l}&{\displaystyle\operatorname*{max}_{\pmb{w},b}\quad\displaystyle\frac{1}{\|\pmb{w}\|}}\\ &{\mathrm{subject}\;\mathrm{to}\;y_{n}(\langle\pmb{w},\pmb{x}_{n}\rangle+b)\geqslant1\quad\mathrm{for\;all}\quad n=1,.\,.\,.\,,N.}\end{array}
$$
我们通常不最大化范数的倒数（如 (12.16) 所示），而是最小化范数的平方。我们还经常包括一个常数 $\frac{1}{2}$ ，它不影响最优的 $\pmb{w},b$ ，但在计算梯度时会得到更整洁的形式。然后，我们的目标变为  

$$
\begin{array}{l}{\displaystyle\operatorname*{min}_{\pmb{w},b}\quad\displaystyle\frac{1}{2}\|\pmb{w}\|^{2}}\\ {\mathrm{subject~to}\,\,y_{n}(\langle\pmb{w},\pmb{x}_{n}\rangle+b)\geqslant1\quad\mathrm{for~all}\quad n=1,.\,.\,.\,,N\,.}\end{array}
$$
方程 (12.18) 称为 硬边缘 SVM 。称其为“硬”是因为该形式不允许任何边缘条件的违反。在第 12.2.4 节中，我们将看到，对于非线性可分的数据，这种 硬边缘 SVM 的平方范数会导致 SVM 的凸二次规划问题（第 12.5 节）。  

硬边缘 SVM  

“硬”条件可以放宽以容纳违反，如果数据不是线性可分的。

### 12.2.3 为什么我们可以将边界设为 1  

在第 12.2.1 节中，我们论证了希望最大化某个值 $r$，它代表最近实例到超平面的距离。在第 12.2.2 节中，我们将数据缩放，使得最近实例到超平面的距离为 1。在本节中，我们将这两个推导联系起来，证明它们是等价的。  

定理 12.1. 最大化边界 $r_{z}$，其中考虑归一化权重如 (12.10) 所示，  

$$
\begin{array}{l}
\underset{\pmb{w},b,r}{\text{max}}\quad\underset{\text{边界}}{\underbrace{r}} \\
\text{subject to}\quad\underset{\text{数据拟合}}{\underbrace{y_{n}(\langle\pmb{w},\pmb{x}_{n}\rangle+b)\geqslant r}},\quad\underset{\text{归一化}}{\underbrace{\|\pmb{w}\|=1}},\quad r>0\,,
\end{array}
$$  

等价于将数据缩放，使得边界为 1：  

$$
\begin{array}{r l}
\underset{\pmb{w},b}{\text{min}} & \underset{\text{边界}}{\underbrace{\frac{1}{2}\|\pmb{w}\|^{2}}} \\
\text{subject to} & \underset{\text{数据拟合}}{\underbrace{y_{n}(\langle\pmb{w},\pmb{x}_{n}\rangle+b)\geqslant1}}\,.
\end{array}
$$  

证明 考虑 (12.20)。由于平方是对非负数的严格单调变换，如果在目标中考虑 $r^{2}$，最大值保持不变。由于 $\|\pmb{w}\|=1$，我们可以用 $\frac{\pmb{w}^{\prime}}{\|\pmb{w}^{\prime}\|}$ 显式地重新参数化方程，得到  

$$
\begin{array}{r l}
& \underset{\pmb{w}^{\prime},b,r}{\text{max}}\quad r^{2} \\
& \text{subject to}\quad y_{n}\left(\left\langle\frac{\pmb{w}^{\prime}}{\|\pmb{w}^{\prime}\|},\pmb{x}_{n}\right\rangle+b\right)\geqslant r,\quad r>0\,.
\end{array}
$$  

注意到 $r>0$ 是因为假设线性可分，因此除以 $r$ 没有问题。方程 (12.22) 明确指出距离 $r$ 是正的。因此，我们可以将第一个约束除以 $r$，得到  

$$
\begin{align}
& \underset{\pmb{w}^{\prime}, b, r}{\text{max}} \quad r^2 \\
& \text{subject to} \quad y_{n} \left( \left\langle \frac{\pmb{w}^{\prime}}{\|\pmb{w}^{\prime}\| r} \right\rangle, \pmb{x}_{n} \right) + \frac{b}{r} \geqslant 1, \quad r > 0.
\end{align}
$$  

重命名参数为 $\pmb{w}^{\prime\prime}$ 和 $b^{\prime\prime}$。由于 $\pmb{w}^{\prime\prime}=\frac{\pmb{w}^{\prime}}{\|\pmb{w}^{\prime}\|\,r}$，重新排列得到 $\|\pmb{w}^{\prime\prime}\|$ 为  

$$
\|\pmb{w}^{\prime\prime}\|=\left\|\frac{\pmb{w}^{\prime}}{\|\pmb{w}^{\prime}\|\,r}\right\|=\frac{1}{r}\cdot\left\|\frac{\pmb{w}^{\prime}}{\|\pmb{w}^{\prime}\|}\right\|=\frac{1}{r}\,.
$$  

将这个结果代入 (12.23)，我们得到  

$$
\begin{array}{r l}
\displaystyle\operatorname*{max}_{\pmb{w^{\prime\prime},b^{\prime\prime}}} & \displaystyle\frac{1}{\|\pmb{w^{\prime\prime}}\|^{2}} \\
\text{subject to} & y_{n}\left(\langle\pmb{w^{\prime\prime}},\pmb{x}_{n}\rangle+b^{\prime\prime}\right)\geqslant1\,.
\end{array}
$$  

最后一步是观察到最大化 $\frac{1}{\|\pmb{w}^{\prime\prime}\|^{2}}$ 与最小化 $\frac{1}{2}\|\pmb{w}^{\prime\prime}\|^{2}$ 导致相同的解，这证明了定理 12.1。  

![](images/f82f87feda10525be03a82b87c5ee7bd3ebe08054f786507e82570d3d845905c.jpg)  
图 12.6 (a) 线性可分和 (b) 非线性可分数据。

### 12.2.4 软边距SVM：几何视角  

在数据不能线性可分的情况下，我们可能希望允许一些样本落在边缘区域内部，甚至位于超平面的错误一侧，如图12.6所示。  

允许一定程度分类错误的模型称为软边距SVM。在本节中，我们将使用几何论证推导出相应的优化问题。在第12.2.5节中，我们将使用损失函数的思想推导出等价的优化问题。使用拉格朗日乘数（第7.2节），在第12.3节中我们将推导出SVM的对偶优化问题。这个对偶优化问题使我们能够观察到SVM的第三种解释：作为将正数据样本和负数据样本对应的凸包之间的线平分的超平面（第12.3.2节）。  

关键的几何思想是引入一个松弛变量$\xi_{n}$，对应于每个样本-标签对$(\pmb{x}_{n},y_{n})$，允许特定的样本位于边缘内部或甚至位于超平面的错误一侧（参见软边距SVM松弛变量）。我们从边缘中减去$\xi_{n}$的值，并约束$\xi_{n}$为非负。为了鼓励样本的正确分类，我们将$\xi_{n}$添加到目标函数中：  

$$
\begin{array}{r l}{\underset{\pmb{w},b,\pmb{\xi}}{\mathrm{min}}}&{\displaystyle\frac{1}{2}\|\pmb{w}\|^{2}+C\sum_{n=1}^{N}\xi_{n}}\\ {\mathrm{subject~to~}}&{y_{n}\big(\langle\pmb{w},\pmb{x}_{n}\rangle+b\big)\geqslant1-\xi_{n}}\\ &{\xi_{n}\geqslant0}\end{array}
$$  

软边距SVM  
正则化参数  
正则化项  

这种正则化有不同的参数化形式，这就是为什么（12.26a）也经常被称为$C$-SVM。  

对于$n=1,\cdot\cdot\cdot,N$。与硬边距SVM的优化问题（12.18）不同，这个优化问题被称为软边距SVM。参数$C>0$权衡了边缘的大小和我们拥有的总松弛量。这个参数被称为正则化参数，因为在接下来的章节中，我们将看到目标函数（12.26a）中的边缘项是一个正则化项。边缘项$\|\pmb{w}\|^{2}$被称为正则化项，而在许多数值优化书籍中，正则化参数会乘以这个项（第8.2.3节）。这与本节的表述不同。在这里，较大的$C$值意味着较低的正则化，因为我们赋予松弛变量更大的权重，从而更优先考虑位于边缘正确一侧之外的样本。  

注释。在软边距SVM的表述（12.26a）中，$\mathbfit{w}$被正则化，但$b$没有被正则化。我们可以通过观察正则化项不包含$b$这一点来看到这一点。未正则化的项$b$使理论分析复杂化（Steinwart和Christmann, 2008, 第1章）并降低计算效率（Fan等人, 2008）。$\diamondsuit$  
![](images/cb1c7e4fe51cffe98b3db79fb2ed7849108154d55fa0ef826dd6d49b2ff17125.jpg)  
图12.7 软边距SVM允许样本位于边缘内部或超平面的错误一侧。松弛变量$\xi$衡量正样本$\pmb{x}_{+}$到正边缘超平面$\langle{\pmb w},{\pmb x}\rangle+b=1$的距离，当$\mathbf{x_{+}}$位于错误一侧时。  
图12.7

### 12.2.5 软边距SVM：损失函数视角  

让我们采用一种不同的方法来推导SVM，遵循经验风险最小化的原则（第8.2节）。对于SVM，我们选择超平面作为假设类，即

$$
f({\pmb x})=\langle{\pmb w},{\pmb x}\rangle+b.
$$
在本节中，我们将看到边距对应于正则化项。剩下的问题是，什么是损失函数？与第9章中考虑回归问题（预测输出是一个实数）不同，在本章中，我们考虑二元分类问题（预测输出是两个标签$\{+1,-1\}$中的一个）。因此，每个单个样本-标签对的误差/损失函数需要适合二元分类。例如，用于回归的平方损失（9.10b）不适合二元分类。  

注释。二元标签之间的理想损失函数是计算预测与标签之间的不匹配数量。这意味着对于应用于示例${\bf x}_{n}$的预测器$f$，我们将输出$f(\pmb{x}_{n})$与标签$y_{n}$进行比较。如果它们匹配，则定义损失为零；如果不匹配，则定义损失为一。这表示为$\mathbf{1}(f(\pmb{x}_{n})\,\neq\,y_{n})$，称为零一损失。不幸的是，零一损失会导致在寻找最佳参数$\pmb{w},b$时的组合优化问题。组合优化问题（与第7章讨论的连续优化问题不同）通常更难解决。$\diamondsuit$  

SVM对应的损失函数是什么？考虑预测器$f(\pmb{x}_{n})$的输出与标签$y_{n}$之间的误差。损失描述了在训练数据上的错误。通过使用切线损失（hinge loss）等价地推导（12.26a）的方式是

$$
\ell(t)=\operatorname*{max}\{0,1-t\}\quad{\mathrm{where}}\quad t=y f(\pmb{x})=y(\langle\pmb{w},\pmb{x}\rangle+b)\,.
$$
如果$f(\pmb{x})$位于超平面的正确一侧（基于相应的标签$y$），并且距离大于1，则意味着$t\geqslant1$，切线损失返回零值。如果$f(\pmb{x})$位于正确一侧但距离超平面太近（$0<t<1$），示例$\pmb{x}$位于边距内，切线损失返回正值。当示例位于超平面的错误一侧（$t<0$）时，切线损失返回更大的正值，呈线性增加。换句话说，只要我们比边距更接近超平面，即使预测正确，也会受到惩罚，且惩罚呈线性增加。切线损失的另一种表达方式是将其视为两个线性部分

$$
\ell(t)=\left\{{0\atop1}-t\quad\mathrm{if}\quad t\geq1\atop t<1\right.,
$$
如图12.8所示。硬边距SVM（12.18）对应的损失定义为

$$
\ell(t)=\left\{\begin{array}{l c l}{{0}}&{{\mathrm{if}}}&{{t\geqslant1}}\\ {{\infty}}&{{\mathrm{if}}}&{{t<1}}\end{array}\right..
$$
![](images/d835bad92b9c9218165cf830f96f48577c773f6ee391515d5ce4d24f7490b388.jpg)  
图12.8 切线损失是零一损失的凸上界。  

这种损失可以解释为不允许任何示例位于边距内。  

对于给定的训练集$\{({\pmb x}_{1},y_{1}),.\,.\,.\,,({\pmb x}_{N},y_{N})\}$，我们寻求最小化总损失，同时使用$\ell_{2}$正则化（见第8.2.3节）来正则化目标。使用切线损失（12.28）给出的是一个无约束优化问题

$$
\operatorname*{min}_{\boldsymbol{w},b}\quad\underbrace{\frac{1}{2}\|\pmb{w}\|^{2}}_{\mathrm{正则化}}+\underbrace{C\sum_{n=1}^{N}\operatorname*{max}\{0,1-y_{n}(\langle\pmb{w},\pmb{x}_{n}\rangle+b)\}}_{\mathrm{误差项}}\,.
$$
正则化项 误差项  

正则化 第12.31式中的第一项称为正则化项或正则化器（见第8.2.3节），第二项称为误差项或损失项。换句话说，边距最大化可以解释为损失项。回想第12.2.4节，项$\scriptstyle{\frac{1}{2}}\|w\|^{2}$直接来源于正则化。  

原则上，第12.31式的无约束优化问题可以直接使用第7.1节描述的（次）梯度下降方法求解。要看到（12.31）和（12.26a）是等价的，观察到切线损失（12.28）实际上由两部分线性部分组成，如（12.29）所示。考虑单个样本-标签对的切线损失（12.28）。我们可以等价地将最小化切线损失转换为最小化松弛变量$\xi$的最小化，同时有两个约束。方程形式为

$$
\operatorname*{min}_{t}\operatorname*{max}\{0,1-t\}
$$
等价于

$$
\begin{array}{r c l}{{}}&{{}}&{{\displaystyle\operatorname*{min}_{{\boldsymbol{\xi}},t}~~{\boldsymbol{\xi}}}}\\ {{}}&{{}}&{{}}\\ {{\mathrm{subject\to}~}}&{{\boldsymbol{\xi}\geqslant0\,,~~~\boldsymbol{\xi}\geqslant1-t\,.}}\end{array}
$$
通过将此表达式代入（12.31）并重新排列其中一个约束，我们恰好得到软边距SVM（12.26a）。  

注释。让我们将本节中选择的损失函数与第9章中线性回归的损失函数进行对比。回想第9.2.1节，为了找到最大似然估计，我们通常最小化负对数似然。此外，由于线性回归带高斯噪声的似然项是高斯的，每个示例的负对数似然是平方误差函数。平方误差函数是寻找最大似然解时最小化的损失函数。$\diamondsuit$

## 12.3 双重支持向量机

在前面章节中，关于变量 $\mathbfit{w}$ 和 $b$ 的支持向量机（SVM）描述被称为原始SVM。回想一下，我们考虑输入 $\pmb{x} \in \mathbb{R}^{D}$，其中 $D$ 是特征的数量。由于 $\mathbfit{w}$ 与 $\pmb{x}$ 的维度相同，这意味着优化问题中的参数数量（即 $\mathbfit{w}$ 的维度）随着特征数量线性增长。

接下来，我们考虑一个等价的优化问题（所谓的双重视角），该问题与特征数量无关。相反，参数的数量随着训练集中的样本数量增加。我们在第10章中看到了类似的想法，我们以一种不依赖于特征数量的方式表达了学习问题。这对于特征数量多于训练数据集中样本数量的问题非常有用。双重SVM还具有额外的优势，即它很容易允许核函数的应用，如本章末尾将看到的。术语“双重”在数学文献中经常出现，而在这种情况下，它指的是凸对偶性。接下来的子章节基本上是凸对偶性的应用，我们在第7.2节中讨论过。

### 12.3.1 通过拉格朗日乘数的凸对偶性

回想原始软间隔SVM（12.26a）。我们将与原始SVM对应的变量 $\mathbfit{w}, \mathbfit{b}$ 和 $\xi$ 称为原始变量。我们使用 $\alpha_{n} \geqslant 0$ 作为与约束（12.26b）对应的拉格朗日乘数，该约束确保样本被正确分类；使用 $\gamma_{n} \geqslant 0$ 作为与松弛变量非负性约束对应的拉格朗日乘数；见（12.26c）。拉格朗日函数则由

$$
\mathfrak{L}(\pmb{w}, b, \xi, \alpha, \gamma) = \frac{1}{2} \|\pmb{w}\|^2 + C \sum_{n=1}^{N} \xi_{n} - \sum_{n=1}^{N} \alpha_{n} \left( y_{n} (\langle \pmb{w}, \pmb{x}_{n} \rangle + b) - 1 + \xi_{n} \right) - \sum_{n=1}^{N} \gamma_{n} \xi_{n}
$$

给出。

在第7章中，我们使用 $\lambda$ 作为拉格朗日乘数。在本节中，我们遵循SVM文献中常用的符号，使用 $\alpha$ 和 $\gamma$。

通过分别对拉格朗日函数（12.34）关于三个原始变量 $\mathbfit{w}, \mathbfit{b},$ 和 $\xi$ 求偏导，我们得到

$$
\begin{array}{l}
\frac{\partial \mathfrak{L}}{\partial \pmb{w}} = \pmb{w}^{\top} - \sum_{n=1}^{N} \alpha_{n} y_{n} \pmb{x}_{n}^{\top}, \\
\frac{\partial \mathfrak{L}}{\partial b} = -\sum_{n=1}^{N} \alpha_{n} y_{n}, \\
\frac{\partial \mathfrak{L}}{\partial \xi_{n}} = C - \alpha_{n} - \gamma_{n}.
\end{array}
$$

我们现在通过将这些偏导数分别设为零来求拉格朗日函数的最大值。通过将（12.35）设为零，我们得到

$$
\pmb{w} = \sum_{n=1}^{N} \alpha_{n} y_{n} \pmb{x}_{n},
$$

表示定理。表示定理实际上是说，最小化经验风险的解位于由样本定义的子空间（第2.4.3节）中。支持向量

这实际上是表示定理（Kimeldorf 和 Wahba, 1970）的一个特例。方程（12.38）表明，原始问题中的最优权重向量是样本 $\pmb{x}_{n}$ 的线性组合。回想第2.6.1节的内容，这意味着优化问题的解位于训练数据的线性空间中。此外，通过将（12.36）设为零得到的约束表明，最优权重向量是样本的仿射组合。表示定理在正则化经验风险最小化的一般设置中成立（Hofmann 等人, 2008; Argyriou 和 Dinuzzo, 2014）。该定理有更一般的版本（Schölkopf 等人, 2001），其存在条件可以在 Yu 等人 (2013) 中找到。

注释。表示定理（12.38）还解释了“支持向量机”这一名称的由来。对于对应的参数 $\alpha_{n} = 0$ 的样本 $\pmb{x}_{n}$ 并不贡献于解 $\mathbfit{w}$。其他 $\alpha_{n} > 0$ 的样本被称为支持向量，因为它们“支持”超平面。$\diamondsuit$

通过将 $\mathbfit{w}$ 的表达式代入拉格朗日函数（12.34），我们得到双重

$$
\begin{array}{c}
\mathfrak{D}(\xi, \alpha, \gamma) = \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} y_{i} y_{j} \alpha_{i} \alpha_{j} \langle \pmb{x}_{i}, \pmb{x}_{j} \rangle - \sum_{i=1}^{N} y_{i} \alpha_{i} \langle \sum_{j=1}^{N} y_{j} \alpha_{j} \pmb{x}_{j}, \pmb{x}_{i} \rangle \\
+ C \sum_{i=1}^{N} \xi_{i} - b \sum_{i=1}^{N} y_{i} \alpha_{i} + \sum_{i=1}^{N} \alpha_{i} - \sum_{i=1}^{N} \alpha_{i} \xi_{i} - \sum_{i=1}^{N} \gamma_{i} \xi_{i}.
\end{array}
$$

注意到这里不再包含原始变量 $\mathbfit{w}$ 的项。通过将（12.36）设为零，我们得到 $\sum_{n=1}^{N} y_{n} \alpha_{n} = 0$。因此，涉及 $b$ 的项也消失了。回想内积是对称且双线性的（见第3.2节）。因此，（12.39）中的前两项涉及相同的对象。这些项（蓝色标记）可以简化，我们得到拉格朗日函数

$$
\mathfrak{D}(\xi, \alpha, \gamma) = -\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} y_{i} y_{j} \alpha_{i} \alpha_{j} \langle \pmb{x}_{i}, \pmb{x}_{j} \rangle + \sum_{i=1}^{N} \alpha_{i} + \sum_{i=1}^{N} (C - \alpha_{i} - \gamma_{i}) \xi_{i}.
$$

该方程的最后一项包含所有涉及松弛变量 $\xi_{i}$ 的项。通过将（12.37）设为零，我们看到（12.40）的最后一项也是零。进一步地，通过使用相同的方程并考虑到拉格朗日乘数 $\gamma_{i}$ 非负，我们得出 $\alpha_{i} \leqslant C$。我们现在得到了SVM的双重优化问题，该问题仅用拉格朗日乘数 $\alpha_{i}$ 表示。回想拉格朗日对偶性（定义7.1），我们最大化双重问题。这等价于最小化负双重问题，从而得到双重SVM

$$
\begin{array}{l}
\operatorname*{min}_{\alpha} \quad \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} y_{i} y_{j} \alpha_{i} \alpha_{j} \langle \pmb{x}_{i}, \pmb{x}_{j} \rangle - \sum_{i=1}^{N} \alpha_{i} \\
\mathrm{subject} \; \mathrm{to} \quad \sum_{i=1}^{N} y_{i} \alpha_{i} = 0 \\
0 \leqslant \alpha_{i} \leqslant C \quad \mathrm{for} \; \mathrm{all} \quad i = 1, \ldots, N.
\end{array}
$$

（12.41）中的等式约束是从将（12.36）设为零得到的。不等式约束 $\alpha_{i} \geqslant 0$ 是不等式约束的拉格朗日乘数的条件（第7.2节）。不等式约束 $\alpha_{i} \leqslant C$ 在上一段中讨论过。

SVM中的不等式约束称为“盒约束”，因为它们限制向量 $\pmb{\alpha} = [\alpha_{1}, \ldots, \alpha_{N}]^{\top} \in \mathbb{R}^{N}$ 的拉格朗日乘数向量位于由0和C定义的每个轴上的盒内。这些轴对齐的盒子在数值求解器中特别高效实现（Dostál, 2009, 第5章）。

一旦我们获得了双重参数 $\alpha$，我们可以通过表示定理（12.38）恢复原始参数 $\mathbfit{w}$。让我们称最优原始参数为 $\pmb{w}^{*}$。然而，如何获得参数 $b^{*}$ 仍是一个问题。考虑一个恰好位于边界上的样本 $\pmb{x}_{n}$，即 $\langle \pmb{w}^{*}, \pmb{x}_{n} \rangle + b = y_{n}$。回想一下，$y_{n}$ 或者是 $+1$ 或者是 $-1$。因此，唯一未知的是 $b$，可以通过

$$
b^{*} = y_{n} - \langle \pmb{w}^{*}, \pmb{x}_{n} \rangle
$$

计算得到。

注释。原则上，可能没有样本恰好位于边界上。在这种情况下，我们应该计算所有支持向量的 $|y_{n} - \langle \pmb{w}^{*}, \pmb{x}_{n} \rangle|$，并取绝对值差异的中位数作为 $b^{*}$ 的值。事实证明，恰好位于边界上的样本是其双重参数严格位于盒约束内的样本，$0 < \alpha_{i} < C$。这可以通过KKT条件推导得出，例如在Schölkopf和Smola (2002) 中。

![](images/2e75b176e562961cc89ae2d029613d659a8ac33361509365f83bcc688fc1b396.jpg)
图12.9 凸包。 (a) 点的凸包，其中一些点位于边界内；(b) 正样本和负样本周围的凸包。

![](images/aa2d21d558052f7865460ababb3a238fd3ede9e02870f258b14887e410f6909e.jpg)
(b) 正样本（蓝色）和负样本（橙色）周围的凸包。两个凸集之间的距离是向量 $\pmb{c} - \pmb{d}$ 的长度。

### 12.3.2 双核SVM：凸包视角  

另一种获得双核SVM的方法是考虑一种替代的几何论证。考虑具有相同标签的样本集 ${\pmb x}_{n}$。我们希望构建一个包含所有样本的凸集，使其成为最小可能的集合。这被称为凸包，并在图12.9中示例说明。

凸包  
让我们首先建立关于点的凸组合的一些直观理解。考虑两个点 $\pmb{x}_{1}$  和 $\pmb{x}_{2}$  以及相应的非负权重 $\alpha_{1},\alpha_{2}\geqslant0$，使得 $\alpha_{1}+\alpha_{2}=1$。方程 $\alpha_{1}{\pmb x}_{1}+\alpha_{2}{\pmb x}_{2}$ 描述了从 $\pmb{x}_{1}$ 到 $\pmb{x}_{2}$  的一条线上的每个点。当添加第三个点 $\pmb{x}_{3}$  以及权重 $\alpha_{3}\geqslant0$ 使得 $\sum_{n=1}^{3}\alpha_{n}=1$ 时，这三个点 ${\pmb x}_{1},{\pmb x}_{2},{\pmb x}_{3}$ 的凸组合覆盖一个二维区域。这个区域的凸包是由每对点对应的边形成的三角形。随着添加更多的点，点的数量超过维度的数量时，一些点将位于凸包内部，如图12.9(a)所示。

一般来说，构建凸包可以通过引入每个样本 ${\pmb x}_{n}$ 对应的非负权重 $\alpha_{n}\geqslant0$ 来实现。然后，凸包可以描述为集合

$$
\operatorname{conv}\left(X\right)=\left\{\sum_{n=1}^{N}\alpha_{n}\pmb{x}_{n}\right\}\quad\mathrm{with}\quad\sum_{n=1}^{N}\alpha_{n}=1\quad\mathrm{and}\quad\alpha_{n}\geqslant0,
$$

对于所有 $n=1,\ldots,N$。如果正类和负类的点云分离，则它们的凸包不重叠。给定训练数据 $(\pmb{x}_{1},y_{1}),\dots,(\pmb{x}_{N},y_{N})$，我们分别形成两个凸包，对应于正类和负类。我们选择一个点 $^c$，它位于正类样本的凸包内，并且最接近负类分布。同样，我们选择一个点 $^d$，它位于负类样本的凸包内，并且最接近正类分布；见图12.9(b)。我们定义 $^d$ 和 $^c$ 之间的差向量为

$$
\pmb{w}:=\pmb{c}-\pmb{d}\,.
$$

选择点 $^c$ 和 $^d$ 如前所述，并要求它们最接近彼此等价于最小化 $\mathbfit{w}$ 的长度/范数，因此我们得到相应的优化问题

$$
\arg\operatorname*{min}_{\pmb{w}}\left\|\pmb{w}\right\|=\arg\operatorname*{min}_{\pmb{w}}\frac{1}{2}\left\|\pmb{w}\right\|^{2}\,.
$$

由于 $^c$ 必须位于正类的凸包内，它可以表示为正类样本的凸组合，即对于非负系数 $\alpha_{n}^{+}$

$$
\pmb{c}=\sum_{n:y_{n}=+1}\alpha_{n}^{+}\pmb{x}_{n}\,.
$$

在(12.46)中，我们使用符号 $n:y_{n}=+1$ 表示 $y_{n}=+1$ 的索引集 $n$。类似地，对于负标签的样本，我们得到

$$
{\pmb d}=\sum_{n:y_{n}=-1}\alpha_{n}^{-}{\pmb x}_{n}\,.
$$

通过将(12.44)，(12.46)和(12.47)代入(12.45)，我们得到目标

$$
\operatorname*{min}_{\alpha}\frac{1}{2}\left\|\sum_{n:y_{n}=+1}\alpha_{n}^{+}\pmb{x}_{n}-\sum_{n:y_{n}=-1}\alpha_{n}^{-}\pmb{x}_{n}\right\|^{2}\,.
$$

令 $_{\alpha}$ 为所有系数的集合，即 $\alpha^{+}$ 和 $\alpha^{-}$ 的连接。回忆一下，我们要求每个凸包的系数之和为一，即

$$
\sum_{n:y_{n}=+1}\alpha_{n}^{+}=1\quad\mathrm{and}\quad\sum_{n:y_{n}=-1}\alpha_{n}^{-}=1\,.
$$

这暗示了约束条件

$$
\sum_{n=1}^{N}y_{n}\alpha_{n}=0\,.
$$

这个结果可以通过展开各个类来观察

$$
\begin{array}{c}
\sum_{n=1}^{N}y_{n}\alpha_{n}=\sum_{n:y_{n}=+1}(+1)\alpha_{n}^{+}+\sum_{n:y_{n}=-1}(-1)\alpha_{n}^{-} \\
=\sum_{n:y_{n}=+1}\alpha_{n}^{+}-\sum_{n:y_{n}=-1}\alpha_{n}^{-}=1-1=0\,.
\end{array}
$$

目标函数(12.48)和约束条件(12.50)，以及假设 $\mathbf{\Omega}\alpha\geqslant\mathbf{0}$，给出了一个受限(凸)优化问题。这个优化问题可以证明与硬边界SVM的对偶形式相同(Bennett and Bredensteiner, 2000a)。

**注释。** 要获得软边界对偶，我们考虑缩减的凸包。缩减的凸包类似于凸包，但对系数 $_{\alpha}$ 的大小有一个上限。$_{\alpha}$ 的最大可能值限制了凸包可以达到的大小。换句话说，$_{\alpha}$ 的上限使凸包缩小到较小的体积(Bennett and Bredensteiner, 2000b)。$\diamondsuit$

## 12.4 核函数  

考虑对偶支持向量机（12.41）的表述。注意，在目标函数中，内积仅发生在示例 $\mathbf{\Delta}\mathbf{x}_{i}$ 和 $\pmb{x}_{j}$ 之间。示例与参数之间没有内积。因此，如果我们考虑一组特征 $\phi(\pmb{x}_{i})$ 来表示 $\mathbf{\Delta}\mathbf{x}_{i}$，对偶支持向量机的唯一变化将是替换内积。这种模块化，即分类方法（支持向量机）的选择和特征表示 $\phi(\pmb{x})$ 的选择可以分别考虑，为我们提供了独立探索两个问题的灵活性。在本节中，我们讨论特征表示 $\phi(\pmb{x})$ 并简要介绍核函数的概念，但不深入技术细节。  

核函数  

核函数的输入 $\mathcal{X}$ 可以非常一般，并不一定局限于 $\mathbb{R}^{D}$。  

由于 $\phi(\pmb{x})$ 可以是非线性函数，我们可以使用假设线性分类器的支持向量机来构建在示例 $\pmb{x}_{n}$ 上是非线性的分类器。这为用户提供了除了软间隔之外的另一种处理不可线性分离的数据集的方法。实际上，有许多算法和统计方法具有我们在对偶支持向量机中观察到的性质：唯一的内积仅发生在示例之间。我们不需要显式地定义非线性特征映射 $\phi(\cdot)$ 并计算示例 $\mathbf{\Delta}\mathbf{x}_{i}$ 和 $\pmb{x}_{j}$ 之间的内积，而是定义一个相似性函数 $k(\pmb{x}_{i},\pmb{x}_{j})$ 作为 $\mathbf{\Delta}\mathbf{x}_{i}$ 和 $\mathbf{\Delta}\mathbf{x}_{j}$ 之间的相似性。对于一类称为核的相似性函数，相似性函数隐含地定义了特征映射 $\phi(\cdot)$。核是由有限个函数 $k:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$ 组成的，对于这些函数存在一个希尔伯特空间 $\mathcal{H}$ 和一个特征映射 $\phi:{\mathcal{X}}\rightarrow{\mathcal{H}}$，使得

$$
k(\pmb{x}_{i},\pmb{x}_{j})=\langle\pmb{\phi}(\pmb{x}_{i}),\pmb{\phi}(\pmb{x}_{j})\rangle_{\mathcal{H}}\ .
$$
![](images/9bd9de43f94b0664e8e6ce66846857488514074ea23ca08f61180579c1588137.jpg)  
(d) 使用多项式（阶数3）核的支持向量机  

![](images/79daf85cd735382fa01121f35e1477eee2e0e0dbd570afea9bbe9b3cac0598f9.jpg)  
(a) 使用线性核的支持向量机  

![](images/1bba8dc34c860eedbcf49e9cc7a38408aed853bb4e0d1af3a3199ae45802bbac.jpg)  
(c) 使用多项式（阶数2）核的支持向量机  

每个核 $k$ 都与一个唯一的再生核希尔伯特空间相关联（Aronszajn, 1950; Berlinet and Thomas-Agnan, 2004）。在这种独特的关联中，$\phi(\pmb{x})\,=\,k(\cdot,\pmb{x})$ 被称为标准特征映射。从内积推广到核函数（12.52）的过程被称为核技巧（Schölkopf and Smola, 2002; Shawe-Taylor and Cristianini, 2004），因为它隐藏了显式的非线性特征映射。  

矩阵 $\boldsymbol{K}\in\mathbb{R}^{N\times N}$，由内积或应用 $k(\cdot,\cdot)$ 到数据集得到，称为格拉姆矩阵，通常简称为核矩阵。核必须是非对称且半正定的函数，使得每个核矩阵 $\kappa$ 都是非对称且半正定的（第3.2.3节）。

![](images/fbff96a37fc7e22e3b88217fb984a4b3f4ebb777153b6a9e954884e6d9105ed6.jpg)  
Gram 矩阵 核矩阵  

$$
\forall z\in\mathbb{R}^{N}:z^{\top}K z\geqslant0\,.
$$  

一些常见的多元实值数据的核函数示例为多项式核、高斯径向基函数核和有理二次核（Schölkopf 和 Smola, 2002; Rasmussen  

© 2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. 由剑桥大学出版社出版 (2020)。  

和 Williams, 2006）。图 12.10 屾示了不同核函数对分离超平面的影响。请注意，我们仍然在求解超平面，即假设函数类仍然是线性的。非线性表面是由核函数引起的。  

注释。 对于初学者而言，不幸的是，“核”这个词有多种含义。在本章中，“核”这个词来源于再生核希尔伯特空间（RKHS）（Aronszajn, 1950; Saitoh, 1988）。我们在线性代数（第 2.7.3 节）中讨论了核的概念，其中核是零空间的另一种说法。机器学习中“核”的第三个常见用法是核密度估计中的平滑核（第 11.5 节）。$\diamondsuit$ 核的选择，以及核的参数，通常使用嵌套交叉验证（第 8.6.1 节）来选择。  

由于显式表示 $\phi(\pmb{x})$ 在数学上等价于核表示 $k(\pmb{x}_{i},\pmb{x}_{j})$，实践者通常会设计核函数，使其计算效率高于显式特征映射之间的内积。例如，考虑多项式核（Schölkopf 和 Smola, 2002），其中显式展开的项数随着输入维度的增加迅速增长（即使是低阶多项式也是如此）。核函数只需要每次输入维度进行一次乘法，这可以提供显著的计算节省。另一个例子是高斯径向基函数核（Schölkopf 和 Smola, 2002; Rasmussen 和 Williams, 2006），其中对应的特征空间是无限维的。在这种情况下，我们无法显式表示特征空间，但仍可以使用核计算一对示例之间的相似性。  

核技巧的另一个有用方面是，原始数据不需要已经被表示为多元实值数据。请注意，内积是在函数 $\phi(\cdot)$ 的输出上定义的，但并不限制输入为实数。因此，函数 $\phi(\cdot)$ 和核函数 $k(\cdot,\cdot)$ 可以在任何对象上定义，例如集合、序列、字符串、图和分布（Ben-Hur 等, 2008; Gärtner, 2008; Shi 等, 2009; Sriperumbudur 等, 2010; Vishwanathan 等, 2010）。

## 12.5 数值解法  

我们通过讨论如何将本章推导出的问题用第7章介绍的概念来表达，来结束对支持向量机（SVM）的讨论。我们考虑两种不同的方法来找到SVM的最优解。首先，我们从8.2.2节的损失函数视角出发，将其表示为一个无约束优化问题。然后，我们将原始SVM和对偶SVM的约束版本表示为标准形式的二次规划问题（7.3.2节）。  

考虑SVM的损失函数视角（12.31）。这是一个凸的无约束优化问题，但 hinge 损失（12.28）是非可微的。因此，我们应用次梯度方法来解决这个问题。然而，hinge 损失在几乎所有地方都是可微的，除了在铰链点 $t=1$ 处。在这一点上，梯度是一组可能的值，位于 $0$ 和 $-1$ 之间。因此，hinge 损失的次梯度 $g$ 给出为  

$$
g(t)={\left\{\begin{array}{l l}{-1}&{t<1}\\ {\left[-1,0\right]}&{t=1}\\ {0}&{t>1}\end{array}\right.}.
$$
利用这个次梯度，我们可以应用第7.1节中介绍的优化方法。  

原始SVM和对偶SVM都导致一个凸的二次规划问题（约束优化）。注意，原始SVM（12.26a）中的优化变量的大小与输入示例的维度 $D$ 相同。对偶SVM（12.41）中的优化变量的大小与示例的数量 $N$ 相同。  

为了将原始SVM表示为二次规划的标准形式（7.45），假设我们使用点积（3.5）作为内积。我们重新排列原始SVM的方程（12.26a），使得优化变量都在右侧，约束条件的不等式与标准形式匹配。这导致优化问题  

$$
\begin{array}{r l}{\underset{\pmb{w},b,\pmb{\xi}}{\operatorname*{min}}}&{\displaystyle\frac{1}{2}\|\pmb{w}\|^{2}+C\sum_{n=1}^{N}\xi_{n}}\\ {\mathrm{subject~to}}&{\displaystyle-y_{n}\pmb{x}_{n}^{\top}\pmb{w}-y_{n}b-\xi_{n}\leqslant-1}\\ &{-\xi_{n}\leqslant0}\end{array}
$$
$n=1,\cdot\cdot\cdot,N$ 。通过将变量 ${\pmb w},{\pmb b},{\pmb x}_{n}$ 连接成一个向量，并仔细收集项，我们得到软间隔SVM的矩阵形式：  

$$
\begin{array}{r l}&{\underset{\substack{\mathbf{w},b,\boldsymbol{\xi}}}{\mathrm{min}}\quad\frac{1}{2}\left[\begin{array}{c}{\pmb{w}}\\ {b}\\ {\pmb{\xi}}\end{array}\right]^{\top}\left[\begin{array}{c c}{\pmb{I}_{D}}&{\mathbf{0}_{D,N+1}}\\ {\mathbf{0}_{N+1,D}}&{\mathbf{0}_{N+1,N+1}}\end{array}\right]\left[\begin{array}{c}{\pmb{w}}\\ {b}\\ {\pmb{\xi}}\end{array}\right]+\left[\mathbf{0}_{D+1,1}\quad C\mathbf{1}_{N,1}\right]^{\top}\left[\begin{array}{c}{\pmb{w}}\\ {b}\\ {\pmb{\xi}}\end{array}\right]}\\ &{\mathrm{subject~to}\;\left[\begin{array}{c c c}{-\pmb{Y}\pmb{X}}&{-\pmb{y}}&{-\pmb{I}_{N}}\\ {\mathbf{0}_{N,D+1}}& &{-\pmb{I}_{N}}\end{array}\right]\left[\begin{array}{c}{\pmb{w}}\\ {b}\\ {\pmb{\xi}}\end{array}\right]\leqslant\left[\begin{array}{c}{-\mathbf{1}_{N,1}}\\ {\mathbf{0}_{N,1}}\end{array}\right]\,.}\end{array}
$$
在上述优化问题中，最小化是在参数 $\lbrack\pmb{w}^{\top},\bar{b},\pmb{\xi}^{\top}]^{\top}\,\in\,\mathbb{R}^{D+1+N}$ 上进行的，我们使用符号 $I_{m}$ 表示大小为 $m\times m$ 的单位矩阵， $\mathbf{0}_{m,n}$ 表示大小为 $m\times n$ 的零矩阵， $\mathbf{1}_{m,n}$ 表示大小为 $m\times n$ 的全1矩阵。此外， $\pmb{y}$ 是标签向量 $[y_{1},\cdot\cdot\cdot\mathbf{\Pi},y_{N}]^{\top}$ ， $\pmb{Y}=\mathrm{diag}(\pmb{y})$ 是一个 $N$ 由 $N$ 的矩阵，其中对角线元素来自 $\pmb{y}$ ，而 $\pmb{X}\in\mathbb{R}^{N\times D}$ 是通过连接所有示例得到的矩阵。  

类似地，我们可以对SVM的对偶版本（12.41）进行项的合并。为了将对偶SVM表示为标准形式，我们首先需要表示核矩阵 $\kappa$，使得每个元素 $K_{i j}=k(\pmb{x}_{i},\pmb{x}_{j})$。如果有一个显式的特征表示 $\mathbf{\delta}\mathbf{x}_{i}$，则定义 $K_{i j}=\langle{\pmb x}_{i},{\pmb x}_{j}\rangle$。为了方便表示，我们引入一个矩阵，除了对角线外都是零，对角线存储标签，即 $\pmb{Y}=\mathrm{diag}(\pmb{y})$。对偶SVM可以写为  

$$
\begin{array}{r l}{\underset{\alpha}{\mathrm{min}}}&{\frac{1}{2}\pmb{\alpha}^{\top}\pmb{Y}\pmb{K}\pmb{Y}\pmb{\alpha}-\mathbf{1}_{N,1}^{\top}\pmb{\alpha}}\\ {\mathrm{subject~to~}}&{\left[\begin{array}{l}{\pmb{y}^{\top}}\\ {-\pmb{y}^{\top}}\\ {-\pmb{I}_{N}}\\ {\pmb{I}_{N}}\end{array}\right]\pmb{\alpha}\leqslant\left[\begin{array}{l}{\mathbf{0}_{N+2,1}}\\ {C\mathbf{1}_{N,1}}\end{array}\right]\,.}\end{array}
$$
注。在第7.3.1节和7.3.2节中，我们介绍了标准形式的约束条件为不等式约束。我们将对偶SVM的等式约束表示为两个不等式约束，即  

$$
A x=b\quad{\mathrm{是被替换为}}\quad A x\leqslant b\quad{\mathrm{和}}\quad A x\geqslant b\,。
$$
特定的凸优化方法软件实现可能提供表达等式约束的能力。$\diamondsuit$  

由于SVM有许多不同的视角，因此有许多方法可以解决由此产生的优化问题。这里介绍的方法，即将SVM问题表示为标准凸优化形式，通常在实践中并不常用。SVM求解器的两种主要实现是Chang和Lin（2011）（开源）和Joachims（1999）。由于SVM具有清晰且定义良好的优化问题，许多基于数值优化技术的方法（Nocedal和Wright，2006）都可以应用（Shawe-Taylor和Sun，2011）。

## 12.6 进一步阅读  

SVM 是研究二分类问题的众多方法之一。其他方法包括感知机、逻辑回归、费舍尔判别、最近邻、朴素贝叶斯和随机森林（Bishop, 2006; Murphy, 2012）。Ben-Hur 等人（2008）提供了一个关于 SVM 和核函数在离散序列上的简短教程。SVM 的发展与经验风险最小化密切相关，详见第 8.2 节。因此，SVM 具有强大的理论性质（Vapnik, 2000; Steinwart 和 Christmann, 2008）。关于核方法的书籍（Schölkopf 和 Smola, 2002）包含了支持向量机的许多细节以及如何优化它们。关于核方法的更广泛的书籍（Shawe-Taylor 和 Cristianini, 2004）还包括了用于不同机器学习问题的许多线性代数方法。  

使用 Legendre–Fenchel 变换的思想可以得到对偶 SVM 的另一种推导（第 7.3.3 节）。推导考虑了 SVM 无约束形式（12.31）中的每一项，并计算它们的凸共轭（Rifkin 和 Lippert, 2007）。对 SVM 的泛函分析观点（也包括正则化方法的观点）感兴趣的读者可以参考 Wahba（1990）的工作。关于核的理论阐述（Aronszajn, 1950; Schwartz, 1964; Saitoh, 1988; Manton 和 Amblard, 2015）需要对线性算子有一个基本的理解（Akhiezer 和 Glazman, 1993）。核的思想已经被推广到巴拿赫空间（Zhang 等人, 2009）和克里恩空间（Ong 等人, 2004; Loosli 等人, 2016）。  

注意到铰链损失有三种等价表示，如（12.28）和（12.29）所示，以及（12.33）中的约束优化问题。形式（12.28）常用于将 SVM 损失函数与其他损失函数进行比较（Steinwart, 2007）。两段形式（12.29）便于计算次梯度，因为每一部分都是线性的。第三种形式（12.33），如第 12.5 节所示，允许使用凸二次规划（第 7.3.2 节）工具。  

由于二分类是机器学习中一个研究得很充分的任务，因此有时也会使用其他词汇，如判别、分离和决策。此外，二分类器可以输出三个量。首先是线性函数本身的输出（通常称为分数），它可以取任何实数值。这个输出可以用于对示例进行排名，二分类可以被视为在排名示例上选择一个阈值（Shawe-Taylor 和 Cristianini, 2004）。二分类器输出的第二个量通常是经过非线性函数处理后将其值限制在一定范围内的输出，例如在区间 [0, 1] 内。常见的非线性函数是 Sigmoid 函数（Bishop, 2006）。当非线性导致概率校准良好时（Gneiting 和 Raftery, 2007; Reid 和 Williamson, 2011），这被称为类别概率估计。二分类器的第三个输出是最终的二元决策 {+1, -1}，这是最常见的假设为分类器输出的量。  

SVM 本身不是一个自然地具有概率解释的二分类器。将线性函数（分数）的原始输出转换为校准的类别概率估计（P(Y=1|X=x)）涉及额外的校准步骤（Platt, 2000; Zadrozny 和 Elkan, 2001; Lin 等人, 2007）。从训练的角度来看，有许多相关的概率方法。我们在第 12.2.5 节末尾提到，损失函数与似然性之间存在关系（也参见第 8.2 节和第 8.3 节）。在训练期间进行良好校准的转换对应的最大似然方法称为逻辑回归，它属于一类称为广义线性模型的方法。从这个角度来看，逻辑回归的细节可以在 Agresti（2002, 第 5 章）和 McCullagh 和 Nelder（1989, 第 4 章）中找到。自然地，可以通过贝叶斯逻辑回归估计分类器输出的后验分布，从而采取更贝叶斯的观点。贝叶斯观点还包括先验的指定，这包括设计选择，如与似然性的共轭（第 6.6.1 节）。此外，可以考虑潜在函数作为先验，这导致高斯过程分类（Rasmussen 和 Williams, 2006, 第 3 章）。

# 参考文献  

Abel, Niels H. 1826.  D´ emonstration de l’Impossibilit´ e de la R´ esolution Alg´ ebrique des Equations G´ en´ erales qui Passent le Quatri\` eme Degr´ . Grøndahl and Søn. Adhikari, Ani, and DeNero, John. 2018.  Computational and Inferential Thinking: The Foundations of Data Science . Gitbooks. Agarwal, Arvind, and Daum´ e III, Hal. 2010. A Geometric View of Conjugate Priors.  Machine Learning ,  81 (1), 99–113. Agresti, A. 2002.  Categorical Data Analysis . Wiley. Akaike, Hirotugu. 1974. A New Look at the Statistical Model Identification.  IEEE Transactions on Automatic Control ,  19 (6), 716–723. Akhiezer, Naum I., and Glazman, Izrail M. 1993.  Theory of Linear Operators in Hilbert Space . Dover Publications. Alpaydin, Ethem. 2010.  Introduction to Machine Learning . MIT Press. Amari, Shun-ichi. 2016.  Information Geometry and Its Applications . Springer. Argyriou, Andreas, and Dinuzzo, Francesco. 2014. A Unifying View of Representer Theorems. In:  Proceedings of the International Conference on Machine Learning . Aronszajn, Nachman. 1950. Theory of Reproducing Kernels.  Transactions of the American Mathematical Society ,  68 , 337–404. Axler, Sheldon. 2015.  Linear Algebra Done Right . Springer. Bakir, G¨ okhan, Hofmann, Thomas, Sch¨ olkopf, Bernhard, Smola, Alexander J., Taskar, Ben, and Vishwanathan, S. V. N. (eds). 2007.  Predicting Structured Data . MIT Press. Barber, David. 2012.  Bayesian Reasoning and Machine Learning . Cambridge University Press. Barndorff-Nielsen, Ole. 2014.  Information and Exponential Families: In Statistical Theory . Wiley. Bartholomew, David, Knott, Martin, and Moustaki, Irini. 2011.  Latent Variable Models and Factor Analysis: A Unified Approach . Wiley. Baydin, Atılım G., Pearlmutter, Barak A., Radul, Alexey A., and Siskind, Jeffrey M. 2018. Automatic Differentiation in Machine Learning: A Survey.  Journal of Machine Learning Research ,  18 , 1–43. Beck, Amir, and Teboulle, Marc. 2003. Mirror Descent and Nonlinear Projected Subgradient Methods for Convex Optimization.  Operations Research Letters ,  31 (3), 167–175. Belabbas, Mohamed-Ali, and Wolfe, Patrick J. 2009. Spectral Methods in Machine Learning and New Strategies for Very Large Datasets.  Proceedings of the National Academy of Sciences , 0810600105. Belkin, Mikhail, and Niyogi, Partha. 2003. Laplacian Eigenmaps for Dimensionality Reduction and Data Representation.  Neural Computation ,  15 (6), 1373–1396. Ben-Hur, Asa, Ong, Cheng Soon, Sonnenburg, S¨ oren, Sch¨ olkopf, Bernhard, and R¨ atsch, Gunnar. 2008. Support Vector Machines and Kernels for Computational Biology.  PLoS Computational Biology ,  4 (10), e1000173.  

Bennett, Kristin P., and Bredensteiner, Erin J. 2000a. Duality and Geometry in SVM Classifiers. In:  Proceedings of the International Conference on Machine Learning . Bennett, Kristin P., and Bredensteiner, Erin J. 2000b. Geometry in Learning. Pages 132–145 of:  Geometry at Work . Mathematical Association of America. Berlinet, Alain, and Thomas-Agnan, Christine. 2004.  Reproducing Kernel Hilbert Spaces in Probability and Statistics . Springer. Bertsekas, Dimitri P. 1999.  Nonlinear Programming . Athena Scientific. Bertsekas, Dimitri P. 2009.  Convex Optimization Theory . Athena Scientific. Bickel, Peter J., and Doksum, Kjell. 2006. Mathematical Statistics, Basic Ideas and Selected Topics . Vol. 1. Prentice Hall. Bickson, Danny, Dolev, Danny, Shental, Ori, Siegel, Paul H., and Wolf, Jack K. 2007. Linear Detection via Belief Propagation. In:  Proceedings of the Annual Allerton Conference on Communication, Control, and Computing . Billingsley, Patrick. 1995.  Probability and Measure . Wiley. Bishop, Christopher M. 1995. Neural Networks for Pattern Recognition . Clarendon Press. Bishop, Christopher M. 1999. Bayesian PCA. In:  Advances in Neural Information Processing Systems . Bishop, Christopher M. 2006.  Pattern Recognition and Machine Learning . Springer. Blei, David M., Kucukelbir, Alp, and McAuliffe, Jon D. 2017. Variational Inference: A Review for Statisticians.  Journal of the American Statistical Association ,  112 (518), 859–877. Blum, Arvim, and Hardt, Moritz. 2015. The Ladder: A Reliable Leaderboard for Machine Learning Competitions. In:  International Conference on Machine Learning . Bonnans, J. Fr´ ed´ eric, Gilbert, J. Charles, Lemar´ echal, Claude, and Sagastiz´ abal, Clau- dia A. 2006.  Numerical Optimization: Theoretical and Practical Aspects . Springer. Borwein, Jonathan M., and Lewis, Adrian S. 2006. Convex Analysis and Nonlinear Optimization . 2nd edn. Canadian Mathematical Society. Bottou, L´ eon. 1998. Online Algorithms and Stochastic Approximations. Pages 9–42 of:  Online Learning and Neural Networks . Cambridge University Press. Bottou, L´ eon, Curtis, Frank E., and Nocedal, Jorge. 2018. Optimization Methods for Large-Scale Machine Learning.  SIAM Review ,  60 (2), 223–311. Boucheron, Stephane, Lugosi, Gabor, and Massart, Pascal. 2013. Concentration Inequalities: A Nonasymptotic Theory of Independence.  Oxford University Press. Boyd, Stephen, and Vandenberghe, Lieven. 2004.  Convex Optimization . Cambridge University Press. Boyd, Stephen, and Vandenberghe, Lieven. 2018.  Introduction to Applied Linear Algebra . Cambridge University Press. Brochu, Eric, Cora, Vlad M., and de Freitas, Nando. 2009. A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning . Tech. rept. TR-2009-023. Department of Computer Science, University of British Columbia. Brooks, Steve, Gelman, Andrew, Jones, Galin L., and Meng, Xiao-Li (eds). 2011.  Handbook of Markov Chain Monte Carlo . Chapman and Hall/CRC. Brown, Lawrence D. 1986.  Fundamentals of Statistical Exponential Families: With Applications in Statistical Decision Theory . Institute of Mathematical Statistics. Bryson, Arthur E. 1961. A Gradient Method for Optimizing Multi-Stage Allocation Processes. In:  Proceedings of the Harvard University Symposium on Digital Computers and Their Applications . Bubeck, S´ ebastien. 2015. Convex Optimization: Algorithms and Complexity.  Foundations and Trends in Machine Learning ,  8 (3-4), 231–357. B¨ uhlmann, Peter, and Van De Geer, Sara. 2011.  Statistics for High-Dimensional Data . Springer.

Burges, Christopher. 2010. Dimension Reduction: A Guided Tour.  Foundations and Trends in Machine Learning ,  2 (4), 275–365. Carroll, J Douglas, and Chang, Jih-Jie. 1970. Analysis of Individual Differences in Multidimensional Scaling via an   $N$ -Way Generalization of “Eckart-Young” Decom- position.  Psychometrika ,  35 (3), 283–319. Casella, George, and Berger, Roger L. 2002.  Statistical Inference . Duxbury. ¸inlar, Erhan. 2011.  Probability and Stochastics . Springer. Chang, Chih-Chung, and Lin, Chih-Jen. 2011. LIBSVM: A Library for Support Vector Machines.  ACM Transactions on Intelligent Systems and Technology ,  2 , 27:1–27:27. Cheeseman, Peter. 1985. In Defense of Probability. In:  Proceedings of the International Joint Conference on Artificial Intelligence . Chollet, Francois, and Allaire, J. J. 2018.  Deep Learning with R . Manning Publications. Codd, Edgar F. 1990.  The Relational Model for Database Management . Addison-Wesley Longman Publishing. Cunningham, John P., and Ghahramani, Zoubin. 2015. Linear Dimensionality Reduc- tion: Survey, Insights, and Generalizations.  Journal of Machine Learning Research , 16 , 2859–2900. Datta, Biswa N. 2010.  Numerical Linear Algebra and Applications . SIAM. Davidson, Anthony C., and Hinkley, David V. 1997.  Bootstrap Methods and Their Appli- cation . Cambridge University Press. Dean, Jeffrey, Corrado, Greg S., Monga, Rajat, and Chen, et al. 2012. Large Scale Distributed Deep Networks. In:  Advances in Neural Information Processing Systems . Deisenroth, Marc P., and Mohamed, Shakir. 2012. Expectation Propagation in Gaus- sian Process Dynamical Systems. Pages 2618–2626 of:  Advances in Neural Informa- tion Processing Systems . Deisenroth, Marc P., and Ohlsson, Henrik. 2011. A General Perspective on Gaussian Filtering and Smoothing: Explaining Current and Deriving New Algorithms. In: Proceedings of the American Control Conference . Deisenroth, Marc P., Fox, Dieter, and Rasmussen, Carl E. 2015. Gaussian Processes for Data-Efficient Learning in Robotics and Control.  IEEE Transactions on Pattern Analysis and Machine Intelligence ,  37 (2), 408–423. Dempster, Arthur P., Laird, Nan M., and Rubin, Donald B. 1977. Maximum Likelihood from Incomplete Data via the EM Algorithm.  Journal of the Royal Statistical Society , 39 (1), 1–38. Deng, Li, Seltzer, Michael L., Yu, Dong, Acero, Alex, Mohamed, Abdel-rahman, and Hinton, Geoffrey E. 2010. Binary Coding of Speech Spectrograms Using a Deep Auto-Encoder. In:  Proceedings of Interspeech . Devroye, Luc. 1986.  Non-Uniform Random Variate Generation . Springer. Donoho, David L., and Grimes, Carrie. 2003. Hessian Eigenmaps: Locally Linear Embedding Techniques for High-Dimensional Data. Proceedings of the National Academy of Sciences ,  100 (10), 5591–5596. Dost´ al, Zden˘ ek. 2009.  Optimal Quadratic Programming Algorithms: With Applications to Variational Inequalities . Springer. Douven, Igor. 2017. Abduction. In:  The Stanford Encyclopedia of Philosophy . Meta- physics Research Lab, Stanford University. Downey, Allen B. 2014. Think Stats: Exploratory Data Analysis . 2nd edn. O’Reilly Media. Dreyfus, Stuart. 1962. The Numerical Solution of Variational Problems.  Journal of Mathematical Analysis and Applications ,  5 (1), 30–45. Drumm, Volker, and Weil, Wolfgang. 2001.  Lineare Algebra und Analytische Geometrie . Lecture Notes, Universit¨ at Karlsruhe (TH). Dudley, Richard M. 2002.  Real Analysis and Probability . Cambridge University Press.  

Eaton, Morris L. 2007.  Multivariate Statistics: A Vector Space Approach . Institute of Mathematical Statistics Lecture Notes. Eckart, Carl, and Young, Gale. 1936. The Approximation of One Matrix by Another of Lower Rank.  Psychometrika ,  1 (3), 211–218. Efron, Bradley, and Hastie, Trevor. 2016.  Computer Age Statistical Inference: Algorithms, Evidence and Data Science . Cambridge University Press. Efron, Bradley, and Tibshirani, Robert J. 1993.  An Introduction to the Bootstrap . Chap- man and Hall/CRC. Elliott, Conal. 2009. Beautiful Differentiation. In:  International Conference on Func- tional Programming . Evgeniou, Theodoros, Pontil, Massimiliano, and Poggio, Tomaso. 2000. Statistical Learning Theory: A Primer.  International Journal of Computer Vision ,  38 (1), 9–13. Fan, Rong-En, Chang, Kai-Wei, Hsieh, Cho-Jui, Wang, Xiang-Rui, and Lin, Chih-Jen. 2008. LIBLINEAR: A Library for Large Linear Classification.  Journal of Machine Learning Research ,  9 , 1871–1874. Gal, Yarin, van der Wilk, Mark, and Rasmussen, Carl E. 2014. Distributed Variational Inference in Sparse Gaussian Process Regression and Latent Variable Models. In: Advances in Neural Information Processing Systems . G¨ artner, Thomas. 2008.  Kernels for Structured Data . World Scientific. Gavish, Matan, and Donoho, David L. 2014. The Optimal Hard Threshold for Singular Values is $4\sqrt{3}$ .  IEEE Transactions on Information Theory ,  60 (8), 5040–5053. Gelman, Andrew, Carlin, John B., Stern, Hal S., and Rubin, Donald B. 2004.  Bayesian Data Analysis . Chapman and Hall/CRC. Gentle, James E. 2004. Random Number Generation and Monte Carlo Methods . Springer. Ghahramani, Zoubin. 2015. Probabilistic Machine Learning and Artificial Intelligence. Nature ,  521 , 452–459. Ghahramani, Zoubin, and Roweis, Sam T. 1999. Learning Nonlinear Dynamical Sys- tems Using an EM Algorithm. In:  Advances in Neural Information Processing Systems . MIT Press.Gilks, Walter R., Richardson, Sylvia, and Spiegelhalter, David J. 1996.  Markov Chain Monte Carlo in Practice . Chapman and Hall/CRC. Gneiting, Tilmann, and Raftery, Adrian E. 2007. Strictly Proper Scoring Rules, Pre- diction, and Estimation.  Journal of the American Statistical Association ,  102 (477), 359–378. Goh, Gabriel. 2017. Why Momentum Really Works.  Distill . Gohberg, Israel, Goldberg, Seymour, and Krupnik, Nahum. 2012.  Traces and Determi- nants of Linear Operators . Birkh¨ auser. Golan, Jonathan S. 2007.  The Linear Algebra a Beginning Graduate Student Ought to Know . Springer. Golub, Gene H., and Van Loan, Charles F. 2012.  Matrix Computations . JHU Press. Goodfellow, Ian, Bengio, Yoshua, and Courville, Aaron. 2016.  Deep Learning . MIT Press.Graepel, Thore, Candela, Joaquin Qui˜ nonero-Candela, Borchert, Thomas, and Her- brich, Ralf. 2010. Web-Scale Bayesian Click-through Rate Prediction for Sponsored Search Advertising in Microsoft’s Bing Search Engine. In:  Proceedings of the Interna- tional Conference on Machine Learning . Griewank, Andreas, and Walther, Andrea. 2003. Introduction to Automatic Differenti- ation. In:  Proceedings in Applied Mathematics and Mechanics . Griewank, Andreas, and Walther, Andrea. 2008.  Evaluating Derivatives, Principles and Techniques of Algorithmic Differentiation . SIAM. Grimmett, Geoffrey R., and Welsh, Dominic. 2014.  Probability: An Introduction . Oxford University Press.  

Grinstead, Charles M., and Snell, J. Laurie. 1997.  Introduction to Probability . American Mathematical Society. Hacking, Ian. 2001.  Probability and Inductive Logic . Cambridge University Press. Hall, Peter. 1992.  The Bootstrap and Edgeworth Expansion . Springer. Hallin, Marc, Paindaveine, Davy, and  Siman, Miroslav. 2010. Multivariate Quan- tiles and Multiple-Output Regression Quantiles: From   $\ell_{1}$  Optimization to Halfspace Depth.  Annals of Statistics ,  38 , 635–669. Hasselblatt, Boris, and Katok, Anatole. 2003. A First Course in Dynamics with a Panorama of Recent Developments . Cambridge University Press. Hastie, Trevor, Tibshirani, Robert, and Friedman, Jerome. 2001.  The Elements of Sta- tistical Learning – Data Mining, Inference, and Prediction . Springer. Hausman, Karol, Springenberg, Jost T., Wang, Ziyu, Heess, Nicolas, and Riedmiller, Martin. 2018. Learning an Embedding Space for Transferable Robot Skills. In: Proceedings of the International Conference on Learning Representations . Hazan, Elad. 2015. Introduction to Online Convex Optimization.  Foundations and Trends in Optimization ,  2 (3–4), 157–325. Hensman, James, Fusi, Nicol\` o, and Lawrence, Neil D. 2013. Gaussian Processes for Big Data. In:  Proceedings of the Conference on Uncertainty in Artificial Intelligence . Herbrich, Ralf, Minka, Tom, and Graepel, Thore. 2007. TrueSkill(TM): A Bayesian Skill Rating System. In:  Advances in Neural Information Processing Systems . Hiriart-Urruty, Jean-Baptiste, and Lemar´ echal, Claude. 2001.  Fundamentals of Convex Analysis . Springer. Hoffman, Matthew D., Blei, David M., and Bach, Francis. 2010. Online Learning for Latent Dirichlet Allocation.  Advances in Neural Information Processing Systems . Hoffman, Matthew D., Blei, David M., Wang, Chong, and Paisley, John. 2013. Stochas- tic Variational Inference.  Journal of Machine Learning Research ,  14 (1), 1303–1347. Hofmann, Thomas, Sch¨ olkopf, Bernhard, and Smola, Alexander J. 2008. Kernel Meth- ods in Machine Learning.  Annals of Statistics ,  36 (3), 1171–1220. Hogben, Leslie. 2013.  Handbook of Linear Algebra . Chapman and Hall/CRC. Horn, Roger A., and Johnson, Charles R. 2013.  Matrix Analysis . Cambridge University Press.Hotelling, Harold. 1933. Analysis of a Complex of Statistical Variables into Principal Components.  Journal of Educational Psychology ,  24 , 417–441. Hyvarinen, Aapo, Oja, Erkki, and Karhunen, Juha. 2001.  Independent Component Anal- ysis . Wiley. Imbens, Guido W., and Rubin, Donald B. 2015.  Causal Inference for Statistics, Social and Biomedical Sciences . Cambridge University Press. Jacod, Jean, and Protter, Philip. 2004.  Probability Essentials . Springer. Jaynes, Edwin T. 2003.  Probability Theory: The Logic of Science . Cambridge University Press.Jefferys, William H., and Berger, James O. 1992. Ockham’s Razor and Bayesian Anal- ysis.  American Scientist ,  80 , 64–72. Jeffreys, Harold. 1961.  Theory of Probability . Oxford University Press. Jimenez Rezende, Danilo, and Mohamed, Shakir. 2015. Variational Inference with Nor- malizing Flows. In:  Proceedings of the International Conference on Machine Learning . Jimenez Rezende, Danilo, Mohamed, Shakir, and Wierstra, Daan. 2014. Stochastic Backpropagation and Approximate Inference in Deep Generative Models. In:  Pro- ceedings of the International Conference on Machine Learning . Joachims, Thorsten. 1999.  Advances in Kernel Methods – Support Vector Learning . MIT Press. Chap. Making Large-Scale SVM Learning Practical, pages 169–184. Jordan, Michael I., Ghahramani, Zoubin, Jaakkola, Tommi S., and Saul, Lawrence K. 1999. An Introduction to Variational Methods for Graphical Models.  Machine Learn- ing ,  37 , 183–233.  

Julier, Simon J., and Uhlmann, Jeffrey K. 1997. A New Extension of the Kalman Filter to Nonlinear Systems. In:  Proceedings of AeroSense Symposium on Aerospace/Defense Sensing, Simulation and Controls . Kaiser, Marcus, and Hilgetag, Claus C. 2006. Nonoptimal Component Placement, but Short Processing Paths, Due to Long-Distance Projections in Neural Systems.  PLoS Computational Biology ,  2 (7), e95. Kalman, Dan. 1996. A Singularly Valuable Decomposition: The SVD of a Matrix.  Col- lege Mathematics Journal ,  27 (1), 2–23. Kalman, Rudolf E. 1960. A New Approach to Linear Filtering and Prediction Problems. Transactions of the ASME – Journal of Basic Engineering ,  82 (Series D), 35–45. Kamthe, Sanket, and Deisenroth, Marc P. 2018. Data-Efficient Reinforcement Learning with Probabilistic Model Predictive Control. In:  Proceedings of the International Conference on Artificial Intelligence and Statistics . Katz, Victor J. 2004.  A History of Mathematics . Pearson/Addison-Wesley. Kelley, Henry J. 1960. Gradient Theory of Optimal Flight Paths.  Ars Journal ,  30 (10), 947–954. Kimeldorf, George S., and Wahba, Grace. 1970. A Correspondence between Bayesian Estimation on Stochastic Processes and Smoothing by Splines.  Annals of Mathemat- ical Statistics ,  41 (2), 495–502. Kingma, Diederik P., and Welling, Max. 2014. Auto-Encoding Variational Bayes. In: Proceedings of the International Conference on Learning Representations . Kittler, Josef, and F¨ oglein, Janos. 1984. Contextual Classification of Multispectral Pixel Data.  Image and Vision Computing ,  2 (1), 13–29. Kolda, Tamara G., and Bader, Brett W. 2009. Tensor Decompositions and Applications. SIAM Review ,  51 (3), 455–500. Koller, Daphne, and Friedman, Nir. 2009.  Probabilistic Graphical Models . MIT Press. Kong, Linglong, and Mizera, Ivan. 2012. Quantile Tomography: Using Quantiles with Multivariate Data.  Statistica Sinica ,  22 , 1598–1610. Lang, Serge. 1987.  Linear Algebra . Springer. Lawrence, Neil D. 2005. Probabilistic Non-Linear Principal Component Analysis with Gaussian Process Latent Variable Models.  Journal of Machine Learning Research , 6 (Nov.), 1783–1816. Leemis, Lawrence M., and McQueston, Jacquelyn T. 2008. Univariate Distribution Relationships.  American Statistician ,  62 (1), 45–53. Lehmann, Erich L., and Romano, Joseph P. 2005. Testing Statistical Hypotheses . Springer. Lehmann, Erich Leo, and Casella, George. 1998.  Theory of Point Estimation . Springer. Liesen, J¨ org, and Mehrmann, Volker. 2015.  Linear Algebra . Springer. Lin, Hsuan-Tien, Lin, Chih-Jen, and Weng, Ruby C. 2007. A Note on Platt’s Probabilistic Outputs for Support Vector Machines.  Machine Learning ,  68 , 267–276. Ljung, Lennart. 1999.  System Identification: Theory for the User . Prentice Hall. Loosli, Ga¨ elle, Canu, St´ ephane, and Ong, Cheng Soon. 2016. Learning SVM in Kre˘ ın Spaces.  IEEE Transactions of Pattern Analysis and Machine Intelligence ,  38 (6), 1204– 1216. Luenberger, David G. 1969.  Optimization by Vector Space Methods . Wiley. MacKay, David J. C. 1992. Bayesian Interpolation.  Neural Computation ,  4 , 415–447. MacKay, David J. C. 1998. Introduction to Gaussian Processes. Pages 133–165 of: Bishop, C. M. (ed),  Neural Networks and Machine Learning . Springer. MacKay, David J. C. 2003. Information Theory, Inference, and Learning Algorithms . Cambridge University Press. Magnus, Jan R., and Neudecker, Heinz. 2007.  Matrix Differential Calculus with Appli- cations in Statistics and Econometrics . Wiley.  

Manton, Jonathan H., and Amblard, Pierre-Olivier. 2015. A Primer on Reproducing Kernel Hilbert Spaces.  Foundations and Trends in Signal Processing ,  8 (1–2), 1–126. Markovsky, Ivan. 2011.  Low Rank Approximation: Algorithms, Implementation, Appli- cations . Springer. Maybeck, Peter S. 1979.  Stochastic Models, Estimation, and Control . Academic Press. McCullagh, Peter, and Nelder, John A. 1989.  Generalized Linear Models . CRC Press. McEliece, Robert J., MacKay, David J. C., and Cheng, Jung-Fu. 1998. Turbo Decoding as an Instance of Pearl’s “Belief Propagation” Algorithm.  IEEE Journal on Selected Areas in Communications ,  16 (2), 140–152. Mika, Sebastian, R¨ atsch, Gunnar, Weston, Jason, Sch¨ olkopf, Bernhard, and M¨ uller, Klaus-Robert. 1999. Fisher Discriminant Analysis with Kernels. Pages 41–48 of: Proceedings of the Workshop on Neural Networks for Signal Processing . Minka, Thomas P. 2001a.  A Family of Algorithms for Approximate Bayesian Inference . Ph.D. thesis, Massachusetts Institute of Technology. Minka, Tom. 2001b. Automatic Choice of Dimensionality of PCA. In:  Advances in Neural Information Processing Systems . Mitchell, Tom. 1997.  Machine Learning . McGraw-Hill. Mnih, Volodymyr, Kavukcuoglu, Koray, and Silver, David, et al. 2015. Human-Level Control through Deep Reinforcement Learning.  Nature ,  518 , 529–533. Moonen, Marc, and De Moor, Bart. 1995.  SVD and Signal Processing, III: Algorithms, Architectures and Applications . Elsevier. Moustaki, Irini, Knott, Martin, and Bartholomew, David J. 2015.  Latent-Variable Mod- eling . American Cancer Society. Pages 1–10. M¨ uller, Andreas C., and Guido, Sarah. 2016.  Introduction to Machine Learning with Python: A Guide for Data Scientists . O’Reilly Publishing. Murphy, Kevin P. 2012.  Machine Learning: A Probabilistic Perspective . MIT Press. Neal, Radford M. 1996.  Bayesian Learning for Neural Networks . Ph.D. thesis, Depart- ment of Computer Science, University of Toronto. Neal, Radford M., and Hinton, Geoffrey E. 1999. A View of the EM Algorithm that Justifies Incremental, Sparse, and Other Variants. Pages 355–368 of:  Learning in Graphical Models . MIT Press. Nelsen, Roger. 2006.  An Introduction to Copulas . Springer. Nesterov, Yuri. 2018.  Lectures on Convex Optimization . Springer. Neumaier, Arnold. 1998. Solving Ill-Conditioned and Singular Linear Systems: A Tu- torial on Regularization.  SIAM Review ,  40 , 636–666. Nocedal, Jorge, and Wright, Stephen J. 2006.  Numerical Optimization . Springer. Nowozin, Sebastian, Gehler, Peter V., Jancsary, Jeremy, and Lampert, Christoph H. (eds). 2014.  Advanced Structured Prediction . MIT Press. O’Hagan, Anthony. 1991. Bayes-Hermite Quadrature.  Journal of Statistical Planning and Inference ,  29 , 245–260. Ong, Cheng Soon, Mary, Xavier, Canu, St´ ephane, and Smola, Alexander J. 2004. Learn- ing with Non-Positive Kernels. In:  Proceedings of the International Conference on Machine Learning . Ormoneit, Dirk, Sidenbladh, Hedvig, Black, Michael J., and Hastie, Trevor. 2001. Learning and Tracking Cyclic Human Motion. In:  Advances in Neural Information Processing Systems . Page, Lawrence, Brin, Sergey, Motwani, Rajeev, and Winograd, Terry. 1999. The PageRank Citation Ranking: Bringing Order to the Web . Tech. rept. Stanford Info- Lab. Paquet, Ulrich. 2008.  Bayesian Inference for Latent Variable Models . Ph.D. thesis, Uni- versity of Cambridge. Parzen, Emanuel. 1962. On Estimation of a Probability Density Function and Mode. Annals of Mathematical Statistics ,  33 (3), 1065–1076.  

Pearl, Judea. 1988.  Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference . Morgan Kaufmann. Pearl, Judea. 2009.  Causality: Models, Reasoning and Inference . 2nd edn. Cambridge University Press. Pearson, Karl. 1895. Contributions to the Mathematical Theory of Evolution. II. Skew Variation in Homogeneous Material.  Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences ,  186 , 343–414. Pearson, Karl. 1901. On Lines and Planes of Closest Fit to Systems of Points in Space. Philosophical Magazine ,  2 (11), 559–572. Peters, Jonas, Janzing, Dominik, and Sch¨ olkopf, Bernhard. 2017.  Elements of Causal Inference: Foundations and Learning Algorithms . MIT Press. Petersen, Kaare B., and Pedersen, Michael S. 2012.  The Matrix Cookbook . Tech. rept. Technical University of Denmark. Platt, John C. 2000. Probabilistic Outputs for Support Vector Machines and Compar- isons to Regularized Likelihood Methods. In:  Advances in Large Margin Classifiers . Pollard, David. 2002. A User’s Guide to Measure Theoretic Probability . Cambridge University Press. Polyak, Roman A. 2016. The Legendre Transformation in Modern Optimization. Pages 437–507 of: Goldengorin, B. (ed),  Optimization and Its Applications in Control and Data Sciences . Springer. Press, William H., Teukolsky, Saul A., Vetterling, William T., and Flannery, Brian P. 2007.  Numerical Recipes: The Art of Scientific Computing . Cambridge University Press.Proschan, Michael A., and Presnell, Brett. 1998. Expect the Unexpected from Condi- tional Expectation.  American Statistician ,  52 (3), 248–252. Raschka, Sebastian, and Mirjalili, Vahid. 2017. Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow . Packt Publish- ing. Rasmussen, Carl E., and Ghahramani, Zoubin. 2001. Occam’s Razor. In:  Advances in Neural Information Processing Systems . Rasmussen, Carl E., and Ghahramani, Zoubin. 2003. Bayesian Monte Carlo. In:  Ad- vances in Neural Information Processing Systems . Rasmussen, Carl E., and Williams, Christopher K. I. 2006.  Gaussian Processes for Ma- chine Learning . MIT Press. Reid, Mark, and Williamson, Robert C. 2011. Information, Divergence and Risk for Binary Experiments.  Journal of Machine Learning Research ,  12 , 731–817. Rifkin, Ryan M., and Lippert, Ross A. 2007. Value Regularization and Fenchel Duality. Journal of Machine Learning Research ,  8 , 441–479. Rockafellar, Ralph T. 1970.  Convex Analysis . Princeton University Press. Rogers, Simon, and Girolami, Mark. 2016.  A First Course in Machine Learning . Chap- man and Hall/CRC. Rosenbaum, Paul R. 2017. Observation and Experiment: An Introduction to Causal Inference . Harvard University Press. Rosenblatt, Murray. 1956. Remarks on Some Nonparametric Estimates of a Density Function.  Annals of Mathematical Statistics ,  27 (3), 832–837. Roweis, Sam T. 1998. EM Algorithms for PCA and SPCA. Pages 626–632 of:  Advances in Neural Information Processing Systems . Roweis, Sam T., and Ghahramani, Zoubin. 1999. A Unifying Review of Linear Gaussian Models.  Neural Computation ,  11 (2), 305–345. Roy, Anindya, and Banerjee, Sudipto. 2014.  Linear Algebra and Matrix Analysis for Statistics . Chapman and Hall/CRC. Rubinstein, Reuven Y., and Kroese, Dirk P. 2016. Simulation and the Monte Carlo Method . Wiley.  

Ruffini, Paolo. 1799.  Teoria Generale delle Equazioni, in cui si Dimostra Impossibile la Soluzione Algebraica delle Equazioni Generali di Grado Superiore al Quarto . Stampe- ria di S. Tommaso d’Aquino. Rumelhart, David E., Hinton, Geoffrey E., and Williams, Ronald J. 1986. Learning Representations by Back-Propagating Errors.  Nature ,  323 (6088), 533–536. Sæmundsson, Steind´ or, Hofmann, Katja, and Deisenroth, Marc P. 2018. Meta Rein- forcement Learning with Latent Variable Gaussian Processes. In:  Proceedings of the Conference on Uncertainty in Artificial Intelligence . Saitoh, Saburou. 1988.  Theory of Reproducing Kernels and its Applications . Longman Scientific and Technical. S¨ arkk¨ a, Simo. 2013.  Bayesian Filtering and Smoothing . Cambridge University Press. Sch¨ olkopf, Bernhard, and Smola, Alexander J. 2002.  Learning with Kernels – Support Vector Machines, Regularization, Optimization, and Beyond . MIT Press. Sch¨ olkopf, Bernhard, Smola, Alexander J., and M¨ uller, Klaus-Robert. 1997. Kernel Principal Component Analysis. In:  Proceedings of the International Conference on Artificial Neural Networks . Sch¨ olkopf, Bernhard, Smola, Alexander J., and M¨ uller, Klaus-Robert. 1998. Nonlinear Component Analysis as a Kernel Eigenvalue Problem.  Neural Computation ,  10 (5), 1299–1319. Sch¨ olkopf, Bernhard, Herbrich, Ralf, and Smola, Alexander J. 2001. A Generalized Representer Theorem. In:  Proceedings of the International Conference on Computa- tional Learning Theory . Schwartz, Laurent. 1964. Sous Espaces Hilbertiens d’Espaces Vectoriels Topologiques et Noyaux Associ´ es.  Journal d’Analyse Math´ ematique ,  13 , 115–256. Schwarz, Gideon E. 1978. Estimating the Dimension of a Model.  Annals of Statistics , 6 (2), 461–464. Shahriari, Bobak, Swersky, Kevin, Wang, Ziyu, Adams, Ryan P., and De Freitas, Nando. 2016. Taking the Human out of the Loop: A Review of Bayesian Optimization. Proceedings of the IEEE ,  104 (1), 148–175. Shalev-Shwartz, Shai, and Ben-David, Shai. 2014.  Understanding Machine Learning: From Theory to Algorithms . Cambridge University Press. Shawe-Taylor, John, and Cristianini, Nello. 2004.  Kernel Methods for Pattern Analysis . Cambridge University Press. Shawe-Taylor, John, and Sun, Shiliang. 2011. A Review of Optimization Methodologies in Support Vector Machines.  Neurocomputing ,  74 (17), 3609–3618. Shental, Ori, Siegel, Paul H., Wolf, Jack K., Bickson, Danny, and Dolev, Danny. 2008. Gaussian Belief Propagation Solver for Systems of Linear Equations. Pages 1863– 1867 of:  Proceedings of the International Symposium on Information Theory . Shewchuk, Jonathan R. 1994.  An Introduction to the Conjugate Gradient Method with- out the Agonizing Pain . Shi, Jianbo, and Malik, Jitendra. 2000. Normalized Cuts and Image Segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence ,  22 (8), 888–905. Shi, Qinfeng, Petterson, James, Dror, Gideon, Langford, John, Smola, Alexander J., and Vishwanathan, S. V. N. 2009. Hash Kernels for Structured Data.  Journal of Machine Learning Research , 2615–2637. Shiryayev, Albert N. 1984.  Probability . Springer. Shor, Naum Z. 1985.  Minimization Methods for Non-Differentiable Functions . Springer. Shotton, Jamie, Winn, John, Rother, Carsten, and Criminisi, Antonio. 2006. Texton- Boost: Joint Appearance, Shape and Context Modeling for Multi-Class Object Recog- nition and Segmentation. In:  Proceedings of the European Conference on Computer Vision . Smith, Adrian F. M., and Spiegelhalter, David. 1980. Bayes Factors and Choice Criteria for Linear Models.  Journal of the Royal Statistical Society B ,  42 (2), 213–220.  

Snoek, Jasper, Larochelle, Hugo, and Adams, Ryan P. 2012. Practical Bayesian Op- timization of Machine Learning Algorithms. In:  Advances in Neural Information Processing Systems . Spearman, Charles. 1904. “General Intelligence,” Objectively Determined and Mea- sured.  American Journal of Psychology ,  15 (2), 201–292. Sriperumbudur, Bharath K., Gretton, Arthur, Fukumizu, Kenji, Sch¨ olkopf, Bernhard, and Lanckriet, Gert R. G. 2010. Hilbert Space Embeddings and Metrics on Proba- bility Measures.  Journal of Machine Learning Research ,  11 , 1517–1561. Steinwart, Ingo. 2007. How to Compare Different Loss Functions and Their Risks. Constructive Approximation ,  26 , 225–287. Steinwart, Ingo, and Christmann, Andreas. 2008.  Support Vector Machines . Springer. Stoer, Josef, and Burlirsch, Roland. 2002.  Introduction to Numerical Analysis . Springer. Strang, Gilbert. 1993. The Fundamental Theorem of Linear Algebra.  The American Mathematical Monthly ,  100 (9), 848–855. Strang, Gilbert. 2003.  Introduction to Linear Algebra . Wellesley-Cambridge Press. Stray, Jonathan. 2016.  The Curious Journalist’s Guide to Data . Tow Center for Digital Journalism at Columbia’s Graduate School of Journalism. Strogatz, Steven. 2014. Writing about Math for the Perplexed and the Traumatized. Notices of the American Mathematical Society ,  61 (3), 286–291. Sucar, Luis E., and Gillies, Duncan F. 1994. Probabilistic Reasoning in High-Level Vision.  Image and Vision Computing ,  12 (1), 42–60. Szeliski, Richard, Zabih, Ramin, and Scharstein, Daniel, et al. 2008. A Compar- ative Study of Energy Minimization Methods for Markov Random Fields with Smoothness-Based Priors.  IEEE Transactions on Pattern Analysis and Machine In- telligence ,  30 (6), 1068–1080. Tandra, Haryono. 2014. The Relationship between the Change of Variable Theorem and the Fundamental Theorem of Calculus for the Lebesgue Integral.  Teaching of Mathematics ,  17 (2), 76–83. Tenenbaum, Joshua B., De Silva, Vin, and Langford, John C. 2000. A Global Geometric Framework for Nonlinear Dimensionality Reduction.  Science ,  290 (5500), 2319– 2323. Tibshirani, Robert. 1996. Regression Selection and Shrinkage via the Lasso.  Journal of the Royal Statistical Society B ,  58 (1), 267–288. Tipping, Michael E., and Bishop, Christopher M. 1999. Probabilistic Principal Compo- nent Analysis.  Journal of the Royal Statistical Society: Series B ,  61 (3), 611–622. Titsias, Michalis K., and Lawrence, Neil D. 2010. Bayesian Gaussian Process Latent Variable Model. In:  Proceedings of the International Conference on Artificial Intelli- gence and Statistics . Toussaint, Marc. 2012.  Some Notes on Gradient Descent . https://ipvs.informatik.uni- stuttgart.de/mlr/marc/notes/gradientDescent.pdf. Trefethen, Lloyd N., and Bau III, David. 1997.  Numerical Linear Algebra . SIAM. Tucker, Ledyard R. 1966. Some Mathematical Notes on Three-Mode Factor Analysis. Psychometrika ,  31 (3), 279–311. Vapnik, Vladimir N. 1998.  Statistical Learning Theory . Wiley. Vapnik, Vladimir N. 1999. An Overview of Statistical Learning Theory.  IEEE Transac- tions on Neural Networks ,  10 (5), 988–999. Vapnik, Vladimir N. 2000.  The Nature of Statistical Learning Theory . Springer. Vishwanathan, S. V. N., Schraudolph, Nicol N., Kondor, Risi, and Borgwardt, Karsten M. 2010. Graph Kernels.  Journal of Machine Learning Research ,  11 , 1201– 1242. von Luxburg, Ulrike, and Sch¨ olkopf, Bernhard. 2011. Statistical Learning Theory: Models, Concepts, and Results. Pages 651–706 of: D. M. Gabbay, S. Hartmann, J. Woods (ed),  Handbook of the History of Logic , vol. 10. Elsevier.  

Wahba, Grace. 1990.  Spline Models for Observational Data . Society for Industrial and Applied Mathematics. Walpole, Ronald E., Myers, Raymond H., Myers, Sharon L., and Ye, Keying. 2011. Probability and Statistics for Engineers and Scientists . Prentice Hall. Wasserman, Larry. 2004.  All of Statistics . Springer. Wasserman, Larry. 2007.  All of Nonparametric Statistics . Springer. Whittle, Peter. 2000.  Probability via Expectation . Springer. Wickham, Hadley. 2014. Tidy Data.  Journal of Statistical Software ,  59 , 1–23. Williams, Christopher K. I. 1997. Computing with Infinite Networks. In:  Advances in Neural Information Processing Systems . Yu, Yaoliang, Cheng, Hao, Schuurmans, Dale, and Szepesv´ ari, Csaba. 2013. Charac- terizing the Representer Theorem. In:  Proceedings of the International Conference on Machine Learning . Zadrozny, Bianca, and Elkan, Charles. 2001. Obtaining Calibrated Probability Esti- mates from Decision Trees and Naive Bayesian Classifiers. In:  Proceedings of the International Conference on Machine Learning . Zhang, Haizhang, Xu, Yuesheng, and Zhang, Jun. 2009. Reproducing Kernel Banach Spaces for Machine Learning.  Journal of Machine Learning Research ,  10 , 2741–2775. Zia, Royce K. P., Redish, Edward F., and McKay, Susan R. 2009. Making Sense of the Legendre Transform.  American Journal of Physics ,  77 (614), 614–622.  

# 索引  

1 -of- $\mathit{K}$ 表示，364

 $\ell_{1}$ 范数，71

 $\ell_{2}$ 范数，72 摄取，258 阿贝尔-鲁菲尼定理，334 阿贝尔群，36 绝对齐次，71 激活函数，315 幂线性映射，63 幂线性子空间，61 阿克aike信息准则，288 代数，17 代数重数，106 分析的，143 先祖采样，340，364 角度，76 结合律，24，26，36 属性，253 增广矩阵，29 自编码器，343 自动微分，161 自同构，49 反向传播，159 基本变量，30 基，44 基向量，45 贝叶斯因子，287 贝叶斯定律，185 贝叶斯规则，185 贝叶斯定理，185 贝叶斯GP-LVM，347 贝叶斯推理，274 贝叶斯信息准则，288 贝叶斯线性回归，303 贝叶斯模型选择，286 贝叶斯网络，278，283 贝叶斯PCA，346 伯努利分布，205 乙太分布，206 双线性映射，72 双射，48 二元分类，370 二项分布，206 盲源分离，346 波尔耳 $\sigma$ -代数，180  

标准基，45 标准特征映射，389 标准联系函数，315 分类变量，180 柯西-施瓦茨不等式，75 变量变换技术，219 特征多项式，104 乔列斯基分解，114 乔列斯基因子，114 乔列斯基分解，114 类，370 分类，315 闭包，36 代码，343 代码方向，105 代码域，58，139 共线，105 列，22 列空间，59 列向量，22，38 完成平方，307 凹函数，236 条件数，230 条件概率，179 条件独立，195 共轭，208 共轭先验，208 凸共轭，242 凸函数，236 凸包，386 凸优化问题，236，239 凸集，236 坐标，50 坐标表示，50 坐标向量，50 相关系数，191 协方差，190 协方差矩阵，190，198 因变量，253 CP 分解，136 交叉协方差，191 交叉验证，258，263 累积分布函数，178，181 d-分离，281  

数据协方差矩阵，318 特征，253 数据点，253 特征映射，254 数据拟合项，302 特征矩阵，296 解码器，343 特征向量，295 深度自编码器，347 菲舍判别分析，136 缺陷，111 菲舍-尼曼定理，210 分子布局，151 正向模式，161 导数，141 自由变量，30 设计矩阵，294，296 完备秩，47 行列式，99 完全SVD，128 对角矩阵，115 线性代数基本定理，116 映射，60 对角化，116 高斯消元法，31 差商，141 高斯混合模型，349 维度，45 高斯过程，316 维度缩减，317 高斯过程潜在变量模型，有向图模型，347，283 347 方向，61 一般线性群，37 方向空间，61 一般解，28，30 距离，75 广义线性模型，272，315 分布，177 生成集，44 分配律，24，26 生成过程，272，286 域，58，139 生成器，344 点积，72 几何重数，108 双边支持向量机，385 吉文斯旋转，94 埃克特-杨定理，131，334 全局最小值，225 特征分解，116 GP-LVM，347 特征空间，106 梯度，146 特征谱，106 格拉姆矩阵，389 特征值，105 格拉姆-施密特正交化，89 特征值方程，105 图模型，278 特征向量，105 群，36 初等变换，28 赫达玛积，23 EM算法，360 硬边界支持向量机，377 显而易见并行，264 海森矩阵，164 经验协方差，192 海森特征映射，136 经验均值，192 海森矩阵，165 经验风险，260 损失函数，260，381 损失项，382 下三角矩阵，101 麦克劳林级数，143 曼哈顿范数，71 MAP，300 MAP估计，269  

边界，374 边缘，190 边缘似然，186，286，306 边缘概率，179 边缘化性质，184 马尔可夫随机场，283 矩阵，22 矩阵分解，98 最大后验概率，300 最大后验概率估计，269 最大似然估计，257 最大似然估计值，296 最大似然估计，265，293 均值，187 均值函数，309 均值向量，198 度量，180 中位数，188 概率，175 最小，44 最小最大不等式，234 不匹配项，302 混合模型，349 混合权重，349 模式，188 模型，251 模型证据，286 模型选择，258 莫尔-彭罗斯伪逆，35 多维尺度分析，136 标量乘法，37 多变量，178 多变量高斯分布，198 多变量泰勒级数，166 自然参数，212 负对数似然，265 嵌套交叉验证，258，284 中性元素，36 非可逆，24 非奇异，24 范数，71 正态分布，197 正态方程，86 正态向量，80 零空间，33，47，58 分子布局，150 奥卡姆剃刀，285 ONB，79 一位热编码，364 有序基，50 正交，77 正交基，79 正交补，79 正交矩阵，78 正交化，77 正交基，79  

外积，38 正则化参数，263，302，380 过拟合，262，271，299 正则化最小二乘法，302 正则化项，263，302，380，382 帕雷尔克，114 表示定理，384 参数，61 责任，352 参数方程，61 反向模式，161 部分导数，146 右奇异向量，119 勒让德变换，242 勒让德-芬彻变换，242 长度，71 似然，185，265，269，291 直线，61，82 线性组合，40 线性流形，61 线性映射，48 线性规划，239 线性子空间，39 线性变换，48 线性相关，40 线性无关，40 联结函数，272 负荷，322 局部最小值，225 对数分区函数，211 对数几率回归，315 对数几率函数，315 损失函数，260，381 损失项，382 下三角矩阵，101 麦克劳林级数，143 曼哈顿范数，71 MAP，300 MAP估计，269  

泰勒多项式，142，166 泰勒级数，142 测试误差，300 测试集，262，284 蒂克霍夫正则化，265 跟踪，103 训练，12 训练误差，300 训练集，260，292 转换函数，315 转换矩阵，51 平移向量，63 转置，25，38 三角不等式，71，76 截断SVD，129 图克尔分解，136 欠拟合，271 无向图模型，283 均匀分布，182 单变量，178 无散度变换，170 上三角矩阵，101 验证集，263，284 变量选择，316 方差，190 向量，37 向量加法，37 向量空间，37 向量空间同构，48 向量空间内积，73 向量子空间，39 弱对偶性，235 零一损失，381