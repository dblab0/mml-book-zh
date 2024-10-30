
# 第五章 向量微积分

机器学习中的许多算法通过调整一组控制模型解释数据程度的期望模型参数来优化目标函数：找到好的参数可以表述为一个优化问题（参见第8.2节和第8.3节）。例如：（i）线性回归（参见第9章），其中我们研究曲线拟合问题，并优化线性权重参数以最大化似然性；（ii）用于降维和数据压缩的神经网络自动编码器，其中参数是每一层的权重和偏置，我们通过反复应用链式法则来最小化重构误差；（iii）高斯混合模型（参见第11章），用于建模数据分布，我们优化每个混合成分的位置和形状参数以最大化模型的似然性。图5.1说明了这些问题，我们通常通过利用梯度信息的优化算法来解决这些问题（参见第7.1节）。图5.2概述了本章概念之间的关系以及它们与其他章节的联系。

本章的核心概念是函数。函数$f$是一个量，它将两个量相互关联。在这本书中，这些量通常是输入$\pmb{x}\in\mathbb{R}^{D}$和目标（函数值$f(\pmb{x})$），我们假设它们是实数值，除非另有说明。这里$\mathbb{R}^{D}$是$f$的定义域，而函数值$f(\pmb{x})$是$f$的值域/陪域。

![](images/bfc51c0e593d2d335dc6608e3adf5a37ca0c93bedfb168e874e17725642d6602.jpg)
（a）回归问题：找到参数，使得曲线很好地解释了观测结果 (穿过)。
（b）使用高斯混合进行密度估计，使得曲线能够很好地解释观测值（叉号），数据（点）能够被很好地解释。

![](images/a38372a8b9c03ed4175fa59bb66aea3dc7c62c31662fb1d8ff205bf27b457b4d.jpg)
图5.2 本章介绍的概念思维导图，以及它们在本书其他部分的使用情况。

第2.7.3节在讨论线性函数的背景下提供了更详细的讨论。我们经常这样写

$$
\begin{array}{r}{f:\mathbb{R}^{D}\rightarrow\mathbb{R}}\\ {\pmb{x}\mapsto f(\pmb{x})}\end{array}
$$
来指定一个函数，其中（5.1a）指明$f$是从$\mathbb{R}^{D}$到$\mathbb{R}$的映射，（5.1b）指明输入$\pmb{x}$到函数值$f(\pmb{x})$的具体赋值。函数$f$为每个输入$\pmb{x}$分配一个唯一的函数值$f(\pmb{x})$。

### 示例 5.1

回顾点积作为内积的一种特殊情况（第3.2节）。在之前的记法中，函数 $f(\pmb{x})=\pmb{x}^{\top}\pmb{x}$ ，$\pmb{x}\,\in\,\mathbb{R}^{2}$ ，可以表示为

$$ 
\begin{align}
f: \mathbb{R}^2 &\to \mathbb{R} \\
\pmb{x} &\mapsto x_1^2 + x_2^2 \,.
\end{align} 
$$

在本章中，我们将讨论如何计算函数的梯度，这对于促进机器学习模型中的学习通常是非常必要的，因为梯度指向最陡峭的上升方向。因此，

![](images/f0abc02815d93aeaf65dcbf4f4a1d5e9d336723b56ecad6433dd182aa70c807b.jpg)
图5.3 函数 $f$ 在 $x_{0}$ 和 $x_{0}+\delta x$ 之间的平均斜率是通过 $f(x_{0})$ 和 $f(x_{0}+\delta x)$ 的割线（蓝色）的斜率，由 $\delta y/\delta x$ 给出。

向量微积分是我们进行机器学习时需要的基本数学工具之一。在本书中，我们假设函数是可微的。通过一些额外的技术定义（我们在此不讨论），许多提出的方法可以扩展到次微分（在某些点上连续但不可微的函数）。在第7章中，我们将讨论带有约束条件的函数的扩展情况。


## 5.1 一元函数的微分

在接下来的内容中，我们将简要回顾一元函数的微分，这可能是你在高中数学中已经熟悉的。我们从一元函数 $y=f(x)$ 的差商开始，其中 $x,y\in\mathbb{R}$，随后我们将使用它来定义导数。

### 定义 5.1 （差商）. 差商

$$
{\frac{\delta y}{\delta x}}:={\frac{f(x+\delta x)-f(x)}{\delta x}}
$$
计算了通过函数 $f$ 图像上两点的割线的斜率。在图 5.3 中，这些点的 $x$ 坐标分别是 $x_{0}$ 和 $x_{0}+\delta x$。

差商也可以被视为函数 $f$ 在 $x$ 和 $x+\delta x$ 之间的平均斜率，如果我们假设 $f$ 是一个线性函数。当 $\delta x\,\rightarrow\,0$ 时，如果 $f$ 是可微的，我们得到 $f$ 在 $x$ 处的切线。切线就是 $f$ 在 $x$ 处的导数。

### 定义 5.2 （导数）. 

更正式地，对于 $h>0$，函数 $f$ 在 $x$ 处的导数定义为极限

$$
{\frac{\mathrm{d}f}{\mathrm{d}x}}:=\operatorname*{lim}_{h\to0}{\frac{f(x+h)-f(x)}{h}}\,，
$$
此时图 5.3 中的割线变成了切线。函数 $f$ 的导数指向 $f$ 最陡峭上升的方向。


### 示例 5.2（多项式的导数）

我们想要计算多项式 $f(x)=x^{n},n\in\mathbb{N}$ 的导数。我们可能已经知道结果是 $n x^{n-1}$，但希望通过导数的定义来推导这个结果。根据导数的定义（5.4），我们得到

$$
\begin{align}
\frac{\mathrm{d}f}{\mathrm{d}x} &= \lim_{h\to0} \frac{f(x+h)-f(x)}{h} \\
&= \lim_{h\to0} \frac{(x+h)^n - x^n}{h} \\
&= \lim_{h\to0} \frac{\sum_{i=0}^{n} \binom{n}{i} x^{n-i} h^i - x^n}{h} \,.
\end{align}
$$

我们看到 $\textstyle x^{n}={\binom{n}{0}}x^{n-0}h^{0}$。从1开始求和，$x^{n}$ 项相互抵消，我们得到

$$
\begin{align}
\frac{\mathrm{d}f}{\mathrm{d}x} &= \lim_{h\to0} \frac{\sum_{i=1}^{n} \binom{n}{i} x^{n-i} h^i}{h} \\
&= \lim_{h\to0} \sum_{i=1}^{n} \binom{n}{i} x^{n-i} h^{i-1} \\
&= \lim_{h\to0} \left( \binom{n}{1} x^{n-1} \sum_{i=2}^{n} \binom{n}{i} x^{n-i} h^{i-1} \right) \\
&= \lim_{h\to0} \left( n x^{n-1} \sum_{i=2}^{n} \binom{n}{i} x^{n-i} h^{i-1} \right) \\
&= n x^{n-1} \,.
\end{align}
$$



### 5.1.1 泰勒级数

泰勒级数是将函数 $f$ 表示为无穷项之和的一种方式。这些项是通过在 $x_{0}$ 处计算 $f$ 的导数来确定的。

泰勒多项式 我们定义 $t^{0}:=1$ 对于所有 $t\in\mathbb{R}$。

定义 5.3 （泰勒多项式）。函数 $f:\mathbb{R}\to\mathbb{R}$ 在 $x_{0}$ 处的泰勒多项式定义为

$$
T_{n}(x):=\sum_{k=0}^{n}\frac{f^{(k)}(x_{0})}{k!}(x-x_{0})^{k}\,,
$$
其中 $f^{(k)}(x_{0})$ 是 $f$ 在 $x_{0}$ 处的第 $k$ 个导数（我们假设它存在），并且 $\frac{f^{(k)}(x_{0})}{k!}$ 是多项式的系数。

泰勒级数 定义 5.4 （泰勒级数）。对于一个光滑函数 $f\in\mathcal C^{\infty}$，$f:\mathbb{R}\to\mathbb{R}$，$f$ 在 $x_{0}$ 处的泰勒级数定义为

$$
T_{\infty}(x)=\sum_{k=0}^{\infty}{\frac{f^{(k)}(x_{0})}{k!}}(x-x_{0})^{k}\,.
$$
当 $x_{0}=0$ 时，我们得到泰勒级数的一个特例，即麦克劳林级数。如果 $f(x)=T_{\infty}(x)$，则称 $f$ 为解析函数。

注释。一般来说，泰勒多项式 $n$ 次是函数的一个近似，它不一定是一个多项式。泰勒多项式在 $x_{0}$ 附近的邻域内与 $f$ 相似。然而，对于 $n$ 次的泰勒多项式，它是多项式 $f$ 的精确表示，其次数为 $k\leqslant n$，因为所有 $f^{(i)}$，$i>k$ 的导数都为零。$\diamondsuit$

$f\in\mathcal C^{\infty}$ 表示 $f$ 是无穷次连续可微的。麦克劳林级数 解析


### 示例 5.3 (泰勒多项式)

我们考虑多项式

$$
f(x)=x^{4}
$$
并寻找在 $x_{0}=1$ 处的泰勒多项式 $T_{6}$。我们首先计算系数 $f^{(k)}(1)$ 对于 $k=0,\cdots,6$：

$$
\begin{align}
f(1) &= 1 \\
f'(1) &= 4 \\
f''(1) &= 12 \\
f^{(3)}(1) &= 24 \\
f^{(4)}(1) &= 24 \\
f^{(5)}(1) &= 0 \\
f^{(6)}(1) &= 0
\end{align}
$$

因此，所需的泰勒多项式为

$$
\begin{align}
T_{6}(x) &= \sum_{k=0}^{6} \frac{f^{(k)}(x_{0})}{k!} (x - x_{0})^{k} \\
&= 1 + 4(x - 1) + 6(x - 1)^{2} + 4(x - 1)^{3} + (x - 1)^{4} + 0
\end{align}
$$

展开并重新排列得

$$
\begin{align}
T_{6}(x) &= \left(1 - 4 + 6 - 4 + 1\right) + x \left(4 - 12 + 12 - 4\right) \\
&\quad + x^2 \left(6 - 12 + 6\right) + x^3 \left(4 - 4\right) + x^4 \\
&\quad = x^4 = f(x) \,.
\end{align}
$$

即，我们得到了原函数的精确表示。

![](images/752484b10af90206cdbc320eb86447d1ae7b3ee736d2b8362f4be5befabb0669.jpg)

图 5.4 泰勒多项式。原函数 $f(x)=\sin(x)+\cos(x)$（黑色实线）由泰勒多项式（虚线）在 $x_{0}=0$ 处近似。更高阶的泰勒多项式更好地全局逼近函数 $f$。$T_{10}$ 在 $[-4,4]$ 区间内已经与 $f$ 相似。


### 示例 5.4（泰勒级数）

考虑图5.4中给出的函数

$$
f(x)=\sin(x)+\cos(x)\in{\mathcal{C}}^{\infty}\,.
$$
我们寻求在$x_{0}=0$处的泰勒级数展开，即$f$的麦克劳林级数展开。我们得到以下导数：

$$
\begin{align}
f(0) &= \sin(0) + \cos(0) = 1 \\
f'(0) &= \cos(0) - \sin(0) = 1 \\
f''(0) &= -\sin(0) - \cos(0) = -1 \\
f^{(3)}(0) &= -\cos(0) + \sin(0) = -1 \\
f^{(4)}(0) &= \sin(0) + \cos(0) = f(0) = 1 \\
&\vdots
\end{align}
$$

我们可以看到一个模式：我们的泰勒级数中的系数只有$\pm1$（因为$\sin(0)=0$），每两次后切换到另一个值。此外，$f^{(k+4)}(0)=f^{(k)}(0)$。

因此，在$x_{0}=0$处的$f$的完整泰勒级数展开式为

$$
\begin{align}
T_{\infty}(x) &\quad = \sum_{k=0}^{\infty} \frac{f^{(k)}(x_0)}{k!} (x - x_0)^k \\
&\quad = 1 + x - \frac{1}{2!} x^2 - \frac{1}{3!} x^3 + \frac{1}{4!} x^4 + \frac{1}{5!} x^5 - \cdots \\
&\quad = 1 - \frac{1}{2!} x^2 + \frac{1}{4!} x^4 + \cdots + x - \frac{1}{3!} x^3 + \frac{1}{5!} x^5 - \cdots \\
&\quad = \sum_{k=0}^{\infty} (-1)^k \frac{1}{(2k)!} x^{2k} + \sum_{k=0}^{\infty} (-1)^k \frac{1}{(2k+1)!} x^{2k+1} \\
&\quad = \cos(x) + \sin(x) \,.
\end{align}
$$

其中我们使用了幂级数表示

$$
\begin{array}{l}{\displaystyle\cos(x)=\sum_{k=0}^{\infty}(-1)^{k}\frac{1}{(2k)!}x^{2k}\,,}\\ {\displaystyle\sin(x)=\sum_{k=0}^{\infty}(-1)^{k}\frac{1}{(2k+1)!}x^{2k+1}\,.}\end{array}
$$
图5.4显示了相应的第一个泰勒多项式$T_{n}$，其中$n=0,1,5,10$。

注释。泰勒级数是幂级数的一种特殊情况

$$
f(x)=\sum_{k=0}^{\infty}a_{k}(x-c)^{k}
$$
其中$a_{k}$是系数，$c$是常数，具有定义5.4中的特殊形式。$\diamondsuit$


### 5.1.2 求导法则

在下面，我们简要陈述基本的求导法则，其中用 $f^{\prime}$ 表示函数 $f$ 的导数。

$$
\begin{align}
\text{乘积法则:} & \quad (f(x)g(x))' = f'(x)g(x) + f(x)g'(x) \\
\text{商法则:} & \quad \left(\frac{f(x)}{g(x)}\right)' = \frac{f'(x)g(x) - f(x)g'(x)}{(g(x))^2} \\
\text{和法则:} & \quad (f(x) + g(x))' = f'(x) + g'(x) \\
\text{复合函数法则:} & \quad (g(f(x)))' = (g \circ f)'(x) = g'(f(x))f'(x)
\end{align}
$$

这里，$g\circ f$ 表示函数的复合 $x\mapsto f(x)\mapsto g(f(x))$。

### 示例 5.5（链式法则）

让我们使用链式法则计算函数 $h(x)=(2x+1)^{4}$ 的导数。设

$$
\begin{array}{l}{{h(x)=(2x+1)^{4}=g(f(x))\,,}}\\ {{f(x)=2x+1\,,}}\\ {{g(f)=f^{4}\,,}}\end{array}
$$
我们得到 $f$ 和 $g$ 的导数为

$$
\begin{array}{l}{{f^{\prime}(x)=2\,,}}\\ {{g^{\prime}(f)=4f^{3}\,,}}\end{array}
$$
因此，$h$ 的导数为

$$
h^{\prime}(x)=g^{\prime}(f)f^{\prime}(x)=(4f^{3})\cdot2\stackrel{(5.34)}{=}4(2x+1)^{3}\cdot2=8(2x+1)^{3}\,,
$$
其中我们使用了链式法则（5.32）并在（5.34）中将 $f$ 的定义代入 $g^{\prime}(f)$ 中。


## 5.2 部分导数和梯度  

在第5.1节中讨论的导数适用于标量变量$x \in \mathbb{R}$的函数$f$。接下来，我们考虑函数$f$依赖于一个或多个变量$\pmb{x} \in \mathbb{R}^{n}$的情况，例如$f(\pmb{x}) = f(x_{1}, x_{2})$。将导数推广到多个变量的函数称为梯度。

我们通过一次改变一个变量并保持其他变量不变来找到函数$f$关于$\pmb{x}$的梯度。然后，梯度是这些部分导数的集合。

部分导数

定义5.2（部分导数）。对于一个函数$f\,:\,\mathbb{R}^{n}\,\rightarrow\,\mathbb{R}$，$\pmb{x} \mapsto f(\pmb{x})$，$\pmb{x} \in \mathbb{R}^{n}$，我们定义$n$个变量$x_{1}, \ldots, x_{n}$的**部分导数**为

$$
\begin{align}
\frac{\partial f}{\partial x_{1}} &= \lim_{h \to 0} \frac{f(x_{1} + h, x_{2}, \ldots, x_{n}) - f(x)}{h} \\
\vdots & \\
\frac{\partial f}{\partial x_{n}} &= \lim_{h \to 0} \frac{f(x_{1}, \ldots, x_{n-1}, x_{n} + h) - f(\mathbf{x})}{h}
\end{align}
$$



并将它们收集在行向量中

$$
\nabla_{\pmb{x}} f = \operatorname{grad} f = \frac{\mathrm{d}f}{\mathrm{d}\pmb{x}} = \left[\frac{\partial f(\pmb{x})}{\partial x_{1}} \quad \frac{\partial f(\pmb{x})}{\partial x_{2}} \quad \cdots \quad \frac{\partial f(\pmb{x})}{\partial x_{n}}\right] \in \mathbb{R}^{1 \times n}
$$

我们可以使用标量导数的结果：每个部分导数是对标量的导数。

其中$n$是变量的数量，$1$是函数$f$的图像/范围/陪域的维度。在这里，我们定义了列向量$\pmb{x} = [x_{1}, \ldots, x_{n}]^{\top} \in \mathbb{R}^{n}$。式(5.40)中的行向量称为$f$的**梯度**或**雅可比**，是第5.1节中导数的一般化。

注释。这种雅可比的定义是向量值函数的雅可比的一般定义的特殊情况，雅可比是部分导数的集合。我们将在第5.3节中再次讨论这一点。$\diamondsuit$
### 示例 5.6（使用链式法则求偏导数）

对于函数 $f(x,y)=(x+2y^{3})^{2}$，我们得到偏导数

$$
\begin{aligned}
\frac{\partial f(x,y)}{\partial x} &= 2(x + 2y^3) \frac{\partial}{\partial x} (x + 2y^3) = 2(x + 2y^3), \\
\frac{\partial f(x,y)}{\partial y} &= 2(x + 2y^3) \frac{\partial}{\partial y} (x + 2y^3) = 12(x + 2y^3) y^2.
\end{aligned}
$$

其中我们使用链式法则（5.32）来计算偏导数。

注释（作为行向量的梯度）：在文献中，通常将梯度向量定义为列向量，遵循向量通常为列向量的惯例。我们将梯度向量定义为行向量的原因有二：首先，我们可以一致地将梯度推广到向量值函数 $f\,:\,\mathbb{R}^{n}\,\rightarrow\,\mathbb{R}^{m}$（此时梯度变为矩阵）。其次，我们可以立即应用多元链式法则而无需关注梯度的维度。我们将在第5.3节中讨论这两个问题。$\diamondsuit$


### 示例 5.7 (梯度)

对于函数 $f(x_{1},x_{2})=x_{1}^{2}x_{2}+x_{1}x_{2}^{3}\in\mathbb{R}$，其偏导数（即，$f$ 关于 $x_{1}$ 和 $x_{2}$ 的导数）为

$$
\begin{array}{l r}{\displaystyle\frac{\partial f(x_{1},x_{2})}{\partial x_{1}}=2x_{1}x_{2}+x_{2}^{3}}\\ {\displaystyle\frac{\partial f(x_{1},x_{2})}{\partial x_{2}}=x_{1}^{2}+3x_{1}x_{2}^{2}}\end{array}
$$
梯度则为

$$
{\frac{\mathrm{d}f}{\mathrm{d}x}}=\left[{\frac{\partial f(x_{1},x_{2})}{\partial x_{1}}}\quad{\frac{\partial f(x_{1},x_{2})}{\partial x_{2}}}\right]=\left[2x_{1}x_{2}+x_{2}^{3}\quad x_{1}^{2}+3x_{1}x_{2}^{2}\right]\in\mathbb{R}^{1\times2}\,.
$$


### 5.2.1 偏导数的基本规则

在多元情况下，其中 $\pmb{x}\in\mathbb{R}^{n}$，我们从学校学到的基本求导规则（例如，和规则、积规则、链规则；参见第5.1.2节）仍然适用。然而，当我们计算关于向量 $\pmb{x}\in\mathbb{R}^{n}$ 的导数时，需要注意：我们现在涉及的是向量和矩阵，矩阵乘法不是交换的（参见第2.2.1节），即顺序很重要。

积规则：$(f g)^{\prime}=f^{\prime}g+f g^{\prime}$，和规则：$(f+g)^{\prime}=f^{\prime}+g^{\prime}$，链规则：$(g(f))^{\prime}=g^{\prime}(f)f^{\prime}$ 以下是积规则、和规则和链规则的一般形式：

$$
\begin{align}
\mathrm{积规则:~}  \frac{\partial}{\partial \pmb{x}} \left( f(\pmb{x}) g(\pmb{x}) \right) &= \frac{\partial f}{\partial \pmb{x}} g(\pmb{x}) + f(\pmb{x}) \frac{\partial g}{\partial \pmb{x}} \\
\mathrm{和规则:~}  \frac{\partial}{\partial \pmb{x}} \left( f(\pmb{x}) + g(\pmb{x}) \right) &= \frac{\partial f}{\partial \pmb{x}} + \frac{\partial g}{\partial \pmb{x}} \\
\mathrm{链规则:~}  \frac{\partial}{\partial \pmb{x}} \left( g \circ f \right) (\pmb{x}) &= \frac{\partial}{\partial \pmb{x}} \left( g(f(\pmb{x})) \right) = \frac{\partial g}{\partial f} \frac{\partial f}{\partial \pmb{x}}
\end{align}
$$


这只是直觉上的理解，但不是数学上正确的，因为偏导数不是一个分数。

让我们更仔细地看看链规则。链规则（5.48）在某种程度上类似于矩阵乘法的规则，其中我们说相邻维度必须匹配才能定义矩阵乘法；参见第2.2.1节。如果我们从左到右看，链规则表现出类似的性质：$\partial f$ 出现在第一个因子的“分母”中和第二个因子的“分子”中。如果我们把因子相乘，乘法是定义的，即 $\partial f$ 的维度匹配，“消去”，使得 $\partial g/\partial x$ 保留下来。

### 5.2.2 链式法则

考虑一个定义在两个变量 $x_{1},x_{2}$ 上的函数 $f\,:\,\mathbb{R}^{2}\,\rightarrow\,\mathbb{R}$。此外，$x_{1}(t)$ 和 $x_{2}(t)$ 本身是 $t$ 的函数。为了计算 $f$ 对 $t$ 的梯度，我们需要应用多元函数的链式法则（5.48）：

$$
\frac{\mathrm{d}f}{\mathrm{d}t} = \left[ \frac{\partial f}{\partial x_1} \quad \frac{\partial f}{\partial x_2} \right] \begin{bmatrix} \frac{\partial x_1(t)}{\partial t} \\ \frac{\partial x_2(t)}{\partial t} \end{bmatrix} = \frac{\partial f}{\partial x_1} \frac{\partial x_1(t)}{\partial t} + \frac{\partial f}{\partial x_2} \frac{\partial x_2(t)}{\partial t}
$$

其中 $\mathrm{d}$ 表示梯度，$\partial$ 表示偏导数。

### 示例 5.8

考虑函数 $f(x_{1},x_{2})=x_{1}^{2}+2x_{2}$，其中 $x_{1}=\sin t$ 和 $x_{2}=\cos t$，则

$$
\begin{align}
\frac{\mathrm{d}f}{\mathrm{d}t} &= \frac{\partial f}{\partial x_1} \frac{\partial x_1}{\partial t} + \frac{\partial f}{\partial x_2} \frac{\partial x_2}{\partial t} \\
&= 2 \sin t \frac{\partial \sin t}{\partial t} + 2 \frac{\partial \cos t}{\partial t} \\
&= 2 \sin t \cos t - 2 (-\sin t) \\
&= 2 \sin t (\cos t + \sin t)
\end{align}

$$
是 $f$ 对 $t$ 的导数。

如果 $f(x_{1},x_{2})$ 是 $x_{1}$ 和 $x_{2}$ 的函数，其中 $x_{1}(s,t)$ 和 $x_{2}(s,t)$ 本身是两个变量 $s$ 和 $t$ 的函数，链式法则给出偏导数

$$
\begin{align}
\frac{\partial f}{\partial s} &= \frac{\partial f}{\partial x_1} \frac{\partial x_1}{\partial s} + \frac{\partial f}{\partial x_2} \frac{\partial x_2}{\partial s}, \\
\frac{\partial f}{\partial t} &= \frac{\partial f}{\partial x_1} \frac{\partial x_1}{\partial t} + \frac{\partial f}{\partial x_2} \frac{\partial x_2}{\partial t}.
\end{align}
$$
梯度可以通过矩阵乘法得到

$$
\frac{\operatorname{d}f}{\operatorname{d}(s,t)} = \frac{\partial f}{\partial \mathbf{x}} \frac{\partial \mathbf{x}}{\partial (s,t)} = \underbrace{\left[\frac{\partial f}{\partial x_1} \quad \frac{\partial f}{\partial x_2}\right]}_{\text{$\cfrac{\partial f}{\partial \mathbf{x}}$}} \underbrace{\left[\begin{matrix}
\cfrac{\partial x_1}{\partial s} & \cfrac{\partial x_1}{\partial t} \\
\cfrac{\partial x_2}{\partial s} & \cfrac{\partial x_2}{\partial t}
\end{matrix}\right]}_{\text{$\cfrac{\partial \mathbf{x}}{\partial (s,t)}$}}
$$

这种将链式法则写成矩阵乘法的形式只有在梯度被定义为行向量时才有意义。否则，我们需要开始转置梯度以匹配矩阵维度。只要梯度是一个向量或矩阵，这可能仍然很简单；然而，当梯度成为一个张量（我们将在接下来讨论）时，转置就不再是一个简单的问题了。

注释（验证梯度实现的正确性）。偏导数的定义为相应差商的极限（见（5.39）），可以用于数值检查计算机程序中梯度的正确性：当我们计算梯度并实现它们时，可以使用有限差分来数值测试我们的计算和实现：我们选择一个很小的值 $h$（例如，$h=10^{-4}$），并比较（5.39）中的有限差分近似与我们对梯度的解析实现。如果误差很小，我们的梯度实现可能是正确的。“很小”可以意味着 $\begin{array}{r}{\sqrt{\frac{\sum_{i}(d h_{i}-d f_{i})^{2}}{\sum_{i}(d h_{i}+d f_{i})^{2}}}<10^{-6}}\end{array}$，其中 $d h_{i}$ 是有限差分近似，$d f_{i}$ 是 $f$ 对第 $i$ 个变量 $x_{i}$ 的解析梯度。链式法则可以写成矩阵乘法的形式。


## 5.3 向量值函数的梯度

到目前为止，我们讨论了映射到实数的函数 $f: \mathbb{R}^{n} \to \mathbb{R}$ 的偏导数和梯度。接下来，我们将梯度的概念推广到向量值函数（向量场）$\pmb{f}: \mathbb{R}^{n} \to \mathbb{R}^{m}$，其中 $n \geqslant 1$，$m > 1$。

对于函数 $\pmb{f}: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ 和向量 $\pmb{x} = [x_{1}, \dots, x_{n}]^{\top} \in \mathbb{R}^{n}$，对应的函数值向量表示为

$$
\pmb{f}(\pmb{x}) = \left[\begin{array}{c} f_{1}(\pmb{x}) \\ \vdots \\ f_{m}(\pmb{x}) \end{array}\right] \in \mathbb{R}^{m}\,.
$$
以这种方式书写向量值函数，我们可以将向量函数 $\pmb{f}: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ 视为函数向量 $[f_{1}, \dots, f_{m}]^{\top}$，其中每个 $f_{i}: \mathbb{R}^{n} \to \mathbb{R}$ 映射到实数。每个 $f_{i}$ 的微分规则与我们在第 5.2 节中讨论的一样。

因此，向量值函数 $\pmb{f}: \mathbb{R}^{n} \to \mathbb{R}^{m}$ 关于 $x_{i} \in \mathbb{R}$，$i = 1, \dots, n$ 的偏导数表示为向量

$$
\frac{\partial \pmb{f}}{\partial x_{i}} = \left[\begin{array}{c} \frac{\partial f_{1}}{\partial x_{i}} \\ \vdots \\ \frac{\partial f_{m}}{\partial x_{i}} \end{array}\right] = \left[\begin{array}{c} \operatorname*{lim}_{h \rightarrow 0} \frac{f_{1}(x_{1}, \dots, x_{i-1}, x_{i} + h, x_{i+1}, \dots, x_{n}) - f_{1}(\pmb{x})}{h} \\ \vdots \\ \operatorname*{lim}_{h \rightarrow 0} \frac{f_{m}(x_{1}, \dots, x_{i-1}, x_{i} + h, x_{i+1}, \dots, x_{n}) - f_{m}(\pmb{x})}{h} \end{array}\right] \in \mathbb{R}^{m}\,.
$$
从 (5.40) 可知，$\pmb{f}$ 关于向量的梯度是偏导数的行向量。在 (5.55) 中，每个偏导数 $\partial \pmb{f} / \partial x_{i}$ 本身是一个列向量。因此，我们通过收集这些偏导数来得到 $\pmb{f}$ 关于 $\pmb{x} \in \mathbb{R}^{n}$ 的梯度：

$$
\frac{\mathrm{d}\pmb{f}(\pmb{x})}{\mathrm{d}\pmb{x}} = \left[\begin{array}{c} \frac{\partial f_{1}(\pmb{x})}{\partial x_{1}} & \cdots & \frac{\partial f_{1}(\pmb{x})}{\partial x_{n}} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_{m}(\pmb{x})}{\partial x_{1}} & \cdots & \frac{\partial f_{m}(\pmb{x})}{\partial x_{n}} \end{array}\right] \in \mathbb{R}^{m \times n}.
$$


雅可比矩阵是函数 $\pmb{f}:\mathbb{R}^{n}\rightarrow\mathbb{R}^{m}$ 的梯度，是一个大小为 $m\times n$ 的矩阵。

### 定义 5.6（雅可比矩阵）。

向量函数 $\pmb{f}:\mathbb{R}^{n}\rightarrow\mathbb{R}^{m}$ 的所有一阶偏导数的集合称为雅可比矩阵。雅可比矩阵 $J$ 是一个 $m\times n$ 的矩阵，定义和排列如下：

$$
\begin{align*}
J &= \nabla_{x} f = \frac{\mathrm{d}f(x)}{\mathrm{d}x} = \left[\frac{\partial f(x)}{\partial x_{1}} \quad \cdots \quad \frac{\partial f(x)}{\partial x_{n}}\right] \\
&= \left[\frac{\partial f_{1}(x)}{\partial x_{1}} \quad \cdots \quad \frac{\partial f_{1}(x)}{\partial x_{n}}\right] \\
&\vdots \\
&= \left[\frac{\partial f_{m}(x)}{\partial x_{1}} \quad \cdots \quad \frac{\partial f_{m}(x)}{\partial x_{n}}\right] \\
\mathbf{x} &= \left[\begin{matrix} x_{1} \\ \vdots \\ x_{n} \end{matrix}\right] \,, \quad J(i,j) = \frac{\partial f_{i}}{\partial x_{j}} \,.
\end{align*}
$$

作为 (5.58) 的一个特例，函数 $f\,:\,\mathbb{R}^{n}\,\rightarrow\,\mathbb{R}^{1}$，将向量 $\pmb{x}\in\mathbb{R}^{n}$ 映射到一个标量（例如，$f(\pmb{x})=\sum_{i=1}^{n}x_{i}$），其雅可比矩阵是一个行向量（维度为 $1\times n$ 的矩阵）；参见 (5.40)。

分子布局 注意。在本书中，我们使用分子布局的导数，即，$\pmb{f}\,\in\,\mathbb{R}^{m}$ 对 $\pmb{x}\in{\mathbb{R}}^{n}$ 的导数 ${\mathrm{d}}{\pmb f}/{\mathrm{d}}{\pmb x}$ 是一个 $m\times n$ 的矩阵，其中 $\pmb{f}$ 的元素定义了行，$\pmb{x}$ 的元素定义了列；参见 (5.58)。还存在分母布局，它是分子布局的转置。在本书中，我们将使用分子布局。$\diamondsuit$

我们将在第 6.7 节中看到雅可比矩阵在概率分布变量变换方法中的应用。变换变量引起的缩放量由行列式给出。

在第 4.1 节中，我们看到行列式可以用来计算平行四边形的面积。如果我们给出两个向量 $\pmb{b}_{1}~=~[1,0]^{\top}$，$\pmb{b}_{2}=[0,1]^{\top}$ 作为单位正方形（蓝色；参见图 5.5）的边，该正方形的面积为

$$
\left|\operatorname{det}\left(\begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}\right)\right| = 1\,.
$$
如果我们取一个以 $\pmb{c}_{1}~=~[-2,1]^{\top}$，$\pmb{c}_{2}~=~[1,1]^{\top}$ 为边的平行四边形（图 5.5 中的橙色），其面积为行列式的绝对值（参见第 4.1 节）

$$
\left|\operatorname{det}\left(\begin{bmatrix}-2 & 1 \\ 1 & 1\end{bmatrix}\right)\right| = |-3| = 3\,.
$$

即该区域的面积正好是单位正方形面积的三倍。我们可以通过找到将单位正方形映射到另一个正方形的映射来找到这个缩放因子。在线性代数术语中，我们实际上执行了从 $(\pmb{b}_{1},\pmb{b}_{2})$ 到 $(\pmb{c}_{\mathrm{1}},\pmb{c}_{\mathrm{2}})$ 的变量变换。在这种情况下，映射是线性的，该映射的行列式的绝对值给出了我们所需的缩放因子。

我们将描述两种方法来确定此映射。首先，我们利用映射是线性的，因此可以使用第2章中的工具来识别此映射。其次，我们将使用本章中讨论的部分导数来找到映射。

方法1：为了开始线性代数方法，我们将 $\{\pmb{b}_{1},\pmb{b}_{2}\}$ 和 $\{\pmb{c}_{1},\pmb{c}_{2}\}$ 都视为 $\mathbb{R}^{2}$ 的基（参见第2.6.1节以回顾）。我们实际上执行的是从 $(\pmb{b}_{1},\pmb{b}_{2})$ 到 $(c_{\mathrm{1}},c_{\mathrm{2}})$ 的基变换，并且我们正在寻找实现基变换的变换矩阵。使用第2.7.2节的结果，我们确定所需的基变换矩阵为

$$
J = \begin{bmatrix}-2 & 1 \\ 1 & 1\end{bmatrix}\,,
$$
使得 ${\cal J}b_{1}=c_{1}$ 和 ${\cal J}b_{\mathrm{2}}=c_{\mathrm{2}}$。矩阵 $J$ 的行列式的绝对值，即我们所寻找的缩放因子，给出为 $|\mathrm{det}(\boldsymbol{J})|=3$，即由 $(c_{\mathrm{1}},c_{\mathrm{2}})$ 所张成的正方形的面积是 $(\pmb{b}_{1},\pmb{b}_{2})$ 所张成的正方形面积的三倍。

方法2：线性代数方法适用于线性变换；对于非线性变换（在第6.7节中变得相关），我们采用更一般的方法，使用部分导数。

对于这种方法，我们考虑一个函数 $\pmb{f}:\mathbb{R}^{2}\rightarrow\mathbb{R}^{2}$，它执行变量变换。在我们的例子中，$\pmb{f}$ 将任何 $\pmb{x}\in\mathbb{R}^{2}$ 的坐标表示从 $(\pmb{b}_{1},\pmb{b}_{2})$ 映射到 $\pmb{y}\in\mathbb{R}^{2}$ 的坐标表示 $(c_{\mathrm{1}},c_{\mathrm{2}})$。我们希望确定映射，以便计算当它被 $\pmb{f}$ 变换时面积（或体积）如何变化。为此，我们需要找出当稍微改变 $\pmb{x}$ 时 $\pmb{f}(\pmb{x})$ 如何变化。这个问题正好由雅可比矩阵 $\textstyle{\frac{\mathrm{d}\pmb{f}}{\mathrm{d}\pmb{x}}}\in\mathbb{R}^{2\times2}$ 给出。由于我们可以写出

$$
\begin{array}{l}{y_{1}=-2x_{1}+x_{2}}\\ {y_{2}=x_{1}+x_{2}}\end{array}
$$
我们得到了 $\pmb{x}$ 和 $\pmb{y}$ 之间的函数关系，这使我们能够得到部分导数

$$
{\frac{\partial y_{1}}{\partial x_{1}}}=-2\,,\quad{\frac{\partial y_{1}}{\partial x_{2}}}=1\,,\quad{\frac{\partial y_{2}}{\partial x_{1}}}=1\,,\quad{\frac{\partial y_{2}}{\partial x_{2}}}=1
$$
并组成雅可比矩阵

$$
J=\left[\begin{array}{l l}{\displaystyle\frac{\partial y_{1}}{\partial x_{1}}}&{\displaystyle\frac{\partial y_{1}}{\partial x_{2}}}\\ {\displaystyle\frac{\partial y_{2}}{\partial x_{1}}}&{\displaystyle\frac{\partial y_{2}}{\partial x_{2}}}\end{array}\right]=\left[\begin{array}{l l}{-2}&{1}\\ {1}&{1}\end{array}\right]\,.
$$
几何上，雅可比行列式给出了当我们变换一个面积或体积时的放大/缩放因子。雅可比行列式

图5.6 (部分)导数的维度。

雅可比矩阵代表我们正在寻找的坐标变换。如果坐标变换是线性的（如我们的例子），它是精确的，并且（5.66）恢复了（5.62）中的基变换矩阵。如果坐标变换是非线性的，雅可比矩阵用线性变换局部近似这个非线性变换。雅可比行列式的绝对值 $|\mathrm{det}(J)|$ 是坐标变换时面积的缩放因子。我们的例子给出 $|\mathrm{det}(J)|=3$。

雅可比行列式和变量变换将在第6.7节中变得相关，当我们变换随机变量和概率分布时。这些变换在使用重参数化技巧训练深度神经网络时非常重要，也称为无限扰动分析。

在本章中，我们遇到了函数的导数。图5.6总结了这些导数的维度。对于 $f:\mathbb{R}\to\mathbb{R}$，梯度只是一个标量（左上角）。对于 $f:\mathbb{R}^{D}\rightarrow\mathbb{R}$，梯度是一个 $1\times D$ 的行向量（右上角）。对于 $\pmb{f}:\mathbb{R}\rightarrow\mathbb{R}^{E}$，梯度是一个 $E\times1$ 的列向量，而对于 $\pmb{f}:\mathbb{R}^{D}\rightarrow\mathbb{R}^{E}$，梯度是一个 $E\times D$ 的矩阵。


### 示例 5.9 (向量值函数的梯度) 我们给定

$$
\pmb{f}(\pmb{x})=\pmb{A}\pmb{x}\,,\qquad\pmb{f}(\pmb{x})\in\mathbb{R}^{M},\quad\pmb{A}\in\mathbb{R}^{M\times N},\quad\pmb{x}\in\mathbb{R}^{N}\,.
$$
为了计算梯度 $\mathrm{d}\pmb{f}/\mathrm{d}\pmb{x}$，我们首先确定 $\mathrm{d}\pmb{f}/\mathrm{d}\pmb{x}$ 的维度：由于 $\pmb{f}:\mathbb{R}^{N}\rightarrow\mathbb{R}^{M}$，可以得出 $\mathrm{d}\pmb{f}/\mathrm{d}\pmb{x}\in\mathbb{R}^{M\times N}$。其次，为了计算梯度，我们确定 $f$ 对每个 $x_{j}$ 的偏导数：

$$
f_{i}({\pmb x})=\sum_{j=1}^{N}A_{i j}x_{j}\implies\frac{\partial f_{i}}{\partial x_{j}}=A_{i j}
$$
我们将偏导数收集在雅可比矩阵中，得到梯度

$$
\frac{\mathrm{d}\pmb{f}}{\mathrm{d}\pmb{x}}=\left[\begin{array}{c c c}{\frac{\partial f_{1}}{\partial x_{1}}}&{\cdot\cdot\cdot}&{\frac{\partial f_{1}}{\partial x_{N}}}\\ {\vdots}&{}&{\vdots}\\ {\frac{\partial f_{M}}{\partial x_{1}}}&{\cdot\cdot\cdot}&{\frac{\partial f_{M}}{\partial x_{N}}}\end{array}\right]=\left[\begin{array}{c c c}{A_{11}}&{\cdot\cdot\cdot}&{A_{1N}}\\ {\vdots}&{}&{\vdots}\\ {A_{M1}}&{\cdot\cdot\cdot}&{A_{M N}}\end{array}\right]=\pmb{A}\in\mathbb{R}^{M\times N}\,.
$$


### 示例 5.10（链式法则）

考虑函数 $h:\mathbb{R}\to\mathbb{R},\,h(t)=(f\circ g)(t)$，其中

$$
\begin{array}{r l}
&{f:\mathbb{R}^{2}\rightarrow\mathbb{R}}\\ 
&{g:\mathbb{R}\rightarrow\mathbb{R}^{2}}\\ 
&{f(\pmb{x})=\exp(x_{1}x_{2}^{2})\,,}\\ 
&{\qquad\pmb{x}=\bigg[\mathscr{x}_{2}\bigg]=g(t)=\bigg[t\cos t\bigg]}\\ 
&{t\sin t}
\end{array}
$$

求 $h$ 关于 $t$ 的梯度。由于 $f:\mathbb{R}^{2}\to\mathbb{R}$ 和 $g:\mathbb{R}\to\mathbb{R}^{2}$，我们注意到

$$
\frac{\partial f}{\partial\pmb{x}}\in\mathbb{R}^{1\times2}\,,\quad\frac{\partial g}{\partial t}\in\mathbb{R}^{2\times1}\,.
$$
所需的梯度通过应用链式法则计算如下：

$$
{\begin{array}{r l}
{{\frac{\mathrm{d}h}{\mathrm{d}t}}={\frac{\partial f}{\partial\mathbf{x}}}{\frac{\partial x}{\partial t}}=\left[{\frac{\partial f}{\partial x_{1}}}\ \ {\frac{\partial f}{\partial x_{2}}}\right]\left[{\frac{\partial x_{1}}{\partial t}}\right]}&{}\\ 
{=\left[\exp(x_{1}x_{2}^{2})x_{2}^{2}\ \ 2\exp(x_{1}x_{2}^{2})x_{1}x_{2}\right]\left[{\frac{\cos t-t\sin t}{\sin t+t\cos t}}\right]}&{}\\ 
{=\exp(x_{1}x_{2}^{2})\left(x_{2}^{2}(\cos t-t\sin t)+2x_{1}x_{2}(\sin t+t\cos t)\right),}
\end{array}}
$$

其中 $x_{1}=t\cos t$ 和 $x_{2}=t\sin t$；参见（5.72）。

我们将在第9章中更详细地讨论这个模型，特别是在线性回归的背景下，我们需要关于参数 $\pmb{\theta}$ 的最小二乘损失 $L$ 的导数。


### 示例5.11（线性模型中的最小二乘损失的梯度）

考虑线性模型

$$
\pmb{y}=\Phi\pmb{\theta}\,,
$$
其中$\pmb{\theta}\in\mathbb{R}^{D}$ 是参数向量，$\Phi\,\in\,\mathbb{R}^{N\times D}$ 是输入特征，$\pmb{y}\in\mathbb{R}^{N}$ 是对应的观测值。我们定义函数

$$
\begin{array}{l}{{L(e):=\|e\|^{2}\,,}}\\ {{e(\pmb\theta):=\pmb y-\Phi\pmb\theta\,.}}\end{array}
$$
我们寻求$\frac{\partial L}{\partial\pmb\theta}$，并将使用链式法则来计算。$L$称为最小二乘损失函数。

在开始计算之前，我们确定梯度的维度为

$$
\frac{\partial L}{\partial\pmb\theta}\in\mathbb{R}^{1\times D}\,.
$$
链式法则允许我们计算梯度为

$$
\frac{\partial L}{\partial\pmb\theta}=\frac{\partial L}{\partial\pmb e}\frac{\partial e}{\partial\theta}\,,
$$
dLdtheta $=$ np.einsum( ${\bf\delta}_{\textrm{n}}$ ,nd’, dLde,dedtheta)

其中第$d$个元素由下式给出

$$
{\frac{\partial L}{\partial\pmb\theta}}[1,d]=\sum_{n=1}^{N}{\frac{\partial L}{\partial e}}[n]{\frac{\partial e}{\partial\pmb\theta}}[n,d]\,.
$$
我们知道$\|e\|^{2}=e^{\top}e$（参见第3.2节），并确定

$$
\frac{\partial\cal L}{\partial e}=2e^{\top}\in\mathbb{R}^{1\times N}\,.
$$
此外，我们得到

$$
\frac{\partial e}{\partial\theta}=-\Phi\in\mathbb{R}^{N\times D}\,,
$$
因此我们所需的导数为

$$
\frac{\partial L}{\partial\pmb\theta}=-2e^{\top}\Phi\stackrel{(5.77)}{=}-\underbrace{2(\pmb y^{\top}-\pmb\theta^{\top}\pmb\Phi^{\top})}_{1\times N}\underbrace{\Phi}_{N\times D}\in\mathbb{R}^{1\times D}\,.
$$
注释。我们可以通过直接考虑函数

$$
L_{2}(\pmb\theta):=\|\pmb y-\Phi\pmb\theta\|^{2}=(\pmb y-\Phi\pmb\theta)^{\top}(\pmb y-\Phi\pmb\theta)\,.
$$
而不使用链式法则，同样得到相同的结果。这种方法对于简单的函数如$L_{2}$仍然实用，但对于深度函数组合则变得不切实际。$\diamondsuit$

![](images/7fbc834d7bacd81e43d6df20b243e32d7d1f236212249523211b0a52c101c9b6.jpg)

我们通过重新调整此梯度来获得梯度张量$\tilde{\pmb{A}}\in\mathbb{R}^{8}$。然后，我们计算梯度$\begin{array}{r}{\frac{\mathrm{d}\tilde{A}}{\mathrm{d}x}\,\in\,\mathbb{R}^{8\times3}}\end{array}$，如上图所示。

图5.7 矩阵相对于向量的梯度计算可视化。我们感兴趣的是计算$\pmb{A}\in\mathbb{R}^{4\times2}$相对于向量$\pmb{x}\in\mathbb{R}^{3}$的梯度。我们知道梯度$\textstyle{\frac{\mathrm{d}A}{\mathrm{d}x}}\in\mathbb{R}^{4\times2\times3}$。我们通过两种等效的方法到达那里：(a) 将偏导数整理成雅可比张量；(b) 将矩阵展平为向量，计算雅可比矩阵，重新调整为雅可比张量。

## 5.4 矩阵的梯度  

我们可以将张量视为多维数组。  

在某些情况下，我们需要对矩阵相对于向量（或其他矩阵）求梯度，这将导致一个多维张量。我们可以将这个张量视为一个收集偏导数的多维数组。例如，如果我们计算一个 $m \times n$ 矩阵 $\pmb{A}$ 相对于一个 $p \times q$ 矩阵 $\pmb{B}$ 的梯度，结果的雅可比矩阵将是 $(m \times n) \times (p \times q)$，即一个四维张量 $J$，其元素由 $J_{ijkl} = \partial A_{ij} / \partial B_{kl}$ 给出。  

矩阵可以通过堆叠矩阵的列（“展平”）转换为向量。  

由于矩阵表示线性映射，我们可以利用 $\mathbb{R}^{m \times n}$ 空间中的 $m \times n$ 矩阵与 $\mathbb{R}^{mn}$ 空间中的 $mn$ 维向量之间的向量空间同构（线性可逆映射）的事实。因此，我们可以将矩阵重塑为长度分别为 $mn$ 和 $pq$ 的向量。使用这些 $mn$ 维向量计算的梯度将得到一个大小为 $mn \times pq$ 的雅可比矩阵。图 5.7 展示了这两种方法。在实际应用中，通常希望将矩阵重塑为向量并继续使用这个雅可比矩阵：链式法则（5.48）简化为简单的矩阵乘法，而在雅可比张量的情况下，我们需要更加注意需要求和的维度。
### 示例 5.12（向量关于矩阵的梯度）

让我们考虑以下例子，其中

$$
\pmb{f}=\pmb{A}\pmb{x}\,,\quad\pmb{f}\in\mathbb{R}^{M},\quad\pmb{A}\in\mathbb{R}^{M\times N},\quad\pmb{x}\in\mathbb{R}^{N}
$$
我们寻求梯度 $\mathrm{d}\pmb{f}/\mathrm{d}\pmb{A}$。我们首先确定梯度的维度为

$$
\frac{\mathrm{d}\pmb{f}}{\mathrm{d}\pmb{A}}\in\mathbb{R}^{M\times(M\times N)}\,.
$$
根据定义，梯度是偏导数的集合：

$$
\frac{\mathrm{d}\pmb{f}}{\mathrm{d}\pmb{A}}=\left[\begin{array}{c}{\frac{\partial f_{1}}{\partial\pmb{A}}}\\ {\vdots}\\ {\frac{\partial f_{M}}{\partial\pmb{A}}}\end{array}\right]\,,\quad\frac{\partial f_{i}}{\partial\pmb{A}}\in\mathbb{R}^{1\times(M\times N)}\,.
$$
为了计算偏导数，我们显式写出矩阵向量乘法：

$$
f_{i}=\sum_{j=1}^{N}A_{i j}x_{j},\quad i=1,\dots,M\,,
$$
偏导数则为

$$
\frac{\partial f_{i}}{\partial A_{i q}}=x_{q}\,.
$$
这使我们能够计算 $f_{i}$ 关于 $\pmb{A}$ 的一行的偏导数，其形式为

$$
\frac{\partial f_{i}}{\partial A_{i,:}}=\pmb{x}^{\top}\in\mathbb{R}^{1\times1\times N}\,,
$$
$$
\frac{\partial f_{i}}{\partial A_{k\neq i,:}}=\mathbf{0}^{\top}\in\mathbb{R}^{1\times1\times N}
$$
我们需要注意正确的维度。由于 $f_{i}$ 映射到 $\mathbb{R}$，并且 $\pmb{A}$ 的每一行是 $1\times N$，我们得到一个 $1\times N$ 的张量作为 $f_{i}$ 关于 $\pmb{A}$ 的一行的偏导数。

我们堆叠偏导数（5.91），并通过

$$
\frac{\partial f_{i}}{\partial A}=\left[\begin{array}{c}{\mathbf{0}^{\top}}\\ {\vdots}\\ {\mathbf{0}^{\top}}\\ {\mathbf{x}^{\top}}\\ {\mathbf{0}^{\top}}\\ {\vdots}\\ {\mathbf{0}^{\top}}\end{array}\right]\in\mathbb{R}^{1\times(M\times N)}\,.
$$
得到所需的梯度（5.87）。

### 示例 5.13（矩阵关于矩阵的梯度）

考虑矩阵 $R\in\mathbb{R}^{M\times N}$ 和 $\pmb{f}:\mathbb{R}^{M\times N}\rightarrow\mathbb{R}^{N\times N}$，其中

$$
\pmb{f}(\pmb{R})=\pmb{R}^{\top}\pmb{R}=:\pmb{K}\in\mathbb{R}^{N\times N}\,,
$$
我们寻求梯度 $\mathrm{d}K/\mathrm{d}R$。

为了解决这个问题，我们先写下我们已知的内容：梯度的维度为

$$
\frac{\mathrm{d}K}{\mathrm{d}R}\in\mathbb{R}^{(N\times N)\times(M\times N)}\,,
$$
这是一个张量。此外，

$$
\frac{\mathrm{d}K_{p q}}{\mathrm{d}R}\in\mathbb{R}^{1\times M\times N}
$$
对于 $p,q\,=\,1,\cdot\cdot\cdot,N$，其中 $K_{p q}$ 是 $\pmb{K}=\pmb{f}(\pmb{R})$ 的 $(p,q)$ 项。记 $R$ 的第 $i$ 列为 $\mathbf{r}_{i}$，则 $\kappa$ 的每一项由 $R$ 的两列的点积给出，即

$$
K_{p q}=\pmb{r}_{p}^{\top}\pmb{r}_{q}=\sum_{m=1}^{M}R_{m p}R_{m q}\,.
$$
当我们现在计算偏导数 $\frac{\partial K_{p q}}{\partial R_{i j}}$ 时，我们得到

$$
\frac{\partial K_{p q}}{\partial R_{i j}}=\sum_{m=1}^{M}\frac{\partial}{\partial R_{i j}}R_{m p}R_{m q}=\partial_{p q i j}\,,
$$
$$
\partial_{p q i j}=\left\{\begin{array}{l l}{R_{i q}}&{\mathrm{if~}j=p,\ p\neq q}\\ {R_{i p}}&{\mathrm{if~}j=q,\ p\neq q}\\ {2R_{i q}}&{\mathrm{if~}j=p,\ p=q}\\ {0}&{\mathrm{otherwise}}\end{array}\right..
$$
从（5.94）我们知道，所需的梯度的维度为 $(N\times N)\times(M\times N)$，并且这个张量的每个元素由（5.98）中的 $\partial_{p q i j}$ 给出，其中 $p,q,j=1,\ldots,N$ 和 $i=1,\dots,M$。


## 5.5 有用的梯度计算恒等式

在以下内容中，我们列出了在机器学习环境中经常需要的一些有用的梯度（Petersen 和 Pedersen, 2012）。这里我们使用 $\operatorname{tr}(\cdot)$ 表示迹（参见定义 4.4），$\operatorname*{det}(\cdot)$ 表示行列式（参见第 4.1 节），$\pmb{f}(\pmb{X})^{-1}$ 表示 $\pmb{f}(\pmb{X})$ 的逆，假设它存在。

$$
\begin{align}
    & \frac{\partial}{\partial X} f(X)^{*} = -f\left(\frac{\partial f(X)}{\partial X}\right)^{*} \\
    & \frac{\partial}{\partial X} \eta f(X) = -\eta \left(\frac{\partial f(X)}{\partial X}\right) \\
    & \frac{\partial}{\partial X} \eta f(X) = -\eta f\left(X\right) \eta \left(f(X) \cdot \frac{\partial f(X)}{\partial X}\right) \\
    & \frac{\partial}{\partial X} \eta f(X) = -\partial f(X) \cdot \frac{\partial f(X)}{\partial X} \eta f(X)^{*} \\
    & \frac{\partial f(X)^{*}}{\partial X} = -f(X)^{*} \cdot \frac{\partial f(X)}{\partial X} \eta f(X)^{*} \\
    & \frac{\partial \alpha^{*} \cdot X^{*}}{\partial X} \eta^{*} = -(X^{*})^{*} \cdot \partial \left(X^{*} - 1\right)^{*} \\
    & \frac{\partial \alpha^{*}}{\partial X} \eta^{*} = -\alpha^{*} \\
    & \frac{\partial \alpha^{*}}{\partial X} \eta^{*} = \alpha^{*} \\
    & \frac{\partial \alpha^{*}}{\partial X} \eta f(X) = \alpha^{*} \\
    & \frac{\partial \alpha^{*}}{\partial X} \eta^{*} = -\alpha^{*} \eta^{*} \\
    & \frac{\partial \alpha^{*} \eta f(X)}{\partial X} = \alpha^{*} \eta \left(B + B^{*}\right) \\
    & \frac{\partial}{\partial X} \eta^{*} = -\alpha^{*} \eta f(X) = -2 (x - A u)^{*} W A.
\end{align}
$$

注释。在这本书中，我们仅涵盖矩阵的迹和转置。然而，我们已经看到导数可以是高维张量，在这种情况下，通常的迹和转置是未定义的。在这种情况下，$D\times D\times E\times F$ 张量的迹将是一个 $E\times F$ 维矩阵。这是张量收缩的一个特例。同样，当我们“转置”一个张量时，我们指的是交换前两个维度。具体来说，在 (5.99) 到 (5.102) 中，当我们处理多元函数 $\pmb{f}(\cdot)$ 并计算矩阵的导数时（并且我们选择不将其向量化，如第 5.4 节所述），我们需要进行张量相关的计算。

## 5.6 反向传播与自动微分

在许多机器学习应用中，我们通过执行梯度下降（参见第7.1节）来找到好的模型参数，这依赖于我们可以计算学习目标相对于模型参数的梯度。对于给定的目标函数，我们可以使用微积分和链式法则来获得相对于模型参数的梯度；参见第5.2.2节。在第5.3节中，我们已经尝到了计算线性回归模型参数的平方损失梯度的滋味。

Tim Vieira 的博客中有一篇关于反向传播和链式法则的良好讨论，网址为 https://tinyurl.com/ycfm2yrw 。

考虑函数

$$
f(x)=\sqrt{x^{2}+\exp(x^{2})}+\cos\left(x^{2}+\exp(x^{2})\right).
$$
通过应用链式法则，并注意到微分是线性的，我们计算梯度

$$
{\begin{array}{l}{{\displaystyle{\frac{\mathrm{d}f}{\mathrm{d}x}}={\frac{2x+2x\exp(x^{2})}{2{\sqrt{x^{2}+\exp(x^{2})}}}}-\sin\left(x^{2}+\exp(x^{2})\right)\left(2x+2x\exp(x^{2})\right)}\ ~}\\ {{\displaystyle\quad=2x\left({\frac{1}{2{\sqrt{x^{2}+\exp(x^{2})}}}}-\sin\left(x^{2}+\exp(x^{2})\right)\right)\left(1+\exp(x^{2})\right)\,.}}\end{array}}
$$
以这种方式显式地写出梯度通常不切实际，因为它通常会导致非常冗长的导数表达式。在实践中，这意味着如果我们不小心，梯度的实现可能会比计算函数本身昂贵得多，这会带来不必要的开销。对于训练深度神经网络模型，反向传播算法（Kelley, 1960; Bryson, 1961; Dreyfus, 1962; Rumelhart et al., 1986）是一种高效地计算误差函数相对于模型参数的梯度的方法。

### 5.6.1 深度网络中的梯度

链式法则在深度学习中被广泛使用的一个领域是，函数值 $\pmb{y}$ 作为多层函数组合计算

$$
\pmb{y}=(f_{K}\circ f_{K-1}\circ\cdot\cdot\cdot\circ f_{1})(\pmb{x})=f_{K}(f_{K-1}(\cdot\cdot\cdot(f_{1}(\pmb{x}))\cdot\cdot\cdot))\,,
$$
其中 $\pmb{x}$ 是输入（例如，图像），$\pmb{y}$ 是观察值（例如，类别标签），每个函数 $f_{i}$ 都有自己的参数。

![](images/4a23c5f076b09c5faf3dbefded21dd4777f2148b9109dbb3a5e1003a8862768e.jpg)
图 5.8 多层神经网络中的前向传播，以计算损失 $L$ 作为输入 $\mathbf{\nabla}_{\mathbf{x}}$ 和参数 $\pmb{A}_{i},\ \pmb{b}_{i}$ 的函数。

我们讨论了各层激活函数相同的情况，以简化符号。

在具有多层的神经网络中，我们在第 $i$ 层有函数 $f_{i}(\pmb{x}_{i-1})=\sigma(A_{i-1}\mathbf{x}_{i-1}+\mathbf{b}_{i-1})$。这里 $\pmb{x}_{i-1}$ 是第 $i-1$ 层的输出，$\sigma$ 是激活函数，例如逻辑sigmoid函数 $\textstyle{\frac{1}{1+e^{-x}}}$、tanh 或修正线性单元（ReLU）。为了训练这些模型，我们需要计算损失函数 $L$ 相对于所有模型参数 $\mathbf{\nabla}A_{j},\mathbf{\nabla}b_{j}$ 的梯度，其中 $j=1,\dots,K$。这也要求我们计算损失函数 $L$ 相对于每一层输入的梯度。例如，如果我们有输入 $_{_{\pmb{x}}}$ 和观察值 $\pmb{y}$，并且网络结构由

$$
\begin{array}{r l}&{\pmb{f}_{0}:=\pmb{x}}\\ &{\pmb{f}_{i}:=\sigma_{i}\big(\pmb{A}_{i-1}\pmb{f}_{i-1}+\pmb{b}_{i-1}\big)\,,\quad i=1,\dots,K\,,}\end{array}
$$
参见图 5.8 的可视化，我们可能对找到 $A_{j},b_{j}$，其中 $j=0,\dots,K-1$，感兴趣，使得平方损失

$$
L(\pmb\theta)=||\pmb y-\pmb f_{\scriptscriptstyle K}(\pmb\theta,\pmb x)||^{2}
$$
最小化，其中 $\pmb\theta=\{A_{0},\pmb b_{0},\dots,A_{K-1},\pmb b_{K-1}\}$。

关于神经网络梯度的更深入讨论可以在 Justin Domke 的讲义中找到，网址为 https://tinyurl. com/yalcxgtv 。

为了获得相对于参数集 $\pmb{\theta}$ 的梯度，我们需要计算相对于每个层参数 $\pmb{\theta}_{j}=\{A_{j},b_{j}\}$ 的偏导数，其中 $j=0,\dots,K-1$。链式法则允许我们确定偏导数为

$$
\begin{array}{r l}&{\frac{\partial L}{\partial\theta_{K-1}}=\frac{\partial L}{\partial f_{K}}\frac{\partial f_{K}}{\partial\theta_{K-1}}}\\ &{\frac{\partial L}{\partial\theta_{K-2}}=\frac{\partial L}{\partial f_{K}}\boxed{\frac{\partial f_{K}}{\partial f_{K-1}}\frac{\partial f_{K-1}}{\partial\theta_{K-2}}}}\\ &{\frac{\partial L}{\partial\theta_{K-3}}=\frac{\partial L}{\partial f_{K}}\frac{\partial f_{K}}{\partial f_{K-1}}\boxed{\frac{\partial f_{K-1}}{\partial f_{K-2}}\frac{\partial f_{K-2}}{\partial\theta_{K-3}}}}\\ &{\qquad\frac{\partial L}{\partial\theta_{i}}=\frac{\partial L}{\partial f_{K}}\frac{\partial f_{K}}{\partial f_{K-1}}\cdot\boxed{\frac{\partial f_{i+2}\partial f_{i+1}}{\partial\theta_{i}}}}\end{array}
$$
橙色项是层输出相对于其输入的偏导数，而蓝色项是层输出相对于其参数的偏导数。假设我们已经计算了偏导数 $\partial L/\partial\pmb{\theta}_{i+1}$，那么大部分计算可以重用以计算 $\partial L/\partial\pmb{\theta}_{i}$。我们需要计算的额外项由方框表示。图 5.9 可视化了损失函数的梯度是如何通过网络反向传递的。

![](images/8203211ca6fd1738181f4ed9bf21b718ec0e2babe62934b60bc7e0dc4f18a2d0.jpg)
图 5.9 多层神经网络中的反向传播，以计算损失函数的梯度。


### 5.6.2 自动微分

结果表明，反向传播是数值分析中一种称为自动微分的一般技术的特殊情况。我们可以将自动微分视为一组技术，通过这些技术可以数值地（而不是符号地）计算函数的确切（精确到机器精度）梯度，通过中间变量并应用链式法则。自动微分应用一系列基本算术运算，例如加法和乘法，以及基本函数，例如  sin 、 cos 、 exp 、 log 。通过将链式法则应用于这些操作，可以自动计算相当复杂的函数的梯度。自动微分适用于一般的计算机程序，并具有前向和反向模式。Baydin等人（2018）对机器学习中的自动微分进行了很好的概述。

图5.10展示了一个简单的图，表示从输入 $x$ 到输出 $y$ 的数据流，通过一些中间变量 $a, b$。如果我们计算导数 $\mathrm{d}y/\mathrm{d}x$，我们将应用链式法则并得到

$$
{\frac{\mathrm{d}y}{\mathrm{d}x}}={\frac{\mathrm{d}y}{\mathrm{d}b}}{\frac{\mathrm{d}b}{\mathrm{d}a}}{\frac{\mathrm{d}a}{\mathrm{d}x}}\,.
$$
直观地，前向和反向模式在乘法顺序上有所不同。由于矩阵乘法的结合性，我们可以选择

$$
{\begin{array}{r}{{\frac{\mathrm{d}y}{\mathrm{d}x}}=\left({\frac{\mathrm{d}y}{\mathrm{d}b}}{\frac{\mathrm{d}b}{\mathrm{d}a}}\right){\frac{\mathrm{d}a}{\mathrm{d}x}}\,,}\\ {{\frac{\mathrm{d}y}{\mathrm{d}x}}={\frac{\mathrm{d}y}{\mathrm{d}b}}\left({\frac{\mathrm{d}b}{\mathrm{d}a}}{\frac{\mathrm{d}a}{\mathrm{d}x}}\right)\,.}\end{array}}
$$
方程（5.120）将是反向模式，因为梯度是沿着图的反向传播的，即与数据流相反。方程（5.121）将是前向模式，其中梯度随着数据从左到右流经图。

自动微分

自动微分不同于符号微分和梯度的数值近似，例如通过使用有限差分。

在一般情况下，我们处理雅可比矩阵，这些矩阵可以是向量、矩阵或张量。

反向模式

前向模式

在接下来的内容中，我们将专注于反向模式自动微分，即反向传播。在神经网络的背景下，输入维度通常远高于标签的维度，反向模式在计算上比前向模式便宜得多。让我们从一个有启发性的例子开始。


### 示例 5.14

考虑函数

$$
f(x)=\sqrt{x^{2}+\exp(x^{2})}+\cos\left(x^{2}+\exp(x^{2})\right)
$$
中间变量

从 (5.109) 式中。如果我们想在计算机上实现函数 $f$，我们可以利用中间变量来节省一些计算量：

$$
\begin{array}{l}{a={x}^{2}\,,}\\ {b=\exp(a)\,,}\\ {c=a+b\,,}\\ {d=\sqrt{c}\,,}\\ {e=\cos(c)\,,}\\ {f=d+e\,.}\end{array}
$$
![](images/27c9a7e903150000a87784981496f3f15cb4fda8b61a0bdc61282aa0a2fad37a.jpg)
图 5.11 计算图，输入为 $x$，函数值为 $f$，中间变量为 $a,b,c,d,e$。

这是应用链式法则时的相同思维方式。请注意，上述方程组所需的运算次数少于直接实现函数 $f(x)$（如 (5.109) 式定义）所需的运算次数。图 5.11 中的相应计算图显示了获取函数值 $f$ 所需的数据流和计算流程。

包含中间变量的方程组可以被视为计算图，这种表示在神经网络软件库的实现中被广泛使用。我们可以通过回忆基本函数的导数定义，直接计算中间变量相对于其相应输入的导数，得到以下结果：

$$
\begin{array}{l}{\displaystyle\frac{\partial a}{\partial x}=2x}\\ {\displaystyle\frac{\partial b}{\partial a}=\exp(a)}\end{array}
$$
$$
\begin{array}{l}{\displaystyle\frac{\partial c}{\partial a}=1=\frac{\partial c}{\partial b}}\\ {\displaystyle\frac{\partial d}{\partial c}=\frac{1}{2\sqrt{c}}}\\ {\displaystyle\frac{\partial e}{\partial c}=-\sin(c)}\\ {\displaystyle\frac{\partial f}{\partial d}=1=\frac{\partial f}{\partial e}\,.}\end{array}
$$
通过查看图 5.11 中的计算图，我们可以从输出开始反向计算 $\partial f/\partial x$，得到

$$
\begin{array}{l}{\displaystyle\frac{\partial f}{\partial c}=\frac{\partial f}{\partial d}\frac{\partial d}{\partial c}+\frac{\partial f}{\partial e}\frac{\partial e}{\partial c}}\\ {\displaystyle\frac{\partial f}{\partial b}=\frac{\partial f}{\partial c}\frac{\partial c}{\partial b}}\\ {\displaystyle\frac{\partial f}{\partial a}=\frac{\partial f}{\partial b}\frac{\partial b}{\partial a}+\frac{\partial f}{\partial c}\frac{\partial c}{\partial a}}\\ {\displaystyle\frac{\partial f}{\partial x}=\frac{\partial f}{\partial a}\frac{\partial a}{\partial x}\,.}\end{array}
$$
注意，我们隐式地应用了链式法则来获得 $\partial f/\partial x$。通过代入基本函数的导数结果，我们得到

$$
\begin{align}
    & \frac{\partial f}{\partial c} = 1 \cdot \frac{1}{2\sqrt{c}} + 1 \cdot (-\sin(c)) \\
    & \frac{\partial f}{\partial b} = \frac{\partial f}{\partial c} \cdot 1 \\
    & \frac{\partial f}{\partial a} = \frac{\partial f}{\partial b} \exp(a) + \frac{\partial f}{\partial c} \cdot 1 \\
    & \frac{\partial f}{\partial x} = \frac{\partial f}{\partial a} \cdot 2x
\end{align}
$$
将上述每个导数视为一个变量，我们观察到计算导数所需的计算复杂度与计算函数本身的复杂度相似。这与直觉相悖，因为导数 $\cfrac{\partial f}{\partial x}$ 的数学表达式（5.110 式）比函数 $f(x)$ 的数学表达式（5.109 式）复杂得多。

自动微分是示例 5.14 的形式化。设 $x_{1},\dots,x_{d}$ 是函数的输入变量，$x_{d+1},.\,.\,.\,,x_{D-1}$ 是中间变量，$x_{D}$ 是输出变量。那么计算图可以表示为：

$$
\mathrm{For}\;i=d+1,\ldots,D:\quad x_{i}=g_{i}\big(x_{\mathrm{Pa}(x_{i})}\big)\,,
$$
其中 $g_{i}(\cdot)$ 是基本函数，$x_{\mathrm{Pa}(x_{i})}$ 是变量 $x_{i}$ 在图中的父节点。给定以这种方式定义的函数，我们可以使用链式法则逐步计算函数的导数。根据定义，$f=x_{D}$，因此

$$
\frac{\partial f}{\partial x_{D}}=1\,.
$$
对于其他变量 $x_{i}$，我们应用链式法则

$$
\frac{\partial f}{\partial x_{i}}=\sum_{x_{j}:x_{i}\in\mathrm{Pa}(x_{j})}\frac{\partial f}{\partial x_{j}}\frac{\partial x_{j}}{\partial x_{i}}=\sum_{x_{j}:x_{i}\in\mathrm{Pa}(x_{j})}\frac{\partial f}{\partial x_{j}}\frac{\partial g_{j}}{\partial x_{i}}\,,
$$
反向自动微分需要解析树。

其中 $\mathrm{Pa}(x_{j})$ 是变量 $x_{j}$ 在计算图中的父节点集合。方程 (5.143) 是函数的前向传播，而 (5.145) 是通过计算图的梯度反向传播。对于神经网络训练，我们反向传播预测误差相对于标签的误差。

上述自动微分方法适用于任何可以表示为计算图的函数，其中基本函数是可微的。实际上，该函数甚至可能不是数学函数，而是一个计算机程序。然而，并非所有计算机程序都可以自动微分，例如，如果我们无法找到可微的基本函数。诸如 for 循环和 if 语句之类的编程结构也需要更多的注意。


## 5.7 高阶导数

到目前为止，我们讨论了梯度，即一阶导数。有时，我们对更高阶的导数感兴趣，例如，当我们想使用牛顿法进行优化时，这需要二阶导数（Nocedal 和 Wright, 2006）。在第 5.1.1 节中，我们讨论了泰勒级数，用于使用多项式近似函数。在多变量情况下，我们可以做同样的事情。在下面，我们将做同样的事情。但让我们从一些符号开始。

考虑一个有两个变量 $x, y$ 的函数 $f: \mathbb{R}^{2} \rightarrow \mathbb{R}$。我们使用以下符号表示更高阶的偏导数（和梯度）：

$\cfrac{\partial^{2}f}{\partial x^{2}}$ 是 $f$ 关于 $x$ 的二阶偏导数。

$\cfrac{\partial^{n}f}{\partial x^{n}}$ 是 $f$ 关于 $x$ 的 $n$ 阶偏导数。

$\cfrac{\partial^{2}f}{\partial y\partial x} = \cfrac{\partial}{\partial y}\left(\cfrac{\partial f}{\partial x}\right)$ 是首先关于 $x$ 偏导，然后关于 $y$ 偏导得到的偏导数。

$\cfrac{\partial^{2}f}{\partial x\partial y}$ 是首先关于 $y$ 偏导，然后关于 $x$ 偏导得到的偏导数。

海森矩阵（Hessian）是所有二阶偏导数的集合。

![](images/f12c23e085030e98b4980c7d9f8e33f2f807c94ac91e1ffea55fe606bb481466.jpg)
图 5.12 函数的线性近似。原始函数 $f$ 在 $x_{0}=-2$ 处使用一阶泰勒级数展开进行线性化。

如果 $f(x,y)$ 是一个二阶（连续）可微函数，则

$$
\frac{\partial^{2}f}{\partial x\partial y} = \frac{\partial^{2}f}{\partial y\partial x}\,,
$$
即，求导的顺序无关紧要，相应的海森矩阵为

$$
H = \left[\begin{array}{l l}
\cfrac{\partial^{2}f}{\partial x^{2}} & \cfrac{\partial^{2}f}{\partial x\partial y} \\
\cfrac{\partial^{2}f}{\partial x\partial y} & \cfrac{\partial^{2}f}{\partial y^{2}}
\end{array}\right]
$$

海森矩阵记为 $\nabla_{x,y}^{2}f(x,y)$。一般地，对于 $\pmb{x} \in \mathbb{R}^{n}$ 和 $f: \mathbb{R}^{n} \to \mathbb{R}$，海森矩阵是一个 $n \times n$ 矩阵。海森矩阵测量函数在 $(x,y)$ 附近的曲率。

注释（海森或场）。如果 $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ 是一个向量场，则海森矩阵是一个 $(m \times n \times n)$ 张量。$\diamond$


## 5.8 线性化与多元泰勒级数

函数 $f$ 的梯度 $\nabla f$ 经常用于在 $\mathbf{\delta}x_{0}$ 附近对 $f$ 进行局部线性近似：

$$
f(\pmb{x})\approx f(\pmb{x}_{0})+(\nabla_{\pmb{x}}f)(\pmb{x}_{0})(\pmb{x}-\pmb{x}_{0})\,.
$$
这里 $(\nabla_{\pmb{x}}f)(\pmb{x}_{0})$ 是 $f$ 关于 $\pmb{x}$ 的梯度，在 $\pmb{x}_{0}$ 处求值。图 5.12 描述了函数 $f$ 在输入 $x_{0}$ 处的线性近似。原始函数被一条直线近似。这种近似在局部是准确的，但离 $x_{0}$ 越远，近似效果越差。方程 (5.148) 是 $f$ 在 $\pmb{x}_{0}$ 处的多元泰勒级数展开的一个特例，其中我们只考虑前两项。接下来我们将讨论更一般的情况，这将允许更好的近似。

![](images/5761b80b50f9c5cb23d7bbe016cba5f336b79fd486c8e76f38f4fa42857bd9f6.jpg)

166

### 定义 5.7 (多元泰勒级数)。

我们考虑一个函数

$$
\begin{array}{r}{f:\mathbb{R}^{D}\rightarrow\mathbb{R}}\\ {\pmb{x}\mapsto f(\pmb{x})\,,\quad\pmb{x}\in\mathbb{R}^{D}\,,}\end{array}
$$
多元泰勒级数

在 $\pmb{x}_{0}$ 处是光滑的。当我们定义差向量 $\pmb{\delta}:=\pmb{x}-\pmb{x}_{0}$ 时，$f$ 在 $(\pmb{x}_{0})$ 处的多元泰勒级数定义为

$$
f({\pmb x})=\sum_{k=0}^{\infty}\frac{D_{{\pmb x}}^{k}f({\pmb x}_{0})}{k!}{\pmb\delta}^{k}\,,
$$
其中 $D_{\mathbfit{x}}^{k}f(\mathbfit{x}_{\mathrm{0}})$ 是 $f$ 关于 $\pmb{x}$ 的第 $k$ 阶（全）导数，在 $\pmb{x}_{0}$ 处求值。

泰勒多项式

向量可以实现为一维数组，矩阵可以实现为二维数组。

定义 5.8 (泰勒多项式)。$f$ 在 $\mathbf{\delta}x_{0}$ 处的泰勒多项式，其次数为 $n$，包含 (5.151) 中级数的前 $n+1$ 个分量，定义为

$$
T_{n}(\pmb{x})=\sum_{k=0}^{n}\frac{D_{\pmb{x}}^{k}f(\pmb{x}_{0})}{k!}\pmb{\delta}^{k}\,.
$$
在 (5.151) 和 (5.152) 中，我们使用了稍微松散的记号 $\delta^{k}$，它表示对于 $\pmb{x}\,\in\,\mathbb{R}^{D}$，$D>1$，以及 $k\,>\,1$ 的向量。注意，$D_{x}^{k}f$ 和 $\delta^{k}$ 都是 $k$ 阶张量，即 $k$ 维数组。$k$ 次 $k$ 阶张量 $\pmb{\delta}^{k}\in\mathbb{R}^{\overbrace{D\times D\times\ldots\times D}^{D\times D\times\ldots\times D}}$ 是通过 $k$ 次外积，记为 $\otimes$，从向量 $\pmb{\delta}\in\mathbb{R}^{D}$ 得到的。例如，

$$
\pmb{\delta}^{2}:=\pmb{\delta}\otimes\pmb{\delta}=\pmb{\delta}\pmb{\delta}^{\top}\,,\quad\pmb{\delta}^{2}[i,j]=\pmb{\delta}[i]\delta[j]
$$
$$
\begin{array}{r}{\delta^{3}:=\delta\otimes\delta\otimes\delta\,,\quad\delta^{3}[i,j,k]=\delta[i]\delta[j]\delta[k]\,.}\end{array}
$$
图 5.13 可视化了两个这样的外积。一般地，我们得到泰勒级数中的项

$$
D_{\pmb{x}}^{k}f(\pmb{x}_{0})\pmb{\delta}^{k}=\sum_{i_{1}=1}^{D}\cdot\cdot\cdot\sum_{i_{k}=1}^{D}D_{\pmb{x}}^{k}f(\pmb{x}_{0})[i_{1},.\,.\,.\,,i_{k}]\delta[i_{1}]\cdot\cdot\cdot\delta[i_{k}]
$$
其中 $D_{\pmb{x}}^{k}f(\pmb{x}_{0})\pmb{\delta}^{k}$ 包含 $k$ 阶多项式。

现在我们已经定义了向量场的泰勒级数，让我们明确写出泰勒级数展开的前几项 $D_{\pmb{x}}^{k}f(\pmb{x}_{0})\pmb{\delta}^{k}$，对于 $k=0,\hdots,3$ 和 $\pmb{\delta}:=\pmb{x}-\pmb{x}_{0}$：

$$
\begin{array}{r l}&{k=0:D_{x}^{0}f(x_{0})\delta^{0}=f(x_{0})\in\mathbb{R}}\\ &{k=1:D_{x}^{1}f(x_{0})\delta^{1}=\nabla_{x}f(x_{0})\displaystyle\frac{\delta}{D\!\times\!D}=\sum_{i=1}^{D}\nabla_{x}f(x_{0})[i]\delta[i]\in\mathbb{R}}\\ &{k=2:D_{x}^{2}f(x_{0})\delta^{2}=\operatorname{tr}(\underbrace{H(x_{0})}_{D\times D}\displaystyle\delta_{D\times1}\displaystyle\delta_{D}^{\top})=\delta^{\top}H(x_{0})\delta}\\ &{\quad=\displaystyle\sum_{i=1}^{D}\sum_{j=1}^{D}H[i,j]\delta[i]\delta[j]\in\mathbb{R}}\\ &{k=3:D_{x}^{3}f(x_{0})\delta^{3}=\displaystyle\sum_{i=1}^{D}\sum_{j=1}^{D}D_{x}^{3}f(x_{0})[i,j,k]\delta[i]\delta[j]\delta[k]\in\mathbb{R}}\end{array}
$$
这里，${\cal H}({\pmb x}_{0})$ 是 $f$ 在 $\pmb{x}_{0}$ 处的海森矩阵。


## 5.9 进一步阅读

有关矩阵微分的详细信息以及所需线性代数的简要回顾，请参阅Magnus和Neudecker (2007)。自动微分有着悠久的历史，我们参考Griewank和Walther (2003)，Griewank和Walther (2008)，以及Elliott (2009)及其参考文献。

在机器学习（以及其他学科）中，我们经常需要计算期望值，即需要求解形式为

$$
\mathbb{E}_{x}[f({\pmb x})]=\int f({\pmb x})p({\pmb x})d{\pmb x}\,.
$$
的积分。

扩展卡尔曼滤波器

无迹变换 拉普拉斯近似

即使 $p(\pmb{x})$ 以方便的形式存在（例如高斯分布），这个积分通常无法解析求解。$f$ 的泰勒级数展开是一种找到近似解的方法：假设 $p({\pmb x})=\mathcal{N}({\pmb\mu},\,{\pmb\Sigma})$ 是高斯分布，则 $f$ 在 $\pmb{\mu}$ 处的一阶泰勒级数展开局部线性化了非线性函数 $f$。对于线性函数，如果 $p(\pmb{x})$ 是高斯分布，我们可以精确计算均值（和协方差）（参见第6.5节）。这一性质被扩展卡尔曼滤波器（Maybeck, 1979）在非线性动态系统的在线状态估计中（也称为“状态空间模型”）广泛利用。其他确定性方法来近似 (5.181) 中的积分包括无迹变换（Julier和Uhlmann, 1997），该方法不需要任何梯度，或者拉普拉斯近似（MacKay, 2003; Bishop, 2006; Murphy, 2012），该方法使用二阶泰勒级数展开（需要海森矩阵）来局部高斯近似 $p(\pmb{x})$ 的模式。


## 练习题

5.1 计算函数 $f(x)=\log(x^{4})\sin(x^{3})$ 的导数 $f^{\prime}(x)$。

$$
f(x)=\log(x^{4})\sin(x^{3})\,.
$$
5.2 计算逻辑Sigmoid函数 $f(x)={\frac{1}{1+\exp(-x)}}$ 的导数 $f^{\prime}(x)$。

$$
f(x)={\frac{1}{1+\exp(-x)}}\,.
$$
5.3 计算函数 $f(x)=\exp(-\frac{1}{2\sigma^{2}}\big(x-\mu\big)^{2})$ 的导数 $f^{\prime}(x)$，其中 $\mu,\ \sigma\in\mathbb{R}$ 是常数。

$$
\begin{array}{r}{f(x)=\exp(-\frac{1}{2\sigma^{2}}\big(x-\mu\big)^{2})\,,}\end{array}
$$
5.4 计算函数 $f(x)=\sin(x)+\cos(x)$ 在 $x_{0}=0$ 处的泰勒多项式 $T_{n}$，其中 $n=0,\ldots,5$。

5.5 考虑以下函数：

$$
\begin{array}{r l}&{f_{1}(\pmb{x})=\sin(x_{1})\cos(x_{2})\,,\quad\pmb{x}\in\mathbb{R}^{2}}\\ &{f_{2}(\pmb{x},\pmb{y})=\pmb{x}^{\top}\pmb{y}\,,\quad\pmb{x},\pmb{y}\in\mathbb{R}^{n}}\\ &{f_{3}(\pmb{x})=\pmb{x}\pmb{x}^{\top}\,,\qquad\pmb{x}\in\mathbb{R}^{n}}\end{array}
$$
a. $\textstyle{\frac{\partial f_{i}}{\partial{\boldsymbol{x}}}}$ 的维度是什么？b. 计算雅可比矩阵。

5.6 分别计算 $f$ 对 $^{t}$ 的导数和 $g$ 对 $_{X}$ 的导数，其中

$$
\begin{array}{r l}&{f(\pmb{t})=\sin(\log(\pmb{t}^{\top}\pmb{t}))\,,\qquad\pmb{t}\in\mathbb{R}^{D}}\\ &{g(\pmb{X})=\operatorname{tr}(\pmb{A}\pmb{X}\pmb{B})\,,\qquad\pmb{A}\in\mathbb{R}^{D\times E},\pmb{X}\in\mathbb{R}^{E\times F},\pmb{B}\in\mathbb{R}^{F\times D}\,,}\end{array}
$$
其中 $\mathrm{tr}(\cdot)$ 表示迹。

5.7 使用链式法则计算以下函数的导数 $\mathrm{d}f/\mathrm{d}x$，并提供每个偏导数的维度。详细描述你的步骤。

a.

$$
f(z)=\log(1+z)\,,\quad z=\pmb{x}^{\top}\pmb{x}\,,\quad\pmb{x}\in\mathbb{R}^{D}
$$
b.

$$
f(\boldsymbol{z})=\sin(\boldsymbol{z})\,,\quad\boldsymbol{z}=\boldsymbol{A}\boldsymbol{x}+\boldsymbol{b}\,,\quad\boldsymbol{A}\in\mathbb{R}^{E\times D},\boldsymbol{x}\in\mathbb{R}^{D}\,,\boldsymbol{b}\in\mathbb{R}^{E\times D}\,,
$$
其中 $\sin(\cdot)$ 作用于 $_z$ 的每个元素。


5.8 计算下列函数的导数 $\mathrm{d}f/\mathrm{d}x$。详细描述你的步骤。

a. 使用链式法则。提供每个偏导数的维度。

$$
\begin{array}{r l}
&{f(z)=\exp(-\frac{1}{2}z)}\\ 
&{\quad z=g(\pmb{y})=\pmb{y}^{\top}\pmb{S}^{-1}\pmb{y}}\\ 
&{\quad\pmb{y}=h(\pmb{x})=\pmb{x}-\pmb{\mu}}
\end{array}
$$

其中 $\pmb{x},\pmb{\mu}\in\mathbb{R}^{D}$, $\pmb{S}\in\mathbb{R}^{D\times D}$。

b.

$$
\boldsymbol{f}(\textbf x)=\mathrm{tr}(\textbf x\textbf x^{\top}+\sigma^{2}\boldsymbol{I})\,,\quad\textbf x\in\mathbb{R}^{D}
$$
这里 $\operatorname{tr}(A)$ 是矩阵 $\pmb{A}$ 的迹，即对角元素之和 $A_{i i}$。提示：显式写出外积。

c. 使用链式法则。提供每个偏导数的维度。你不需要显式计算偏导数的乘积。

$$
\begin{array}{r l}
&{\pmb{f}=\operatorname{tanh}(\pmb{z})\in\mathbb{R}^{M}}\\ 
&{\pmb{z}=\pmb{A x}+\pmb{b},\quad\pmb{x}\in\mathbb{R}^{N},\pmb{A}\in\mathbb{R}^{M\times N},\pmb{b}\in\mathbb{R}^{M}.}
\end{array}
$$

这里，$\operatorname{tanh}$ 函数应用于 $\pmb{z}$ 的每个分量。

5.9 我们定义

$$
\begin{array}{r l}
&{\mathscr{g}(\pmb{x},z,\pmb{\nu}):=\log\mathscr{p}(\pmb{x},z)-\log\mathscr{q}(z,\pmb{\nu})}\\ 
&{\quad\quad\quad z:=t(\pmb{\epsilon},\pmb{\nu})}
\end{array}
$$

对于可微函数 $p,q,t$ 和 $\pmb{x}\in\mathbb{R}^{D},z\in\mathbb{R}^{E},\pmb{\nu}\in\mathbb{R}^{F},\pmb{\epsilon}\in\mathbb{R}^{G}$。通过使用链式法则，计算梯度

$$
\frac{\mathrm{d}}{\mathrm{d}\nu}g(\pmb{x},z,\nu)\,.
$$