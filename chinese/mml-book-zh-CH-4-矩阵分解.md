
# 第四章 矩阵分解

在第2章和第3章中，我们研究了操作和测量向量、向量投影以及线性映射的方法。映射和向量变换可以方便地描述为矩阵执行的操作。此外，数据通常也以矩阵形式表示，例如，矩阵的行代表不同的人，列描述人的不同特征，如体重、身高和社会经济地位。在本章中，我们将介绍矩阵的三个方面：如何总结矩阵，如何分解矩阵，以及如何使用这些分解进行矩阵近似。

我们首先考虑一些方法，这些方法允许我们用少数几个数字来描述矩阵，这些数字可以表征矩阵的整体属性。我们将在行列式（第4.1节）和特征值（第4.2节）部分讨论正方形矩阵的重要特例。这些特征数字具有重要的数学后果，并允许我们快速掌握矩阵的有用属性。从这里我们将继续讨论矩阵分解方法：矩阵分解的一个类比是数字的因式分解，例如将21分解为素数 $7\cdot3$ 。因此，矩阵分解也经常被称为矩阵因式分解。矩阵分解用于通过可解释的矩阵因子来描述矩阵的不同表示形式。

我们将首先介绍对称正定矩阵的平方根操作，即Cholesky分解（第4.3节）。从这里我们将研究两种相关的矩阵因式分解方法。第一种方法称为矩阵对角化（第4.4节），它允许我们通过选择适当的基来表示线性映射，使用对角变换矩阵。第二种方法是奇异值分解（第4.5节），它将这种因式分解扩展到非正方形矩阵，并被认为是线性代数中的基本概念之一。这些分解对于表示数值数据的矩阵非常有用，因为这些矩阵通常非常大且难以分析。我们将在第4.7节中以矩阵分类的形式系统地概述矩阵的类型及其特征属性。

本章中介绍的方法将在后续的数学章节中变得非常重要，例如第6章，但也在应用章节中，例如第10章中的降维和第11章中的密度估计。本章的整体结构在图4.1中的思维导图中有所描绘。

![](images/d22a159df54b1bb6b0ae7c2d557122d311ea4591842be30c9850eb609a1ca73f.jpg)

图4.1 本章介绍的概念的思维导图，以及它们在本书其他部分中的应用位置。

## 4.1 行列式和迹

行列式是线性代数中的重要概念。行列式是用于分析和求解线性方程组的数学对象。行列式仅定义为正方形矩阵 $\pmb{A}\in\mathbb{R}^{n\times n}$，即行数和列数相同的矩阵。在本书中，我们用 $\operatorname*{det}(A)$ 或有时用 $|A|$ 表示行列式，以便

行列式符号 $|A|$ 不应与绝对值混淆。

$$
\operatorname{det}(A) = \left| \begin{array}{cccc}
a_{11} & a_{12} & \dots & a_{1n} \\
a_{21} & a_{22} & \dots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \dots & a_{nn}
\end{array} \right|
$$

正方形矩阵 $\pmb{A}\in\mathbb{R}^{n\times n}$ 的行列式是一个将 $\pmb{A}$ 映射到实数的函数。

在提供一般 $n\times n$ 矩阵的行列式定义之前，让我们先看看一些有启发性的例子，并定义一些特殊矩阵的行列式。

### 例4.1（测试矩阵可逆性）

让我们从探索一个正方形矩阵 $\pmb{A}$ 是否可逆（见第2.2.2节）开始。对于最小的情况，我们已经知道当矩阵 $\pmb{A}$ 是一个 $1\times1$ 矩阵，即它是一个标量数时，$\pmb{A}=\pmb{a}\implies\pmb{A}^{-1}=\frac{1}{\pmb{a}}$。因此，$a\cdot\frac{1}{a}=1$ 成立，当且仅当 $a\neq0$。

对于 $2\times2$ 矩阵，根据逆矩阵的定义（定义2.3），我们知道 $\boldsymbol{A}\boldsymbol{A}^{-1}=\boldsymbol{I}$。然后，根据（2.24），矩阵 $A$ 的逆为

$$
\pmb{A}^{-1}=\frac{1}{a_{11}a_{22}-a_{12}a_{21}}\left[\begin{array}{c c}{a_{22}}&{-a_{12}}\\ {-a_{21}}&{a_{11}}\end{array}\right].
$$
因此，$\pmb{A}$ 可逆当且仅当

$$
a_{11}a_{22}-a_{12}a_{21}\neq0\,.
$$
这个量是 $\pmb{A}\in\mathbb{R}^{2\times2}$ 的行列式，即

$$
\operatorname*{det}(A)={\binom{a_{11}}{a_{21}}}{\binom{a_{12}}{a_{22}}}=a_{11}a_{22}-a_{12}a_{21}\,.
$$
例4.1已经指出了行列式与逆矩阵存在的关系。下一个定理对 $n\times n$ 矩阵陈述了相同的结果。

### 定理4.1

对于任何正方形矩阵 $\pmb{A}\in\mathbb{R}^{n\times n}$，矩阵 $\pmb{A}$ 可逆当且仅当 $\operatorname*{det}(A)\neq0$。

我们有小矩阵行列式的显式（封闭形式）表达式，这些表达式以矩阵元素的形式给出。对于 $n=1$，

$$
\operatorname*{det}(\pmb{A})=\operatorname*{det}(a_{11})=a_{11}\,.
$$
对于 $n=2$，

$$
\operatorname*{det}(A) = \left| \begin{array}{cc}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{array} \right| = a_{11}a_{22} - a_{12}a_{21},
$$

我们在前面的例子中已经观察到这一点。对于 $n=3$（称为Sarrus法则），

$$
\begin{vmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{vmatrix}
= a_{11}a_{22}a_{33} + a_{12}a_{23}a_{31} + a_{13}a_{21}a_{32} \\
- a_{13}a_{22}a_{31} - a_{11}a_{23}a_{32} - a_{12}a_{21}a_{33}
$$
为了帮助记忆Sarrus法则中的乘积项，可以尝试追踪矩阵中三重乘积的元素。

我们称一个正方形矩阵 $\pmb{T}$ 为上三角矩阵，如果 $T_{i j}=0$ 对于 $i>j$，即矩阵在其对角线以下的元素为零。类似地，我们定义下三角矩阵为在其对角线以上的元素为零的矩阵。对于一个三角矩阵 $\pmb{T}\in\mathbb{R}^{n\times n}$，行列式是其对角线元素的乘积，即

$$
\operatorname*{det}(\pmb{T})=\prod_{i=1}^{n}T_{i i}\,.
$$

### 示例 4.2 (行列式作为体积的度量)

当我们考虑行列式作为从一组 $n$ 个向量生成的对象到 $\mathbb{R}^{n}$ 的映射时，行列式的概念是自然的。结果表明，行列式 $\operatorname*{det}(A)$ 是由矩阵 $\pmb{A}$ 的列生成的 $n$ 维平行六面体的带符号体积。

对于 $n=2$，矩阵的列形成一个平行四边形；见图 4.2。随着向量之间的角度变小，平行四边形的面积也会缩小。考虑两个向量 $b$ 和 $g$，它们是矩阵 $\pmb{A}=[\pmb{b},\pmb{g}]$ 的列。那么，矩阵 $\pmb{A}$ 的行列式的绝对值是顶点为 $\mathbf{0},b,g,b+g$ 的平行四边形的面积。特别地，如果 $b$ 和 $g$ 是线性相关的，即 $\mathbf{b}=\lambda\mathbf{g}$，对于某个 $\lambda\in\mathbb{R}$，它们不再形成一个二维平行四边形。因此，相应的面积为 0。相反，如果 $b$ 和 $g$ 是线性独立的，并且是标准基向量 $e_{1},e_{2}$ 的倍数，则它们可以表示为 $b = \begin{bmatrix}b \\0\end{bmatrix}$ 和 $g = \begin{bmatrix}0 \\g\end{bmatrix}$，行列式为 $\left|\begin{array}{ll}b & 0\\ 0 & g\end{array}\right|=bg-0=bg$。

行列式的符号表示生成向量 $b$ 和 $g$ 相对于标准基 $(e_{1},e_{2})$ 的方向。在我们的图中，将顺序翻转为 $g,b$ 会交换矩阵 $\pmb{A}$ 的列，并且会反转阴影区域的方向。这变成了熟悉的公式：面积 $=$ 高度 $\times$ 长度。这种直觉可以扩展到更高维度。在 $\mathbb{R}^{3}$ 中，我们考虑三个向量 $r,b,g\in\mathbb{R}^{3}$，它们生成平行六面体的边（见图 4.3）。$3\times3$ 矩阵 $[r,\ b,\ g]$ 的行列式的绝对值是该平行六面体的体积。因此，行列式作为度量函数，测量由矩阵列向量组成的带符号体积。

上三角矩阵 下三角矩阵

行列式是矩阵列生成的平行六面体的带符号体积。图 4.2 由向量 $b$ 和 $g$ 生成的平行四边形（阴影区域）的面积是 $|\mathrm{det}([b,\,g])|$。

![](images/64f385aff0519978910fa634a9fb20e128e79b06926d14e072de66329970b2b3.jpg)  

图4.3 由向量$\pmb{r},\pmb{b},\pmb{g}$张成的平行六面体（阴影部分）的体积为

$|\mathrm{det}([{\boldsymbol{\mathbf{r}}},\,b,\,{\boldsymbol{\mathbf{\mathit{g}}}}])|$。  

![](images/f8e46dca3ad30a54ff41ab03c181368c6fdd70babaec63499702c58e04370aa5.jpg)  

行列式的符号指示了张成向量的方向。  

考虑三个线性独立的向量$r,g,b\in\mathbb{R}^{3}$，给定为

$$
r={\left[\begin{array}{c}{2}\\ {0}\\ {-8}\end{array}\right]}\,,\quad g={\left[\begin{array}{c}{6}\\ {1}\\ {0}\end{array}\right]}\,,\quad b={\left[\begin{array}{c}{1}\\ {4}\\ {-1}\end{array}\right]}\ .
$$
将这些向量作为矩阵的列写成

$$
A=[r,\ g,\ b]=\left[{\begin{array}{c c c}{2}&{6}&{1}\\ {0}&{1}&{4}\\ {-8}&{0}&{-1}\end{array}}\right]
$$
允许我们计算所需的体积为

$$
V=|\mathrm{det}(\mathbf{A})|=186\,.
$$
计算$n\times n$矩阵的行列式需要一个通用算法来解决$n>3$的情况，我们将在接下来的部分中探讨这个问题。定理4.2将计算$n\times n$矩阵行列式的问题转化为计算$(n-1)\times(n-1)$矩阵行列式的问题。通过递归应用拉普拉斯展开（定理4.2），我们最终可以通过计算$2\times2$矩阵的行列式来计算$n\times n$矩阵的行列式。

### 定理 4.2 （拉普拉斯展开）. 

考虑一个矩阵 $\pmb{A}\,\in\,\mathbb{R}^{n\times n}$。则对于所有 $j=1,\dotsc,n$：

$\operatorname*{det}(\pmb{A}_{k,j})$ 称为一个 余子式 ，而 $(-1)^{k+j}\operatorname*{det}(A_{k,j})$ 称为 代数余子式 。

1. 按列 $j$ 展开

$$
\operatorname*{det}(A)=\sum_{k=1}^{n}(-1)^{k+j}a_{k j}\operatorname*{det}(A_{k,j})\,.
$$
2. 按行 $j$ 展开

$$
\operatorname*{det}(\pmb{A})=\sum_{k=1}^{n}(-1)^{k+j}a_{j k}\operatorname*{det}(\pmb{A}_{j,k})\,.
$$
这里 $\pmb{A}_{k,j}\in\mathbb{R}^{(n-1)\times(n-1)}$ 是从矩阵 $\pmb{A}$ 中删除第 $k$ 行和第 $j$ 列后得到的子矩阵。

### 示例 4.3 (拉普拉斯展开)

让我们计算矩阵

$$
A=\begin{bmatrix}
1 & 2 & 3 \\
3 & 1 & 2 \\
0 & 0 & 1
\end{bmatrix}
$$

的行列式，使用拉普拉斯展开沿第一行展开。应用公式 (4.13) 得到

$$
\begin{array}{r l}
\left|
\begin{array}{ccc}
1 & 2 & 3 \\
3 & 1 & 2 \\
0 & 0 & 1
\end{array}
\right| &= (-1)^{1+1}\cdot1\left|
\begin{array}{cc}
1 & 2 \\
0 & 1
\end{array}
\right| \\
&+ (-1)^{1+2}\cdot2\left|
\begin{array}{cc}
3 & 2 \\
0 & 1
\end{array}
\right| + (-1)^{1+3}\cdot3\left|
\begin{array}{cc}
3 & 1 \\
0 & 0
\end{array}
\right|
\end{array}
$$

我们使用公式 (4.6) 计算所有 $2\times2$ 矩阵的行列式，得到

$$
\operatorname*{det}(\pmb{A})=1(1-0)-2(3-0)+3(0-0)=-5
$$
为了完整性，我们可以将此结果与使用萨鲁斯法则 (4.7) 计算行列式的结果进行比较：

$$
\operatorname*{det}(A)=1\cdot1\cdot1+3\cdot0\cdot3+0\cdot2\cdot2-0\cdot1\cdot3-1\cdot0\cdot2-3\cdot2\cdot1=1-6=-5
$$
对于 $\pmb{A}\in\mathbb{R}^{n\times n}$，行列式具有以下性质：

矩阵乘积的行列式等于相应行列式的乘积，$\operatorname*{det}(\pmb{A}\pmb{B})=\operatorname*{det}(\pmb{A})\operatorname*{det}(\pmb{B})$。行列式对转置不变，即 $\operatorname*{det}(\pmb{A})=\operatorname*{det}(\pmb{A}^{\top})$。如果 $\pmb{A}$ 是可逆的，则 $\operatorname*{det}(\pmb{A}^{-1})=\cfrac{1}{\operatorname*{det}(\pmb{A})}$。相似矩阵（定义 2.22）具有相同的行列式。因此，对于线性映射 $\Phi:V\to V$，所有变换矩阵 $\pmb{A}_{\Phi}$ 具有相同的行列式。因此，行列式对于线性映射的选择基是不变的。添加一列/行的倍数到另一列/行不会改变 $\operatorname*{det}(A)$。将一列/行乘以 $\lambda\in\mathbb{R}$ 会将 $\operatorname*{det}(A)$ 缩放 $\lambda$。特别地，$\operatorname*{det}(\lambda\pmb{A})=\lambda^{n}\operatorname*{det}(\pmb{A})$。交换两行/列会改变 $\operatorname*{det}(A)$ 的符号。

由于最后三个性质，我们可以使用高斯消元法（见第 2.1 节）将 $\pmb{A}$ 转换为行阶梯形来计算 $\operatorname*{det}(A)$。当 $\pmb{A}$ 转换为三角形形式，即对角线下方的所有元素均为 0 时，可以停止高斯消元法。回忆公式 (4.8)，三角矩阵的行列式是对角线元素的乘积。

### 定理 4.3. 

一个 $n \times n$ 的方阵 $\pmb{A}$ 有 $\operatorname*{det}(A) \neq 0$ 当且仅当 $\operatorname{rk}(A) = n$。换句话说，$A$ 可逆当且仅当它是满秩的。

当数学主要由手工完成时，行列式的计算被认为是分析矩阵可逆性的一种基本方法。然而，现代机器学习方法使用直接数值方法，这些方法超越了显式计算行列式的传统方法。例如，在第 2 章中，我们学习了通过高斯消元法计算逆矩阵的方法。因此，高斯消元法也可以用于计算矩阵的行列式。

行列式将在接下来的章节中扮演重要的理论角色，特别是在我们学习特征值和特征向量（第 4.2 节）时，通过特征多项式。

定义 4.4. 一个 $n \times n$ 的方阵 $\pmb{A}$ 的迹定义为

$$
\operatorname{tr}(A) := \sum_{i=1}^{n} a_{ii}\,,
$$
即，迹是矩阵 $\pmb{A}$ 的对角元素之和。迹具有以下性质：

$$
\begin{array}{r l}
&{\bullet\,\,\mathrm{tr}(A+B) = \mathrm{tr}(A) + \mathrm{tr}(B)\,\,\text{对于}\,\,A,B \in \mathbb{R}^{n \times n}}\\
&{\bullet\,\,\mathrm{tr}(\alpha A) = \alpha \mathrm{tr}(A)\,,\alpha \in \mathbb{R}\,\,\text{对于}\,\,A \in \mathbb{R}^{n \times n}}\\
&{\bullet\,\,\mathrm{tr}(I_{n}) = n}\\
&{\bullet\,\,\mathrm{tr}(AB) = \mathrm{tr}(BA)\,\,\text{对于}\,\,A \in \mathbb{R}^{n \times k},B \in \mathbb{R}^{k \times n}}
\end{array}
$$

可以证明，只有唯一一个函数同时满足这四个性质——即迹（Gohberg 等人，2012）。

迹在循环置换下是不变的。

矩阵乘积的迹的性质更为一般。具体来说，迹在循环置换下是不变的，即

$$
\operatorname{tr}(A K L) = \operatorname{tr}(K L A)
$$
对于矩阵 $\pmb{A} \in \mathbb{R}^{a \times k}, \pmb{K} \in \mathbb{R}^{k \times l}, \pmb{L} \in \mathbb{R}^{l \times a}$。这一性质推广到任意数量矩阵的乘积。作为（4.19）的一个特例，对于两个向量 $\pmb{x}, \pmb{y} \in \mathbb{R}^{n}$，

$$
\operatorname{tr}(\boldsymbol{x} \boldsymbol{y}^{\intercal}) = \operatorname{tr}(\boldsymbol{y}^{\intercal} \boldsymbol{x}) = \boldsymbol{y}^{\intercal} \boldsymbol{x} \in \mathbb{R}\,.
$$
给定一个线性映射 $\Phi: V \rightarrow V$，其中 $V$ 是一个向量空间，我们通过矩阵表示的迹来定义这个映射的迹。对于 $V$ 的给定基，我们可以通过变换矩阵 $\pmb{A}$ 来描述 $\Phi$。那么 $\Phi$ 的迹就是 $\pmb{A}$ 的迹。对于 $V$ 的不同基，变换矩阵 $\Phi$ 可以通过基变换的形式 $S^{-1} A S$ 来获得（见第 2.7.2 节）。对于相应的 $\Phi$ 的迹，这意味着

$$
\operatorname{tr}(B) = \operatorname{tr}(S^{-1} A S) \overset{(4.19)}{=} \operatorname{tr}(A S S^{-1}) = \operatorname{tr}(A)\,.
$$
因此，虽然线性映射的矩阵表示依赖于基，但线性映射 $\Phi$ 的迹与基无关。

在本节中，我们介绍了行列式和迹作为表征方阵的函数。结合我们对行列式和迹的理解，我们现在可以定义一个描述矩阵 $\pmb{A}$ 的重要方程，该方程将在接下来的章节中广泛使用。

定义（特征多项式）。对于 $\lambda \in \mathbb{R}$ 和一个 $n \times n$ 的方阵 $\pmb{A}$，

$$
\begin{array}{r l}
&{p_{A}(\lambda) := \operatorname*{det}(A - \lambda I)}\\
&{\qquad = c_{0} + c_{1} \lambda + c_{2} \lambda^{2} + \cdots + c_{n-1} \lambda^{n-1} + (-1)^{n} \lambda^{n}\,,}
\end{array}
$$

特征多项式

$c_{0}, \ldots, c_{n-1} \in \mathbb{R}_{*}$，是矩阵 $\pmb{A}$ 的特征多项式。特别地，

$$
\begin{array}{r c l}
&{}&{c_{0} = \operatorname*{det}(A)\,,}\\
&{}&{c_{n-1} = (-1)^{n-1} \mathrm{tr}(A)\,.}
\end{array}
$$

特征多项式（4.22a）将允许我们计算特征值和特征向量，这将在下一节中介绍。

## 4.2 特征值与特征向量

我们现在将了解一种新的方法来描述矩阵及其相关的线性映射。回顾第2.7.1节的内容，每个线性映射在给定有序基的情况下都有一个唯一的变换矩阵。我们可以通过进行“特征”分析来解释线性映射及其相关的变换矩阵。正如我们将要看到的，线性映射的特征值将告诉我们一组特殊的向量，即特征向量，是如何被线性映射变换的。

定义4 令 $\pmb{A} \in \mathbb{R}^{n\times n}$ 是一个方阵。如果 $\lambda \in \mathbb{R}$ 是 $\pmb{A}$ 的特征值，且 $\pmb{x} \in \mathbb{R}^{n} \backslash \{\mathbf{0}\}$ 是对应的特征向量，满足

$$
\pmb{A} \pmb{x} = \lambda \pmb{x}\,.
$$
我们称（4.25）为特征值方程。

“特征”是德语单词，意思是“特征”、“自我”或“自身”。

特征值 特征向量

特征值方程 备注。在线性代数文献和软件中，通常约定特征值按降序排列，因此最大的特征值及其对应的特征向量被称为第一个特征值及其对应的特征向量，第二大的特征值及其对应的特征向量被称为第二个特征值及其对应的特征向量，依此类推。然而，教科书和出版物可能有不同的排序方式或根本没有排序。如果未明确说明，我们不会假设存在排序。 $\diamondsuit$

以下陈述是等价的：

$\lambda$ 是 $\pmb{A} \in \mathbb{R}^{n\times n}$ 的特征值。存在一个 $\pmb{x} \in \mathbb{R}^{n} \backslash \{\mathbf{0}\}$ 使得 $\pmb{A} \pmb{x} = \lambda \pmb{x}$，或者等价地，$(\pmb{A} - \lambda \pmb{I}_{n}) \pmb{x} = \mathbf{0}$ 可以非平凡地求解，即 $\pmb{x} \neq \mathbf{0}$。$\operatorname{rk}(\pmb{A} - \lambda \pmb{I}_{n}) < n$。$\operatorname{det}(\pmb{A} - \lambda \pmb{I}_{n}) = 0$。


### 定义 4.7 (共线与同向)。

指向相同方向的两个向量称为**同向向量**。如果两个向量指向相同方向或相反方向，则称这两个向量为**共线向量**。

同向共线

备注 (特征向量的非唯一性)。如果向量 $x$ 是矩阵 $\pmb{A}$ 的特征向量，对应特征值 $\lambda$，那么对于任意 $c\in\mathbb{R}\backslash\{0\}$，都有 $c\pmb{x}$ 是矩阵 $\pmb{A}$ 的特征向量，对应相同的特征值，因为

$$
\pmb{A}(c\pmb{x})=c\pmb{A}\pmb{x}=c\lambda\pmb{x}=\lambda(c\pmb{x})\,.
$$
因此，所有与 $\pmb{x}$ 共线的向量都是矩阵 $\pmb{A}$ 的特征向量。

### 定理 4.8。

$\lambda\in\mathbb{R}$ 是矩阵 $\pmb{A}\in\mathbb{R}^{n\times n}$ 的特征值当且仅当 $\lambda$ 是矩阵 $\pmb{A}$ 的特征多项式 $p_{A}(\lambda)$ 的根。

代数重数

特征子空间 特征谱 谱

### 定义 4.9。

设方阵 $\pmb{A}$ 有一个特征值 $\lambda_{i}$。$\lambda_{i}$ 的**代数重数**是指特征多项式中该根出现的次数。

### 定义 4.10 (特征子空间与特征谱)。

对于 $\pmb{A}\in\mathbb{R}^{n\times n}$，所有与特征值 $\lambda$ 相关联的特征向量构成的子空间称为矩阵 $\pmb{A}$ 的**特征子空间**，记为 $E_{\lambda}$。所有特征值的集合称为矩阵 $\pmb{A}$ 的**特征谱**，或简称为**谱**。

如果 $\lambda$ 是矩阵 $\pmb{A}\in\mathbb{R}^{n\times n}$ 的特征值，则对应的特征子空间 $E_{\lambda}$ 是齐次线性方程组 $(A-\lambda I)\pmb{x}=\mathbf{0}$ 的解空间。从几何角度来看，对应于非零特征值的特征向量指向的方向是线性映射拉伸的方向。特征值是拉伸的比例因子。如果特征值为负，则拉伸的方向会被翻转。


### 示例 4.4（单位矩阵的情况）

$\textbf{\textit{I}}\in\mathrm{~\mathbb{R}^{n\times n}~}$ 的特征多项式为 $p_{I}(\lambda)\ =\operatorname*{det}(I-\lambda I)=(1-\lambda)^{n}=0$，只有一个特征值 $\lambda=1$，出现 $n$ 次。此外，对于所有非零向量 $\pmb{x}\in\mathbb{R}^{n}\backslash\{\mathbf{0}\}$，都有 ${\pmb I}{\pmb x}=\lambda{\pmb x}=1{\pmb x}$ 成立。因此，单位矩阵的唯一特征空间 $E_{1}$ 跨越 $n$ 维，所有 $n$ 个标准基向量都是 $\pmb{I}$ 的特征向量。

关于特征值和特征向量的一些有用性质包括：

一个矩阵 $\pmb{A}$ 及其转置 $A^{\top}$ 具有相同的特征值，但不一定具有相同的特征向量。特征空间 $E_{\lambda}$ 是 $A-\lambda I$ 的零空间，因为

$$
\begin{array}{r l}
&{A\pmb{x}=\lambda\pmb{x}\iff A\pmb{x}-\lambda\pmb{x}=\mathbf{0}}\\
&{\iff(\pmb{A}-\lambda\pmb{I})\pmb{x}=\mathbf{0}\iff\pmb{x}\in\ker(\pmb{A}-\lambda\pmb{I}).}
\end{array}
$$

相似矩阵（见定义 2.22）具有相同的特征值。因此，线性映射 $\Phi$ 的特征值与它的变换矩阵的基的选择无关。这使得特征值，连同行列式和迹，成为线性映射的关键特征参数，因为它们在基变换下都是不变的。对称正定矩阵总是具有正实特征值。


### 示例 4.5 (计算特征值、特征向量和特征空间)

让我们找到 $2\times2$ 矩阵的特征值和特征向量

$$
A=\begin{pmatrix}4\\1\end{pmatrix}\ .
$$
步骤 1：特征多项式。根据特征值的定义，对于非零向量 $\textbf{x}$，存在一个数 $\lambda$ 使得 $\pmb{A}\pmb{x}=\lambda\pmb{x}$，即 $(A-\lambda I)\pmb{x}=\mathbf{0}$。由于 $\mathbf{x}\neq\mathbf{0}$，这意味着 $A-\lambda I$ 的核（零空间）包含非零向量。因此，$A-\lambda I$ 不可逆，从而 $\operatorname*{det}(A-\lambda I)=0$。因此，我们需要计算特征多项式（4.22a）的根来找到特征值。

步骤 2：特征值。特征多项式为

$$
{\begin{array}{r l}
&{p_{A}(\lambda)=\operatorname*{det}(A-\lambda I)}\\ 
&{\qquad=\operatorname*{det}\left(\left[\begin{array}{cc}
4 & 2\\ 
1 & 3
\end{array}\right]-\left[\begin{array}{cc}
\lambda & 0\\ 
0 & \lambda
\end{array}\right]\right)=\left|\begin{array}{cc}
4-\lambda & 2\\ 
1 & 3-\lambda
\end{array}\right|}\\ 
&{\qquad=(4-\lambda)(3-\lambda)-2\cdot1\,.}
\end{array}}
$$

我们因式分解特征多项式，得到

$$
p(\lambda)=(4-\lambda)(3-\lambda)-2\cdot1=10-7\lambda+\lambda^{2}=(2-\lambda)(5-\lambda)
$$
给出根 $\lambda_{1}=2$ 和 $\lambda_{2}=5$。

步骤 3：特征向量和特征空间。我们通过寻找满足以下条件的向量 $\pmb{x}$ 来找到这些特征值对应的特征向量

$$
\begin{array}{r}
\left[\begin{array}{cc}
4-\lambda & 2\\ 
1 & \lambda-3
\end{array}\right]\pmb{x}=\mathbf{0}\,.
\end{array}
$$

对于 $\lambda=5$，我们得到

$$
\left[\begin{array}{cc}
4-5 & 2\\ 
1 & 3-5
\end{array}\right]\left[\begin{array}{c}
x_{1}\\ 
x_{2}
\end{array}\right]=\left[\begin{array}{cc}
-1 & 2\\ 
1 & -2
\end{array}\right]\left[\begin{array}{c}
x_{1}\\ 
x_{2}
\end{array}\right]=\mathbf{0}\,.
$$

我们解这个齐次线性方程组，得到解空间

$$
E_{5}=\mathrm{span}\left[\begin{array}{c}
2\\ 
1
\end{array}\right]\;.
$$

这个特征空间是一维的，因为它只有一个基向量。

类似地，我们通过解齐次线性方程组来找到 $\lambda=2$ 对应的特征向量

$$
\begin{array}{r}
\left[\begin{array}{cc}
4-2 & 2\\ 
1 & 3-2
\end{array}\right]\pmb{x}=\left[\begin{array}{cc}
2 & 2\\ 
1 & 1
\end{array}\right]\pmb{x}=\mathbf{0}\,.
\end{array}
$$

这意味着任何向量 $\pmb{x}=\left[\begin{array}{c}x_{1}\\ x_{2}\end{array}\right]$，其中 $x_{2}=-x_{1}$，例如 $\begin{pmatrix}1\\-1\end{pmatrix}$，都是特征值为 2 的特征向量。对应的特征空间为 $E_{2}=\mathrm{span}\left[\begin{array}{c}1\\ -1\end{array}\right]$ (4.35)。

示例 4.5 中的两个特征空间 $E_{5}$ 和 $E_{2}$ 是一维的，因为它们各自由一个向量生成。然而，在其他情况下，我们可能会有多个相同的特征值（见定义 4.9），特征空间可能具有超过一维的维度。

几何重数定义 4.11。设 $\lambda_{i}$ 是方阵 $\pmb{A}$ 的一个特征值。则 $\lambda_{i}$ 的几何重数是与 $\lambda_{i}$ 相关联的线性无关特征向量的数量。换句话说，它是由与 $\lambda_{i}$ 相关联的特征向量生成的特征空间的维数。

备注。一个特定特征值的几何重数必须至少为 1，因为每个特征值至少有一个关联的特征向量。一个特征值的几何重数不能超过其代数重数，但可能更低。$\diamondsuit$


### 示例 4.6

矩阵 $\pmb{A}=\left[\begin{array}{c c}{2}&{1}\\ {0}&{2}\end{array}\right]$ 有两个重复的特征值 $\lambda_{1}=\lambda_{2}=2$，代数重数为 2。然而，特征值只有一个不同的单位特征向量 $\pmb{x}_{1}=\left[^{1}_{0}\right]$，因此几何重数为 1。

### 二维图形直观理解

在几何学中，这种类型的沿轴剪切且保持面积不变的性质也被称为卡瓦列里等面积原理（Katz, 2004）。

让我们通过不同的线性映射来获得行列式、特征向量和特征值的一些直观理解。图 4.4 描述了五个变换矩阵 $A_{1},\ldots,A_{5}$ 及其对原点为中心的正方形网格点的影响：

$A_{1}={\left[\begin{array}{l l}{{\frac{1}{2}}}&{0}\\ {0}&{2}\end{array}\right]}$。两个特征向量的方向对应于 $\mathbb{R}^{2}$ 中的正交基向量，即两个主轴。垂直轴被拉伸了两倍（特征值 $\lambda_{1}=2$），水平轴被压缩了 $\frac{1}{2}$ 倍（特征值 $\lambda_{2}=\frac{1}{2}$）。该映射保持面积不变 $(\operatorname*{det}(A_{1})=1=2\cdot\frac{1}{2})$。

$A_{2}=\left[\begin{array}{c c}{{1}}&{{\frac{1}{2}}}\\ {{0}}&{{1}}\end{array}\right]$ 对应于沿水平轴剪切的映射，即如果点位于垂直轴的正半部分，则向右剪切，反之则向左剪切。该映射保持面积不变 $(\operatorname*{det}(A_{2})=1)$。特征值 $\lambda_{1}=1=\lambda_{2}$ 重复，特征向量共线（此处为了强调画在两个相反的方向）。这表明映射仅沿一个方向（水平轴）作用。

$A_{3}=\begin{array}{l l}{{\bigl[\cos\bigl({\frac{\pi}{6}}\bigr)\,}}&{{-\sin\bigl({\frac{\pi}{6}}\bigr)\,\bigr]}}\\ {{\bigl[\sin\bigl({\frac{\pi}{6}}\bigr)\,}}&{{\cos\bigl({\frac{\pi}{6}}\bigr)\,\bigr]}}\end{array}=\frac{1}{2}\left[\begin{array}{l l}{{\sqrt{3}}}&{{-1}}\\ {{1}}&{{\sqrt{3}}}\end{array}\right]$。矩阵 $A_{3}$ 将点逆时针旋转 $\frac{\pi}{6}$ 弧度（即 30 度），并且只有复数特征值，反映该映射是一个旋转（因此没有画出特征向量）。旋转必须保持体积不变，因此行列式为 1。有关旋转的更多细节，请参见第 3.9 节。

$A_{4}=\left[{1\atop-1}\right]^{}~{-1\atop1}$ 表示在标准基下的映射，将二维域压缩成一维。由于一个特征值为 0，对应于 $\lambda_{1}=0$ 的特征向量方向的空间被压缩，而正交的特征向量（红色）拉伸空间 $\lambda_{2}=2$ 倍。因此，图像的面积为 0。

$A_{5}=\left[\overset{\cdot}{1}_{\frac{1}{2}}\quad\overset{\cdot}{1}\right]$ 是剪切和拉伸映射，将空间拉伸了 75%。其行列式的绝对值为 $|\operatorname*{det}(A_{5})|=\frac{3}{4}$。它沿特征向量 $\lambda_{2}$ 方向拉伸空间 1.5 倍，沿正交的特征向量方向压缩 0.5 倍。

![](images/a8636c524da03f78a6163194da8fcb92988b2e008a612e13f639242e2f92884b.jpg)

图 4.4 行列式和特征空间。五个线性映射及其相关变换矩阵 $\pmb{A}_{i}\in\mathbb{R}^{2\times2}$ 将 400 个颜色编码的点 $\pmb{x}\in\mathbb{R}^{2}$（左列）投影到目标点 $\mathbf{\nabla}A_{i}\mathbf{x}$（右列）。中间列描绘了第一个特征向量，沿其关联的特征值 $\lambda_{1}$ 拉伸，第二个特征向量沿其特征值 $\lambda_{2}$ 拉伸。每一行描绘了五个变换矩阵 $\pmb{A}_{i}$ 中的一个相对于标准基的作用。


### 示例 4.7 (生物神经网络的特征谱)

图 4.5 Caenorhabditis elegans 神经网络 (Kaiser 和 Hilgetag, 2006)。(a) 对称化连接矩阵；(b) 特征谱。

![](images/40327e64b8f1799fcfbac10902881259d15cb4aeef5331af37ba0ff8b23e8b1d.jpg)

分析和学习网络数据的方法是机器学习方法的重要组成部分。理解网络的关键在于网络节点之间的连接性，特别是两个节点是否相互连接。在数据科学应用中，研究捕获这种连接性的矩阵通常很有用。

我们构建了 C.Elegans 蠕虫完整神经网络的连接/邻接矩阵 $\pmb{A}\in\mathbb{R}^{277\times277}$。每一行/列代表该蠕虫大脑中的 277 个神经元中的一个。连接矩阵 $\pmb{A}$ 中的元素 $a_{i j}$ 的值为 1，如果神经元 $i$ 通过突触与神经元 $j$ 通信；否则为 0。连接矩阵不是对称的，这意味着特征值可能不是实数值。因此，我们计算连接矩阵的对称化版本 $\begin{array}{r}{\pmb{A}_{s y m}:=\pmb{A}+\pmb{A}^{\top}}\end{array}$。这个新的矩阵 $A_{s y m}$ 如图 4.5(a) 所示，并且只有当两个神经元相互连接时（白色像素），矩阵中的元素 $a_{i j}$ 才是非零的，无论连接的方向如何。在图 4.5(b) 中，我们展示了 $A_{s y m}$ 的对应特征谱。水平轴显示按降序排列的特征值的索引。垂直轴显示对应的特征值。这种特征谱的 $S$ 形状对于许多生物神经网络是典型的。这种特征谱背后的机制是神经科学研究的一个活跃领域。


### 定理 4.12
矩阵 $\pmb{A}\in\mathbb{R}^{n\times n}$ 的特征向量 $\pmb{x}_{1},\ldots,\pmb{x}_{n}$，如果该矩阵有 $n$ 个不同的特征值 $\lambda_{1},\dots,\lambda_{n}$，那么这些特征向量是线性无关的。

该定理表明，具有 $n$ 个不同特征值的矩阵的特征向量可以构成 $\mathbb{R}^{n}$ 的一组基。

### 定义 4.13
如果一个 $n\times n$ 的方阵 $\pmb{A}\in\mathbb{R}^{n\times n}$ 拥有少于 $n$ 个线性无关的特征向量，则称该矩阵为**缺陷矩阵**。

一个非缺陷矩阵 $\pmb{A}\in\mathbb{R}^{n\times n}$ 不一定需要 $n$ 个不同的特征值，但它确实需要特征向量构成 $\mathbb{R}^{n}$ 的一组基。对于一个缺陷矩阵，其特征子空间的维数之和小于 $n$。具体来说，一个缺陷矩阵至少有一个特征值 $\lambda_{i}$，其代数重数 $m>1$，而几何重数小于 $m$。

**注释** 一个缺陷矩阵不能有 $n$ 个不同的特征值，因为不同的特征值对应的特征向量是线性无关的（定理 4.12）。$\diamondsuit$

### 定理 4.14
给定一个矩阵 $\pmb{A}\in\mathbb{R}^{m\times n}$，我们总是可以通过定义

$$
\begin{array}{r}{\boldsymbol{S}:=\boldsymbol{A}^{\intercal}\boldsymbol{A}\,.}\end{array}
$$
得到一个对称且半正定的矩阵 $\b{S}\in\mathbb{R}^{n\times n}$。

**注释** 如果 $\operatorname{rk}(A)=n$，则 $\pmb{S}:=\pmb{A}^{\top}\pmb{A}$ 是对称且正定的。

理解定理 4.14 的原因对于如何使用对称矩阵是有启发性的：对称性要求 $\bar{\boldsymbol{S}}=\boldsymbol{S}^{\intercal}$，通过插入 (4.36) 我们得到 $S=A^{\top}A=(A^{\top}A)^{\top}=S^{\top}$。此外，半正定性（第 3.2.3 节）要求 $\mathbf{x}^{\,\top}{\cal S}\mathbf{x}\;\geqslant\;0$，通过插入 (4.36) 我们得到 ${\pmb x}^{\top}{\pmb S}{\pmb x}\,=\,{\pmb x}^{\top}{\pmb A}^{\top}\bar{\pmb A}{\pmb x}\,=\,({\pmb x}^{\top}{\pmb A}^{\top})({\pmb A}{\pmb x})\,=$ $(A{\pmb x})^{\top}(A{\pmb x})\;\geqslant\;0$，因为点积计算的是平方和（这些平方和本身是非负的）。

### 定理 4.15 (谱定理)
如果 $\pmb{A}\in\mathbb{R}^{n\times n}$ 是对称的，则存在一个对应向量空间 $V$ 的标准正交基，该基由矩阵 $A$ 的特征向量组成，且每个特征值都是实数。

谱定理的一个直接推论是，对称矩阵 $\pmb{A}$ 的特征分解存在（具有实特征值），并且我们可以找到一个由特征向量组成的标准正交基，使得 $\pmb{A}=\pmb{P}\pmb{D}\pmb{P}^{\top}$，其中 $\pmb{D}$ 是对角矩阵，$\pmb{P}$ 的列向量是特征向量。


### 示例 4.8

考虑矩阵

$$
A=\left[{\begin{array}{rrr}
3 & 2 & 2 \\
2 & 3 & 2 \\
2 & 2 & 3
\end{array}}\right]\,.
$$

矩阵 $\pmb{A}$ 的特征多项式为

$$
p_{A}(\lambda)=-(\lambda-1)^{2}(\lambda-7)\,,
$$
因此我们得到特征值 $\lambda_{1} = 1$ 和 $\lambda_{2} = 7$，其中 $\lambda_{1}$ 是重特征值。按照计算特征向量的标准程序，我们得到特征空间

$$
E_{1}=\operatorname{span}[\underbrace{\left[{\begin{array}{c}{-1}\\ {1}\\ {0}\end{array}}\right]}_{=:x_{1}},\underbrace{\left[{\begin{array}{c}{-1}\\ {0}\\ {1}\end{array}}\right]}_{=:x_{2}}],\quad E_{7}=\operatorname{span}[\underbrace{\left[{\begin{array}{c}{1}\\ {1}\\ {1}\end{array}}\right]}_{=:x_{3}}].
$$

我们看到 $\pmb{x}_{3}$ 与 $\pmb{x}_{1}$ 和 $\pmb{x}_{2}$ 都正交。然而，由于 ${\pmb x}_{1}^{\top}{\pmb x}_{2} \neq 0$，它们不是正交的。谱定理（定理 4.15）表明存在一个正交基，但我们现在得到的基不是正交的。然而，我们可以构造一个正交基。

为了构造这样的基，我们利用了 $\pmb{x}_{1}$ 和 $\pmb{x}_{2}$ 是与同一特征值 $\lambda$ 相关联的特征向量这一事实。因此，对于任意 $\alpha, \beta \in \mathbb{R}$，有

$$
\pmb{A}(\alpha\pmb{x}_{1}+\beta\pmb{x}_{2})=\pmb{A}\pmb{x}_{1}\alpha+\pmb{A}\pmb{x}_{2}\beta=\lambda(\alpha\pmb{x}_{1}+\beta\pmb{x}_{2})\,,
$$
即，$\pmb{x}_{1}$ 和 $\pmb{x}_{2}$ 的任意线性组合也是与 $\lambda$ 相关联的 $\pmb{A}$ 的特征向量。Gram-Schmidt 算法（第 3.8.3 节）是一种从一组基向量中使用这样的线性组合迭代构造正交/标准正交基的方法。因此，即使 $\pmb{x}_{1}$ 和  $\pmb{x}_{2}$ 不是正交的，我们也可以应用 Gram-Schmidt 算法并找到与 $\lambda_{1} = 1$ 相关联的正交特征向量（并且与 $\scriptstyle{\mathcal{X}}_{3.}^{.}$ 正交）。在我们的示例中，我们将得到

$$
{\pmb x}_{1}^{\prime}=\left[\begin{array}{c}
-1 \\
1 \\
0
\end{array}\right],\quad{\pmb x}_{2}^{\prime}=\frac{1}{2}\left[\begin{array}{c}
-1 \\
-1 \\
2
\end{array}\right]\,,
$$

它们彼此正交，与 $\pmb{x}_{3}$ 正交，并且是与 $\lambda_{1}=1$ 相关联的 $\pmb{A}$ 的特征向量。

在我们结束对特征值和特征向量的讨论之前，将这些矩阵特征与行列式和迹的概念联系起来是有用的。

### 定理 4.16. 

矩阵 $\pmb{A} \in \mathbb{R}^{n \times n}$ 的行列式是其特征值的乘积，即

$$
\operatorname{det}(A)=\prod_{i=1}^{n}\lambda_{i}\,,
$$
其中 $\lambda_{i} \in \mathbb{C}$ 是 $\pmb{A}$ 的（可能重复的）特征值。

![](images/925b96f006a57b50c03a6444f06420f96143a58c66c6906bb83b38d83f37bacf.jpg)


### 定理 4.17

矩阵 $\pmb{A}\in\mathbb{R}^{n\times n}$ 的迹是其特征值之和，即

$$
\mathrm{tr}(\pmb{A})=\sum_{i=1}^{n}\lambda_{i}\,,
$$

其中 $\lambda_{i}\in\mathbb{C}$ 是 $\pmb{A}$ 的特征值（可能重复）。

让我们提供这两个定理的几何直观。考虑一个矩阵 $\pmb{A}\in\mathbb{R}^{2\times2}$，它具有两个线性独立的特征向量 ${\pmb x}_{1},{\pmb x}_{2}$。在这个例子中，我们假设 $(\pmb{x}_{1},\pmb{x}_{2})$ 是 $\mathbb{R}^{2}$ 的标准正交基（ONB），因此它们是正交的，并且它们所张成的正方形的面积为 1；见图 4.6。从第 4.1 节我们知道，行列式计算单位正方形在变换 $\pmb{A}$ 下面积的变化。在这个例子中，我们可以显式地计算面积的变化：使用 $\pmb{A}$ 映射特征向量给出向量 ${\pmb v}_{1}\,=\,A{\pmb x}_{1}\,=\,\lambda_{1}{\pmb x}_{1}$ 和 ${\pmb v}_{2}\,=\,{\pmb A}{\pmb x}_{2}\,=\,\lambda_{2}{\pmb x}_{2}$，即新的向量 ${\pmb v}_{i}$ 是特征向量 ${\pmb x}_{i}$ 的缩放版本，缩放因子是相应的特征值 $\lambda_{i}$。${\pmb v}_{1},{\pmb v}_{2}$ 仍然是正交的，并且它们所张成的矩形的面积是 $|\lambda_{1}\lambda_{2}|$。

由于 ${\pmb x}_{1},{\pmb x}_{2}$（在我们的例子中）是正交的，我们可以直接计算单位正方形的周长为 $2(1+1)$。使用 $\pmb{A}$ 映射特征向量创建一个矩形，其周长为 $2(|\lambda_{1}|+|\lambda_{2}|)$。因此，特征值的绝对值之和告诉我们单位正方形在变换矩阵 $\pmb{A}$ 下周长的变化。

### 示例 4.9 (Google 的 PageRank — 网页作为特征向量)

Google 使用矩阵 $\pmb{A}$ 的最大特征值对应的特征向量来确定网页在搜索中的排名。PageRank 算法的思想是由 Larry Page 和 Sergey Brin 在斯坦福大学于 1996 年开发的，该思想认为任何网页的重要性可以通过链接到它的网页的重要性来近似。为此，他们将所有网站写成一个巨大的有向图，显示哪些页面链接到哪些页面。PageRank 通过计算指向网页 $a_{i}$ 的页面数量来确定网页 $a_{i}$ 的权重（重要性）$x_{i}\,\geqslant\,0$。此外，PageRank 还考虑了链接到 $a_{i}$ 的网页的重要性。用户的导航行为通过该图的转移矩阵 $\pmb{A}$ 来建模，该矩阵告诉我们用户以多大的点击概率会到达不同的网站。矩阵 $\pmb{A}$ 具有如下性质：对于任何初始排名/重要性向量 ${\pmb x}$，序列 ${\pmb x},{\pmb A}{\pmb x},{\pmb A}^{2}{\pmb x},.\,.$ 收敛到一个向量 $\mathbf{\Psi}_{\pmb{x}}^{*}$。这个向量称为 PageRank，并满足 $A x^{*}=x^{*}$，即它是矩阵 $\pmb{A}$ 的特征向量（对应的特征值为 1）。在归一化 $\mathbf{\Psi}_{\mathbf{X}}^{*}$，使得 $\|\pmb{x}^{*}\|=1$ 后，我们可以将向量的条目解释为概率。更多关于 PageRank 的详细信息和不同视角可以在原始技术报告（Page et al., 1999）中找到。

## 4.3 Cholesky 分解

在机器学习中，我们经常遇到许多特殊类型的矩阵，这些矩阵有许多因式分解的方法。在正实数中，我们有平方根运算，它可以将数字分解为相同的组成部分，例如，$9 = 3 \cdot 3$。对于矩阵，我们需要小心地计算正数量的平方根运算。对于对称、正定矩阵（见第3.2.3节），我们可以选择多种平方根等价运算。Cholesky 分解/Cholesky 因式分解为对称、正定矩阵提供了一种实用的平方根等价运算。

**定理 4.18 (Cholesky 分解)**. 对称、正定矩阵 $\pmb{A}$ 可以分解为 $\pmb{A}=\pmb{L}\pmb{L}^{\top}$ 的形式，其中 $\pmb{L}$ 是一个下三角矩阵，其对角元素为正：

$$
{\left[\begin{array}{l l l}{a_{11}}&{\cdot\cdot\cdot}&{a_{1n}}\\ {\vdots}&{\cdot\cdot\cdot}&{\vdots}\\ {a_{n1}}&{\cdot\cdot\cdot}&{a_{nn}}\end{array}\right]}={\left[\begin{array}{l l l}{l_{11}}&{\cdot\cdot\cdot}&{0}\\ {\vdots}&{\cdot\cdot\cdot}&{\vdots}\\ {l_{n1}}&{\cdot\cdot\cdot}&{l_{nn}}\end{array}\right]}{\left[\begin{array}{l l l}{l_{11}}&{\cdot\cdot\cdot}&{l_{n1}}\\ {\vdots}&{\cdot\cdot\cdot}&{\vdots}\\ {0}&{\cdot\cdot\cdot}&{l_{nn}}\end{array}\right]}\ .
$$
Cholesky 因子 $\pmb{L}$ 被称为 $\pmb{A}$ 的 Cholesky 因子，并且 $\pmb{L}$ 是唯一的。


### 示例 4.10 (Cholesky 分解)

考虑一个对称且正定的矩阵 $\pmb{A}\,\in\,\mathbb{R}^{3\times3}$。我们感兴趣的是找到它的 Cholesky 分解 $\pmb{A}=\pmb{L}\pmb{L}^{\top}$，即

$$
\pmb{A}=\left[\begin{array}{ccc}
a_{11} & a_{21} & a_{31} \\
a_{21} & a_{22} & a_{32} \\
a_{31} & a_{32} & a_{33}
\end{array}\right]=\pmb{L}\pmb{L}^{\top}=\left[\begin{array}{ccc}
l_{11} & 0 & 0 \\
l_{21} & l_{22} & 0 \\
l_{31} & l_{32} & l_{33}
\end{array}\right]\left[\begin{array}{ccc}
l_{11} & l_{21} & l_{31} \\
0 & l_{22} & l_{32} \\
0 & 0 & l_{33}
\end{array}\right]\,.
$$

展开右边的乘积得到

$$
\pmb{A}=\left[\begin{array}{ccc}
l_{11}^{2} & l_{21}l_{11} & l_{31}l_{11} \\
l_{21}l_{11} & l_{21}^{2}+l_{22}^{2} & l_{31}l_{21}+l_{32}l_{22} \\
l_{31}l_{11} & l_{31}l_{21}+l_{32}l_{22} & l_{31}^{2}+l_{32}^{2}+l_{33}^{2}
\end{array}\right]\,.
$$

比较 (4.45) 式的左边和 (4.46) 式的右边，可以看出对角元素 $l_{i i}$ 有一个简单的模式：

$$
l_{11}=\sqrt{a_{11}}\,,\quad l_{22}=\sqrt{a_{22}-l_{21}^{2}}\,,\quad l_{33}=\sqrt{a_{33}-\left(l_{31}^{2}+l_{32}^{2}\right)}\,.
$$
同样地，对于对角线下方的元素 $(l_{i j}$，其中 $i>j)$，也有一个重复的模式：

$$
l_{21}=\frac{1}{l_{11}}a_{21}\,,\quad l_{31}=\frac{1}{l_{11}}a_{31}\,,\quad l_{32}=\frac{1}{l_{22}}(a_{32}-l_{31}l_{21})\,.
$$
因此，我们构造了任何对称且正定的 $3\times3$ 矩阵的 Cholesky 分解。关键的实现是，我们可以根据矩阵 $\pmb{A}$ 的值 $a_{i j}$ 和先前计算的 $l_{i j}$ 值，反向计算出 $L$ 的分量 $l_{i j}$。

Cholesky 分解是机器学习中数值计算的重要工具。在这里，对称正定矩阵需要频繁操作，例如，多元高斯变量的协方差矩阵（见第 6.5 节）是对称且正定的。该协方差矩阵的 Cholesky 分解允许我们从高斯分布中生成样本。它还允许我们对随机变量进行线性变换，这在计算深度随机模型中的梯度时得到了广泛应用，例如变分自动编码器（Jimenez Rezende 等，2014；Kingma 和 Welling，2014）。Cholesky 分解还允许我们高效地计算行列式。给定 Cholesky 分解 $\bar{A}=L L^{\top}$，我们知道 $\operatorname*{det}(A)\stackrel{.}{=}\operatorname*{det}(L)\operatorname*{det}(L^{\top})=\operatorname*{det}(L)^{2}$。由于 $\pmb{L}$ 是一个三角矩阵，行列式只是其对角线元素的乘积，即 $\begin{array}{r}{\operatorname*{det}(\pmb{A})\,=\,\prod_{i}l_{i i}^{2}}\end{array}$。因此，许多数值软件包使用 Cholesky 分解来提高计算效率。


## 4.4 对角化和约当标准型

对角矩阵是一种矩阵，其所有非对角线元素均为零，即它们的形式为

$$
D={\left[\begin{array}{ccc}
c_{1} & \cdot\cdot\cdot & 0 \\
\vdots & \cdot\cdot & \vdots \\
0 & \cdot\cdot\cdot & c_{n}
\end{array}\right]}\,.
$$

它们允许快速计算行列式、幂和逆矩阵。行列式是其对角线元素的乘积，矩阵幂 $D^{k}$ 由每个对角线元素的 $k$ 次幂给出，逆矩阵 ${\bf\bar{\cal D}}^{-1}$ 是其对角线元素的倒数，前提是所有对角线元素均非零。

在本节中，我们将讨论如何将矩阵转换为对角矩阵的形式。这是我们在第 2.7.2 节中讨论的基变换和第 4.2 节中讨论的特征值的重要应用。

回忆一下，如果存在一个可逆矩阵 $P$，使得 $D=P^{-1}AP$，则两个矩阵 $A$ 和 $D$ 是相似的（定义 2.22）。更具体地说，我们将研究与包含矩阵 $\pmb{A}$ 的特征值的对角矩阵相似的矩阵 $\pmb{A}$。

### 定义 4.19 (可对角化)。

矩阵 $\pmb{A}\in\mathbb{R}^{n\times n}$ 是可对角化的，如果它与一个对角矩阵相似，即存在一个可逆矩阵 $P\in\mathbb{R}^{n\times n}$，使得 $D=P^{-1}AP$。

在以下内容中，我们将看到对角化矩阵 $\pmb{A}\in\mathbb{R}^{n\times n}$ 是一种表示相同线性映射但使用另一个基的方法（见第 2.6.1 节），该基由矩阵 $\pmb{A}$ 的特征向量组成。

设 $\pmb{A}\in\mathbb{R}^{n\times n}$，$\lambda_{1},\dots,\lambda_{n}$ 是一组标量，$\pmb{p}_{1},\ldots,\pmb{p}_{n}$ 是 $\mathbb{R}^{n}$ 中的一组向量。我们定义 $P:=[\pmb{p}_{1},.\,.\,.\,,\pmb{p}_{n}]$，并设 ${D}\in\mathbb{R}^{n\times n}$ 是一个对角矩阵，其对角线元素为 $\lambda_{1},\dots,\lambda_{n}$。那么我们就可以证明

$$
AP=PD
$$
当且仅当 $\lambda_{1},\ldots,\lambda_{n}$ 是 $\pmb{A}$ 的特征值，$\pmb{p}_{1},\ldots,\pmb{p}_{n}$ 是 $\pmb{A}$ 的相应特征向量。


我们可以看到这个陈述成立，因为

$$
\begin{array}{l}
AP = A[p_{1}, \ldots, p_{n}] = [A p_{1}, \ldots, A p_{n}] \,, \\
PD = [p_{1}, \ldots, p_{n}] \left[\begin{array}{ccc}
\lambda_{1} & & 0 \\
& \ddots & \\
0 & & \lambda_{n}
\end{array}\right] = [\lambda_{1} p_{1}, \ldots, \lambda_{n} p_{n}] \,.
\end{array}
$$

因此，(4.50) 表明

$$
\begin{array}{l}{A\pmb{p}_{1}=\lambda_{1}\pmb{p}_{1}}\\ {\vdots}\\ {A\pmb{p}_{n}=\lambda_{n}\pmb{p}_{n}\,.}\end{array}
$$
因此，矩阵 $\pmb{P}$ 的列必须是 $\pmb{A}$ 的特征向量。

定义对角化要求 $\pmb{P} \in \mathbb{R}^{n \times n}$ 是可逆的，即 $P$ 具有满秩（定理 4.3）。这意味着我们需要有 $n$ 个线性无关的特征向量 $\pmb{p}_{1},\ldots,\pmb{p}_{n}$，即 $\pmb{p}_{i}$ 形成 $\textstyle\mathbb{R}^{n}$ 的一组基。

定理 4.20 (特征分解)。方阵 $\pmb{A}\in\mathbb{R}^{n\times n}$ 可以分解为

$$
\begin{array}{r}{\pmb{A}=\pmb{P}\pmb{D}\pmb{P}^{-1}\,,}\end{array}
$$
其中 $\b{P}\in\mathbb{R}^{n\times n}$ 且 $_D$ 是一个对角矩阵，其对角元素是 $A$ 的特征值，当且仅当 $A$ 的特征向量形成 $\textstyle\mathbb{R}^{n}$ 的一组基。

![](images/604c94d6bb251af94fe3d29eb57b2cb26d3b8083e7ec18029d992fd13c30113b.jpg)

### 定理 4.20 

表明，只有非缺陷矩阵才能被对角化，并且矩阵 $\pmb{P}$ 的列是 $\pmb{A}$ 的 $n$ 个特征向量。对于对称矩阵，我们可以得到特征值分解的更强结果。
### 定理 4.21. 

对称矩阵 $\b{S}\in\mathbb{R}^{n\times n}$ 总是可以对角化的。
### 定理 4.21

直接来源于谱定理 4.15。此外，谱定理表明我们可以找到 $\mathbb{R}^{n}$ 的一组正交基向量。这使得 $P$ 成为一个正交矩阵，使得 $\pmb{D}=\pmb{P}^{\top}\pmb{A}\bar{\pmb{P}}$。

**注释** 矩阵的 Jordan 标准型提供了一种分解方法，适用于缺陷矩阵（Lang, 1987），但超出了本书的范围。$\diamondsuit$

图 4.7  本征分解背后的直观理解，作为连续变换。从左上到左下：$P^{-1}$ 执行基变换（这里在 $\mathbb{R}^{2}$ 中绘制，并表示为类似旋转的操作），从标准基到本征基。从左下到右下：$\pmb{D}$ 沿重新映射的正交本征向量进行缩放，这里表示为一个圆被拉伸成椭圆。从右下到右上：$P$ 撤销基变换（表示为反向旋转），并恢复原始坐标系。

### 几何直观理解特征分解

我们可以将矩阵的特征分解解释如下（参见图4.7）：设 $\pmb{A}$ 是线性映射相对于标准基 $e_{i}$（蓝色箭头）的变换矩阵。$P^{-1}$ 执行从标准基到特征基的基变换。然后，对角矩阵 $_D$ 沿这些轴按特征值 $\lambda_{i}$ 缩放向量。最后，$P$ 将这些缩放后的向量变换回标准/规范坐标，得到 $\lambda_{i}\pmb{p}_{i}$。


### 示例 4.11 (特征值分解)

让我们计算矩阵 $A=\frac{1}{2}\left[2\begin{bmatrix}5 & -2 \\ -2 & 5\end{bmatrix}\right]$ 的特征值分解。步骤 1：计算特征值和特征向量。矩阵 $\pmb{A}$ 的特征多项式为

$$
\begin{array}{rl}
\det(A - \lambda I) &= \det\left(\left[\begin{array}{cc}
\frac{5}{2} - \lambda & -1 \\
-1 & \frac{5}{2} - \lambda
\end{array}\right]\right) \\
&= \left(\frac{5}{2} - \lambda\right)^2 - 1 \\
&= \lambda^2 - 5\lambda + \frac{21}{4} \\
&= \left(\lambda - \frac{7}{2}\right)\left(\lambda - \frac{3}{2}\right).
\end{array}
$$

因此，矩阵 $\pmb{A}$ 的特征值为 $\lambda_{1}=\frac{7}{2}$ 和 $\lambda_{2}=\frac{3}{2}$（特征多项式的根），对应的（归一化）特征向量通过

$$
\mathbf{A}\mathbf{p}_{1}=\frac{7}{2}\mathbf{p}_{1}\,,\quad\mathbf{A}\mathbf{p}_{2}=\frac{3}{2}\mathbf{p}_{2}\,.
$$
得到

$$
{\pmb p}_{1}=\frac{1}{\sqrt{2}}\left[\begin{array}{c}1 \\ -1\end{array}\right]\,,\quad{\pmb p}_{2}=\frac{1}{\sqrt{2}}\left[\begin{array}{c}1 \\ 1\end{array}\right]\,.
$$
步骤 2：检查存在性。特征向量 $\pmb{p}_{1},\pmb{p}_{2}$ 形成 $\mathbb{R}^{2}$ 的一组基。因此，矩阵 $\pmb{A}$ 可以对角化。

步骤 3：构造矩阵 $P$ 以对角化 $\pmb{A}$。我们将矩阵 $\pmb{A}$ 的特征向量收集到 $P$ 中，使得

$$
P=[\pmb{p}_{1},\;\pmb{p}_{2}]=\frac{1}{\sqrt{2}}\left[\begin{array}{cc}1 & 1 \\ -1 & 1\end{array}\right]\,.
$$
我们得到

$$
\begin{array}{r}{P^{-1}A P=\left[\begin{array}{cc}\frac{7}{2} & 0 \\ 0 & \frac{3}{2}\end{array}\right]=D\,.}
\end{array}
$$

图 4.7 可视化了矩阵 $A=\left[\begin{array}{cc}5 & -2 \\ -2 & 5\end{array}\right]$ 的特征值分解，作为一系列线性变换的序列。

等价地，我们得到（利用 $\boldsymbol{P}^{-1}=\boldsymbol{P}^{\intercal}$，因为在这个例子中特征向量 $\pmb{p}_{1}$ 和 $\pmb{p}_{2}$ 形成一个正交基）

$$
\underbrace{{\frac{1}{2}}\left[\begin{array}{cc}5 & -2 \\ -2 & 5\end{array}\right]}_{A}=\underbrace{{\frac{1}{\sqrt{2}}}\left[\begin{array}{cc}1 & 1 \\ -1 & 1\end{array}\right]}_{P}\underbrace{\left[\begin{array}{cc}\sqrt{2} & 0 \\ 0 & \sqrt{2}\end{array}\right]}_{D}\underbrace{{\frac{1}{\sqrt{2}}}\left[\begin{array}{cc}1 & -1 \\ 1 & 1\end{array}\right]}_{P^{-1}}\,.
$$
对角矩阵 $D$ 可以高效地进行幂运算。因此，我们可以通过特征值分解（如果存在）找到矩阵 $\pmb{A}\in\mathbb{R}^{n\times n}$ 的幂，使得

$$
\begin{array}{r}{A^{k}=(P D P^{-1})^{k}=P D^{k}P^{-1}\,.}
\end{array}
$$

计算 $D^{k}$ 是高效的，因为我们对每个对角元素单独应用该操作。

假设特征值分解 $\pmb{A}=\pmb{P}\pmb{D}\pmb{P}^{-1}$ 存在。则

$$
\operatorname{det}(A)=\operatorname{det}(P D P^{-1})=\operatorname{det}(P)\operatorname{det}(D)\operatorname{det}(P^{-1})
$$
$$
=\operatorname{det}(D)=\prod_{i}d_{i i}
$$
允许高效地计算矩阵 $\pmb{A}$ 的行列式。

特征值分解要求方阵。对一般矩阵进行分解是有用的。在下一节中，我们将介绍一种更一般的矩阵分解技术，即奇异值分解。


## 4.5 奇异值分解

矩阵的奇异值分解（SVD）是线性代数中一个核心的矩阵分解方法。它被称为“线性代数的基本定理”（Strang, 1993），因为它可以应用于所有矩阵，而不仅仅是方阵，并且总是存在。此外，正如我们将在下面探讨的那样，矩阵 $\pmb{A}$ 的 SVD 表示一个线性映射 $\Phi:V\,\rightarrow\,W$，量化了这两个向量空间之间的几何变化。我们推荐 Kalman (1996) 和 Roy 和 Banerjee (2014) 的工作，以获得 SVD 数学的更深入概述。

### 定理 4.22 (SVD 定理). 

设 $\pmb{A}\in\mathbb{R}^{m\times n}$ 是一个秩为 $r\in[0,\operatorname*{min}(m,n)]$ 的长方矩阵。矩阵 $\pmb{A}$ 的 SVD 是以下形式的分解：

奇异值分解，其中 $U\in\mathbb{R}^{m\times m}$ 是一个正交矩阵，其列向量为 $\pmb{u}_{i},\;i=1,\ldots,m$，$V\in\mathbb{R}^{n\times n}$ 是一个正交矩阵，其列向量为 $\pmb{v}_{j},\;j=1,\ldots,n$。此外，$\Sigma$ 是一个 $m\times n$ 的矩阵，其对角线元素为 $\Sigma_{i i}=\sigma_{i}\geqslant0$，非对角线元素为 $\Sigma_{i j}=0,\;i\neq j$。

对角线元素 $\sigma_{i},\;i=1,\dots,r$，称为奇异值，$\pmb{u}_{i}$ 称为左奇异向量，$\pmb{v}_{j}$ 称为右奇异向量。按照惯例，奇异值是按降序排列的，即 $\sigma_{1}\geqslant\sigma_{2}\geqslant\ldots\geqslant\sigma_{r}\geqslant0$。

奇异值矩阵 $\pmb{\Sigma}$ 是唯一的，但需要一些注意。观察到 $\pmb{\Sigma}\in\mathbb{R}^{m\times n}$ 是一个长方矩阵。特别地，$\pmb{\Sigma}$ 的大小与 $\pmb{A}$ 相同。这意味着 $\pmb{\Sigma}$ 有一个对角子矩阵，其中包含奇异值，并且需要额外的零填充。具体来说，如果 $m>n$，则矩阵 $\pmb{\Sigma}$ 在前 $n$ 行具有对角结构，其余部分由左奇异向量和右奇异向量组成。

奇异值矩阵

![](images/ee6dfe31e1543331e1e305a3b44d6f6c09531ffbbf497eb1495fe97bb5b9d593.jpg)

图 4.8 矩阵 $\pmb{A}\in\mathbb{R}^{3\times2}$ 的 SVD 的直观理解。从左上到左下：$V^{\top}$ 在 $\mathbb{R}^{2}$ 中执行基变换。从左下到右下：$\pmb{\Sigma}$ 在 $\mathbb{R}^{2}$ 和 $\mathbb{R}^{3}$ 之间进行缩放和映射。右下角的椭圆位于 $\mathbb{R}^{3}$ 中。第三个维度垂直于椭圆盘的表面。从右下到右上：$U$

$$
\pmb{\Sigma}=\left[\begin{array}{c c c}
{\sigma_{1}}&{0}&{0}\\ 
{0}&{\ddots}&{0}\\ 
{0}&{0}&{\sigma_{n}}\\ 
{0}&{\ldots}&{0}\\ 
{\vdots}&&{\vdots}\\ 
{0}&{\ldots}&{0}
\end{array}\right]\;.
$$

如果 $m<n$，矩阵 $\pmb{\Sigma}$ 在前 $m$ 列具有对角结构，其余列由 0 组成：

$$
\pmb{\Sigma} = \left[\begin{matrix}
\sigma_{1} & 0 & 0 & 0 & \cdots & 0 \\ 
0 & \ddots & 0 & \vdots & & \vdots \\ 
0 & 0 & \sigma_{m} & 0 & \cdots & 0
\end{matrix}\right]\,.
$$

**注释**. 任何矩阵 $\pmb{A}\in\mathbb{R}^{m\times n}$ 都存在 SVD。

### 4.5.1 SVD 的几何直观

SVD 为描述变换矩阵 $\pmb{A}$ 提供了几何直观。接下来，我们将讨论 SVD 如何通过在基底上执行一系列线性变换来实现。在示例 4.12 中，我们将 SVD 的变换矩阵应用于 $\mathbb{R}^{2}$ 中的一组向量，这使我们能够更清晰地可视化每个变换的效果。

矩阵的 SVD 可以解释为一个线性映射 $\Phi:\mathbb{R}^{n}\rightarrow\mathbb{R}^{m}$ 的分解，分解为三个操作；见图 4.8。SVD 的直观理解在表面上与我们的特征值分解直观理解具有相似的结构，见图 4.7：从广义上讲，SVD 通过 $V^{\top}$ 进行基底变换，然后通过奇异值矩阵 $\pmb{\Sigma}$ 进行缩放和维度的增加（或减少）。最后，它通过 $U$ 进行第二次基底变换。SVD 涉及许多重要的细节和注意事项，因此我们将更详细地回顾我们的直观理解。

假设我们给定一个线性映射 $\Phi$ 的变换矩阵：$\mathbb{R}^{n}\rightarrow\mathbb{R}^{m}$，相对于标准基底 $B$ 和 $C$。此外，假设 $\textstyle\mathbb{R}^{n}$ 的第二个基底 $\tilde{B}$ 和 $\mathbb{R}^{m}$ 的第二个基底 $\tilde{C}$。那么

1. 矩阵 $V$ 在域 $\textstyle\mathbb{R}^{n}$ 中从基底 $\tilde{B}$（由图 4.8 左上角的红色和橙色向量 $\scriptstyle{\mathbf{v}}_{1}$ 和 $\scriptstyle{\mathbf{v}}_{2}$ 表示）变换到标准基底 $B$。$V^{\top}=V^{-1}$ 从 $B$ 变换到 $\tilde{B}$。红色和橙色向量现在与图 4.8 左下角的正交基底对齐。

回顾基底变换（第 2.7.2 节）、正交矩阵（定义 3.8）和正交基底（第 3.5 节）是有用的。

2. 在将坐标系统变更为 $\tilde{B}$ 后，$\pmb{\Sigma}$ 通过奇异值 $\sigma_{i}$ 对新坐标进行缩放（并添加或删除维度），即 $\pmb{\Sigma}$ 是 $\Phi$ 相对于 $\tilde{B}$ 和 $\tilde{C}$ 的变换矩阵，由图 4.8 右下角的红色和橙色向量表示，它们被拉伸并位于 $e_{\mathrm{1}\mathbf{\Gamma}}\mathbf{-}\mathbf{e}_{\mathrm{2}}$ 平面中，该平面现在嵌入在第三个维度中。

3. $U$ 在陪域 $\mathbb{R}^{m}$ 中从基底 $\tilde{C}$ 变换到 $\mathbb{R}^{m}$ 的标准基底，由图 4.8 右上角的红色和橙色向量表示，它们从 $e_{\mathrm{1}^{-}}e_{\mathrm{2}}$ 平面中旋转出来。

SVD 表达了域和陪域中的基底变换。这与特征值分解不同，特征值分解在同一个向量空间中操作，其中相同的基底变换被应用并随后取消。使 SVD 特别的地方在于这两个不同的基底通过奇异值矩阵 $\pmb{\Sigma}$ 同时链接在一起。

### 4.5.2 奇异值分解的构造

接下来我们将讨论为什么奇异值分解（SVD）存在，并展示如何详细计算它。一般矩阵的SVD与方阵的特征值分解有一些相似之处。

注释。比较SPD矩阵的特征值分解

$$
\pmb{S}=\pmb{S}^{\top}=\pmb{P D P}^{\top}
$$

与相应的SVD

$$
\boldsymbol{S}=\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}\,.
$$
如果我们设置

$$
U=P=V\,,\quad D=\Sigma\,,
$$
我们会看到SPD矩阵的SVD就是它们的特征值分解。

在接下来的内容中，我们将探讨为什么定理4.22成立以及如何构造SVD。计算矩阵$\pmb{A}\,\in\,\mathbb{R}^{m\times n}$的奇异值等价于找到两个正交基集$U\;=\;(\pmb{u}_{1},.\,.\,.\,,\pmb{u}_{m})$和$V\,=$$(\pmb{v}_{1},\dots,\pmb{v}_{n})$，分别对应于$\mathbb{R}^{m}$的共域和$\mathbb{R}^{n}$的域。从这些有序基中，我们将构造矩阵$U$和$V$。

我们的计划是首先构造右奇异向量的正交集$\pmb{v}_{1},\ldots,\pmb{v}_{n}\in\mathbb{R}^{n}$，然后构造左奇异向量的正交集$\pmb{u}_{1},\dots,\pmb{u}_{m}\in\mathbb{R}^{m}$。接下来，我们将这两个正交集联系起来，并要求$\mathbf{v}_{i}$的正交性在$\pmb{A}$的变换下保持不变。这是很重要的，因为我们知道$\pmb{A}\pmb{v}_{i}$形成的是一组正交向量。然后，我们将这些图像通过标量因子进行归一化，这些因子将被证明是奇异值。

让我们从构造右奇异向量开始。谱定理（定理4.15）告诉我们，对称矩阵的特征向量构成一个正交基（ONB），这也意味着它可以被对角化。此外，从定理4.14中，我们可以总是构造一个对称且正定的矩阵$\pmb{A}^{\top}\pmb{A}~\in~\mathbb{R}^{n\times n}$，从矩形矩阵$\textbf{\textit{A}}\in$ $\mathbb{R}^{m\times n}$。因此，我们总是可以对角化$\pmb{A}^{\top}\pmb{A}$并得到

$$
\begin{array}{r}{\pmb{A}^{\top}\pmb{A}=\pmb{P}\pmb{D}\pmb{P}^{\top}=\pmb{P}\left[\begin{array}{l l l}{\lambda_{1}}&{\cdot\cdot\cdot}&{0}\\ {\vdots}&{\ddots}&{\vdots}\\ {0}&{\cdot\cdot\cdot}&{\lambda_{n}}\end{array}\right]\pmb{P}^{\top}\,,}\end{array}
$$
其中$P$是一个正交矩阵，由正交特征基组成。$\lambda_{i}\,\geqslant\,0$是$\mathbf{\bar{A}}^{\top}A$的特征值。假设$\pmb{A}$的SVD存在，并将（4.64）代入（4.71）。这得到

$$
\begin{array}{r}{\pmb{A}^{\top}\pmb{A}=(\pmb{U}\pmb{\Sigma}\pmb{V}^{\top})^{\top}(\pmb{U}\pmb{\Sigma}\pmb{V}^{\top})=\pmb{V}\pmb{\Sigma}^{\top}\pmb{U}^{\top}\pmb{U}\pmb{\Sigma}\pmb{V}^{\top}\;,}\end{array}
$$
其中$U,V$是正交矩阵。因此，由于$\pmb{U}^{\top}\pmb{U}=\pmb{I}$，我们得到

$$
\begin{array}{r}{\pmb{A}^{\top}\pmb{A}=\pmb{V}\pmb{\Sigma}^{\top}\pmb{\Sigma}\pmb{V}^{\top}=\pmb{V}\left[\begin{array}{l l l}{\sigma_{1}^{2}}&{0}&{0}\\ {0}&{\ddots}&{0}\\ {0}&{0}&{\sigma_{n}^{2}}\end{array}\right]\pmb{V}^{\top}\,.}\end{array}
$$
现在比较（4.71）和（4.73），我们得到

$$
\begin{array}{l}{{\pmb{V}^{\top}=\pmb{P}^{\top}\,,}}\\ {{\quad\sigma_{i}^{2}=\lambda_{i}\,.}}\end{array}
$$
因此，$\pmb{A}^{\top}\pmb{A}$的特征向量组成$P$的向量是$\pmb{A}$的右奇异向量（见（4.74））。$\pmb{A}^{\top}\pmb{A}$的特征值是$\pmb{\Sigma}$的平方奇异值（见（4.75））。


为了获得左奇异向量 $U$，我们遵循类似的步骤。我们首先计算对称矩阵 $\boldsymbol{A}\boldsymbol{A}^{\intercal}\in\mathbb{R}^{m\times m}$（而不是之前的 $\pmb{A}^{\top}\pmb{A}\in\mathbb{R}^{n\times n}$）。$A$ 的SVD给出

$$
\begin{array}{r l}
&A A^{\top} = (U \Sigma V^{\top})(U \Sigma V^{\top})^{\top} \\
&\qquad = (U \Sigma V^{\top})(V \Sigma^{\top} U^{\top}) \\
&\qquad = U \Sigma V^{\top} V \Sigma^{\top} U^{\top} \\
&\qquad = U \Sigma (\Sigma^{\top}) U^{\top} \\
&\qquad = U \Sigma \Sigma^{\top} U^{\top} \\
&\qquad = U \left[\begin{array}{ccc}
\sigma_1^2 & & \\
& \ddots & \\
& & \sigma_m^2
\end{array}\right] U^{\top}
\end{array}
$$


谱定理告诉我们 $A A^{\top}\,=\,S D S^{\top}$ 可以对角化，并且我们可以找到 $A A^{\top}$ 的一组正交归一化特征向量，这些特征向量被收集在 $s$ 中。$A A^{\top}$ 的正交归一化特征向量是左奇异向量 $U$，并形成SVD的共域中的正交归一化基。

接下来的问题是矩阵 $\pmb{\Sigma}$ 的结构。由于 $A A^{\top}$ 和 $\pmb{A}^{\top}\pmb{A}$ 具有相同的非零特征值（见第106页），SVD中这两种情况的 $\pmb{\Sigma}$ 矩阵的非零项必须相同。

最后一步是将我们迄今为止接触的所有部分连接起来。我们有一个正交归一化的右奇异向量集合 $V$。为了完成SVD的构造，我们需要将它们与正交归一化的向量 $U$ 连接起来。为了达到这个目标，我们利用了 $\mathbf{\delta}_{\mathbf{\ell}i}$ 在 $\pmb{A}$ 下的像必须正交的性质。我们可以通过使用第3.4节的结果来证明这一点。我们要求 $\pmb{A}\pmb{v}_{i}$ 和 $\pmb{A}\pmb{v}_{j}$ 之间的内积对于 $i \neq j$ 必须为0。对于任意两个正交特征向量 $\mathbf{\boldsymbol{v}}_{i}, \mathbf{\boldsymbol{v}}_{j}, i \neq j$，有

$$
(A{\pmb v}_{i})^{\top}(A{\pmb v}_{j})={\pmb v}_{i}^{\top}(A^{\top}A){\pmb v}_{j}={\pmb v}_{i}^{\top}(\lambda_{j}{\pmb v}_{j})=\lambda_{j}{\pmb v}_{i}^{\top}{\pmb v}_{j}=0\,.
$$
对于 $m \geqslant r$ 的情况，$\{\pmb{A}\pmb{v}_{1}, \dots, \pmb{A}\pmb{v}_{r}\}$ 是 $\mathbb{R}^{m}$ 中的一个 $r$ 维子空间的基。

为了完成SVD的构造，我们需要正交归一化的左奇异向量：我们归一化右奇异向量 $\pmb{A}\pmb{v}_{i}$ 的像，并得到

$$
\mathbf{\delta}\mathbf{u}_{i}:=\frac{\mathbf{\delta}A\mathbf{\delta}v_{i}}{\|A\mathbf{\delta}v_{i}\|}=\frac{1}{\sqrt{\lambda_{i}}}A\mathbf{\delta}v_{i}=\frac{1}{\sigma_{i}}A\mathbf{\delta}v_{i}\,,
$$
其中最后一个等式来自（4.75）和（4.76b），表明 $A A^{\top}$ 的特征值满足 $\sigma_{i}^{2}=\lambda_{i}$。

因此，$\pmb{A}^{\top}\pmb{A}$ 的特征向量，即我们已知的右奇异向量 $\mathbf{v}_{i}$，以及它们在 $\pmb{A}$ 下的归一化像，即左奇异向量 $\mathbf{u}_{i}$，形成了两个自洽的正交归一化基，通过奇异值矩阵 $\pmb{\Sigma}$ 连接起来。

让我们重新排列（4.78）以获得奇异值方程

$$
A{\pmb v}_{i}=\sigma_{i}{\pmb u}_{i}\,,\quad i=1,.\,.\,.\,,r\,.
$$
这个方程与特征值方程（4.25）非常相似，但左右两边的向量不是相同的。

对于 $n < m$ 的情况，（4.79）仅对 $i \leqslant n$ 成立，但（4.79）对 $i > n$ 的 $\mathbf{u}_{i}$ 没有说明。然而，我们通过构造知道它们是正交归一化的。相反，对于 $m < n$ 的情况，（4.79）仅对 $i \leqslant m$ 成立。对于 $i > m$，我们有 $\mathbf{\nabla}A\mathbf{v}_{i}=\mathbf{0}$，并且我们仍然知道 $\mathbf{\delta}v_{i}$ 形成一个正交归一化的集合。这意味着SVD还提供了 $\pmb{A}$ 的核（零空间）的正交归一化基，即向量集合 $_{_{\pmb{x}}}$，满足 $\mathbf{\nabla}A\mathbf{x}=\mathbf{0}$（见第2.7.3节）。

将 $\mathbf{\delta}_{\mathbf{\ell}i}$ 作为 $V$ 的列，将 $\mathbf{u}_{i}$ 作为 $U$ 的列，得到

$$
A V=U\Sigma\,,
$$
其中 $\pmb{\Sigma}$ 的维度与 $\pmb{A}$ 相同，并且对于行 $q, \ldots, r$ 具有对角结构。因此，右乘 $V^{\top}$ 得到 $\pmb{A}=\pmb{U}\pmb{\Sigma}\pmb{V}^{\top}$，这是 $\pmb{A}$ 的SVD。


### 示例 4.13 (计算SVD)

让我们找到矩阵的奇异值分解

$$
\begin{array}{r}{\pmb{A}=\left[\begin{array}{l l l}{1}&{0}&{1}\\ {-2}&{1}&{0}\end{array}\right]\,.}\end{array}
$$
SVD要求我们计算右奇异向量 $\pmb{v}_{j}$，奇异值 $\sigma_{k}$，以及左奇异向量 $\mathbf{\mathbf{\delta}}\mathbf{\mathbf{u}}_{i}$。

步骤1：右奇异向量作为 $\pmb{A}^{\top}\pmb{A}$ 的特征基。我们首先计算

$$
\pmb{A}^{\top}\pmb{A}=\left[\begin{array}{l l}{1}&{-2}\\ {0}&{1}\\ {1}&{0}\end{array}\right]\left[\begin{array}{l l l}{1}&{0}&{1}\\ {-2}&{1}&{0}\end{array}\right]=\left[\begin{array}{l l l}{5}&{-2}&{1}\\ {-2}&{1}&{0}\\ {1}&{0}&{1}\end{array}\right]\,.
$$
我们通过 $\pmb{A}^{\top}\pmb{A}$ 的特征值分解来计算奇异值和右奇异向量 $\pmb{v}_{j}$，其形式为

我们得到右奇异向量为矩阵 $P$ 的列，即

$$
V=P=\left[\begin{array}{c c c}{{\frac{5}{\sqrt{30}}}}&{{0}}&{{\frac{-1}{\sqrt{6}}}}\\ {{\frac{-2}{\sqrt{30}}}}&{{\frac{1}{\sqrt{5}}}}&{{\frac{-2}{\sqrt{6}}}}\\ {{\frac{1}{\sqrt{30}}}}&{{\frac{2}{\sqrt{5}}}}&{{\frac{1}{\sqrt{6}}}}\end{array}\right]\;.
$$
步骤2：奇异值矩阵。

由于奇异值 $\sigma_{i}$ 是 $\pmb{A}^{\top}\pmb{A}$ 的特征值的平方根，我们直接从 $_D$ 中得到它们。由于 $\operatorname{rk}(A)=2$，只有两个非零奇异值：$\sigma_{1}=\sqrt{6}$ 和 $\sigma_{2}=1$。奇异值矩阵必须与 $\pmb{A}$ 有相同的大小，我们得到

$$
\pmb{\Sigma} = \begin{bmatrix}
\sqrt{6} & 0 & 0 \\
0 & 1 & 0
\end{bmatrix}
$$


步骤3：左奇异向量作为右奇异向量的归一化图像。

我们通过计算右奇异向量在 $\pmb{A}$ 下的图像并将其归一化来找到左奇异向量，即除以相应的奇异值。我们得到

$$
\begin{array}{l}{{\displaystyle{\pmb u}_{1}=\frac{1}{\sigma_{1}}{\pmb A}{\pmb v}_{1}=\frac{1}{\sqrt{6}}\left[\begin{array}{l l l}{{1}}&{{0}}&{{1}}\\ {{-2}}&{{1}}&{{0}}\end{array}\right]\left[\frac{\frac{5}{\sqrt{3}0}}{\frac{-2}{\sqrt{3}0}}\right]=\left[\begin{array}{l}{{\frac{1}{\sqrt{5}}}}\\ {{-\frac{2}{\sqrt{5}}}}\end{array}\right]\,,}}\\ {{\displaystyle{\pmb u}_{2}=\frac{1}{\sigma_{2}}{\pmb A}{\pmb v}_{2}=\frac{1}{1}\left[\begin{array}{l l l}{{1}}&{{0}}&{{1}}\\ {{-2}}&{{1}}&{{0}}\end{array}\right]\left[\begin{array}{l}{{0}}\\ {{\frac{1}{\sqrt{5}}}}\\ {{\frac{2}{\sqrt{5}}}}\end{array}\right]=\left[\begin{array}{l}{{\frac{2}{\sqrt{5}}}}\\ {{\frac{1}{\sqrt{5}}}}\end{array}\right]\,,}}\\ {{\displaystyle{\pmb U}=[{\pmb u}_{1},{\pmb u}_{2}]=\frac{1}{\sqrt{5}}\left[\begin{array}{l l}{{1}}&{{2}}\\ {{-2}}&{{1}}\end{array}\right]\,.}}\end{array}
$$
注意，在计算机上，这里展示的方法具有较差的数值行为，通常不通过 $\pmb{A}^{\top}\pmb{A}$ 的特征值分解来计算 $\pmb{A}$ 的SVD。


### 4.5.3 特征值分解与奇异值分解

让我们考虑特征值分解 $A=P D P^{-1}$ 和奇异值分解 $A=U \boldsymbol{\Sigma} V^{\top}$，并回顾过去章节中的核心元素。

对于任何矩阵 $\mathbb{R}^{m \times n}$，奇异值分解总是存在。特征值分解仅定义为方阵 $\mathbb{R}^{n \times n}$，并且只有当我们能找到 $\mathbb{R}^{n}$ 的一组特征向量时才存在。特征值分解矩阵 $P$ 中的向量不一定正交，即基变换不是简单的旋转和平移。另一方面，奇异值分解矩阵 $U$ 和 $V$ 中的向量是正交的，因此它们确实代表了旋转。特征值分解和奇异值分解都是三个线性映射的组合：1. 域中的基变换 2. 每个新基向量的独立缩放和从域到陪域的映射 3. 陪域中的基变换

![](images/5dbefd2b69b9019190e308ace6a54219e3bd13fbf4b776fdb2d82fa6eba5e1fc.jpg)

图4.10 三个人对四部电影的评分及其奇异值分解。

特征值分解和奇异值分解之间的一个关键区别在于，奇异值分解中域和陪域可以是不同维度的向量空间。

在奇异值分解中，左奇异向量矩阵 $U$ 和右奇异向量矩阵 $V$ 通常不是彼此的逆（它们在不同的向量空间中执行基变换）。在特征值分解中，基变换矩阵 $P$ 和 $\bar{P}^{-1}$ 是彼此的逆。在奇异值分解中，对角矩阵 $\boldsymbol{\Sigma}$ 中的元素都是实数且非负，这在特征值分解的对角矩阵中通常不是真的。特征值分解和奇异值分解通过它们的投影紧密相关：

- $\boldsymbol{A}$ 的左奇异向量是 $A A^{\top}$ 的特征向量。
- $\boldsymbol{A}$ 的右奇异向量是 $\boldsymbol{A}^{\top} \boldsymbol{A}$ 的特征向量。
- $\boldsymbol{A}$ 的非零奇异值是 $A A^{\top}$ 和 $\boldsymbol{A}^{\top} \boldsymbol{A}$ 的非零特征值的平方根。

对于对称矩阵 $\boldsymbol{A} \in \mathbb{R}^{n \times n}$，特征值分解和奇异值分解是相同的，这可以从谱定理4.15得出。


### 示例 4.14（在电影评分和消费者中寻找结构）

让我们通过分析有关人们及其偏好电影的数据，为SVD添加一个实际的解释。考虑三个观众（阿里、贝特丽丝、查恩德拉）对四部不同电影（《星球大战》、《银翼杀手》、《天使爱美丽》、《高卢英雄传》）的评分。他们的评分值在0（最差）和5（最好）之间，并编码在一个数据矩阵$\pmb{A}\,\in\,\mathbb{R}^{4\times3}$中，如图4.10所示。每一行代表一部电影，每一列代表一个用户。因此，每部电影的评分列向量，对应每个观众的是$\pmb{x}_{\mathrm{Ali}}$，$\pmb{x}_{\mathrm{Beatrix}}$，$\pmb{x}_{\mathrm{Chandra}}$。

使用SVD对矩阵$\pmb{A}$进行因式分解为我们提供了一种捕捉人们如何对电影进行评分的关系的方法，特别是如果存在一种将哪些人喜欢哪些电影联系起来的结构。将SVD应用于我们的数据矩阵$\pmb{A}$会做出一些假设：

1. 所有观众都使用相同的线性映射一致地对电影进行评分。

2. 评分中没有错误或噪声。

3. 我们将左奇异向量$\mathbf{\mathbf{\mathit{u}}}_{i}$解释为典型的电影，将右奇异向量$\pmb{v}_{j}$解释为典型的观众。

这两个“空间”只有在数据本身涵盖了足够多样化的观众和电影时才有意义。

我们假设任何观众的特定电影偏好都可以表示为$\pmb{v}_{j}$的线性组合。同样，任何电影的受欢迎程度也可以表示为$\mathbf{\mathbf{\mathit{u}}}_{i}$的线性组合。因此，SVD域中的向量可以解释为“空间”中的典型观众，而SVD共域中的向量则可以解释为“空间”中的典型电影。让我们检查我们的电影-用户矩阵的SVD。第一个左奇异向量$\pmb{u}_{1}$对两部科幻电影具有较大的绝对值，并且具有较大的第一个奇异值（图4.10中的红色阴影）。因此，这将一种类型的用户与一组特定的电影（科幻主题）联系起来。同样，第一个右奇异向量$\scriptstyle{\mathbf{v}}_{1}$对阿里和贝特丽丝具有较大的绝对值，他们对科幻电影给出了高评分（图4.10中的绿色阴影）。这表明$\scriptstyle{\mathbf{v}}_{1}$反映了科幻爱好者的概念。


同样地，$\pmb{u}_{2}$ 似乎捕捉到了法国艺术电影的主题，而 $\scriptstyle{\mathbf{v}}_{2}$ 表明 Chandra 接近于这种电影的理想化爱好者。一个理想化的科幻电影爱好者是纯粹主义者，只喜欢科幻电影，因此一个科幻电影爱好者 $\scriptstyle{\mathbf{v}}_{1}$ 对除科幻主题外的所有事物都给出零评分——这种逻辑由奇异值矩阵 $\pmb{\Sigma}$ 的对角子结构所暗示。因此，一部特定的电影可以通过其如何分解（线性地）为典型电影来表示。同样地，一个人也可以通过其如何分解（通过线性组合）为电影主题来表示。

简要讨论奇异值分解（SVD）的术语和约定是有益的，因为文献中使用了不同的版本。尽管这些差异可能会令人困惑，但数学本身是不变的。

完整的 SVD 为了方便表示和抽象，我们使用一种 SVD 表示法，其中 SVD 被描述为具有两个正方形的左奇异向量矩阵和右奇异向量矩阵，但奇异值矩阵是非正方形的。我们的定义（4.64）对于 SVD 有时被称为完整的 SVD。一些作者以稍微不同的方式定义 SVD，并专注于正方形的奇异矩阵。对于 $\pmb{A}\in\mathbb{R}^{m\times n}$ 和 $m\geqslant n$，

$$
\mathbf{\Pi}_{m\times n}^{A}=\mathbf{\Pi}_{m\times n}\mathbf{\Pi}_{n\times n}^{\sum}\mathbf{\Pi}_{n\times n}^{V^{\top}}.
$$
有时这种形式被称为简化 SVD（例如，Datta (2010)）简化 SVD 或 SVD（例如，Press et al. (2bk07)）。这种替代格式只是改变了矩阵的构造方式，但保持了 SVD 的数学结构不变。这种替代表示法的便利之处在于 $\pmb{\Sigma}$ 是对角的，就像特征值分解一样。

在第 4.6 节中，我们将学习使用 SVD 的矩阵近似技术，这有时也被称为截断 SVD。截断 SVD 可以定义秩为 $\cdot r$ 的矩阵 $\pmb{A}$ 的 SVD，使得 $U$ 是一个 $m\times r$ 矩阵，$\pmb{\Sigma}$ 是一个 $r\times r$ 的对角矩阵，$V$ 是一个 $r\times n$ 矩阵。这种构造与我们的定义非常相似，并确保对角矩阵 $\pmb{\Sigma}$ 的对角线上的所有条目都是非零的。这种替代表示法的主要便利之处在于 $\pmb{\Sigma}$ 是对角的，就像特征值分解一样。

假设 SVD 只适用于 $m\times n$ 矩阵且 $m>n$ 的情况在实际应用中是不必要的。当 $m < n$ 时，SVD 分解将产生 $\pmb{\Sigma}$ 有更多零列而不是零行，并且因此奇异值 $\sigma_{m+1},\cdot\cdot\cdot,\sigma_{n}$ 为 0。

SVD 在机器学习的各种应用中都有使用，从曲线拟合中的最小二乘问题到求解线性方程组。这些应用利用了 SVD 的各种重要性质，包括其与矩阵秩的关系以及其能够用较低秩矩阵近似给定秩矩阵的能力。用 SVD 替换矩阵通常具有使计算更稳健地抵抗数值舍入误差的优势。正如我们将在下一节中探讨的那样，SVD 能够以一种原则性的方式用“更简单”的矩阵近似矩阵，这为从降维和主题建模到数据压缩和聚类的机器学习应用打开了大门。

## 4.6 矩阵近似

我们考虑了SVD作为将矩阵 $\pmb{A}=\pmb{U}\pmb{\Sigma}\pmb{V}^{\top}\in\mathbb{R}^{m\times n}$ 分解为三个矩阵乘积的方法，其中 $U\in\mathbb{R}^{m\times m}$ 和 $V\in\mathbb{R}^{n\times n}$ 是正交矩阵，$\Sigma$ 包含其主对角线上的奇异值。我们不再进行完整的SVD分解，而是研究如何利用SVD将矩阵 $\pmb{A}$ 表示为一些更简单的（低秩）矩阵 $\pmb{A}_{i}$ 的和，这种表示方法比完整的SVD计算起来更便宜。

我们构造一个秩为1的矩阵 $\pmb{A}_{i}\in\mathbb{R}^{m\times n}$ 为

$$
\begin{array}{r}{\mathbf{A}_{i}:=\mathbf{u}_{i}\mathbf{v}_{i}^{\top}\,,}\end{array}
$$
它是 $U$ 和 $V$ 的第 $i$ 个正交列向量的外积。图4.11显示了巨石阵的一张图片，该图片可以由矩阵 $\pmb{A}\in\mathbb{R}^{1432\times1910}$ 表示，以及一些外积 $A_{i}$，如（4.90）定义的那样。

![](images/ddd8225259fc782718d3492c9f28112ab53fdc4df719fdc50b36b422795ed675.jpg)

一个秩为 $r$ 的矩阵 $\pmb{A}\in\mathbb{R}^{m\times n}$ 可以表示为秩为1的矩阵 $A_{i}$ 的和，即

$$
\pmb{A}=\sum_{i=1}^{r}\sigma_{i}\pmb{u}_{i}\pmb{v}_{i}^{\top}=\sum_{i=1}^{r}\sigma_{i}\pmb{A}_{i}\,,
$$
其中外积矩阵 $A_{i}$ 由第 $i$ 个奇异值 $\sigma_{i}$ 权重。我们可以看到为什么（4.91）成立：奇异值矩阵 $\pmb{\Sigma}$ 的对角结构只乘以匹配的左奇异向量和右奇异向量 $\mathbf{\mathit{u}}_{i}\mathbf{\mathit{v}}_{i}^{\top}$，并按相应的奇异值 $\sigma_{i}$ 进行缩放。所有项 $\pmb{\Sigma}_{i j}\pmb{u}_{i}\pmb{v}_{j}^{\top}$ 对于 $i\neq j$ 都会消失，因为 $\pmb{\Sigma}$ 是一个对角矩阵。任何项 $i>r$ 都会消失，因为相应的奇异值为0。


在 (4.90) 中，我们介绍了秩为 1 的矩阵 $A_{i}$。我们将 $r$ 个独立的秩为 1 的矩阵相加，得到秩为 $\cdot r$ 的矩阵 $\pmb{A}$；见 (4.91)。如果求和不包括所有矩阵 $A_{i}$，$i=1,.\,.\,.,r$，而是只到一个中间值 $k<r$，我们得到一个秩为 $\cdot k$ 的近似矩阵

$$
\widehat{\pmb{A}}(k):=\sum_{i=1}^{k}\sigma_{i}\pmb{u}_{i}\pmb{v}_{i}^{\top}=\sum_{i=1}^{k}\sigma_{i}\pmb{A}_{i}
$$
$\pmb{A}$ 的秩为 $\operatorname{rk}(\widehat{\pmb{A}}(k))\:=\:k$。图 4.12 显示了原始图像 $\pmb{A}$（巨石阵）的低秩近似 $\widehat{\pmb{A}}(k)$。岩石的形状在秩为 5 的近似中变得越来越明显和清晰可辨。虽然原始图像需要 $432 \cdot 1910 = 2,735,120$ 个数字，但秩为 5 的近似只需要存储五个奇异值和五个左、右奇异向量（432 维和 1910 维）的总和为 $5 \cdot (1432 + 1910 + 1) = 16,715$ 个数字——仅占原始图像的 $0.6\%$。

为了衡量 $\pmb{A}$ 和其秩为 $\cdot k$ 的近似矩阵 $\widehat{\pmb{A}}(k)$ 之间的差异（误差），我们需要范数的概念。在第 3.1 节中，我们已经使用了

![](images/f90a18ecb4e3adc61beb9f57da4b5107d59d3b55109b0dd27bbebcd20422ab37.jpg)

(d) 秩为 3 的近似 $\widehat{\pmb{A}}(3)$。(e) 秩为 4 的近似 $\widehat{\pmb{A}}(4)$。(f) 秩为 5 的近似 $\widehat{\pmb{A}}(5)$。

向量的范数来衡量向量的长度。通过类比，我们也可以定义矩阵的范数。

### 定义 4.23 (矩阵的范数)。

对于 $\pmb{x} \in \mathbb{R}^{n} \backslash \{\mathbf{0}\}$，矩阵 $\pmb{A} \in \mathbb{R}^{m \times n}$ 的谱范数定义为

$$
\|\pmb{A}\|_{2} := \operatorname*{max}_{\pmb{x}} \frac{\|\pmb{A}\pmb{x}\|_{2}}{\|\pmb{x}\|_{2}}\,.
$$
我们在矩阵范数（左侧）中引入下标，类似于向量的欧几里得范数（右侧），其下标为 2。谱范数（4.93）确定了任何向量 $\pmb{x}$ 在乘以 $\pmb{A}$ 后最多可以变得多长。

### 定理 4.24。

矩阵 $\pmb{A}$ 的谱范数是其最大的奇异值 $\sigma_{1}$。
我们把证明这个定理作为练习。

### 定理 4.25 (Eckart-Young 定理 (Eckart 和 Young, 1936))。

考虑一个矩阵 $\pmb{A} \in \mathbb{R}^{m \times n}$，其秩为 $r$。令 $\pmb{B} \in \mathbb{R}^{m \times n}$ 为秩为 $k$ 的矩阵。对于任何 $k \leqslant r$，有

$$
\widehat{\pmb{A}}(k) = \operatorname{argmin}_{\operatorname{rk}(B)=k} \left\|\pmb{A} - \pmb{B}\right\|_{2}\,,
$$
$$
\left\|\pmb{A} - \widehat{\pmb{A}}(k)\right\|_{2} = \sigma_{k+1}\,.
$$
Eckart-Young 定理明确地说明了我们通过使用秩为 $\cdot k$ 的近似矩阵来近似 $\pmb{A}$ 时引入的误差。我们可以将使用 SVD 获得的秩为 $\cdot k$ 的近似矩阵解释为将全秩矩阵 $\pmb{A}$ 投影到秩最多为 $\cdot k$ 的矩阵的低维空间上的投影。在所有可能的投影中，SVD 在谱范数的意义下最小化了 $\pmb{A}$ 和任何秩为 $\cdot k$ 的近似矩阵之间的误差。

我们可以回顾一些步骤来理解为什么 (4.95) 应该成立。

我们观察到 $A - \widehat{A}(k)$ 的差是一个包含剩余秩为 1 的矩阵之和的矩阵

$$
\widehat{A} - \widehat{A}(k) = \sum_{i=k+1}^{r} \sigma_{i} \mathbf{u}_{i} \mathbf{v}_{i}^{\top}\,.
$$
根据定理 4.24，我们立即得到 $\sigma_{k+1}$ 作为差矩阵的谱范数。让我们仔细看看 (4.94)。如果我们假设存在另一个秩为 $\mathrm{rk}(B) \leqslant k$ 的矩阵 $B$，使得

$$
\left\|A - B\right\|_{2} < \left\|A - \widehat{A}(k)\right\|_{2}\,,
$$
那么存在一个 $(n-k)$ 维的零空间 $Z \subseteq \mathbb{R}^{n}$，使得 $\pmb{x} \in Z$ 意味着 $B x = \mathbf{0}$。然后可以得出

$$
\left\|\left(A \pmb{x}\right)\right\|_{2} = \left\|\left(\pmb{A} - \pmb{B}\right) \pmb{x}\right\|_{2}\,,
$$

通过使用 Cauchy-Schwartz 不等式（3.17）的一个版本，该版本涵盖了矩阵的范数，我们得到

$$
\left\|A \pmb{x}\right\|_{2} \leqslant \left\|\pmb{A} - \pmb{B}\right\|_{2} \left\|\pmb{x}\right\|_{2} < \sigma_{k+1} \left\|\pmb{x}\right\|_{2}\,.
$$
然而，存在一个 $(k+1)$ 维的子空间，使得 $\left\|A \pmb{x}\right\|_{2} \geqslant \sigma_{k+1} \left|\left|\pmb{x}\right|\right|_{2}$，该子空间由矩阵 $\pmb{A}$ 的右奇异向量 $\pmb{v}_{j}$，$j \leqslant k+1$，张成。将这两个空间的维数相加得到一个大于 $n$ 的数，因为这两个空间中必须存在一个非零向量。这是与秩-零度定理（定理 2.24）在第 2.7.3 节中的矛盾。

Eckart-Young 定理表明，我们可以使用 SVD 以一种原则性和最优的方式（在谱范数的意义上）将秩为 $\cdot r$ 的矩阵 $\pmb{A}$ 降低为秩为 $\cdot k$ 的矩阵 $\widehat{\pmb{A}}$。我们可以将 $\pmb{A}$ 通过秩为 $\cdot k$ 的矩阵进行近似解释为一种有损压缩的形式。因此，矩阵的低秩近似在许多机器学习应用中出现，例如图像处理、噪声过滤和病态问题的正则化。此外，它在降维和主成分分析中起着关键作用，如我们在第 10 章中将看到的。


### 示例 4.15 (在电影评分和消费者中的结构发现（续）)

回到我们的电影评分示例，我们现在可以应用低秩近似概念来近似原始数据矩阵。回想一下，我们的第一个奇异值捕捉了电影中的科幻主题以及科幻电影爱好者的概念。因此，通过仅使用电影评分矩阵的秩-1分解中的第一个奇异值项，我们得到了预测评分

$$
\mathbf{\Delta}A_{1} = \mathbf{u}_{1}\mathbf{v}_{1}^{\top} = \left[\begin{matrix}
-0.6710 \\
-0.7197 \\
-0.0939 \\
-0.1515
\end{matrix}\right]
\left[\begin{matrix}
-0.7367 & -0.6515 & -0.1811 \\
-0.7367 & -0.6515 & -0.1811
\end{matrix}\right]
$$

$$
= \left[\begin{matrix}
0.4943 & 0.4372 & 0.1215 \\
0.5302 & 0.4689 & 0.1303 \\
0.0692 & 0.0612 & 0.0170 \\
0.1116 & 0.0987 & 0.0274
\end{matrix}\right]\,.
$$

这个第一个秩-1近似$A_{1}$是有启发性的：它告诉我们阿里和贝特丽丝喜欢科幻电影，如《星球大战》和《银翼杀手》（条目值$>0.4$），但未能捕捉到钱德拉对其他电影的评分。这并不令人惊讶，因为钱德拉的电影类型并未被第一个奇异值捕捉到。第二个奇异值为我们提供了更好的秩-1近似，以捕捉那些电影主题爱好者的评分：

$$
\pmb{A}_{2}=\pmb{u}_{2}\pmb{v}_{2}^{\top}=\left[\begin{array}{l}{0.0236}\\ {0.2054}\\ {-0.7705}\\ {-0.6030}\end{array}\right]\left[0.0852\quad0.1762\quad-0.9807\right]
$$
$$
=\left[\begin{array}{l l l}{{0.0020}}&{{0.0042}}&{{-0.0231}}\\ {{0.0175}}&{{0.0362}}&{{-0.2014}}\\ {{-0.0656}}&{{-0.1358}}&{{0.7556}}\\ {{-0.0514}}&{{-0.1063}}&{{0.5914}}\end{array}\right]\;.
$$
在这个第二个秩-1近似$A_{2}$中，我们很好地捕捉了钱德拉的评分和电影类型，但没有捕捉到科幻电影。这使我们考虑秩-2近似$\hat{\pmb{A}}(2)$，其中我们结合了前两个秩-1近似

$$
{\widehat{\pmb{A}}}(2)=\sigma_{1}\pmb{A}_{1}+\sigma_{2}\pmb{A}_{2}=\left[\begin{array}{l l l}{4.7801}&{4.2419}&{1.0244}\\ {5.2252}&{4.7522}&{-0.0250}\\ {0.2493}&{-0.2743}&{4.9724}\\ {0.7495}&{0.2756}&{4.0278}\end{array}\right]\,.
$$
$\hat{\pmb{A}}(2)$与原始电影评分表相似

$$
\pmb{A}=\left[\begin{array}{l l l}{5}&{4}&{1}\\ {5}&{5}&{0}\\ {0}&{0}&{5}\\ {1}&{0}&{4}\end{array}\right]\,,
$$
这表明我们可以忽略$A_{3}$的贡献。我们可以这样解释，即在数据表中没有证据表明存在第三个电影主题/电影爱好者的类别。这也意味着我们示例中的整个电影主题/电影爱好者的空间是一个由科幻电影和法国艺术电影及其爱好者所构成的二维空间。

![示意图](images/102270d1c6e911f2c90b89b4e2699ad488dfd37e517a0972fed4073624d539f7.jpg)

## 4.7 矩阵谱系

在第2章和第3章中，我们介绍了线性代数和解析几何的基础知识。在本章中，我们探讨了矩阵和线性映射的基本特征。图4.13描绘了不同类型矩阵（黑色箭头表示“是子集”）之间的关系谱系树以及我们可以对它们执行的操作（蓝色表示）。我们考虑所有实矩阵 $\pmb{A}\in\mathbb{R}^{n\times m}$。对于非方矩阵（其中 $n\neq m$），S存在，正如我们在本章中所看到的。专注于方矩阵 $\pmb{A}\in\mathbb{R}^{n\times n}$，行列式告诉我们一个方矩阵是否具有逆矩阵，即，它是否属于正则、可逆矩阵的类别。如果 $n\times n$ 方矩阵具有 $n$ 个线性独立的特征向量，则该矩阵是非缺陷的，并且存在特征分解（定理4.12）。我们知道，重复的特征值可能导致缺陷矩阵，这些矩阵不能被对角化。

“谱系”一词描述了我们如何捕捉个体或群体之间的关系，并源自希腊语中的“部落”和“来源”这两个词。

非奇异和非缺陷矩阵不是相同的。例如，旋转矩阵将是可逆的（行列式非零），但在实数中不能被对角化（特征值不保证是实数）。

我们进一步探讨有效的方矩阵 $n\times n$。$\pmb{A}$ 是正交的，如果条件 $\mathbf{A}^{\top}\mathbf{A}=\mathbf{A}\mathbf{A}^{\top}$ 成立。此外，如果更严格的条件 $\mathbf{A}^{\intercal}\mathbf{A}=\mathbf{A}\mathbf{A}^{\intercal}=\mathbf{I}$ 成立，则 $\pmb{A}$ 被称为正交矩阵（见定义3.8）。正交矩阵的集合是正则（可逆）矩阵的子集，并满足 $\mathbf{\bar{A}}^{\top}=\mathbf{A}^{-1}$。

正交矩阵有一个经常遇到的子集，即对称矩阵 $\pmb{S}\in\mathbb{R}^{n\times n}$，它们满足 $\overset{\cdot}{\boldsymbol{S}}=\boldsymbol{S}^{\intercal}$。对称矩阵只有实特征值。对称矩阵的一个子集是正定矩阵 $P$，它们满足条件 $\pmb{x}^{\top}\pmb{P x}>0$ 对所有 $\pmb{x}\in\mathbb{R}^{n}\backslash\{\mathbf{0}\}$。在这种情况下，存在唯一的Cholesky分解（定理4.18）。正定矩阵只有正特征值，并且总是可逆的（即，行列式非零）。

对称矩阵的另一个子集是对角矩阵 $_D$。对角矩阵在乘法和加法下封闭，但不一定形成群（这只有在所有对角元素都不为零，使得矩阵可逆时才成立）。一个特殊的对角矩阵是单位矩阵 $\pmb{I}$。

## 4.8 进一步阅读

本章的大部分内容建立了基础数学，并将其与研究映射的方法联系起来，这些方法是机器学习中支撑软件解决方案和几乎所有理论构建块的核心。使用行列式、特征谱和特征空间对矩阵进行表征，提供了分类和分析矩阵的基本特征和条件。这扩展到所有形式的数据表示和涉及数据的映射，以及评估这些矩阵上计算操作的数值稳定性（Press 等人，2007）。

行列式是用于矩阵求逆和计算特征值的基本工具。然而，对于几乎所有但最小的实例，高斯消元法的数值计算优于行列式（Press 等人，2007）。尽管如此，行列式仍然是一个强大的理论概念，例如，通过行列式的符号来获得基向量方向的直观理解。特征向量可以用于执行基变换，将数据转换为有意义的正交特征向量的坐标。同样，矩阵分解方法，如 Cholesky 分解，在我们计算或模拟随机事件时经常出现（Rubinstein 和 Kroese，2016）。因此，Cholesky 分解使我们能够计算重新参数化技巧，即我们希望对随机变量进行连续微分，例如，在变分自动编码器中（Jimenez Rezende 等人，2014；Kingma 和 Welling，2014）。

特征分解是使我们能够提取描述线性映射的有意义且可解释信息的基本工具。

主成分分析 Fisher 判别分析 多维尺度分析

Isomap Laplacian 特征映射 Hessian 特征映射 谱聚类

Tucker 分解 CP 分解

因此，特征分解是称为谱方法的一般类机器学习算法的基础，这些算法执行正定核的特征分解。这些谱分解方法涵盖了统计数据分析的经典方法，例如：

主成分分析（PCA（Pearson，1901），参见第 10 章），其中寻求一个低维子空间，该子空间解释了数据中大部分变化。Fisher 判别分析，旨在确定数据分类的分离超平面（Mika 等人，1999）。多维尺度分析（MDS）（Carroll 和 Chang，1970）。

这些方法的计算效率通常来自找到最佳秩-$\cdot k$ 近似，该近似适用于对称、半正定矩阵。更现代的谱方法示例具有不同的起源，但每个方法都需要计算正定核的特征向量和特征值，例如 Isomap（Tenenbaum 等人，2000），Laplacian 特征映射（Belkin 和 Niyogi，2003），Hessian 特征映射（Donoho 和 Grimes，2003），以及谱聚类（Shi 和 Malik，2000）。这些核心计算通常由低秩矩阵近似技术（Belabbas 和 Wolfe，2009）支撑，我们在这里通过 SVD 遇到了这些技术。

SVD 允许我们发现与特征分解相同类型的信息。然而，SVD 更广泛地适用于非方矩阵和数据表。这些矩阵分解方法在我们希望识别数据中的异质性时变得相关，例如，通过近似进行数据压缩（例如，存储 $n\times m$ 值，而不是存储 $(n+m)k$ 值），或者当我们希望进行数据预处理时（例如，去相关设计矩阵中的预测变量）（Ormoneit 等人，2001）。SVD 操作于矩阵，我们可以将其解释为具有两个索引（行和列）的矩形数组。将矩阵结构扩展到更高维数组称为张量。结果表明，SVD 是一种更广泛分解家族的特殊情况，这些分解操作于这样的张量（Kolda 和 Bader，2009）。例如，Tucker 分解（Tucker，1966）或 CP 分解（Carroll 和 Chang，1970）。

SVD 低秩近似在机器学习中经常用于计算效率的原因。这是因为它减少了我们可能需要在潜在非常大的数据矩阵上执行的非零乘法操作的数量（Trefethen 和 Bau III，1997）。此外，低秩近似用于操作可能包含缺失值的矩阵，以及用于有损压缩和降维的目的（Moonen 和 De Moor，1995；Markovsky，2011）。

## 练习题

4.1 使用拉普拉斯展开（使用第一行）和萨鲁斯法则计算以下矩阵的行列式：

$$
\begin{array}{r}{\pmb{A}=\left[\begin{array}{ccc}1 & 3 & 5 \\ 2 & 4 & 6 \\ 0 & 2 & 4\end{array}\right]\,.}\end{array}
$$
4.2 高效地计算以下矩阵的行列式：

$$
\left[{\begin{array}{ccccc}2 & 0 & 1 & 2 & 0 \\ 2 & -1 & 0 & 1 & 1 \\ 0 & 1 & 2 & 1 & 2 \\ -2 & 0 & 2 & -1 & 2 \\ 2 & 0 & 0 & 1 & 1\end{array}}\right]\,.
$$
4.3 计算以下矩阵的特征空间：

a.

$$
\pmb{A} := \left[\begin{array}{cc}
1 & 0 \\
1 & 1
\end{array}\right]
$$

b.

$$
\pmb{\mathcal{B}} := \left[\begin{array}{cc}
-2 & 2 \\
2 & 1
\end{array}\right]
$$

4.4 计算以下矩阵的所有特征空间：

$$
A=\left[{\begin{array}{cccc}0 & -1 & 1 & 1 \\ -1 & 1 & -2 & 3 \\ 2 & -1 & 0 & 0 \\ 1 & -1 & 1 & 0\end{array}}\right]\,.
$$
4.5 矩阵的对角化与其可逆性无关。确定以下四个矩阵是否可以对角化和/或可逆：

$$
\left[\begin{array}{cc}
1 & 0 \\
0 & 1
\end{array}\right]\,,\quad
\left[\begin{array}{cc}
1 & 0 \\
0 & 0
\end{array}\right]\,,\quad
\left[\begin{array}{cc}
1 & 1 \\
0 & 1
\end{array}\right]\,,\quad
\left[\begin{array}{cc}
0 & 1 \\
0 & 0
\end{array}\right]\,.
$$

4.6 计算以下变换矩阵的特征空间。它们是否可以对角化？

a. 对于

$$
A={\left[\begin{array}{ccc}
2 & 3 & 0 \\
1 & 4 & 3 \\
0 & 0 & 1
\end{array}\right]}
$$

b. 对于

$$
A={\left[\begin{array}{cccc}
1 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{array}\right]}
$$


4.7 以下矩阵是否可以对角化？如果可以，确定它们的对角形式，并给出一个基，使得变换矩阵在此基下是对角的。如果不能，给出它们不能对角化的理由。

a.
$$
A={\left[\begin{array}{cc}
0 & 1 \\
-8 & 4
\end{array}\right]}
$$

b.
$$
A={\left[\begin{array}{ccc}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{array}\right]}
$$

$$
A={\left[\begin{array}{cccc}
5 & 4 & 2 & 1 \\
0 & 1 & -1 & -1 \\
-1 & -1 & 3 & 0 \\
1 & 1 & -1 & 2
\end{array}\right]}
$$

$$
A={\left[\begin{array}{ccc}
5 & -6 & -6 \\
-1 & 4 & 2 \\
3 & -6 & -4
\end{array}\right]}
$$

4.8 求矩阵的奇异值分解

$$
A = \begin{bmatrix}
    3 & 2 & 2 \\
    2 & 3 & -2
\end{bmatrix}
$$
4.9 求矩阵的奇异值分解

$$
A = \begin{bmatrix}
    -2 & 2 \\
    1 & 1
\end{bmatrix}
$$
4.10 求矩阵的秩1近似

$$
A = \begin{bmatrix}
3 & 2 & 2 \\
2 & 3 & -2 \\
2 & -2 & 3
\end{bmatrix}
$$

4.11 证明对于任意 $\pmb{A}\,\in\,\mathbb{R}^{m\times n}$，矩阵 $\pmb{A}^{\top}\pmb{A}$ 和 $\pmb{A A}^{\top}$ 具有相同的非零特征值。

4.12 证明对于 $\mathbfit{\textbf{x}\ne\mathbf{0}}$，定理 4.24 成立，即证明

$$
\operatorname*{max}_{\pmb{x}}\frac{\|\pmb{A}\pmb{x}\|_{2}}{\|\pmb{x}\|_{2}}=\sigma_{1}\,,
$$
其中 $\sigma_{1}$ 是矩阵 $\pmb{A}\in\mathbb{R}^{m\times n}$ 的最大奇异值。****