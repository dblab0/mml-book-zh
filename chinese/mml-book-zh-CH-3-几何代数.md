
# 第三章 几何代数

在第2章中，我们从一般且抽象的角度研究了向量、向量空间和线性映射。在本章中，我们将为这些概念添加几何解释和直观理解。特别是，我们将研究几何向量，并计算它们的长度和两个向量之间的距离或角度。为了做到这一点，我们为向量空间配备了一个内积，从而诱导了向量空间的几何结构。内积及其相应的范数和度量捕捉了相似性和距离的直观概念，我们在第12章中使用这些概念来发展支持向量机。然后，我们将使用向量之间的长度和角度的概念来讨论正交投影，这在讨论第10章中的主成分分析和第9章中的最大似然估计回归时将发挥核心作用。图3.1给出了本章中介绍的概念之间的关系以及它们与书中其他部分的联系。

![](images/55dd20be380dc5c4ec1aa755c03091d38ebcb8ff1fe66d82205a0a13b4978fc4.jpg)  
图3.1 本章介绍的概念的思维导图，以及它们在书中其他部分的应用。  

![](images/260a596bebc10ee742421fd4b2b94bc416a611ad8916fda99b028b1e85ec489f.jpg)  
图3.3  对于不同的范数，红色线条表示范数为1的向量集合。左：曼哈顿范数；右：欧几里得距离。 
## 3.1 范数  

当我们思考几何向量时，即从原点出发的有向线段，直观上向量的长度是该有向线段终点与原点之间的距离。接下来，我们将使用范数的概念来讨论向量的长度。  

### 定义3.1（范数）。

向量空间$V$上的范数是一个函数  

$$
\begin{array}{r l}&{\|\cdot\|:V\to\mathbb{R}\,,}\\ &{\quad\quad\;\pmb{x}\mapsto\|\pmb{x}\|\,,}\end{array}
$$
其中每个向量$x$对应其长度$\|x\|\in\mathbb{R}$，对于所有$\lambda\in\mathbb{R}$和$\pmb{x},\pmb{y}\in V$，以下条件成立：  

绝对同态：$\|\lambda{\pmb{x}}\|=|\lambda|\|{\pmb{x}}\|$  
三角不等式：$\|\pmb{x}+\pmb{y}\|\leqslant\|\pmb{x}\|+\|\pmb{y}\|$  
正定性：$\|\pmb{x}\|\geqslant0$，且$\|{\pmb{x}}\|=0\iff{\pmb{x}}={\bf0}$  

从几何角度来看，三角不等式表示对于任何三角形，任意两边之和必须大于或等于剩余边的长度；见图3.2的插图。定义3.1是基于一般向量空间$V$（第2.4节），但在本书中，我们仅考虑有限维向量空间$\mathbb{R}^{n}$。回想一下，对于向量$\pmb{x}\in\mathbb{R}^{n}$，我们用下标表示向量的元素，也就是说，$x_{i}$是向量$x$的第$i$个元素。

### 示例3.1（曼哈顿范数）  

$\mathbb{R}^{n}$上的曼哈顿范数定义为$\pmb{x}\in\mathbb{R}^{n}$时的  

$$
\left\|{\pmb{x}}\right\|_{1}:=\sum_{i=1}^{n}\left|x_{i}\right|,
$$
其中$|\cdot|$表示绝对值。图3.3的左图展示了$\pmb{x}\,\in\,\mathbb{R}^{2}$的向量，使得$\|\pmb{x}\|_{1}\,=\,1$。曼哈顿范数也被称为$\ell_{1}$范数。

范数  
长度  
绝对同态  
三角不等式  
正定性  

图3.2 三角不等式。  

![](images/2d0f6398949246a3c3e8ad6507463855e5b6646334f250f056e1a4efc04f1050.jpg)  

曼哈顿范数  
$\ell_{1}$范数  

### 示例3.2（欧几里得范数）  

欧几里得范数  

欧几里得距离$\ell_{2}$范数  

$\mathbb{R}^{n}$上的欧几里得范数定义为  

$$
\|x\|_{2}:=\sqrt{\sum_{i=1}^{n}x_{i}^{2}}=\sqrt{{\pmb{x}}^{\top}{\pmb{x}}}
$$
并计算了向量$\pmb{x}$从原点的距离。图3.3的右图展示了$\pmb{x}\,\in\,\mathbb{R}^{2}$的向量，使得$\|\pmb{x}\|_{2}\,=\,1$。欧几里得范数也被称为$\ell_{2}$范数。  

注释。在本书中，除非另有说明，否则我们将默认使用欧几里得范数（3.4）。$\diamondsuit$

## 3.2 内积  

内积允许引入直观的几何概念，如向量的长度和两个向量之间的角度或距离。内积的主要目的是确定向量是否正交。  

### 3.2.1 点积  

标量积 点积  

我们可能已经熟悉了一种特定类型的内积，即$\mathbb{R}^{n}$中的标量积/点积，其给出为  

$$
\pmb{x}^{\top}\pmb{y}=\sum_{i=1}^{n}x_{i}y_{i}\,.
$$
我们将称这种特定的内积为本书中的点积。然而，内积是更通用的概念，具有特定的性质，我们现在将介绍。  

### 3.2.2 一般内积  

双线性映射 回顾第2.7节中的线性映射，我们可以根据加法和标量乘法重排映射。一个**双线性映射**$\Omega$是一个接受两个参数的映射，并且在每个参数上都是线性的，即当我们查看向量空间$V$时，对于所有$\pmb{x},\pmb{y},z\in V,\ \lambda,\psi\in\mathbb{R}$，有  

$$
\begin{array}{r l}&{\Omega(\lambda\pmb{x}+\psi\pmb{y},z)=\lambda\Omega(\pmb{x},z)+\psi\Omega(\pmb{y},z)}\\ &{\Omega(\pmb{x},\lambda\pmb{y}+\psi\pmb{z})=\lambda\Omega(\pmb{x},\pmb{y})+\psi\Omega(\pmb{x},z)\,.}\end{array}
$$
这里，(3.6)断言$\Omega$在第一个参数上是线性的，(3.7)断言$\Omega$在第二个参数上也是线性的（参见(2.87)）。  

### 定义3.2. 

令$V$为向量空间，$\Omega:V\times V\rightarrow\mathbb{R}$为接受两个向量并映射到实数的双线性映射。则  

$\Omega$称为**对称**如果$\Omega({\pmb{x}},{\pmb y})\,=\,\Omega({\pmb y},{\pmb{x}})$对于所有${\pmb{x}},{\pmb y}\,\in\,V$，即参数的对称顺序不重要。$\Omega$称为**正定**如果对于所有${\pmb{x}}\in V\backslash\{{\bf0}\}:\Omega({\pmb{x}},{\pmb{x}})>0\,,\quad\Omega({\bf0},{\bf0})=0\,.$  

### 定义3.3. 

令$V$为向量空间，$\Omega:V\times V\rightarrow\mathbb{R}$为接受两个向量并映射到实数的双线性映射。则  

一个**正定**且**对称**的双线性映射$\Omega:V\times V\rightarrow\mathbb{R}$称为**内积**。我们通常写$\langle \pmb{x},\pmb{y}\rangle$而不是$\Omega({\pmb{x}},{\pmb y})$。对$( \left(V,\left\langle\cdot,\cdot\right\rangle\right)$⟨·  ·⟩)称为**内积空间**或（实）**具有内积的向量空间**。如果我们使用在(3.5)中定义的点积，我们称$( \left(V,\left\langle\cdot,\cdot\right\rangle\right)$为**欧几里得向量空间**。  

我们将这些空间称为内积空间在本书中。  

内积 内积空间 具有内积的向量空间 欧几里得向量空间  

### 示例3.3（不是点积的内积）

考虑$V=\mathbb{R}^{2}$。如果定义  

$$
\langle x,y\rangle:=x_{1}y_{1}-\left(x_{1}y_{2}+x_{2}y_{1}\right)+2x_{2}y_{2}
$$
则$\langle\cdot,\cdot\rangle$是一个内积，但与点积不同。证明将是一个练习。  

### 3.2.3 对称、正定矩阵  

对称、正定矩阵在机器学习中扮演重要角色，并通过内积定义。在第4.3节中，我们将再次讨论对称、正定矩阵的矩阵分解上下文。对称正半定矩阵的关键定义在核的定义（第12.4节）。  

一个$n$维向量空间$V$与内积$\langle\cdot,\cdot\rangle：V\times V\to\mathbb{R}$× →（见定义3.3）和有序基$B=(b_{1},\cdots,b_{n})$的$V$。回想第2.6.1节，任何向量$\pmb{x},\pmb{y}\in V$n作为基础的表示为$\begin{array}{r}{{\pmb{x}}=\sum_{i=1}^{n}\psi_{i}{\pmb{b}}_{i}\in V}\end{array}$   和$\begin{array}{r}{\pmb{y}=\sum_{j=1}^{n}\lambda_{j}\pmb{b}_{j}\in V}\end{array}$   对于适当的$\psi_{i},\lambda_{j}\in\mathbb{R}$   。由于内积的双线性性质，对于所有$\pmb{x},\pmb{y}\in V$有  

$$
\langle{\pmb{x}},{\pmb y}\rangle=\left\langle\sum_{i=1}^{n}\psi_{i}{\pmb{b}}_{i},\sum_{j=1}^{n}\lambda_{j}{\pmb{b}}_{j}\right\rangle=\sum_{i=1}^{n}\sum_{j=1}^{n}\psi_{i}\left\langle{\pmb{b}}_{i},{\pmb{b}}_{j}\right\rangle\lambda_{j}={\hat{\pmb{x}}}^{\top}A{\hat{\pmb y}}\,，
$$
其中$A_{i j}:=\langle{b_{i},b_{j}}\rangle$和$\hat{\pmb{x}},\hat{\pmb{y}}$是$\pmb{x}$和$\pmb{ y}$相对于基B的坐标。这意味着内积$\langle\cdot,\cdot\rangle$通过A唯一确定。内积的对称性也意味着A是对称的。此外，内积的正定性意味着  

$$
\forall{\pmb{x}}\in V\backslash\{{\bf0}\}:{\pmb{x}}^{\top}A{\pmb{x}}>0\,。
$$
对称、正定正定半定  

### 定义3.4（对称、正定矩阵）。

满足( 11)的对称矩阵$\pmb{A}\ \in\ \mathbb{R}^{n\times n}$称为**对称正定**，或简称为**正定**。如果仅在(3.11)中满足≥，则A称为**对称正半定**。


### 示例3.4（对称、正定矩阵）

考虑矩阵

$$
A_{1}=\left[9\quad6\right],\quad A_{2}=\left[9\quad6\right]\,.
$$
$A_{1}$ 是正定的，因为它是对称的，并且

$$
\begin{array}{r l}{\pmb{x}^{\top}\pmb{A}_{1}\pmb{x}=\left[x_{1}\quad x_{2}\right]\left[\begin{array}{l l}{9}&{6}\\ {6}&{5}\end{array}\right]\left[x_{1}\right]}&{}\\ &{=9x_{1}^{2}+12x_{1}x_{2}+5x_{2}^{2}=(3x_{1}+2x_{2})^{2}+x_{2}^{2}>0}\end{array}
$$
对于所有 ${\pmb{x}}\in V\backslash\{{\bf0}\}$。相比之下，$A_{2}$ 是对称的，但不是正定的，因为 $\pmb{x}^{\top}\pmb{A}_{2}\pmb{x}=9x_{1}^{2}+12x_{1}x_{2}+3x_{2}^{2}=(3x_{1}+2x_{2})^{2}-x_{2}^{2}$ 可以小于0，例如对于 $\pmb{x}=[2,-3]^{\top}$。

如果 $\pmb{A}\in\mathbb{R}^{n\times n}$ 是对称的、正定的，那么

$$
\langle{\pmb{x}},{\pmb y}\rangle=\hat{{\pmb{x}}}^{\top}{\pmb A}\hat{{\pmb y}}
$$
根据有序基 $B$ 定义了一个内积，其中 $\hat{\pmb{x}}$ 和 $\hat{\pmb{y}}$ 是相对于 $B$ 的向量 $\pmb{x},\pmb{y}\in V$ 的坐标表示。

### 定理3.5

对于实数值、有限维的向量空间 $V$ 和有序基 $B$ 的 $V_{-}$，内积 $\langle\cdot,\cdot\rangle:V\times V\rightarrow\mathbb{R}$ 存在当且仅当存在一个对称、正定矩阵 $\pmb{A}\in\mathbb{R}^{n\times n}$，使得

$$
\langle{\pmb{x}},{\pmb y}\rangle=\hat{\pmb{x}}^{\top}{\pmb A}\hat{\pmb y}\,.
$$
如果 $\pmb{A}\,\in\,\mathbb{R}^{n\times n}$ 是对称的且正定的，则：

$\pmb{A}$ 的零空间（核）仅包含 0，因为 ${\pmb{x}}^{\top}{\pmb A}{\pmb{x}}>0$ 对于所有 $\pmb{x}\neq\mathbf{0}$。这意味着 $\pmb{Ax}\neq\mathbf{0}$ 如果 $\pmb{x}\neq{\bf0}$。$\pmb{A}$ 的对角元素 $a_{i i}$ 是正的，因为 $a_{i i}=\boldsymbol{e}_{i}^{\top}\boldsymbol{A}\boldsymbol{e}_{i}>0,$，其中 $e_{i}$ 是 $\textstyle\mathbb{R}^{n}$ 中的标准基的第 $i$ 个向量。


## 3.3 向量的长度与距离

在第3.1节中，我们已经讨论了可以用来计算向量长度的范数。内积和范数之间存在着密切的关系，任何内积都可以诱导出一个范数：

$$
\|\pmb{x}\|:=\sqrt{\langle\pmb{x},\pmb{x}\rangle}
$$
内积诱导出范数。以自然的方式，使得我们可以通过内积来计算向量的长度。然而，并不是每一个范数都是由内积诱导出的。曼哈顿范数（3.3）就是一个没有对应内积的范数的例子。接下来，我们将专注于由内积诱导出的范数，并引入几何概念，如长度、距离和角度。

**注释**（柯西-施瓦茨不等式）：
对于内积向量空间$\left(V,\left\langle\cdot,\cdot\right\rangle\right)$，诱导出的范数$\|\cdot\|$满足柯西-施瓦茨不等式：

$$
|\left\langle{\pmb{x}},{\pmb y}\right\rangle|\leqslant\|{\pmb{x}}\|\|{\pmb y}\|\,.
$$
柯西-施瓦茨不等式

### 示例3.5（使用内积计算向量的长度）

在几何学中，我们经常对向量的长度感兴趣。现在我们可以使用内积来通过（3.16）计算它们。假设$\pmb{x}=[1,1]^{\top}\in\mathbb{R}^{2}$。如果我们使用点积作为内积，那么根据（3.16）我们得到：

$$
\|x\|={\sqrt{x^{\top}x}}={\sqrt{1^{2}+1^{2}}}={\sqrt{2}}
$$
作为向量$x$的长度。现在让我们选择不同的内积：

$$
\langle{\pmb{x}},{\pmb y}\rangle:={\pmb{x}}^{\top}\left[\begin{array}{c c}{1}&{-\frac{1}{2}}\\ {-\frac{1}{2}}&{1}\end{array}\right]{\pmb y}=x_{1}y_{1}-\frac{1}{2}\big(x_{1}y_{2}+x_{2}y_{1}\big)+x_{2}y_{2}\,.
$$
如果我们计算向量的范数，那么这个内积在$x_{1}$和$x_{2}$具有相同符号（且$x_{1}x_{2}>0$）的情况下返回的值比点积小；否则，它返回的值比点积大。使用这个内积，我们得到：

$$
\langle{\pmb{x}},{\pmb{x}}\rangle=x_{1}^{2}-x_{1}x_{2}+x_{2}^{2}=1-1+1=1\implies\|{\pmb{x}}\|=\sqrt{1}=1\,,
$$
这意味着使用这个内积，向量$\pmb{x}$比使用点积时显得“更短”。

### 定义3.6（距离与度量）
：考虑内积空间$\left(V,\left\langle\cdot,\cdot\right\rangle\right)$。则

$$
d(\pmb{x},\pmb{y}):=\|\pmb{x}-\pmb{y}\|=\sqrt{\langle\pmb{x}-\pmb{y},\pmb{x}-\pmb{y}\rangle}
$$
被称为$\pmb{x}$和$\pmb{ y}$之间的**距离**，对于${\pmb{x}},{\pmb{ y}}\in{\pmb{ V}}$。如果我们使用点积作为内积，那么距离被称为**欧几里得距离**。欧几里得距离映射

$$
\begin{array}{r l}&{d:V\times V\to\mathbb{R}}\\ &{\quad(\pmb{x},\pmb{y})\mapsto d(\pmb{x},\pmb{y})}\end{array}
$$
被称为**度量**。

注释。向量之间的距离并不需要内积：一个范数就足够了。如果我们有一个由内积诱导出的范数，距离可能会根据内积的选择而变化。$\diamondsuit$

一个度量$d$满足以下条件：

1. **正定性**：即$d(\pmb{x},\pmb{y})\,\geqslant\,0$对于所有$\pmb{x},\pmb{y}\,\in\,V$，且$d({\pmb{x}},{\pmb y})=0\iff\pmb{x}=\pmb{y}$。

2. **对称性**：即$d(\pmb{x},\pmb{y})=d(\pmb{y},\pmb{x})$对于所有$\pmb{x},\pmb{y}\in V$。

3. **三角不等式**：$d(\pmb{x},z)\leqslant d(\pmb{x},\pmb{y})+d(\pmb{y},z)$对于所有$x,y,z\in V$。

注释。乍一看，内积和度量的属性列表看起来非常相似。然而，通过比较定义3.3与定义3.6，我们可以发现$\langle\pmb{x},\pmb{y}\rangle$和$d(\pmb{x},\pmb{y})$的行为方向相反。非常相似的$\pmb{x}$和$\pmb{ y}$会导致内积返回较大的值，而度量返回较小的值。$\diamondsuit$

图3.4 当限制在$[0,\pi]$区间内时，$f(\omega)=\cos(\omega)$返回区间$[-1,1]$内的唯一数值。

![](images/5ae51d06b9be371b782cec5663db76846953523667af22eb2084c730fa5cd51e.jpg)

## 3.4 角度与正交性  

除了允许定义向量的长度以及两个向量之间的距离外，内积还通过定义两个向量之间的角度 $\omega$ 来捕捉向量空间的几何结构。我们使用柯西-施瓦茨不等式（3.17）来在内积空间中定义两个向量 $\mathbfit{x},\mathbfit{y}$ 之间的角度 $\omega$，这种定义与我们在 $\mathbb{R}^{2}$ 和 $\mathbb{R}^{3}$ 中的直观理解一致。假设 $\mathbfit{x}\neq\mathbf{0},\mathbfit{y}\neq\mathbf{0}$，则有  

$$
-1\leqslant\frac{\langle\mathbf{\mathscr{x}},\mathbf{\mathscr{y}}\rangle}{\|\mathbf{\mathscr{x}}\|\,\|\mathbf{\mathscr{y}}\|}\leqslant1\,.
$$
因此，存在唯一的 $\omega\in[0,\pi]$，如图3.4所示，且  

$$
\cos\omega=\frac{\langle{\pmb x},{\pmb y}\rangle}{\|{\pmb x}\|\,\|{\pmb y}\|}\,.
$$
数字 $\omega$ 是向量 ${\pmb{x}}$ 和 ${\pmb{y}}$ 之间的 角度 。直观上，两个向量之间的角度告诉我们它们的方向相似程度。例如，使用点积，向量 ${\pmb{x}}$ 和 ${\pmb{y}}=4{\pmb{x}}$，即 ${\pmb{y}}$ 是 ${\pmb{x}}$ 的缩放版本，其角度为 0：它们的方向相同。  

**例3.6（向量之间的角度）** 计算向量 ${\pmb{x}}=[1,1]^{\top}\in\mathbb{R}^{2}$ 和 ${\pmb{y}}=[1,2]^{\top}\in\mathbb{R}^{2}$ 之间的角度；见图3.5，我们使用点积作为内积。然后我们得到  

$$
\cos\omega=\frac{\left\langle\pmb{x},\pmb{y}\right\rangle}{\sqrt{\left\langle\pmb{x},\pmb{x}\right\rangle\left\langle\pmb{y},\pmb{y}\right\rangle}}=\frac{\pmb{x}^{\top}\pmb{y}}{\sqrt{\pmb{x}^{\top}\pmb{x}\pmb{y}^{\top}\pmb{y}}}=\frac{3}{\sqrt{10}}\,,
$$
因此，两个向量之间的角度为 $\operatorname{arccos}\left(\frac{3}{\sqrt{10}}\right)\approx0.32\,{\mathrm{rad}}$，这对应于大约 $32^{\circ}$。  

内积的一个关键特征是它还允许我们表征正交的向量。  

### 定义3.7（正交性）。

两个向量 ${\pmb{x}}$ 和 ${\pmb{y}}$ 正交当且仅当 $\langle{\pmb x},{\pmb y}\rangle=0$，我们记作 ${\pmb{x}}\perp{\pmb{y}}$。如果此外 $\|\pmb{x}\|=1=\|\pmb{y}\|$，即这些向量是单位向量，则 ${\pmb{x}}$ 和 ${\pmb{y}}$ 是 正交单位向量。  

**图3.5** 两个向量 ${\mathbfit{x}},{\mathbfit{y}}$ 之间的角度 $\omega$ 是通过内积计算得出的。  

![](images/16dead139722b481d45d60e3feece53a4aa2d99e091c1136b688b5b75fd7c833.jpg)  
正交单位向量  

这一定义的一个推论是零向量与向量空间中的每个向量正交。  

**注释**。正交性是垂直概念在不必是点积的双线性形式中的推广。在我们的上下文中，几何上我们可以将正交向量视为相对于特定内积具有直角。$\diamondsuit$
### 示例3.7（正交向量）

![](images/064ec054d5f999c46187ee0fee5451046586abdef27b97cdbad3bba86b4fd68a.jpg)  
图3.6：两个向量$\pmb{x},\pmb{y}$之间的角度$\omega$可能根据内积的不同而变化。

考虑两个向量$\pmb{x}=[1,1]^{\top},\pmb{y}=[-1,1]^{\top}\in\mathbb{R}^{2}$；见图3.6。我们感兴趣的是使用两种不同的内积来确定它们之间的角度$\omega$。使用点积作为内积，得到$\pmb{x}$和$\pmb{ y}$之间的角度为$90^{\circ}$，因此$\pmb{x}\perp\pmb{ y}$。然而，如果我们选择内积

$$
\langle\pmb{x},\pmb{y}\rangle=\pmb{x}^{\top}\left[\begin{array}{c c}{2}&{0}\\ {0}&{1}\end{array}\right]\pmb{y}\,，
$$
我们得到$\pmb{x}$和$\pmb{ y}$之间的角度$\omega$由

$$
\cos\omega={\frac{\langle\pmb{x},\pmb{y}\rangle}{\|\pmb{x}\|\|\pmb{y}\|}}=-{\frac{1}{3}}\implies\omega\approx1.91\,\mathrm{rad}\approx109.5^{\circ}\,，
$$
因此$x$和$\pmb{ y}$不是正交的。因此，相对于一个内积正交的向量，相对于不同的内积可能不正交。

### 定义3.8（正交矩阵）

一个$n\times n$的方阵$\pmb{A}\ \in\ \mathbb{R}^{n\times n}$是正交矩阵，当且仅当它的列是正交标准向量，使得

$$
\pmb{A}\pmb{A}^{\top}=\pmb{I}=\pmb{A}^{\top}\pmb{A}\,，
$$
这暗示

$$
\begin{array}{r}{\boldsymbol{A}^{-1}=\boldsymbol{A}^{\intercal}\,,}\end{array}
$$
通常称这些矩阵为“正交”，但更精确的描述应该是“正交标准”。使用正交矩阵进行变换会保持距离和角度不变。

即，通过简单转置矩阵得到逆矩阵。使用正交矩阵进行变换时，向量$x$的长度不会改变。对于点积，我们得到

$$
\left\|A\pmb{x}\right\|^{2}=(A\pmb{x})^{\top}(A\pmb{x})=\pmb{x}^{\top}A^{\top}A\pmb{x}=\pmb{x}^{\top}\pmb{I}\pmb{x}=\pmb{x}^{\top}\pmb{x}=\left\|\pmb{x}\right\|^{2}\,。
$$
此外，使用正交矩阵$\pmb{A}$变换两个向量$\pmb{x},\pmb{y}$时，它们之间内积测量的角度也不会改变。假设点积作为内积，变换后的向量$A\pmb{x}$和$A\pmb{y}$的角度给出为

$$
\cos\omega={\frac{(A\pmb{x})^{\top}(A\pmb{y})}{\|A\pmb{x}\|\,\|A\pmb{y}\|}}={\frac{\pmb{x}^{\top}A^{\top}A\pmb{y}}{\sqrt{\pmb{x}^{\top}A^{\top}A\pmb{x}\pmb{y}^{\top}A^{\top}A\pmb{y}}}}={\frac{\pmb{x}^{\top}\pmb{y}}{\|\pmb{x}\|\,\|\pmb{y}\|}}\,，
$$
这给出恰好是$\pmb{x}$和$\pmb{ y}$之间的角度。这意味着正交矩阵$\pmb{A}$，其中$\pmb{A}^{\top}\breve{=}\pmb{A}^{-1}$，既保持角度又保持距离。实际上，正交矩阵定义的变换是旋转（可能包括翻转）。在第3.9节中，我们将讨论更多关于旋转的细节。

## 3.5 正交标准基

在第2.6.1节中，我们描述了基向量的特性，并发现在一个$n$维向量空间中，我们需要$n$个基向量，即$n$个线性独立的向量。在第3.3节和第3.4节中，我们使用内积来计算向量的长度和向量之间的角度。接下来，我们将讨论基向量彼此正交，且每个基向量长度为1的特殊情况。我们将这种基称为正交标准基。

让我们更正式地引入这个概念。

### 定义3.9（正交标准基）：

考虑一个$n$维向量空间$V$和$V$的一个基$\{b_{1},\ldots,b_{n}\}$。如果对于所有$i,j=1,\dotsc,n$有

$$
\begin{array}{l}{\langle\pmb{b}_{i},\pmb{b}_{j}\rangle=0\quad\mathrm{for}\;i\neq j}\\ {\langle\pmb{b}_{i},\pmb{b}_{i}\rangle=1}\end{array}
$$
则该基称为正交标准基（ONB）。如果仅满足（3.33），则该基称为正交基。请注意，（3.34）表明每个基向量的长度/范数为1。

回顾第2.6.1节，我们可以使用高斯消元法来找到一个向量空间由一组向量生成的基。假设我们给出一组非标准化基向量$\{b_{1},\cdots,b_{n}\}$。我们将它们连接成矩阵$\mathbf{\tilde{B}}=[\tilde{\pmb{b}}_{1},\cdots,\tilde{\pmb{b}}_{n}]$，并应用高斯消元法到增广矩阵（第2.3.2节）$[\tilde{B}\tilde{B}^{\top}|\tilde{B}]$来获得一个正交标准基。这种逐步构建正交标准基$\{b_{1},\ldots,b_{n}\}$的方法称为格拉姆-施密特过程（Strang, 2003）。

### 示例3.8（正交标准基）

欧几里得向量空间$\textstyle\mathbb{R}^{n}$的标准基是一个正交标准基，其中的内积是向量的点积。在$\mathbb{R}^{2}$中，向量

$$
{\pmb{b}}_{1}=\frac{1}{\sqrt{2}}\left[\begin{array}{c}{{1}}\\ {{1}}\end{array}\right],\quad{\pmb{b}}_{2}=\frac{1}{\sqrt{2}}\left[\begin{array}{c}{{1}}\\ {{-1}}\end{array}\right]
$$
形成一个正交标准基，因为$\pmb{b}_{1}^{\top}\pmb{b}_{2}=0$且$\|b_{1}\|=1=\|b_{2}\|$。

我们将利用正交标准基的概念在第12章和第10章中讨论支持向量机和支持主成分分析。

## 3.6 正交补集  

定义了正交性之后，我们现在将探讨彼此正交的向量空间。这将在第10章中从几何角度讨论线性降维时发挥重要作用。  

考虑一个$D$维向量空间$V$和一个$M$维子空间$U\subseteq V$。$U$的正交补集$U^{\perp}$是$V$中的一个$(D-M)$维正交子空间，包含$V$中所有与$U$中的每个补集向量正交的向量。此外，$U\cap U^{\perp}=\{\mathbf{0}\}$，因此任何向量$\pmb{x}\in V$可以

![](images/444f07dbdac70cbd68e683c2483fac4f4739798dd7dca7daaef8d8ac206629d1.jpg)  
图3.7：三维向量空间中的平面$U$可以通过其法向量来描述，该法向量决定了其正交补集$U^{\perp}$。  

唯一分解为  

$$
\pmb{x}=\sum_{m=1}^{M}\lambda_{m}\pmb{b}_{m}+\sum_{j=1}^{D-M}\psi_{j}\pmb{b}_{j}^{\perp},\quad\lambda_{m},\;\psi_{j}\in\mathbb{R}\,，
$$
其中$(b_{1},\cdots,b_{M})$是$U$的基，$(b_{1}^{\perp},\cdots,b_{D-M}^{\perp})$是$U^{\perp}$的基。  

法向量  

因此，正交补集也可以用来描述三维向量空间中的平面$U$（二维子空间）。更具体地说，向量$\pmb{w}$，$\|w\|=1$，且与平面$U$正交，是$U^{\perp}$的基向量。图3.7展示了这一设置。所有与$\pmb{w}$正交的向量（通过构造）必须位于平面$U$中。向量$\pmb{w}$称为$U$的法向量。  

通常情况下，正交补集可以用来描述$n$维向量空间和仿射空间中的超平面。


## 3.7 函数的内积  

到目前为止，我们探讨了内积的性质，用于计算长度、角度和距离。我们专注于有限维向量的内积。接下来，我们将探讨不同类型的向量的内积示例：函数的内积。  

我们之前讨论的内积是为具有有限个元素的向量定义的。我们可以将向量$\pmb{x}\in\mathbb{R}^{n}$视为具有$n$个函数值的函数。内积的概念可以推广到具有无限个元素（可数无限）的向量和连续值函数（不可数无限）。然后，向量各个分量的求和（例如见式（3.5））转化为积分。  

两个函数$u:\mathbb{R}\rightarrow\mathbb{R}$和$v:\mathbb{R}\rightarrow\mathbb{R}$的内积可以定义为定积分  

$$
\langle u,v\rangle:=\int_{a}^{b}u(x)v(x)d x
$$
其中$a,b<\infty$分别为下限和上限。与我们通常的内积一样，我们可以定义范数和正交性，通过观察内积。如果（3.37）的值为0，则函数$u$和$v$正交。为了使上述内积在数学上精确，我们需要处理测度和积分的定义，从而定义希尔伯特空间。此外，与有限维向量上的内积不同，函数上的内积可能发散（具有无限值）。所有这些都需要深入探讨实分析和泛函分析的更复杂细节，本书不涵盖这些内容。  

### 示例3.9（函数的内积） 

如果我们选择$u=\sin(x)$和$v=\cos(x)$，式（3.37）的被积函数$f(x)=u(x)v(x)$如图3.8所示。我们看到这个函数是奇函数，即$f(-x)=-f(x)$。因此，从$a=-\pi$到$b=\pi$的这个乘积的积分值为0。因此，$\sin$和$\cos$是正交函数。  

注释。如果我们将积分范围从$-\pi$到$\pi$，则集合中的函数  

$$
\{1,\cos(x),\cos(2x),\cos(3x),\cdots\}
$$
![](images/2dfba01d88a2f77b55a919e9966827017e8db57f54be64262d782ba8467ba946.jpg)  
图3.8 $f(x)=\sin(x)\cos(x)$。  

如果从$-\pi$到$\pi$积分，则这个集合中的函数是正交的，即任何一对函数都是相互正交的。式（3.38）中的函数集合覆盖了在$[-\pi,\pi)$区间内偶函数和周期函数的一个大子空间，将函数投影到这个子空间是傅里叶级数的基本思想。$\diamondsuit$  

在第6.4.6节中，我们将探讨另一种非传统的内积类型：随机变量的内积。

## 3.8 正交投影  

投影是线性变换的一个重要类别（除了旋转和反射），在图形学、编码理论、统计学和机器学习中扮演着重要角色。在机器学习中，我们经常处理高维数据。高维数据往往难以分析或可视化。然而，高维数据通常具有这样的性质：只有少数维度包含大部分信息，而大多数其他维度对于描述数据的关键属性并不重要。当我们压缩或可视化高维数据时，会丢失信息。为了最小化这种压缩损失，我们理想地在数据中找到最具有信息量的维度。正如我们在第1章中讨论的那样，数据可以表示为向量，在本章中，我们将讨论一些用于数据压缩的基本工具。更具体地说，我们可以将原始的高维数据投影到低维特征空间，并在这一低维空间中学习更多关于数据集的信息并提取相关模式。例如，机器学习中的“特征”是一个常见的数据表示术语。

![](images/b3e53c0e81224a2492d48ece4df0e982906250213a9cb2e78cf163f98c419129.jpg)  
图3.9：将二维数据集（蓝色点）投影到一维子空间（直线）上的正交投影（橙色点）。

学习算法，如皮尔逊（1901年）和霍特林（1933年）的主成分分析（PCA）以及深度神经网络（例如深度自动编码器（邓等，2010年）），大量利用了维度减少的思想。接下来，我们将专注于正交投影，我们在第10章中使用它进行线性维度减少，在第12章中用于分类。即使是我们在第9章中讨论的线性回归，也可以通过正交投影来解释。对于给定的低维子空间，正交投影保留了高维数据尽可能多的信息，并最小化原始数据与对应投影之间的差异/误差。图3.9给出了这样的正交投影的一个示例。在详细讨论如何获得这些投影之前，让我们定义什么是投影。


### 定义3.0（投影）：

设$V$为向量空间，$U~\subseteq~V$为$V$的一个子空间。线性映射$\pi\,:\,V\,\rightarrow\,U$称为投影，如果$\pi^{2}=\pi\circ\pi=\pi$。

由于线性映射可以由变换矩阵表示（见第2.7节），上述定义同样适用于特殊类型的变换矩阵，即投影矩阵$P_{\pi}$，它们具有$\boldsymbol{P}_{\pi}^{2}=\boldsymbol{P}_{\pi}$的性质。

接下来，我们将推导内积空间$\left(\mathbb{R}^{n},\left\langle\cdot,\cdot\right\rangle\right)$中向量在子空间上的正交投影。我们将从一维子空间开始，也称为线。如果不特别说明，我们假设点积$\langle{\pmb{x}},{\pmb y}\rangle={\pmb{x}}^{\top}{\pmb y}$作为内积。

## 3.8.1 投影到一维子空间（线）

假设我们给出了一条线（一维子空间），通过基础向量$\pmb{b}\,\in\,\mathbb{R}^{n}$。这条线是一维子空间$U\,\subseteq\,\mathbb{R}^{n}$，由$b$生成。当我们对$\pmb{x}\,\in\,\mathbb{R}^{n}$进行投影到$U$时，我们寻求的是在$\pmb{x}$中与$U$最接近的向量$\pi_{U}({\pmb{x}})\,\in\,U$。使用几何论证，让我们描述投影$\pi_{U}({\pmb{x}})$的一些性质（图3.10(a)作为插图）：
图3.10(a)
![](images/9038b14389c2185e00da9cba67edc85bb72cb8dd82bfa75dab5c9d77f0541959.jpg)
图3.10(b)
![](images/5c2e886575cffb5ddd255e01631f2ccbfce5994fb38715f6db363124287ce451.jpg)
投影$\pi_{U}({\pmb{x}})$与$\pmb{x}$最接近，“最接近”意味着距离$||\pmb{x}-\pi_{U}(\pmb{x})||$最小。因此，从$\pi_{U}({\pmb{x}})$到$\pmb{x}$的线段$\pi_{U}({\pmb{x}})-{\pmb{x}}$与$U$正交，并且因此与$U$的基础向量$b$正交。正交条件给出$\langle\pi_{U}({\pmb{x}})-{\pmb{x}},{\pmb{b}}\rangle=0$，因为向量之间的角度通过内积定义。投影$\pi_{U}({\pmb{x}})$将$x$投影到$U$，必须是$U$中的元素，并且因此是基础向量$b$的倍数，该向量生成$U$。因此，$\pi_{U}({\pmb{x}})=\lambda b$，其中$\lambda\in\mathbb{R}$。

接下来的三个步骤中，我们确定坐标$\lambda$，投影$\pi_{U}({\pmb{x}})\in U$，以及将任何$\pmb{x}\in\mathbb{R}^{n}$映射到$U$的投影矩阵$P_{\pi}$：

1. 寻找坐标$\lambda$。正交条件给出
$$
\langle{\pmb{x}}-\pi_{U}({\pmb{x}}),{\pmb{b}}\rangle=0\,\,{\overset{\pi_{U}({\pmb{x}})=\lambda b}{\longleftrightarrow}}\,\,\langle{\pmb{x}}-\lambda{\pmb{b}},{\pmb{b}}\rangle=0\,。
$$
现在我们可以利用内积的双线性性质，得出
$$
\left\langle{\pmb{x}},{\pmb{b}}\right\rangle-\lambda\left\langle{\pmb{b}},{\pmb{b}}\right\rangle=0\iff\lambda=\frac{\left\langle{\pmb{x}},{\pmb{b}}\right\rangle}{\left\langle{\pmb{b}},{\pmb{b}}\right\rangle}=\frac{\left\langle{\pmb{b}},{\pmb{x}}\right\rangle}{\|{\pmb{b}}\|^{2}}\,。
$$
对于一般的内积，我们得到
$\lambda=\langle{\pmb{x}},{\pmb{b}}\rangle$如果$\|\pmb{b}\|=1$。在最后一步中，我们利用了内积是对称的性质。如果我们选择$\langle\cdot,\cdot\rangle$为点积，我们得到
$$
\lambda={\frac{\boldsymbol{b}^{\top}{\boldsymbol{x}}}{\boldsymbol{b}^{\top}{\boldsymbol{b}}}}={\frac{\boldsymbol{b}^{\top}{\boldsymbol{x}}}{\|{\boldsymbol{b}}\|^{2}}}\,。
$$
如果$\|b\|=1$，则投影的坐标$\lambda$由$\pmb{b}^{\top}\pmb{x}$给出。

2. 寻找投影点$\pi_{U}({\pmb{x}})\in U$。由于$\pi_{U}({\pmb{x}})=\lambda b$，我们立即使用(3.40)得到
$$
\pi_{U}(\pmb{x})=\lambda\pmb{b}=\frac{\langle\pmb{x},\pmb{b}\rangle}{\|\pmb{b}\|^{2}}\pmb{b}=\frac{\pmb{b}^{\top}\pmb{x}}{\|\pmb{b}\|^{2}}\pmb{b}\,，
$$
其中最后一个等式仅对点积成立。我们还可以通过定义3.1计算$\pi_{U}({\pmb{x}})$的长度
$$
\left\|\pi_{U}({\pmb{x}})\right\|=\left\|\lambda{\pmb{b}}\right\|=\left|\lambda\right|\left\|{\pmb{b}}\right\|。
$$
因此，我们的投影长度为$|\lambda|$乘以$b$的长度。这也增加了直观性，即$\pi_{U}({\pmb{x}})$相对于生成我们一维子空间$U$的基础向量$b$的坐标是$|\lambda|$。如果使用点积作为内积，我们得到
$$
\Vert\pi_{U}({\pmb{x}})\Vert\stackrel{(3.42)}{=}\frac{|{\pmb{b}}^{\top}{\pmb{x}}|}{\Vert{\pmb{b}}\Vert^{2}}\,\Vert{\pmb{b}}\Vert\stackrel{(3.25)}{=}|\cos\omega|\,\Vert{\pmb{x}}\Vert\,\Vert{\pmb{b}}\Vert\frac{\Vert{\pmb{b}}\Vert}{\Vert{\pmb{b}}\Vert^{2}}=|\cos\omega|\,\Vert{\pmb{x}}\Vert\,。
$$
这里的$\omega$是$\pmb{x}$和$b$之间的角度。这个等式应该熟悉于三角学：如果$\|\pmb{x}\|=1$，则$\pmb{x}$位于单位圆上。由此可知，将$\pmb{x}$投影到由$b$生成的水平轴上正好是$\cos\omega$，对应向量$\pi_{U}({\pmb{x}})$的长度为$|\cos\omega|$。图3.10(b)给出了这个设置的插图。

3. 投影矩阵$P_{\pi}$的寻找

我们知道投影是一个线性映射（见定义3.10）。因此，存在一个投影矩阵$P_{\pi}$，使得$\pi_{U}({\pmb{x}})\,=\,P_{\pi}{\pmb{x}}$。使用点积作为内积和

$$
\pi_{U}({\pmb{x}})=\lambda{\pmb{b}}=b\lambda=b\frac{{\pmb{b}}^{\top}{\pmb{x}}}{\|{\pmb{b}}\|^{2}}=\frac{{\pmb{b}}^{\top}}{\|{\pmb{b}}\|^{2}}{\pmb{x}}\,，
$$
我们立即看到

$$
P_{\pi}={\frac{b b^{\top}}{\|b\|^{2}}}\,。
$$
投影矩阵$P_{\pi}$总是对称的。

注意$\boldsymbol{b}\boldsymbol{b}^{\top}$（以及因此$P_{\pi}$）是一个对称矩阵（秩为1），且$\|\pmb{b}\|^{2}=\langle\pmb{b},\pmb{b}\rangle$是一个标量。

投影矩阵$P_{\pi}$将$\pmb{x}\in\mathbb{R}^{n}$投影到通过原点的方向为$b$的线（等价地，由$b$生成的子空间$U$）。

注释。投影$\pi_{U}({\pmb{x}})\in\mathbb{R}^{n}$仍然是一个$n$维向量，而不是标量。然而，我们不再需要$n$个坐标来表示投影，如果我们希望以基础向量$b$为基准来表示它，只需要一个坐标$\lambda$。$\diamondsuit$

![](images/44a6cc449d46e0f1cb7b538d2c490e67e4dfdbd19632ffb832f63d34283ab1ad.jpg)  
图3.11：投影到二维子空间$U$，其基为${\pmb{b}}_{1},{\pmb{b}}_{2}$。$\pmb{x}\in\mathbb{R}^{3}$投影到$U$可以表示为${\pmb{b}}_{1},{\pmb{b}}_{2}$的线性组合，而$\pmb{x}-\pi_{U}({\pmb{x}})$与$b_{1}$和$b_{2}$都正交。

### 示例3.10（线上的投影）

求投影矩阵$P_{\pi}$，将向量投影到通过原点的由$\pmb{b}=\bar{\left[\begin{array}{l l l}{1}&{2}&{2}\end{array}\right]}^{\top}$生成的线（通过原点的线）。$\pmb{b}$是方向和一维子空间（通过原点的线）的基础。

使用(3.46)，我们得到

$$
\pmb{P}_{\pi}=\frac{{\pmb{b}}{\pmb{b}}^{\top}}{{\pmb{b}}^{\top}{\pmb{b}}}=\frac{1}{9}\left[\begin{array}{c c c}{1}&{2}&{2}\end{array}\right]=\frac{1}{9}\left[\begin{array}{c c c}{1}&{2}&{2}\\ {2}&{4}&{4}\\ {2}&{4}&{4}\end{array}\right]\,。
$$
现在让我们选择一个特定的$\pmb{x}$，看看它是否位于由$\pmb{b}$生成的子空间中。对于$\pmb{x}=\left[1\quad1\quad1\right]^{\top}$，投影是

$$
\pi_{U}({\pmb{x}})=P_{\pi}{\pmb{x}}=\frac{1}{9}\left[\begin{array}{c c c}{1}&{2}&{2}\\ {2}&{4}&{4}\\ {2}&{4}&{4}\end{array}\right]\left[\begin{array}{c}{1}\\ {1}\\ {1}\end{array}\right]=\frac{1}{9}\left[\begin{array}{c}{5}\\ {10}\\ {10}\end{array}\right]\in\mathrm{span}[\left[\begin{array}{c}{1}\\ {2}\\ {2}\end{array}\right]。
$$
注意，将$P_{\pi}$应用于$\pi_{U}({\pmb{x}})$不会改变任何事情，即${\cal P}_{\pi}\pi_{U}({\pmb{x}})=\pi_{U}({\pmb{x}})$。这是预期的，因为根据定义3.10，我们知道投影矩阵$P_{\pi}$满足对于所有$\pmb{x}$，${\cal P}_{\pi}^{2}{\pmb{x}}={\cal P}_{\pi}{\pmb{x}}$。

注释。根据第4章的结果，我们可以证明$\pi_{U}({\pmb{x}})$是$P_{\pi}$的特征向量，对应的特征值是1。$\diamondsuit$

### 3.8.2 对一般子空间的投影

接下来，我们考虑将向量$\pmb{x}\ \in\ \mathbb{R}^{n}$投影到$\mathbb{R}^{n}$中的低维子空间$U\,\subseteq\,\mathbb{R}^{n}$。图3.11给出了一个示例。

假设$(\pmb{b}_{1},\dots,\pmb{b}_{m})$是$U$的有序基。任何投影$\pi_{U}({\pmb{x}})$到$U$的向量必须是$U$的元素，因此可以表示为基向量$\pmb{b}_{1},\ldots,\pmb{b}_{m}$的线性组合，使得$\begin{array}{r}{\pi_{U}({\pmb{x}})=\sum_{i=1}^{m}\lambda_{i}{\pmb{b}}_{i}}\end{array}$。如同一维情况，我们遵循三个步骤来找到投影$\pi_{U}({\pmb{x}})$和投影矩阵$P_{\pi}$：

1. 找到投影$\lambda_{1},\cdots,\lambda_{m}$的坐标（相对于$U$的基），使得线性组合

$$
\begin{array}{r l}&{\pi_{U}(\pmb{x})=\displaystyle\sum_{i=1}^{m}\lambda_{i}\pmb{b}_{i}=B\pmb{\lambda}\,,}\\ &{\pmb{B}=[\pmb{b}_{1},\cdots,\pmb{b}_{m}]\in\mathbb{R}^{n\times m},\quad\pmb{\lambda}=[\lambda_{1},\cdots,\lambda_{m}]^{\top}\in\mathbb{R}^{m}\,,}\end{array}
$$
最接近$\pmb{x}\ \in\ \mathbb{R}^{n}$。如同一维情况，“最接近”意味着连接$\pi_{U}({\pmb{x}})\,\in\,U$和$\pmb{x}\,\in\,\mathbb{R}^{n}$的向量必须与$U$的所有基向量正交。因此，我们得到$m$个同时条件（假设点积作为内积）

$$
\begin{array}{l}{{\langle\pmb{b}_{1},\pmb{x}-\pi_{U}(\pmb{x})\rangle=\pmb{b}_{1}^{\top}(\pmb{x}-\pi_{U}(\pmb{x}))=0}}\\ {{\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\vdots}}\\ {{\langle\pmb{b}_{m},\pmb{x}-\pi_{U}(\pmb{x})\rangle=\pmb{b}_{m}^{\top}(\pmb{x}-\pi_{U}(\pmb{x}))=0}}\end{array}
$$
，其中$\pi_{U}({\pmb{x}})=B\lambda$，可以写为

$$
\begin{array}{c}{{{\pmb{b}}_{1}^{\top}({\pmb{x}}-{\pmb{B}}{\pmb\lambda})=0}}\\ {{\vdots}}\\ {{{\pmb{b}}_{m}^{\top}({\pmb{x}}-{\pmb{B}}{\pmb\lambda})=0}}\end{array}
$$
，这样我们得到一个齐次线性方程组

$$
\begin{bmatrix}  
b_1^T \\
\vdots \\
b_m^T  
\end{bmatrix}  
\left[ x - B\lambda \right] = 0 \quad \Leftrightarrow \quad B^T (x - B\lambda) = 0 
$$

$$
\Leftrightarrow \quad B^T B \lambda = B^T x.
$$

正则方程伪逆

最后一个表达式称为正则方程。由于$\pmb{b}_{1},\ldots,\pmb{b}_{m}$是$U$的基，因此线性独立，$\boldsymbol{B}^{\intercal}\boldsymbol{B}\in\mathbb{R}^{m\times m}$是可逆的，可以求逆。这允许我们求解系数/坐标

$$
\pmb{\lambda}=(\pmb{B}^{\top}\pmb{B})^{-1}\pmb{B}^{\top}\pmb{x}\,。
$$
矩阵$(\boldsymbol{B}^{\intercal}\boldsymbol{B})^{-1}\boldsymbol{B}^{\intercal}$也称为$B$的伪逆，可以计算非方阵$B$。它只需要$\boldsymbol{B}^{\intercal}\boldsymbol{B}$是正定的，这在$B$是满秩的情况下成立。在实际应用中（例如线性回归），我们经常添加一个“抖动项”$\epsilon I$到$\boldsymbol{B}^{\intercal}\boldsymbol{B}$以确保数值稳定性和正定性。这种“岭”可以通过贝叶斯推断严格推导。详情请参阅第9章。

### 示例3.11（投影到二维子空间）

对于子空间$U=\operatorname{span}\left[\begin{bmatrix}1\\1\\1\end{bmatrix},\begin{bmatrix}0\\1\\2\end{bmatrix}\right]\subseteq\mathbb{R}^{3}$和$\pmb{x}=\left[\begin{array}{l}{6}\\ {0}\\ {0}\end{array}\right]\in\mathbb{R}^{3}$，我们需要找到坐标$\lambda$在子空间$U$中的$x$的表示，投影点$\pi_{U}({\pmb{x}})$和投影矩阵$P_{\pi}$。

首先，我们看到生成集$U$是一个基础（线性独立），并写出$U$的基向量矩阵$B=\left[{\begin{array}{r r}{1}&{0}\\ {1}&{1}\\ {1}&{2}\end{array}}\right]$。其次，我们计算矩阵$\pmb{B}^{\top}\pmb{B}$和向量$\pmb{B}^{\top}\pmb{x}$为

$$
\pmb{B}^{\top}\pmb{B}=\left[\begin{array}{r r}{1}&{1}&{1}\\ {0}&{1}&{2}\end{array}\right]\left[\begin{array}{r r}{1}&{0}\\ {1}&{1}\\ {1}&{2}\end{array}\right]=\left[\begin{array}{r r}{3}&{3}\\ {3}&{5}\end{array}\right]\,,\quad\pmb{B}^{\top}\pmb{x}=\left[\begin{array}{r r}{1}&{1}&{1}\\ {0}&{1}&{2}\end{array}\right]\left[\begin{array}{l}{6}\\ {0}\\ {0}\end{array}\right]=\left[\begin{array}{l}{6}\\ {0}\end{array}\right]\,。
$$
第三步，我们解正则方程$\pmb{B}^{\top}\pmb{B}\lambda=\pmb{B}^{\top}\pmb{x}$以找到$\lambda$：

$$
\begin{array}{r}{\Big[\begin{array}{c c}{3}&{3}\\ {3}&{5}\end{array}\Big]\;\Big[\lambda_{1}\Big]=\Big[\begin{array}{c c}{6}\\ {0}\end{array}\Big]\iff\lambda=\Big[\begin{array}{c c}{5}\\ {-3}\end{array}\Big]\ .}\end{array}
$$
第四步，将$x$投影到$U$，即投影到矩阵$B$的列空间，可以直接计算为

$$
\pi_{U}({\pmb{x}})=\pmb{B}\pmb{\lambda}=\left[{5\atop2}\atop-1\right]\,。
$$
投影误差投影误差也称为重构误差。

对应的投影误差是原始向量与投影到$U$的向量之间的差向量的范数，即

$$
\begin{array}{r}{\|\pmb{x}-\pi_{U}(\pmb{x})\|=\Big|\Big|\big[1\quad-2\quad1\big]^{\top}\Big|\Big|=\sqrt{6}\,.}\end{array}
$$
第五步，对于任何$\pmb{x}\in\mathbb{R}^{3}$，投影矩阵给出为

$$
\pmb{P}_{\pi}=\pmb{B}(\pmb{B}^{\top}\pmb{B})^{-1}\pmb{B}^{\top}=\frac{1}{6}\left[\begin{array}{l l l}{5}&{2}&{-1}\\ {2}&{2}&{2}\\ {-1}&{2}&{5}\end{array}\right]\,。
$$
为了验证结果，我们可以（a）检查$\pi_{U}({\pmb{x}})\,-\,{\pmb{x}}$是否与$U$的所有基向量正交，以及（b）验证$P_{\pi}=P_{\pi}^{2}$（见定义3.10）。

注释。投影$\pi_{U}({\pmb{x}})$虽然是$\mathbb{R}^{n}$中的向量，但它们位于$m$维子空间$U\,\subseteq\,\mathbb{R}^{n}$中。然而，为了表示投影向量，我们只需要$m$个坐标$\lambda_{1},\dots,\lambda_{m}$，相对于$U$的基向量$b_{1},\ldots,b_{m}$。$\diamondsuit$我们可以通过投影来找到不可解线性方程系统的近似解。

 最小二乘解

**注释。** 在一般内积的向量空间中，当我们计算角度和距离时，必须注意，这些是由内积定义的。$\diamondsuit$投影允许我们考虑线性系统${\pmb A x}={\pmb{b}}$没有解的情况。回想一下，这意味着${\pmb{b}}$不在$\pmb{A}$的跨度中，即向量${\pmb{b}}$不在由$\pmb{A}$的列生成的子空间中。鉴于线性方程无法精确解决，我们可以找到一个近似解。想法是找到在$\pmb{A}$的列生成的子空间中与${\pmb{b}}$最接近的向量，即计算${\pmb{b}}$到$\pmb{A}$的列生成的子空间的正交投影。这个问题在实践中经常出现，解决方案称为过定系统的最小二乘解（假设点积作为内积）。这部分在第9.4节中有更详细的讨论。使用重构误差（3.63）是导出主成分分析（第10.3节）的一种可能方法。

注释。我们刚刚考虑了将向量$\pmb{x}$投影到子空间$U$，其基向量为$\{\boldsymbol{b}_{1},\ldots,\boldsymbol{b}_{k}\}$。如果这个基是正交基，即（3.33）和（3.34）得到满足，投影方程（3.58）大大简化为

$$
\pi_{U}({\pmb{x}})={\pmb{b}}{\pmb{b}}^{\top}{\pmb{x}}
$$
因为$\boldsymbol{B}^{\intercal}\boldsymbol{B}=\boldsymbol{I}$，坐标为

$$
\pmb{\lambda}=\pmb{B}^{\top}\pmb{x}\,。
$$
这意味着我们不再需要从（3.58）计算逆矩阵，这节省了计算时间。$\diamondsuit$

### 3.8.3 格拉姆-施密特正交化

投影是格拉姆-施密特方法的核心，该方法允许我们从$n$维向量空间$V$中的任意基$(\pmb{b}_{1},.\cdot\cdot,\pmb{b}_{n})$构造出一个正交/正规化的基$(\pmb{u}_{1},\cdot\cdot\cdot,\pmb{u}_{n})$。这样的基总是存在的（Liesen和Mehrmann, 2015），且$\left[b_{1},\dots,b_{n}\right]=$ $\operatorname{span}[\pmb{u}_{1},.\,.\,,\pmb{u}_{n}]$。格拉姆-施密特正交化方法通过以下步骤从任意基$(\pmb{b}_{1},.\cdot\cdot,\pmb{b}_{n})$构造出$V$的正交基$(\pmb{u}_{1},\cdot\cdot\cdot,\pmb{u}_{n})$：

格拉姆-施密特正交化

$$
\begin{array}{r l}&{\pmb{u}_{1}:=\pmb{b}_{1}}\\ &{\pmb{u}_{k}:=\pmb{b}_{k}-\pi_{\mathrm{span}[\pmb{u}_{1},\ldots,\pmb{u}_{k-1}]}(\pmb{b}_{k})\,,\quad k=2,\cdots,n\,.}\end{array}
$$
在(3.68)中，第$k$个基向量$b_{k}$被投影到由前$k-1$个构造出的正交向量$\pmb{u}_{1},\dots,\pmb{u}_{k-1}$生成的子空间上；见第3.8.2节。然后从$b_{k}$中减去这个投影，得到一个向量$\pmb{u}_{k}$，该向量与$(k\mathrm{~-~}1)$维子空间$\pmb{u}_{1},.\cdot\cdot,\pmb{u}_{k-1}$正交。重复此过程对所有$n$个基向量$b_{1},\ldots,b_{n}$，即可得到$V$的正交基$(\pmb{u}_{1},\cdot\cdot\cdot,\pmb{u}_{n})$。如果我们将$\pmb{u}_{k}$归一化，我们得到一个正规化基，其中$\|\pmb{u}_{k}\|=1$对$k=1,\ldots,n$成立。

### 示例3.12（格拉姆-施密特正交化）

![](images/6fcea0d5c9947938b0de5787df0e8b47bfeec7cd38cac1820d10850cf766b14a.jpg)  
图3.12 格拉姆-施密特正交化。 (a) $\mathbb{R}^{2}$中的非正交基$(\pmb{b}_{1},\pmb{b}_{2})$；(b) 第一个构造出的基向量$\mathbf{\mathit{u}}_{1}$和向量$b_{2}$在$\mathrm{span[}\pmb{u}_{1}]$上的正交投影；(c) $\mathbb{R}^{2}$中的正交基$(\pmb{u}_{1},\pmb{u}_{2})$。

$$
b_{1}={\binom{2}{0}}\ ,\quad b_{2}={\binom{1}{1}}\ ;
$$
参见图3.12(a)。使用格拉姆-施密特方法，我们构造出$\mathbb{R}^{2}$的正交基$(\pmb{u}_{1},\pmb{u}_{2})$如下（假设点积作为内积）：

$$
\begin{array}{r l r}&{\mathbf{u}_{1}:=\mathbf{b}_{1}=\bigg[\begin{array}{c}{2}\\ {0}\end{array}\bigg]\;,}&{(3.70)}\\ &{\mathbf{u}_{2}:=\mathbf{b}_{2}-\pi_{\mathrm{span}[\mathbf{u}_{1}]}(\mathbf{b}_{2})\overset{(3.45)}{=}\mathbf{b}_{2}-\frac{\mathbf{u}_{1}\mathbf{u}_{1}^{\top}}{\left\Vert\mathbf{u}_{1}\right\Vert^{2}}\mathbf{b}_{2}=\bigg[\begin{array}{c}{1}\\ {1}\end{array}\bigg]-\bigg[\begin{array}{c}{1}\\ {0}\end{array}\bigg]\overset{[1]}{\leq}\bigg[\begin{array}{c}{0}\\ {1}\end{array}\bigg]\;.}\end{array}
$$
![](images/0026d8f353c1dbdee971e00acdf8a391395ac801050818963b3fcf5cbfa34b25.jpg)  
图3.13 投影到一个仿射空间。 (a) 原始设置；(b) 通过${\mathbf{-}}{\mathbf{x}}_{\mathrm{0}}$将设置移位，使得${\pmb{x}}-{\pmb{x}}_{0}$可以投影到方向空间$U$；(c) 投影被移回到${\pmb{x}}_{0}+{\pmb\pi}_{U}({\pmb{x}}-{\pmb{x}}_{0})$，给出最终的正交投影$\pi_{L}({\pmb{x}})$。

这些步骤在图3.12(b)和(c)中进行了说明。我们立即看到$\pmb{u}_{1}$和$\pmb{u}_{2}$是正交的，即$\mathbf{u}_{1}^{\top}\mathbf{u}_{2}=0$。


### 3.8.4 投影到仿射子空间

到目前为止，我们讨论了如何将向量投影到较低维的子空间$U$。接下来，我们将提供一种将向量投影到仿射子空间的解决方案。

考虑图3.13(a)中的设置。我们给定一个仿射空间$L=\pmb{x}_{0}+U$，其中$b_{1},b_{2}$是$U$的基向量。为了确定向量$x$在$L$上的正交投影$\pi_{L}({\pmb{x}})$，我们将问题转化为一个我们已知如何解决的问题：将向量投影到子空间。为了达到这个目的，我们从$x$和$L$中减去支撑点$\pmb{x}_{0}$，使得$L-\pmb{x}_{0}=U$恰好是向量子空间$U$。我们现在可以使用在第3.8.2节中讨论的子空间上的正交投影，并得到投影$\pi_{U}({\pmb{x}}-{\pmb{x}}_{0})$，如图3.13(b)所示。这个投影现在可以通过将$\pmb{x}_{0}$加回来，从而将投影转换回$L$，这样我们就可以得到仿射空间$L$上的正交投影为

$$
\pi_{L}({\pmb{x}})={\pmb{x}}_{0}+\pi_{U}({\pmb{x}}-{\pmb{x}}_{0})\,,
$$
其中$\pi_{U}(\cdot)$是向子空间$U$（即$L$的方向空间）的正交投影，如图3.13(c)所示。

从图3.13中可以看出，向量$x$与仿射空间$L$之间的距离与$x-\pmb{x}_{0}$与$U$之间的距离相同，即

$$
\begin{array}{c}{{d({\pmb{x}},L)=\|{\pmb{x}}-\pi_{L}({\pmb{x}})\|=\|{\pmb{x}}-({\pmb{x}}_{0}+\pi_{U}({\pmb{x}}-{\pmb{x}}_{0}))\|}}\\ {{=d({\pmb{x}}-{\pmb{x}}_{0},\pi_{U}({\pmb{x}}-{\pmb{x}}_{0}))=d({\pmb{x}}-{\pmb{x}}_{0},U)\,.}}\end{array}
$$
我们将使用仿射子空间上的投影来在第12.1节中推导分离超平面的概念。

![](images/b27f4c3c82c42bcb93d185f291ff371ca851d8c27f4e520e9857d7c4afca19c3.jpg)  
## 3.9 旋转

如第3.4节所述，线性映射具有正交变换矩阵的两个特点是长度和角度的保持。接下来，我们将更详细地研究描述旋转的特定正交变换矩阵。

旋转是一种线性映射（更具体地说，是欧几里得向量空间的自同构），它绕原点旋转平面一个角度$\theta$，即原点是一个固定点。对于正角度$\theta>0$，通常我们按照逆时针方向旋转。一个例子如图3.14所示，其中变换矩阵为

旋转矩阵
$$
\pmb{R}=\left[\begin{array}{l l}{-0.38}&{-0.92}\\ {0.92}&{-0.38}\end{array}\right]\,.
$$
旋转的应用领域包括计算机图形学和机器人学。例如，在机器人学中，了解如何旋转机器人臂的关节以拾取或放置物体通常很重要，如图3.15所示。

  

![](images/26826a632b7e25f313c6a9b6fb91eec2ad8635a64bd3b9c997562016b74fb5ef.jpg)  
图3.16 在$\mathbb{R}^{2}$中绕角度$\theta$旋转标准基。

### 3.9.1 二维空间中的旋转

考虑$\mathbb{R}^{2}$的标准基$\left\{e_{1}=\left[\begin{array}{c}1\\0\end{array}\right], e_{2}=\left[\begin{array}{c}0\\1\end{array}\right]\right\}$，它定义了$\mathbb{R}^{2}$中的标准坐标系统。我们的目标是通过角度$\theta$旋转这个坐标系统，如图3.16所示。请注意，旋转后的向量仍然线性独立，因此是$\mathbb{R}^{2}$的一个基。这意味着旋转执行了基的改变。

旋转矩阵

旋转$\Phi$是线性映射，因此我们可以用旋转矩阵$R(\theta)$来表示它们。三角函数（见图3.16）允许我们确定旋转轴的坐标（$\Phi$的像）相对于$\mathbb{R}^{2}$中的标准基。我们得到

$$
\Phi(e_{1})=\left[\begin{array}{c}
\cos\theta \\
\sin\theta
\end{array}\right],\quad
\Phi(e_{2})=\left[\begin{array}{c}
-\sin\theta \\
\cos\theta
\end{array}\right].
$$

因此，执行基改变到旋转坐标$R(\theta)$的旋转矩阵是

$$
{\pmb R}(\theta)=\left[\Phi(e_{1})\quad\Phi(e_{2})\right]=\left[\begin{array}{cc}
\cos\theta &-\sin\theta \\
\sin\theta &\cos\theta
\end{array}\right].
$$

### 3.9.2 三维空间中的旋转

与$\mathbb{R}^{2}$的情况不同，在$\mathbb{R}^{3}$中，我们可以围绕一条一维轴旋转任何二维平面。指定一般旋转矩阵的最简单方法是指定标准基向量$e_{1},e_{2},e_{3}$的像应该如何旋转，并确保这些像$R e_{1}$，$R e_{2}$，$R e_{3}$彼此正交。然后，我们可以通过组合标准基的像来获得一个通用的旋转矩阵$R$。

为了有意义地定义旋转角度，我们需要定义当我们操作在三维以上时，“逆时针”意味着什么。我们采用的约定是，当从轴的“末端向原点”看时，轴上的“逆时针”（平面）旋转是指围绕轴的旋转。在$\mathbb{R}^{3}$中，因此有三个（平面）围绕三个标准基向量的旋转（见图3.17）：

![](images/2771294af6f3ffae8edc1256076403d0f12668dbb9dc54a4be8b29baf3e30479.jpg)  
图3.17 在$\mathbb{R}^{3}$中，围绕$e_{3}$轴以角度$\theta$旋转向量（灰色）。旋转后的向量用蓝色表示。

围绕$e_{1}$轴的旋转

$$
{\pmb R}_{1}(\theta)=\left[\Phi(e_{1})\quad\Phi(e_{2})\quad\Phi(e_{3})\right]=\left[\begin{array}{c c c}{1}&{0}&{0}\\ {0}&{\cos\theta}&{-\sin\theta}\\ {0}&{\sin\theta}&{\cos\theta}\end{array}\right]\,.
$$
在这里，$e_{1}$坐标保持不变，我们在$e_{2}e_{3}$平面上执行逆时针旋转。

围绕$e_{2}$轴的旋转

$$
{\cal R}_{2}(\theta)=\left[\begin{array}{c c c}{{\cos\theta}}&{{0}}&{{\sin\theta}}\\ {{0}}&{{1}}&{{0}}\\ {{-\sin\theta}}&{{0}}&{{\cos\theta}}\end{array}\right]\,.
$$
如果我们围绕$e_{2}$轴旋转$e_{1}e_{3}$平面，我们需要从$e_{2}$轴的“尖端”向原点看。

围绕$e_{3}$轴的旋转

$$
\begin{array}{r}{\pmb{R}_{3}(\theta)=\left[\begin{array}{c c c}{\cos\theta}&{-\sin\theta}&{0}\\ {\sin\theta}&{\cos\theta}&{0}\\ {0}&{0}&{1}\end{array}\right]\,.}\end{array}
$$
图3.17说明了这一点。

### 3.9.3 任意维度空间中的旋转

从二维和三维推广到n维欧几里得向量空间的旋转的一般化可以直观地描述为固定n-2维，并将旋转限制在n维空间中的二维平面。如同三维情况一样，我们可以旋转任何平面（$\textstyle\mathbb{R}^{n}$的二维子空间）。

定义3.11（盖文斯旋转）. 让$V$为一个n维欧几里得向量空间，$\Phi:V\to V$为一个自同构，其变换矩阵为

$$
\begin{array}{r}{R_{i j}(\theta):=\left[\begin{array}{c c c c c}{I_{i-1}}&{\mathbf{0}}&{\cdots}&{\cdots}&{\mathbf{0}}\\ {\mathbf{0}}&{\cos\theta}&{\mathbf{0}}&{-\sin\theta}&{\mathbf{0}}\\ {\mathbf{0}}&{\mathbf{0}}&{I_{j-i-1}}&{\mathbf{0}}&{\mathbf{0}}\\ {\mathbf{0}}&{\sin\theta}&{\mathbf{0}}&{\cos\theta}&{\mathbf{0}}\\ {\mathbf{0}}&{\cdots}&{\cdots}&{\mathbf{0}}&{I_{n-j}}\end{array}\right]\in\mathbb{R}^{n\times n}\,,}\end{array}
$$
对于$q\,\leqslant\,i\,<\,j\,\leqslant\,n$和$\theta\,\in\,\mathbb{R}$。那么$R_{i j}(\theta)$被称为盖文斯旋转。本质上，$R_{i j}(\theta)$是n阶单位矩阵$I_{n}$，其中

$$
r_{i i}=\cos\theta\,,\quad r_{i j}=-\sin\theta\,,\quad r_{j i}=\sin\theta\,,\quad r_{j j}=\cos\theta\,。
$$
在二维（即，$n=2.$）情况下，我们得到（3.76）作为特殊情况。

### 3.9.4 旋转的性质

旋转表现出一系列有用的性质，这些性质可以通过将它们视为正交矩阵（定义3.8）来推导：

旋转保持距离，即$\|\pmb{x}-\pmb{y}\|=\|\pmb{R_{\theta}(x)}-\pmb{R_{\theta}(y)}\|$。换句话说，旋转在变换后不会改变任意两点之间的距离。旋转保持角度，即$\mathbf{\Delta}R_{\theta}\mathbf{\Delta}x$和$\scriptstyle R_{\theta}y$之间的角度等于$\pmb{x}$和$\pmb{ y}$之间的角度。三维（或更多）空间中的旋转通常不满足交换律。因此，应用旋转的顺序很重要，即使它们围绕相同的点旋转。只有在二维空间中向量旋转是可交换的，即$\pmb{R}(\phi)\pmb{R}(\theta)\,=\,\pmb{R}(\theta)\pmb{R}(\phi)$对于所有$\phi,\theta\in[0,2\pi)$成立。如果它们围绕相同的点旋转（例如，原点），它们仅形成一个阿贝尔群（乘法）。

## 3.10 进一步阅读

在本章中，我们对分析几何的一些重要概念进行了简要概述，这些概念将在本书后续章节中使用。对于一些我们介绍的概念的更广泛和深入的概述，我们推荐以下优秀的书籍：Axler (2015) 和 Boyd and Vandenberghe (2018)。

内积允许我们确定向量（子）空间的特定基，其中每个向量与其他向量正交（正交基）使用格拉姆-施密特方法。这些基在优化和求解线性方程系统时的算法中非常重要。例如，Krylov子空间方法，如共轭梯度或广义最小残差方法（GMRES）（Stoer and Burlirsch, 2002），最小化彼此正交的残差误差。

在机器学习中，内积在核方法（Sch¨ olkopf and Smola, 2002）的背景下非常重要。核方法利用了许多线性算法可以仅通过内积计算来表达的事实。然后，“核技巧”允许我们在（可能无限维的）特征空间中隐式计算这些内积，甚至无需明确知道这个特征空间。这使得许多在机器学习中使用的算法非线性化，例如用于降维的核主成分分析（Sch¨ olkopf et al., 1997）。高斯过程（Rasmussen and Williams, 2006）也属于核方法类别，并且是概率回归（拟合数据点的曲线）的当前前沿技术。核的概念在第12章中进一步探讨。

投影在计算机图形学中经常使用，例如生成阴影。在优化中，通常使用正交投影（迭代）来最小化残差误差。这在机器学习中也有应用，例如在线性回归中，我们希望找到一个（线性）函数，它最小化残差误差，即数据对线性函数的正交投影的长度（Bishop, 2006）。我们将在第9章中进一步探讨这一点。主成分分析（Pearson, 1901; Hotelling, 1933）也使用投影来减少高维数据的维度。我们将在第10章中更详细地讨论这一点。

## 练习  

3.1 证明对于所有$\pmb{x}=[x_{1},x_{2}]^{\top}\in\mathbb{R}^{2}$和$\pmb{y}=[y_{1},y_{2}]^{\top}\in\mathbb{R}^{2}$，定义为$\langle{\pmb{x}},{\pmb y}\rangle:=x_{1}y_{1}-\left(x_{1}y_{2}+x_{2}y_{1}\right)+2(x_{2}y_{2})$的$\langle\cdot,\cdot\rangle$是内积。  

3.2 考虑$\mathbb{R}^{2}$，对于$\mathbb{R}^{2}$中的所有$\pmb{x}$和$\pmb{y}$，定义为$\langle\pmb{x},\pmb{y}\rangle:=\pmb{x}^{\top}\underbrace{\left[\begin{array}{r r}{2}&{0}\\ {1}&{\!2}\end{array}\right]}_{=:\pmb{A}}\pmb{y}$的$\langle\cdot,\cdot\rangle$。$\langle\cdot,\cdot\rangle$是内积吗？  

3.3 计算$\pmb{x}=\left[\begin{array}{c}{1}\\ {2}\\ {3}\end{array}\right]$和$\pmb y=\left[\!\begin{array}{c}{-1}\\ {-1}\\ {0}\end{array}\right]$之间的距离，使用  

a.   $\langle\pmb{x},\pmb{y}\rangle:=\pmb{x}^{\top}\pmb{y}$

3.4 计算向量$\pmb{x}=\left[\begin{array}{l}{1}\\ {2}\end{array}\right]$和$\pmb y=\left[\!\begin{array}{l}{-1}\\ {-1}\end{array}\right]$之间的夹角，使用  

3.5 考虑欧几里得向量空间$\mathbb{R}^{5}$，使用点积。给定子空间$U\subseteq\mathbb{R}^{5}$和$\pmb{x}\in\mathbb{R}^{5}$为  

$$
U=\operatorname{span}\left[\left[\begin{array}{l}{0}\\ {-1}\\ {2}\\ {0}\\ {2}\end{array}\right],\left[\begin{array}{l}{1}\\ {-3}\\ {1}\\ {-1}\\ {2}\end{array}\right],\left[\begin{array}{l}{-3}\\ {4}\\ {1}\\ {2}\\ {1}\end{array}\right],\left[\begin{array}{l}{-1}\\ {-3}\\ {5}\\ {0}\\ {7}\end{array}\right]\right],\quad\mathbf{x}=\left[\begin{array}{l}{-1}\\ {-9}\\ {-1}\\ {4}\\ {1}\end{array}\right]~.
$$
a. 确定$\pi_{U}({\pmb{x}})$，即$x$在$U$上的正交投影。  

b. 计算距离$d({\pmb{x}},U)$。  

3.6 考虑$\mathbb{R}^{3}$，使用内积  

$$
\langle\pmb{x},\pmb{y}\rangle:=\pmb{x}^{\top}\left[\begin{array}{c c c}{2}&{1}&{0}\\ {1}&{2}&{-1}\\ {0}&{-1}&{2}\end{array}\right]\pmb{y}\,.
$$
此外，定义$e_{1},e_{2},e_{3}$为$\mathbb{R}^{3}$的标准/标准基。  

a. 确定$e_{2}$在$U=\operatorname{span}[e_{1},e_{3}]$上的正交投影$\pi_{U}(e_{2})$。  

提示：正交性通过内积定义。  

b. 计算距离$d(e_{2},U)$。  

c. 绘制场景：标准基向量和$\pi_{U}(e_{2})$。  

3.7 让$V$为一个向量空间，$\pi$为$V$的同态。  

a. 证明$\pi$为投影当且仅当$\mathrm{id}_{V}-\pi$也是投影，其中$\mathrm{id}_{V}$是$V$上的单位同态。  

b. 假设$\pi$为投影。计算$\mathrm{Im}(\mathrm{id}_{V}\!-\!\pi)$和$\ker(\mathrm{id}_{V}\!-\!\pi)$作为$\operatorname{Im}(\pi)$和$\ker(\pi)$的函数。


3.8 使用格拉姆-施密特方法，将二维子空间$U\subseteq\mathbb{R}^{3}$的基$B=(b_{1},b_{2})$转换为$U$的正交基$C=(\pmb{c}_{1},\pmb{c}_{2})$，其中  

$$
b_{1}:={\small\left[\begin{array}{c}{1}\\ {1}\\ {1}\end{array}\right]}\ ,\quad b_{2}:={\small\left[\begin{array}{c}{-1}\\ {2}\\ {0}\end{array}\right]}\ .
$$
3.9 让$n\in\mathbb{N}$，并且让$x_{1},\ldots,x_{n}\,>\,0$为$n$个正实数，使得$x_{1}+\cdot\cdot+x n=1$。使用柯西-施瓦茨不等式并证明  

a.   $\textstyle\sum_{i=1}^{n}x_{i}^{2}\geqslant{\frac{1}{n}}$ $\textstyle\sum_{i=1}^{n}{\frac{1}{x_{i}}}\geqslant n^{2}$ 提示：考虑$\textstyle\mathbb{R}^{n}$上的点积。然后，选择特定的向量$\pmb{x},\pmb{y}\in\mathbb{R}^{n}$，并应用柯西-施瓦茨不等式。  

3.10 旋转向量  

$$
\pmb{x}_{1}:=\left[\begin{array}{c}{2}\\ {3}\end{array}\right]\,,\quad\pmb{x}_{2}:=\left[\begin{array}{c}{0}\\ {-1}\end{array}\right]
$$
以$30^{\circ}$的角度旋转。
