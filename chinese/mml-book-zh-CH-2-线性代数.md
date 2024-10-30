
# 第二章 线性代数  

在将直观概念形式化时，一种常见的方法是构建一组对象（符号）和一组用于操作这些对象的规则。这被称为代数。线性代数是向量及其操作规则的研究。我们大多数人从学校中学到的向量通常称为“几何向量”，通常用字母上方的小箭头表示，例如 → 和 → 。在本书中，我们将讨论更一般的向量概念，并使用粗体字母表示它们，例如 $\pmb{x}$ 和 $\pmb{y}$ 。  

一般来说，向量是能够相加并乘以标量以产生另一种相同类型对象的特殊对象。从抽象数学的角度来看，任何满足这两个属性的对象都可以被视为向量。以下是这样的向量对象的一些示例：

1. 几何向量。这个向量的例子可能在高中数学和物理中很熟悉。几何向量（见图2.1(a)）是定向段，可以在至少二维空间中绘制。两个几何向量 $\vec{\pmb{x}}$ 和 $\vec{y}$ 可以相加，使得 $\vec{\pmb{x}}+\vec{\pmb{y}}=\vec{\pmb{z}}$ 是另一个几何向量。此外，乘以标量 $\lambda\,{\vec{\pmb{x}}},\,\lambda\,\in\,\mathbb{R}$ 是几何向量。实际上，它是原始向量缩放了 $\lambda$ 。因此，几何向量是之前介绍的向量概念的实例。将向量解释为几何向量使我们能够利用关于方向和大小的直觉来推理数学操作。

2. 多项式也是向量；见图2.1(b)：两个多项式可以

![](images/518da7db5d39d0b98af97a987801f798ffa2b647080b7ce9f88a0441378d36f2.jpg)  
(a) 几何向量。  

![](images/837c41e6fc96b75e463d46701b9404a9244206cdebe94a19b704a407d531165f.jpg)  
(b) 多项式。  

可以相加，结果是另一个多项式；它们也可以乘以一个标量 $\lambda\,\in\,\mathbb{R}$，结果也是一个多项式。因此，多项式是（相对不寻常的）向量实例。请注意，多项式与几何向量非常不同。虽然几何向量是具体的“绘制”，多项式是抽象的概念。然而，它们在之前描述的意义上都是向量。

3. 音频信号是向量。音频信号表示为一系列数字。我们可以将音频信号相加，其和是一个新的音频信号。如果缩放音频信号，我们也会得到一个音频信号。因此，音频信号也是向量的一种类型。

4. $\textstyle\mathbb{R}^{n}$（$n$个实数的元组）的元素是向量。$\textstyle\mathbb{R}^{n}$比多项式更抽象，这是我们在本书中关注的概念。例如，
$$
\begin{align}
\pmb{a}=\left[\begin{array}{c} 1 \\ 2 \\ 3 \end{array}\right]\in\mathbb{R}^{3}\tag{2.1}
\end{align}
$$

在计算机实现时，请务必检查数组操作是否确实执行了向量操作。Pavel Grinfeld的线性代数系列：http://tinyurl.com/nahclwm Gilbert Strang的线性代数课程：http://tinyurl.com/bdfbu8s5 3Blue1Brown的线性代数系列：https://tinyurl.com/h5g4kps  

是一个数字三元组。对于 $\pmb{a},\pmb{b}\in\mathbb{R}^{n}$ 组件：$\pmb{a}+\pmb{b}=\pmb{c}\in\mathbb{R}^{n}$ 。将 $\pmb{a}\ \in\ \mathbb{R}^{n}$ 乘以 $\lambda\,\in\,\mathbb{R}$ 产生一个缩放向量 $\lambda\,\in\,\mathbb{R}^{n}$ 。将向量视为 $\textstyle\mathbb{R}^{n}$ 的元素还有一个额外的好处，即它大致对应于计算机上的实数数组。许多编程语言支持数组操作，这允许方便地实现涉及向量操作的算法。  

线性代数关注这些向量概念之间的相似性。我们可以将它们相加并乘以标量。我们将主要关注 $\textstyle\mathbb{R}^{n}$ 中的向量，因为线性代数中的大多数算法都是在 $\textstyle\mathbb{R}^{n}$ 中形式化的。在第8章中，我们将看到我们通常将数据表示为 $\textstyle\mathbb{R}^{n}$ 中的向量。在本书中，我们将专注于有限维向量空间，在这种情况下，任何类型的向量与 $\textstyle\mathbb{R}^{n}$ 之间存在一对一的对应关系。当方便时，我们将利用对几何向量的直觉并考虑基于数组的算法。  

数学中的一个主要思想是“闭包”的概念。这个问题是：在我的提议的操作下可以得到的所有事物的集合是什么？对于向量而言：从一个小集合的向量开始，通过将它们相加和缩放，可以得到什么样的向量集合？这导致了一个向量空间（第2.4节）。向量空间的概念及其属性构成了机器学习的许多基础。本章介绍的概念在图2.2中进行了总结。  

本章主要基于Drumm和Weil（2001）、Strang（2003）、Hogben（2013）、Liesen和Mehrmann（2015）的讲座笔记和书籍，以及Pavel Grinfeld的线性代数系列。其他优秀的


![](images/2b1d6ea8fa56697d0d978ff71859383cdbfe966e2b6ac66fb1fb47da2d3ab234.jpg)  
图2.2 本章介绍的概念思维导图，以及它们在本书其他部分的应用。  

资源包括麻省理工学院的Gilbert Strang线性代数课程和3Blue1Brown的线性代数系列。  

线性代数在机器学习和一般数学中扮演着重要角色。本章介绍的概念在第3章进一步扩展，包括几何的概念。在第5章，我们将讨论向量微积分，其中矩阵操作的原理知识是必不可少的。在第10章，我们将使用投影（将在第3.8节中介绍）进行降维，使用主成分分析（PCA）。在第9章，我们将讨论线性回归，其中线性代数在解决最小二乘问题中扮演核心角色。


## 2.1 线性方程组  

线性方程组在线性代数中占据核心地位。许多问题可以被表述为线性方程组，而线性代数则提供了求解这些方程组的工具。

### 示例 2.1  

一家公司生产产品 $N_{1},\ldots,N_{n}$，需要资源 $R_{1},\ldots,R_{m}$。为了生产单位产品 $N_{j}$，需要 $a_{ij}$ 单位的资源 $R_{i}$，其中 $i=1,\ldots,m$，且 $j=1,\ldots,n$。

目标是找到最优生产计划，即确定如果总共拥有 $b_{i}$ 单位的资源 $R_{i}$，并且理想情况下不浪费任何资源，应该生产多少单位的产品 $N_{j}$。

如果生产了 $x_{1},\ldots,x_{n}$ 单位对应的产品，那么总共需要 $R_{i}$ 资源的单位数为：

$$
\begin{align}
a_{i1}x_{1}+\cdot\cdot\cdot+a_{in}x_{n}\tag{2.2}
\end{align}
$$
因此，最优生产计划 $(x_{1},\ldots,x_{n})\in\mathbb{R}^{n}$ 必须满足以下线性方程组：

$$
\begin{align}
\begin{array}{c}{a_{11}x_{1}+\cdot\cdot\cdot+a_{1n}x_{n}=b_{1}}\\ {\vdots}\\ {a_{m1}x_{1}+\cdot\cdot\cdot+a_{mn}x_{n}=b_{m}}\end{array},\tag{2.3}
\end{align}
$$
其中 $\boldsymbol{a}_{ij}\in\mathbb{R}$ 且 $b_{i}\in\mathbb{R}$。

方程 (2.3) 是线性方程组的一般形式，$x_{1},\ldots,x_{n}$ 是这个系统的未知数。满足 (2.3) 的每一个 $\mathbb{R}^{n}$ 中的 $n$ 元组 $(x_{1},\cdot\cdot\cdot,x_{n})$ 是线性方程系统的解。


### 示例 2.2  

线性方程组  
$$
\begin{align}
\begin{array}{r r r r r r r}{x_{1}}&{{}+}&{x_{2}}&{{}+}&{x_{3}}&{{}=}&{3}\\ {x_{1}}&{{}-}&{x_{2}}&{{}+}&{2x_{3}}&{{}=}&{2}\\ {2x_{1}}& &{{}}&{+}&{3x_{3}}&{{}=}&{1}\end{array}\tag{2.4}
\end{align}
$$
没有解：将前两个方程相加得到 $2x_{1}+3x_{3}=5$，这与第三个方程（3）相矛盾。

让我们看看线性方程组 

$$
\begin{align}
\begin{array}{r r r r r r r r}{x_{1}}&{+}&{x_{2}}&{+}&{x_{3}}&{=}&{3}\\ {x_{1}}&{-}&{x_{2}}&{+}&{2x_{3}}&{=}&{2}\\ &{}&{x_{2}}&{+}&{x_{3}}&{=}&{2}\end{array}\tag{2.5}
\end{align}
$$
从第一个和第三个方程中可以得出 $x_{1}=1$。从 $(1)+(2)$，我们得到 $2x_{1}+3x_{3}=5$，即 $x_{3}=1$。从 (3)，我们得到 $x_{2}=1$。因此，$(1,1,1)$ 是唯一可能的且唯一的解（验证 $(1,1,1)$ 是否是解，只需将其代入）。

作为第三个例子，我们考虑 

$$
\begin{align}
\begin{array}{r r r r r r r}{x_{1}}&{{}+}&{x_{2}}&{{}+}&{x_{3}}&{{}=}&{3}\\ {x_{1}}&{{}-}&{x_{2}}&{{}+}&{2x_{3}}&{{}=}&{2}\\ {2x_{1}}&&{{}}&{+}&{3x_{3}}&{{}=}&{5}\end{array}\tag{2.6}
\end{align}
$$
由于 $(1)+(2)=(3)$，我们可以省略第三个方程（冗余）。从 (1) 和 (2)，我们得到 $2x_{1}=5-3x_{3}$ 和 $2x_{2}=1+x_{3}$。我们定义 $x_{3}=a\in\mathbb{R}$ 为一个自由变量，使得任何三元组  

$$
\begin{align}
\left(\frac{5}{2}-\frac{3}{2}a,\frac{1}{2}+\frac{1}{2}a,a\right),\quad a\in\mathbb{R}\tag{2.7}
\end{align}
$$
![](images/76e3382696b8f2cb9168921cd5bfe60129ad2816499169cde74191b1b1aaccef.jpg)  
图2.3 两个变量的两个线性方程系统的解空间可以几何地解释为两条线的交集。每个线性方程代表一条线。

$(x_{1},x_{2})$ 是线性方程组的解，即我们得到一个包含无限多个解的解集。

一般来说，对于实值线性方程组，我们得到没有解、恰好一个解或无限多个解。线性回归（第9章）在我们无法解决线性方程组时，解决了一个类似于示例2.1的版本。

**注释**（线性方程组的几何解释） 。 在包含两个变量 $x_{1},x_{2}$ 的线性方程组中，每个线性方程在 $x_{1}x_{2}$ 平面上定义一条线。由于线性方程组的解必须同时满足所有方程，解集是这些线的交集。这个交集集可以是一条线（如果线性方程描述的是同一条线），一个点，或者空集（当线平行时）。图2.3给出了系统

$$
\begin{array}{c}{4x_{1}+4x_{2}=5}\\ {2x_{1}-4x_{2}=1}\end{array}\tag{2.8}
$$
的解空间是点 $(x_{1},x_{2})=(1,\frac{1}{4})$ 的示例。对于三个变量，每个线性方程在三维空间中确定一个平面。当我们交这些平面，即同时满足所有线性方程时，我们可以得到一个解集，该解集是一个平面、一条线、一个点或空集（当平面没有共同的交点时）。$\diamondsuit$

为了系统地解决线性方程组，我们将引入一个有用的紧凑表示法。我们将系数 $a_{ij}$ 收集到向量中，并将这些向量收集到矩阵中。换句话说，我们将从（2.3）的形式写成以下形式：

$$
\left[\begin{array}{c}{a_{11}}\\ {\vdots}\\ {a_{m1}}\end{array}\right]x_{1} + \left[\begin{array}{c}{a_{12}}\\ {\vdots}\\ {a_{m2}}\end{array}\right]x_{2} + \cdot\cdot\cdot + \left[\begin{array}{c}{a_{1n}}\\ {\vdots}\\ {a_{m n}}\end{array}\right]x_{n} = \left[\begin{array}{c}{b_{1}}\\ {\vdots}\\ {b_{m}}\end{array}\right]\tag{2.9}
$$

$$
\iff \left[\begin{array}{c c c}{a_{11}}&{\cdot\cdot\cdot\cdot}&{a_{1n}}\\ {\vdots}&&{\vdots}\\ {a_{m1}}&{\cdot\cdot\cdot}&{a_{m n}}\end{array}\right]\left[\begin{array}{c}{x_{1}}\\ {\vdots}\\ {x_{n}}\end{array}\right]=\left[\begin{array}{c}{b_{1}}\\ {\vdots}\\ {b_{m}}\end{array}\right].\tag{2.10}
$$

在接下来的内容中，我们将仔细研究这些矩阵，并定义计算规则。我们将在第2.3节中回到解决线性方程的问题。

## 2.2 矩阵  

矩阵在线性代数中扮演核心角色。它们可以紧凑地表示线性方程组，但正如我们在第2.7节中将看到的，它们也表示线性函数（线性映射）。在讨论一些有趣的话题之前，让我们首先定义矩阵是什么，以及我们能对矩阵执行哪些操作。我们将在第4章中看到更多矩阵的性质。

### 定义 2.1（矩阵）

对于$m,n\in\mathbb{N}$，实值$(m,n)$矩阵$\pmb{A}$是一个由元素$a_{ij}$组成的$m\!\cdot\!n$元组，其中$i=1,\ldots,m$，$j=1,\dots,n$，按照包含$m$行和$n$列的矩形方案排列：

$$
\pmb{A}=\left[\begin{array}{c c c c}{a_{11}}&{a_{12}}&{\cdots\cdot}&{a_{1n}}\\ {a_{21}}&{a_{22}}&{\cdots\cdot}&{a_{2n}}\\ {\vdots}&{\vdots}&&{\vdots}\\ {a_{m1}}&{a_{m2}}&{\cdots\cdot}&{a_{m n}}\end{array}\right]\,,\quad a_{ij}\in\mathbb{R}\,.\tag{2.11}
$$

图 2.4 通过堆叠其列，矩阵$A$可以表示为一个长向量$\pmb{a}$。

![](images/fff2bb36d25e8bae8bf6488aeba02ea76b883c65031f36026d1d8c4bf1e6d2fb.jpg)

注意矩阵的大小。$\textsf{C}=$ np.einsum(’il, lj’, A, B)

按照惯例，$(1,n)$矩阵被称为 行 ，而$(m,1)$矩阵被称为 列 。这些特殊的矩阵也被称为 行/列向量 。

$\mathbb{R}^{m\times n}$ 是所有实值$(m,n)$矩阵的集合。$\pmb{A}\,\in\,\mathbb{R}^{m\times n}$ 可以通过将矩阵的所有$n$列堆叠成一个长向量等效地表示为$\pmb{a}\ \in\ \mathbb{R}^{mn}$，如图2.4所示。

### 2.2.1 矩阵加法与乘法

两个矩阵$\pmb{A}\in\mathbb{R}^{m\times n}$ 和$\pmb{B}\in\mathbb{R}^{m\times n}$ 的和被定义为元素级别的和，即：

$$
\pmb{A}+\pmb{B}:=\left[\begin{array}{c c c}{a_{11}+b_{11}}&{\cdot\cdot\cdot}&{a_{1n}+b_{1n}}\\ {\vdots}&{}&{\vdots}\\ {a_{m1}+b_{m1}}&{\cdot\cdot\cdot}&{a_{m n}+b_{m n}}\end{array}\right]\in\mathbb{R}^{m\times n}\,.\tag{2.12}
$$
对于$\pmb{A}\,\in\,\mathbb{R}^{m\times n}$ 和$\pmb{B}\,\in\,\mathbb{R}^{n\times k}$，矩阵乘积$\boldsymbol{C}=\boldsymbol{A}\boldsymbol{B}\in\mathbb{R}^{m\times k}$ 中的元素$c_{ij}$计算为：
$$
c_{ij}=\sum_{l=1}^{n}a_{il}b_{lj},\qquad i=1,\ldots,m,\quad j=1,\ldots,k.\tag{2.13}
$$
这意味着，为了计算元素$c_{ij}$，我们乘以矩阵$\pmb{A}$的第$i$行中的元素与矩阵$B$的第$j$列中的元素，并将它们相加。在第3.2节中，我们将称此为对应行和列的点积。在需要明确表示我们正在执行乘法的情况下，我们使用符号$A\cdot B$来表示乘法（明确显示“·”）。

注释。矩阵只能在“相邻”维度匹配时相乘。例如，一个$n\times k$矩阵$\pmb{A}$可以与$k\times m$矩阵$B$相乘，但只能从左侧进行：$\pmb{A}$有$n$列，$B$有$n$行，因此我们可以计算$a_{il}b_{lj}$，其中$l=1,\dots,n$。通常，两个向量$a,b$之间的点积表示为$\mathbf{\nabla}a^{\top}b$或$\langle\pmb{a},\pmb{b}\rangle$。
$$
\underbrace{A}_{n\times k}\underbrace{B}_{k\times m}=\underbrace{C}_{n\times m}\tag{2.14}
$$
如果$m\neq n$，矩阵乘积$BA$是未定义的，因为相邻维度不匹配。$\diamondsuit$

**注释。** 矩阵乘法不是在矩阵元素上进行元素级别的操作定义的，即$c_{ij}\neq a_{ij}b_{ij}$（即使矩阵$A,B$的大小选择得当）。这种元素级别的乘法在编程语言中经常出现，当我们将（多维）数组相互相乘时，被称为 Hadamard 乘积。$\diamondsuit$

### 示例 2.3

对于$$
\pmb{A}=\left[\begin{array}{r r r}{1}&{2}&{3}\\ {3}&{2}&{1}\end{array}\right]\in\mathbb{R}^{2\times3},\pmb{B}=\left[\begin{array}{r r r}{0}&{2}\\ {1}&{-1}\\ {0}&{1}\end{array}\right]\in\mathbb{R}^{3\times2},
$$我们观察到
$$
\pmb{A}\pmb{B}=\left[\begin{array}{r r r}{1}&{2}&{3}\\ {3}&{2}&{1}\end{array}\right]\left[\begin{array}{r r}{0}&{2}\\ {1}&{-1}\\ {0}&{1}\end{array}\right]=\left[\begin{array}{r}{2}&{3}\\ {2}&{5}\end{array}\right]\in\mathbb{R}^{2\times2},\tag{2.15}
$$
$$
\pmb{B}\pmb{A}=\left[\begin{array}{r r}{0}&{2}\\ {1}&{-1}\\ {0}&{1}\end{array}\right]\left[\begin{array}{r r r}{1}&{2}&{3}\\ {3}&{2}&{1}\end{array}\right]=\left[\begin{array}{r r r}{6}&{4}&{2}\\ {-2}&{0}&{2}\\ {3}&{2}&{1}\end{array}\right]\in\mathbb{R}^{3\times3}\,.\tag{2.16}
$$
图 2.5 即使两个矩阵乘法 $AB$ 和 $BA$ 都是定义的，结果的维度可以不同。

从这个例子中，我们已经可以看到矩阵乘法不是可交换的，即$AB\neq BA$；参见图 2.5 中的插图。
![](images/b814e11af258698142477bf5eb85e27f8172b1e6b0043c34d6255662b72a7b76.jpg)
### 定义 2.2（单位矩阵）。
在$\mathbb{R}^{n\times n}$中，我们定义单位矩阵

$$
\pmb{I}_{n}:=\left[\begin{array}{l l l l l l}{1}&{0}&{\cdots}&{0}&{\cdots}&{0}\\ {0}&{1}&{\cdots}&{0}&{\cdots}&{0}\\ {\vdots}&{\vdots}&{\ddots}&{\vdots}&{\ddots}&{\vdots}\\ {0}&{0}&{\cdots}&{1}&{\cdots}&{0}\\ {\vdots}&{\vdots}&{\ddots}&{\vdots}&{\ddots}&{\vdots}\\ {0}&{0}&{\cdots}&{0}&{\cdots}&{1}\end{array}\right]\in\mathbb{R}^{n\times n}\tag{2.17}
$$
单位矩阵是一个包含对角线上的1和其余位置为0的$n\times n$矩阵。

结合律：  
$$
\forall A\in\mathbb{R}^{m\times n},B\in\mathbb{R}^{n\times p},C\in\mathbb{R}^{p\times q}:(A B)C=A(B C)\tag{2.18}
$$
分配律：
$$
\begin{align}
\forall A,B\in\mathbb{R}^{m\times n},C,D\in\mathbb{R}^{n\times p}: (A+B)C &= AC + BC \tag{2.19a} \\
A(C+D) &= AC + AD \tag{2.19b}
\end{align}
$$
与单位矩阵相乘：  
$$
\forall A\in\mathbb{R}^{m\times n}:I_{m}A=A I_{n}=A\tag{2.20}
$$
请注意，当$m\neq n$时，$\pmb{I}_m\neq\pmb{I}_n$。


### 2.2.2 逆矩阵与转置  

一个方阵具有相同的列数和行数。

### 定义 2.3（逆矩阵）. 

考虑一个$\pmb{A}\in\mathbb{R}^{n\times n}$。设矩阵$\boldsymbol{B}\,\in\,\mathbb{R}^{n\times n}$满足$AB=I_{n}=BA$。B称为A的逆矩阵，并用$A^{-1}$表示。

不幸的是，并非每个矩阵$\pmb{A}$都拥有逆矩阵$A^{-1}$。如果这个逆矩阵存在，$\pmb{A}$被称为可逆的/可正规的/非奇异的，否则称为奇异的/不可逆的。当矩阵的逆存在时，它是唯一的。在第2.3节中，我们将讨论通过解线性方程组来计算矩阵逆的方法。

注释（$2\times2$矩阵的逆矩阵存在性）. 考虑一个矩阵

$$
\begin{array}{r}{A:=\left[\begin{array}{l l}{a_{11}}&{a_{12}}\\ {a_{21}}&{a_{22}}\end{array}\right]\in\mathbb{R}^{2\times2}\,.}\end{array}\tag{2.21}
$$
如果我们将$\pmb{A}$与

$$
A^{\prime}:={\biggl[}{\begin{array}{c c}{a_{22}}&{-a_{12}}\\ {-a_{21}}&{a_{11}}\end{array}}{\biggr]}\tag{2.22}
$$
相乘，我们得到

$$
A A^{\prime}=\left[{\begin{array}{c c}{a_{11}a_{22}-a_{12}a_{21}}&{0}\\ {0}&{a_{11}a_{22}-a_{12}a_{21}}\end{array}}\right]=(a_{11}a_{22}-a_{12}a_{21})I\,\tag{2.23}
$$
因此，

$$
\pmb{A}^{-1}=\frac{1}{a_{11}a_{22}-a_{12}a_{21}}\left[\begin{array}{c c}{a_{22}}&{-a_{12}}\\ {-a_{21}}&{a_{11}}\end{array}\right]\tag{2.24}
$$
当且仅当$a_{11}a_{22}-a_{12}a_{21}\neq0$。在第4.1节中，我们将看到$a_{\mathrm{11}}a_{\mathrm{22}}-a_{\mathrm{12}}a_{\mathrm{21}}$是$2\times2$矩阵的行列式。此外，我们可以使用行列式来检查矩阵是否可逆。♢

### 示例 2.4（逆矩阵）  

矩阵  
$$
A=\left[{\begin{array}{r r r}{1}&{2}&{1}\\ {4}&{4}&{5}\\ {6}&{7}&{7}\end{array}}\right]\,,\quad B=\left[{\begin{array}{r r r}{-7}&{-7}&{6}\\ {2}&{1}&{-1}\\ {4}&{5}&{-4}\end{array}}\right]
$$
互为逆矩阵，因为$AB=I=BA$。  

### 定义 2.4（转置）  
对于$\pmb{A}\,\in\,\mathbb{R}^{m\times n}$和$\pmb{B}\,\in\,\mathbb{R}^{n\times m}$，其中$b_{ij}=a_{ji}$，称为A的转置。我们写作$\pmb{B}=\pmb{A}^{\top}$。  

通常，$A^{\top}$可以通过将$\pmb{A}$的列写为$A^{\top}$的行来获得。逆矩阵和转置的重要性质如下：  

$$
\begin{align}
AA^{-1} &= I = A^{-1}A & \tag{2.26} \\
(AB)^{-1} &= B^{-1}A^{-1} & \tag{2.27} \\
(A+B)^{-1} &\neq A^{-1}+B^{-1} & \tag{2.28} \\
(A^{\top})^{\top} &= A & \tag{2.29} \\
(AB)^{\top} &= B^{\top}A^{\top} & \tag{2.30} \\
(A+B)^{\top} &= A^{\top}+B^{\top} & \tag{2.31}
\end{align}
$$
转置矩阵的主对角线（有时称为“主对角线”，“主对角线”，“主要对角线”或“主要对角线”）是集合$A_{ij}$的元素，其中$i=j$。标量情况下的（2.28）是$\textstyle{\frac{1}{2+4}}={\frac{1}{6}}\neq{\frac{1}{2}}+{\frac{1}{4}}$。  

### 定义 2.5（对称矩阵）  

矩阵$\pmb{A}\,\in\,\mathbb{R}^{n\times n}$是对称的，如果$\pmb{A}=\pmb{A}^{\top}$。  

请注意，只有$(n,n)$矩阵可以是对称的。通常，我们称$(n,n)$矩阵也称为方阵，因为它们具有相同的行数和列数。此外，如果$\pmb{A}$可逆，则$\pmb{A}^{\top}$也是可逆的，并且$(\pmb{A}^{-1})^{\top}=(\pmb{A}^{\top})^{-1}=:\pmb{A}^{-\top}$。  

**注释** （对称矩阵的和与乘积）  
对称矩阵$A,B\in\mathbb{R}^{n\times n}$的和总是对称的。然而，尽管它们的乘积总是定义的，但通常不是对称的：
$$
{\left[\begin{array}{l l}{1}&{0}\\ {0}&{0}\end{array}\right]}{\left[\begin{array}{l l}{1}&{1}\\ {1}&{1}\end{array}\right]}={\left[\begin{array}{l l}{1}&{1}\\ {0}&{0}\end{array}\right]}\,.\tag{2.32}
$$
### 2.2.3 矩阵与标量相乘  

让我们看看当矩阵乘以标量$\lambda\in\mathbb{R}$时会发生什么。设$\pmb{A}\,\in\,\mathbb{R}^{m\times n}$和$\lambda\in\mathbb{R}$。$\lambda A\,=\,K$，$K_{ij}\,=\,\lambda\,a_{ij}$。实际上，$\lambda$会缩放矩阵$A$中的每个元素。对于$\lambda,\psi\in\mathbb{R}$，以下成立：

结合律：

$(\lambda\psi)C=\lambda(\psi C),\quad C\in\mathbb{R}^{m\times n}$

$\lambda(BC)=(\lambda B)C=B(\lambda C)=(BC)\lambda$，$B  ∈ R^{m\times n}$，$C\in\mathbb{R}^{n\times k}$。请注意，这允许我们移动标量值。$(\lambda\boldsymbol{C})^{\top}=\boldsymbol{C}^{\top}\lambda^{\top}=\boldsymbol{C}^{\top}\lambda=\lambda\boldsymbol{C}^{\top}$，因为$\lambda=\lambda^{\top}$对于所有$\lambda\in\mathbb{R}$。

分配律：

$$
\begin{array}{r l}
&(\lambda+\psi)\dot{\pmb{C}}=\dot{\lambda}\pmb{C}+\psi\pmb{C},\quad\pmb{C}\in\mathbb{R}^{m\times n}\\
&\lambda(\pmb{B}+\pmb{C})=\lambda\pmb{B}+\lambda\pmb{C},\quad\pmb{B},\pmb{C}\in\mathbb{R}^{m\times n}
\end{array}
$$
### 示例 2.5（分配律）  

如果定义  

$$
C:=\left[{\begin{array}{c c}{1}&{{2}}\\ {3}&{4}\end{array}}\right]\,,\,\tag{2.33}
$$ 
那么对于任何$\lambda,\psi\in\mathbb{R}$，我们得到  

$$
\begin{align}
(\lambda + \psi)\mathbf{C} &= \begin{bmatrix}
(\lambda + \psi) \cdot 1 & (\lambda + \psi) \cdot 2 \\
(\lambda + \psi) \cdot 3 & (\lambda + \psi) \cdot 4
\end{bmatrix}
= \begin{bmatrix}
\lambda + \psi & 2\lambda + 2\psi \\
3\lambda + 3\psi & 4\lambda + 4\psi
\end{bmatrix} \tag{2.34a} \\
&= \begin{bmatrix}
\lambda & 2\lambda \\
3\lambda & 4\lambda
\end{bmatrix}
+
\begin{bmatrix}
\psi & 2\psi \\
3\psi & 4\psi
\end{bmatrix}
= \lambda\mathbf{C}
+
\psi\mathbf{C} \tag{2.34b}
\end{align}
$$

### 2.2.4 线性方程组的紧凑表示  

如果我们考虑线性方程组  

$$
\begin{array}{c}{2x_{1}+3x_{2}+5x_{3}=1}\\ {4x_{1}-2x_{2}-7x_{3}=8}\\ {9x_{1}+5x_{2}-3x_{3}=2}\end{array}\tag{2.35}
$$
并使用矩阵乘法的规则，我们可以将这个方程组以更紧凑的形式表示为  

$$ \begin{bmatrix} 2 & 3 & 5 \\ 4 & -2 & -7 \\ 9 & 5 & -3 \end{bmatrix} \begin{bmatrix} x_{1} \\ x_{2} \\ x_{3} \end{bmatrix} = \begin{bmatrix} 1 \\ 8 \\ 2 \end{bmatrix},\, \tag{2.36}$$
请注意，$x_{1}$ 放大第一列，$x_{2}$ 放大第二列，而$x_{3}$ 放大第三列。  

通常，线性方程组可以以矩阵形式紧凑地表示为 $A x=b$；参见（2.3），其中 $A x$ 是矩阵$\pmb{A}$的列的（线性）组合。我们将在第2.5节中更详细地讨论线性组合。

## 2.3 解线性方程组  

在（2.3）中，我们介绍了线性方程组的一般形式，即  

$$
\begin{align}
a_{11}x_{1} + \cdots + a_{1n}x_{n} &= b_{1} \\
&\vdots \\
a_{m1}x_{1} + \cdots + a_{mn}x_{n} &= b_{m}
\end{align}\tag{2.37}
$$  
其中$\boldsymbol{a}_{ij}\,\in\,\mathbb{R}$和$b_{i}\,\in\,\mathbb{R}$是已知的常数，而$x_{j}$是未知数，$i=1,\ldots,m$，$j=1,\dots,n$。到目前为止，我们看到矩阵可以作为表示线性方程组的紧凑方式，因此我们可以写出$\pmb{A}\pmb{x}=\pmb{b}$，参见（2.10）。此外，我们定义了基本的矩阵操作，如矩阵的加法和乘法。接下来，我们将专注于解决线性方程组，并提供求解矩阵逆的算法。  

### 2.3.1 特殊解与一般解  

在讨论如何一般性地解决线性方程组之前，让我们先看一个例子。考虑以下线性方程组  

$$
\begin{bmatrix}
1 & 0 & 8 & -4\\
0 & 1 & 2 & 12
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
x_4
\end{bmatrix}
=
\begin{bmatrix}
42 \\
8
\end{bmatrix}.\tag{2.38}
$$  
该方程组有两个方程和四个未知数。因此，通常我们期望有无限多个解。这个方程组特别容易处理，其中前两列分别包含一个1和一个0。请记住，我们想要找到标量$x_{1},\dots,x_{4}$，使得$\textstyle\sum_{i=1}^{4}x_{i}{\pmb c}_{i}=b$，其中我们定义$c_{i}$为矩阵的第$i$列，而$b$是（2.38）的右侧。通过取42倍的第一列和8倍的第二列，我们可以立即找到（2.38）问题的解。

$$
b=\left[{48\atop8}\right]=42\left[{1\atop0}\right]+8\left[{0\atop1}\right].\tag{2.39}
$$
因此，一个解是$[42,8,0,0]^{\top}$。这个解被称为特殊解或特定解。然而，这不是这个线性方程组的唯一解。为了捕获所有其他解，我们需要创造性地使用矩阵的列生成0：将0添加到特殊解中不会改变特殊解。为此，我们用前两列（这种非常简单的形式）表示第三列  

$$
\left[{8\atop2}\right]=8\left[{\begin{array}{l}{1}\\ {0}\end{array}}\right]+2\left[{\begin{array}{l}{0}\\ {1}\end{array}}\right]\tag{2.40}
$$
这样我们得到$\mathbf{0}=8\pmb{c}_{1}+2\pmb{c}_{2}-1\pmb{c}_{3}+0\pmb{c}_{4}$，即$(x_{1},x_{2},x_{3},x_{4})=(8,2,-1,0)$。实际上，通过将这个解乘以$\lambda_{1}\in\mathbb{R}$的任意缩放，可以产生0向量，即  

$$
\left[\begin{array}{c c c c}{{1}}&{{0}}&{{8}}&{{-4}}\\ {{0}}&{{1}}&{{2}}&{{12}}\end{array}\right]\left(\lambda_{1}\left[\begin{array}{c}{{8}}\\ {{2}}\\ {{-1}}\\ {{0}}\end{array}\right]\right)=\lambda_{1}(8{\pmb c}_{1}+2{\pmb c}_{2}-{\pmb c}_{3})={\bf0}\,.\tag{2.41}
$$
遵循同样的推理，我们用前两列表示矩阵（2.38）中的第四列，并生成另一个非平凡的0版本  

$$
\left[\begin{array}{c c c c}{{1}}&{{0}}&{{8}}&{{-4}}\\ {{0}}&{{1}}&{{2}}&{{12}}\end{array}\right]\left(\lambda_{2}\left[\begin{array}{c}{{-4}}\\ {{12}}\\ {{0}}\\ {{-1}}\end{array}\right]\right)=\lambda_{2}(-4\pmb{c}_{1}+12\pmb{c}_{2}-\pmb{c}_{4})=\bf{0}\tag{2.42}
$$
对于任何$\lambda_{2}\in\mathbb{R}$的通用解。将所有内容放在一起，我们得到（2.38）中的方程系统的所有解，称为通用解，为集合  

$$
\left\{{\pmb{x}}\in\mathbb{R}^{4}:{\pmb{x}}=\left[{\begin{array}{r}{42}\\ {8}\\ {0}\\ {0}\end{array}}\right]+\lambda_{1}\left[{\begin{array}{r}{8}\\ {2}\\ {-1}\\ {0}\end{array}}\right]+\lambda_{2}\left[{\begin{array}{r}{-4}\\ {12}\\ {0}\\ {-1}\end{array}}\right],\lambda_{1},\lambda_{2}\in\mathbb{R}\right\}.\tag{2.43}
$$
注释。我们遵循的通用方法包括以下三个步骤：  

1. 找到$A x=b$的特定解。  

2. 找到所有解$\mathbf{\nabla}A\mathbf{x}=\mathbf{0}$。  

3. 将步骤1和2的解组合成通用解。  

通用解和特定解都不是唯一的。  

前面例子中的线性方程组容易求解，因为（2.38）中的矩阵具有这种特别方便的形式，允许我们通过直观的方法找到特定和通用解。然而，一般的方程组不是这种简单形式。幸运的是，存在一种构造性的算法方法，可以将任何线性方程组转换为这种特别简单的形式：**高斯消元法**。高斯消元的关键在于线性方程组的基本变换，这些变换将方程组转换为简单形式。然后，我们可以将步骤应用于我们刚刚在（2.38）示例中讨论的简单形式。

### 2.3.2 基本变换  

解决线性方程组的关键是**基本变换**，这些变换保持解集不变，但将方程组转换为更简单的形式：

1. 交换两个方程（矩阵中表示方程组的行）
2. 将一个方程（行）乘以一个常数$\lambda\in\mathbb{R}\backslash\{0\}$
3. 两个方程（行）相加

### 示例 2.6  

对于$a\in\mathbb{R}$，我们寻找以下方程组的所有解：

$$
{\begin{array}{r r r r r r r r r r}{-2x_{1}}&{+}&{4x_{2}}&{-}&{2x_{3}}&{-}&{x_{4}}&{+}&{4x_{5}}&{=}&{-3}\\ {4x_{1}}&{-}&{8x_{2}}&{+}&{3x_{3}}&{-}&{3x_{4}}&{+}&{x_{5}}&{=}&{2}\\ {x_{1}}&{-}&{2x_{2}}&{+}&{x_{3}}&{-}&{x_{4}}&{+}&{x_{5}}&{=}&{0}\\ {x_{1}}&{-}&{2x_{2}}&{{}}&{{}}&{-}&{3x_{4}}&{+}&{4x_{5}}&{=}&{a}\end{array}}\tag{2.44}
$$
我们首先将这个方程组转换为紧凑的矩阵形式$Ax=b$。我们不再明确提及变量$x$，而是构建**增广矩阵**（形式为$\left[A\,|\,b\right]$）
$$
\left[\begin{array}{rrrrr|r}
-2 & 4 & -2 & -1 & 4 & -3 \\
4 & -8 & 3 & -3 & 1 & 2 \\
1 & -2 & 1 & -1 & 1 & 0 \\
1 & -2 & 0 & -3 & 4 & a
\end{array}\right]
\quad
\begin{aligned}
\text{与 } R_{3} \text{ 交换} \\
\text{} \\
\text{与 } R_{1} \text{ 交换} \\
\text{}
\end{aligned}
$$
我们使用竖线将（2.44）中的左侧与右侧分开。使用$\rightsquigarrow$表示使用基本变换对增广矩阵进行转换。

交换行 1 和行 3 导致  

$$
\left[{\begin{array}{r r r r r | r}{1}&{-2}&{1}&{-1}&{1}&{0}\\ {4}&{-8}&{3}&{-3}&{1}&{2}\\ {-2}&{4}&{-2}&{-1}&{4}&{-3}\\ {1}&{-2}&{0}&{-3}&{4}&{a}\end{array}}\right]
\begin{aligned}
\text{} \\
-4R_1 \\
+2R_1 \\
-R_1
\end{aligned}
$$
增广矩阵$\left[A\,|\,b\right]$紧凑地表示线性方程组$\pmb{Ax}=\pmb{b}$。

现在，当我们应用所指示的转换（例如，从行 2 减去行 1 的四倍）时，我们得到  

$$
\begin{array}{r l}&{}\left[\begin{array}{r r r r r | r}
{1}&{-2}&{1}&{-1}&{1}&{0}\\ 
{0}&{0}&{-1}&{1}&{-3}&{2}\\ 
{0}&{0}&{0}&{-3}&{6}&{-3}\\ 
{0}&{0}&{-1}&{-2}&{3}&{a}\end{array}\right]
\begin{aligned}
\text{} \\
\text{} \\
\text{} \\
-R_2-R_3
\end{aligned}
\\{\rightarrow}&{}\left[\begin{array}{r r r r r | r}
{1}&{-2}&{1}&{-1}&{1}&{0}\\ 
{0}&{0}&{-1}&{1}&{-3}&{2}\\ 
{0}&{0}&{0}&{-3}&{6}&{-3}\\
{0}&{0}&{0}&{0}&{0}&{a+1}\end{array}\right]
\begin{aligned}
\text{} \\
&.(-1) \\
&.(-\frac{1}{3}) \\
\text{} 
\end{aligned}
\\{\rightarrow}&{{}\left[\begin{array}{r r r r r | r}
{1}&{-2}&{1}&{-1}&{1}&{0}\\ 
{0}&{0}&{1}&{-1}&{3}&{-2}\\ 
{0}&{0}&{0}&{1}&{-2}&{1}\\ 
{0}&{0}&{0}&{0}&{0}&{a+1}\end{array}\right]}\end{array}
$$
阶梯形矩阵 这个（增广）矩阵处于方便的形式，称为**阶梯形矩阵**（REF）。将这种紧凑表示恢复为包含我们寻求的变量的明确表示，我们得到

$$
\begin{align}
x_{1} - 2x_{2} + x_{3} - x_{4} + x_{5} &= 0 \\
x_{3} - x_{4} + 3x_{5} &= -2 \\
x_{4} - 2x_{5} &= 1 \\
0 &= a + 1
\end{align}\tag{2.45}
$$

仅当$a=-1$时，此系统可以求解。一个特殊解是  

$$
{\begin{array}{r}{{\left[\begin{array}{l}{x_{1}}\\ {x_{2}}\\ {x_{3}}\\ {x_{4}}\\ {x_{5}}\end{array}\right]}={\left[\begin{array}{r}{2}\\ {0}\\ {-1}\\ {1}\\ {0}\end{array}\right]}~.}\end{array}}\tag{2.46}
$$
通用解，捕获所有可能解的集合是  

$$
\left\{{\pmb{x}}\in\mathbb{R}^{5}:{\pmb{x}}=\left[{\begin{array}{l}{2}\\ {0}\\ {-1}\\ {1}\\ {0}\end{array}}\right]+\lambda_{1}\left[{\begin{array}{l}{2}\\ {1}\\ {0}\\ {0}\\ {0}\end{array}}\right]+\lambda_{2}\left[{\begin{array}{l}{2}\\ {0}\\ {-1}\\ {2}\\ {1}\end{array}}\right]\,,\quad\lambda_{1},\lambda_{2}\in\mathbb{R}\right\}.\tag{2.47}
$$
接下来，我们将详细说明一种构造性方法，用于获得线性方程组的特殊解和通用解。

主元引导系数 在其他文本中，有时要求主元为1。基本变量 自由变量

**注释（主元和阶梯结构）** 。 行中的第一个非零数字称为**主元**，并且总是严格位于其上方行的主元右侧。因此，任何阶梯形矩阵总是具有“阶梯”结构。$\diamondsuit$

### 定义 2.6（阶梯形矩阵） 。 
矩阵处于**阶梯形矩阵**状态，如果所有只包含零的行位于矩阵底部；相应地，所有包含至少一个非零元素的行位于只包含零的行之上。仅查看非零行，从左到右的第一个非零数字（也称为**主元**或**引导系数**）总是严格位于其上方行的主元右侧。  

**注释（基本变量和自由变量）** 。 阶梯形矩阵中的主元对应的变量称为**基本变量**，而其他变量称为**自由变量**。例如，在（2.45）中，$x_{1},x_{3},x_{4}$是基本变量，而$x_{2},x_{5}$是自由变量。$\diamondsuit$

**注释（获取特殊解）**。 阶梯形矩阵使我们更容易确定特殊解。为此，我们使用主元列表示方程组的右侧，使得$\begin{array}{r}{\pmb{b}\overset{\cdot}{=}\sum_{i=1}^{P}\lambda_{i}\pmb{p}_{i}}\end{array}$，其中$\pmb{p}_{i}$，$i=1,\dots,P,$是主元列。$\lambda_{i}$最容易确定，如果我们从最右侧的主元列开始并逐步向左。  

在前面的例子中，我们会尝试找到$\lambda_{1},\lambda_{2},\lambda_{3}$，使得  

$$
\lambda_{1}\left[{\begin{array}{c c c c c}{1}\\ {0}\\ {0}\\ {0}\end{array}}\right]+\lambda_{2}\left[{\begin{array}{c c c c c}{1}\\ {1}\\ {0}\\ {0}\end{array}}\right]+\lambda_{3}\left[{\begin{array}{c c c c c}{-1}\\ {-1}\\ {1}\\ {0}\end{array}}\right]=\left[{\begin{array}{c c c c c}{0}\\ {-2}\\ {1}\\ {0}\end{array}}\right]\,.\tag{2.48}
$$
从这里，我们直接找到$\lambda_{3}=1,\lambda_{2}=-1,\lambda_{1}=2$。当我们放在一起时，我们不能忘记非主元列，我们隐式将系数设置为0。因此，我们得到特殊解$\pmb{x}=[2,0,-1,1,0]^{\top}$。$\diamondsuit$
注释（简化阶梯形矩阵） 。 方程组处于**简化阶梯形矩阵**状态（也称为**行简化阶梯形矩阵**或**行标准形式**），如果它处于阶梯形矩阵状态。每个主元为1。主元是其列中唯一的非零元素。  

简化阶梯形矩阵在第2.3.3节中扮演重要角色，因为它允许我们以直接方式确定线性方程组的通用解。  

**注释（高斯消元）** 。 高斯消元是一种算法，通过执行基本变换将线性方程组转换为简化阶梯形矩阵。

### 示例 2.7（简化阶梯形矩阵）

验证以下矩阵是否处于简化阶梯形矩阵状态（主元以粗体显示）：

$$
A = \begin{bmatrix}  
\pmb{1} & 3 & 0 & 0 & 3 \\
0 & 0 & \pmb{1} & 0 & 9 \\
0 & 0 & 0 & \pmb{1} & -4  
\end{bmatrix}.\tag{2.49}
$$

找到$A x = 0$的解的关键思路是关注**非主元列**，我们需要将这些列表示为**主元列**的线性组合。简化阶梯形矩阵使这一过程相对简单，我们将非主元列表示为左侧主元列的和与倍数：第二列是第一列的3倍（我们可以忽略第二列右侧的主元列）。因此，为了得到0，我们需要从第一列的三倍中减去第二列。接下来，我们观察第五列，这是我们的第二个非主元列。第五列可以表示为第一主元列的3倍，第二主元列的9倍，以及第三主元列的-4倍。我们需要跟踪主元列的索引，并将此表示转换为第一列的3倍，第二列的0倍（这是一个非主元列），第三列的9倍（这是我们的第二主元列），以及第四列的-4倍（这是第三主元列）。然后我们需要从第五列中减去，以得到0。最终，我们仍然在解决一个齐次方程组。

总结，对于$\pmb{A}\pmb{x}=\mathbf{0},\pmb{x}\in\mathbb{R}^{5}$的所有解，给出如下：

$$
\left\{\pmb{x}\in\mathbb{R}^{5}:\pmb{x}=\lambda_{1}\left[\begin{array}{l}{3}\\ {-1}\\ {0}\\ {0}\\ {0}\end{array}\right]+\lambda_{2}\left[\begin{array}{l}{3}\\ {0}\\ {9}\\ {-4}\\ {-1}\end{array}\right]\,,\quad\lambda_{1},\lambda_{2}\in\mathbb{R}\right\}\,.\tag{2.50}
$$
### 2.3.3 负一技巧

以下，我们介绍一个实用技巧，用于读取齐次线性方程组$\mathbf{\mathit{A}x} = \mathbf{0}$的解$\pmb{x}$，其中$\pmb{A}\in\mathbb{R}^{k\times n}$，$\pmb{x}\in\mathbb{R}^{n}$。

首先，我们假设$\pmb{A}$处于简化阶梯形矩阵状态，且没有仅包含零的行，即：

$$
A=\begin{bmatrix}
 0 & \cdots& 0& 1 & * & \cdots & * & 0 & * & \cdots & * & 0 & * & \cdots & *\\
 \vdots &  & \vdots & 0 & 0 & \cdots & 0 & 1 & * & \cdots & * & \vdots & \vdots &  & \vdots\\
 \vdots &  & \vdots & \vdots & \vdots &  & \vdots & 0 & \vdots &  & \vdots & \vdots & \vdots &  & \vdots\\
 \vdots &  & \vdots & \vdots & \vdots &  & \vdots & \vdots & \vdots &  & \vdots & 0 & \vdots &  & \vdots\\
 0 & \cdots & 0 & 0 & 0 & \cdots & 0 & 0 & 0 & \cdots & 0 & 1 & * & \cdots & *
\end{bmatrix},\tag{2.51}
$$

其中$*$可以是任意实数，但每行的第一个非零元素必须为1，对应列中的其他元素必须为0。具有主元（以粗体标记）的列$j_{1},\dots,j_{k}$对应于单位向量$e_{1},\ldots,e_{k}\in\mathbb{R}^{k}$。我们将这个矩阵扩展为一个$n\times n$矩阵$\tilde{\pmb{A}}$，通过在缺失主元的对角线上添加形式为（2.52）的$n-k$行：

$$
\left[\begin{array}{l l l l l l l l}{0}&{\cdots}&{0}&{-1}&{0}&{\cdots}&{0}\end{array}\right]\tag{2.52}
$$
使得增广矩阵$\tilde{\pmb{A}}$的对角线元素要么为1，要么为-1。然后，包含-1作为主元的$\tilde{\pmb{A}}$的列是齐次方程组$\mathbf{\mathit{A}x}=\mathbf{0}$的解。更准确地说，这些列构成了$\mathbf{\mathit{A}x}=\mathbf{0}$解空间的基础（见第2.6.1节），我们稍后将称之为**核**或**零空间**（见第2.7.3节）。

### 示例 2.8（负一技巧）

让我们回顾一下（2.49）中的矩阵，该矩阵已经处于简化阶梯形矩阵状态：

$$
\pmb{A} = \begin{bmatrix}
1 & 3 & 0 & 0 & 3 \\
0 & 0 & 1 & 0 & 9 \\
0 & 0 & 0 & 1 & -4
\end{bmatrix}.\tag{2.53}
$$

现在，我们通过在对角线上缺失主元的地方添加形式（2.52）的行，将此矩阵扩充为一个$5\times5$矩阵，得到：

$$
\tilde{\pmb{A}}=\left[\begin{array}{c c c c c}{{1}}&{{3}}&{{0}}&{{0}}&{{3}}\\ {{0}}&{{-{\bf1}}}&{{0}}&{{0}}&{{0}}\\ {{0}}&{{0}}&{{1}}&{{0}}&{{9}}\\ {{0}}&{{0}}&{{0}}&{{1}}&{{-4}}\\ {{0}}&{{0}}&{{0}}&{{0}}&{{-{\bf1}}}\end{array}\right]\;.\tag{2.54}
$$
从这种形式，我们可以直接读取$\pmb{A}\mathbf{x}=\mathbf{0}$的解，通过选取包含-1在对角线上的$\tilde{\pmb{A}}$的列：

$$
\left\{{\pmb{x}}\in\mathbb{R}^{5}:{\pmb{x}}=\lambda_{1}\left[\begin{array}{l}{3}\\ {-1}\\ {0}\\ {0}\\ {0}\end{array}\right]+\lambda_{2}\left[\begin{array}{l}{3}\\ {0}\\ {9}\\ {-4}\\ {-1}\end{array}\right]\,,\quad\lambda_{1},\lambda_{2}\in\mathbb{R}\right\}\,，\tag{2.55}
$$
这与我们通过“直觉”获得的（2.50）中的解相同。


**计算逆矩阵**

为了计算$\pmb{A}\,\in\,\mathbb{R}^{n\times n}$的逆矩阵$\pmb{A}^{-1}$，我们需要找到一个矩阵$\boldsymbol{X}$，使得$\boldsymbol{A}\boldsymbol{X}\,=\,\boldsymbol{I}_{n}$成立。此时，$\pmb{X}\,=\,\pmb{A}^{-1}$。我们可以将这个条件写成一组同时线性方程$\boldsymbol{A}\boldsymbol{X}\,=\,\boldsymbol{I}_{n}$的形式，其中我们解出$X=[{\pmb{x}}_{1}|\cdot\cdot\cdot|{\pmb{x}}_{n}]$。我们使用增广矩阵的表示法来紧凑地表示这一组线性方程组，并得到：

$$
\begin{array}{r l}{\left[\boldsymbol{A}|\boldsymbol{I}_{n}\right]}&{{}\sim\cdots\cdots\sim\quad\left[\boldsymbol{I}_{n}|\boldsymbol{A}^{-1}\right].}\end{array}\tag{2.56}
$$
这意味着，如果我们将增广方程组化简为简化阶梯形矩阵，我们就可以在方程组的右侧读出逆矩阵。因此，计算矩阵的逆矩阵等同于解线性方程组。

### 示例 2.9（通过高斯消元计算逆矩阵）  

为了确定矩阵$A$的逆矩阵，
$$
\pmb{A}=\begin{bmatrix}  
1 & 0 & 2 & 0 \\
1 & 1 & 0 & 0 \\
1 & 2 & 0 & 1 \\
1 & 1 & 1 & 1
\end{bmatrix}\tag{2.57}
$$
我们首先写出增广矩阵：

$$
\begin{bmatrix}  
1 & 0 & 2 & 0 & \big| & 1 & 0 & 0 & 0\\
1 & 1 & 0 & 0 & \big| & 0 & 1 & 0 & 0\\
1 & 2 & 0 & 1 & \big| & 0 & 0 & 1 & 0\\
1 & 1 & 1 & 1 & \big| & 0 & 0 & 0 & 1
\end{bmatrix}
$$

然后，我们使用高斯消元法将其化简为简化阶梯形矩阵：

$$
\begin{bmatrix}  
1 & 0 & 0 & 0 & \big| & -1 & 2 & -2 & 2 \\
0 & 1 & 0 & 0 & \big| & 1 & -1 & 2 & -2 \\
0 & 0 & 1 & 0 & \big| & 1 & -1 & 1 & -1 \\
0 & 0 & 0 & 1 & \big| & -1 & 0 & -1 & 2  
\end{bmatrix}
$$

这样，我们得到的右侧矩阵即为所求的逆矩阵：

$$
\pmb{A}^{-1}=\left[\begin{array}{c c c c}{-1}&{2}&{-2}&{2}\\ {1}&{-1}&{2}&{-2}\\ {1}&{-1}&{1}&{-1}\\ {-1}&{0}&{-1}&{2}\end{array}\right]\,。\tag{2.58}
$$
我们可以通过执行乘法$A A^{-1}$并观察结果是否为单位矩阵$I_{4}$来验证（2.58）确实是逆矩阵。

### 2.3.4 解线性方程组的算法

以下，我们简要讨论解形如$Ax=b$的线性方程组的方法。我们假设存在解。如果不存在解，则需要采用近似解法，这部分内容不在本章讨论。一种解决近似问题的方法是使用线性回归的策略，我们在第9章详细讨论。

在特殊情况下，我们可能能够确定逆矩阵$A^{-1}$，使得方程$\pmb{Ax}=\pmb{b}$的解为${\pmb{x}}\ =\ {\pmb A}^{-1}{\pmb{b}}$。然而，这仅在矩阵$\pmb{A}$为方阵且可逆的情况下可能实现，这通常不是常见情况。否则，在轻微假设下（即$\pmb{A}$的列线性独立），我们可以使用变换：

$$
\mathbf{A}\mathbf{x}=\mathbf{b}\iff\mathbf{A}^{\top}\mathbf{A}\mathbf{x}=\mathbf{A}^{\top}\mathbf{b}\iff\mathbf{x}=(\mathbf{A}^{\top}\mathbf{A})^{-1}\mathbf{A}^{\top}\mathbf{b}\tag{2.59}
$$
并使用Moore-Penrose伪逆$(\pmb{A}^{\top}\pmb{A})^{-1}\pmb{A}^{\top}$来确定解（2.59），该解也对应于最小范数最小二乘解。这种方法的缺点是需要进行大量的矩阵乘法计算和计算$\pmb{A}^{\top}\pmb{A}$的逆，此外，从数值精确性的角度来看，通常不推荐计算逆矩阵或伪逆矩阵。因此，以下我们简要讨论解决线性方程组的其他方法。

高斯消元在计算行列式（第4.1节）、检查一组向量是否线性独立（第2.5节）、计算矩阵的逆（第2.2.2节）、计算矩阵的秩（第2.6.2节）以及确定向量空间的基（第2.6.1节）时扮演着重要角色。高斯消元是一种直观且构建性的方法，用于解决具有数千个变量的线性方程组。然而，对于具有数百万个变量的系统，由于所需的算术操作数量随同时方程的数量呈三次方增长，因此这种方法在实践中不可行。

在实际应用中，大量线性方程组通过**间接方法**解决，例如静态迭代方法（如Richardson方法、Jacobi方法、Gauss-Seidel方法和逐次过松弛方法）或Krylov子空间方法（如共轭梯度法、最小残量法或双共轭梯度法）。我们参考Stoer和Burlirsch（2002）、Strang（2003）和Liesen和Mehrmann（2015）的书籍获取更多细节。

设$\pmb{x}_{*}$是方程$Ax=b$的解。这些迭代方法的核心思想是设置形式为：

$$
\pmb{x}^{(k+1)}=C\pmb{x}^{(k)}+\pmb{d}\tag{2.60}
$$
其中$C$和$\pmb{d}$是适合的参数，使得在每次迭代中减少残差误差$\|\pmb{x}^{(k+1)}-\pmb{x}_{*}\|$并最终收敛到$\pmb{x}_{*}$。我们将在第3.1节中引入范数$\|\cdot\|$，它允许我们计算向量之间的相似性。

## 2.4 向量空间

到目前为止，我们研究了线性方程组及其求解方法（第2.3节）。我们看到，线性方程组可以通过矩阵向量表示（2.10）进行紧凑表示。接下来，我们将更深入地探讨向量空间，即向量存在的结构化空间。

在本章开始时，我们非正式地将向量定义为可以相加并乘以标量的对象，且保持相同类型的对象。现在，我们准备正式化这一概念，并将从介绍群的概念开始，群是元素集合及其定义在这些元素上的运算，该运算保持集合结构的完整性。

### 2.4.1 群

群在计算机科学中扮演着重要角色。除了提供对集合操作的基本框架外，它们在密码学、编码理论和图形学中有着广泛的应用。
### 定义 2.7
（考虑集合$\mathcal{G}$和在其上定义的**运算**$\otimes:\mathcal{G}\times\mathcal{G}\to\mathcal{G}$。则$(\mathcal{G},\otimes)$称为一个**群**，如果以下条件成立：

1. **闭包**：$\mathcal{G}$在$\otimes$下闭包，即$\forall x,y\in\mathcal{G}:x\otimes y\in\mathcal{G}$

2. **结合律**：$\forall x,y,z\in\mathcal{G}:(x\otimes y)\otimes z=x\otimes(y\otimes z)$

3. **单位元素**：$\exists e\in\mathcal{G}\,\forall x\in\mathcal{G}:x\otimes e=x$和$e\otimes x=x$

4. **逆元素**：$\forall x\in{\mathcal{G}}\,\exists y\in{\mathcal{G}}:x\otimes y=e$和$y\otimes x=e$，其中$e$是单位元素。我们通常写$x^{-1}$来表示$x$的逆元素。

注释。逆元素相对于运算$\otimes$定义，并不一定意味着$\frac{1}{x}$。  

**阿贝尔群**（交换群）如果此外$\forall x,y\in\mathcal{G}:x\otimes y=y\otimes x$，则$(\mathcal{G},\otimes)$是一个**阿贝尔群**（交换群）。

### 示例 2.10（群）

让我们看看一些带有相关运算的集合示例，并判断它们是否构成群：

**注释**$\mathbb{N}_{0}:=\mathbb{N}\cup\{0\}$

$(\mathbb{Z},+)$ 是一个阿贝尔群。

$(\mathbb{N}_{0},+)$ 不是群：尽管$(\mathbb{N}_{0},+)$具有单位元素（0），但其缺失逆元素。

$(\mathbb{Z},\cdot)$ 不是群：尽管$(\mathbb{Z},\cdot)$包含单位元素（1），但对于任何$z\in\mathbb{Z}$，$z\neq\pm1$，缺失相应的元素。

$(\mathbb{R},\cdot)$ 不是群，因为0没有逆元素。

$(\mathbb{R}\backslash\{0\},\cdot)$ 是阿贝尔群。

$(\mathbb{R}^{n},+),(\mathbb{Z}^{n},+),n\in\mathbb{N}$ 是阿贝尔群，如果+定义为分量级运算，即
$$(x_{1},\cdots,x_{n})+(y_{1},\cdots,y_{n})=(x_{1}+y_{1},\cdots,x_{n}+y_{n})\tag{2.61}$$
那么，$(x_{1},\cdots\,,x_{n})^{-1}\;:=\;(-x_{1},\cdots\,,-x_{n})$是逆元素，$e=(0,\cdots\ ,0)$是单位元素。$(\mathbb{R}^{m\times n},+)$，即$m\times n$矩阵的集合，在分量级加法定义下是阿贝尔群。

- 闭包和结合律直接从矩阵乘法的定义得出。

- 单位元素：单位矩阵$\scriptstyle{I_{n}}$是$(\mathbb{R}^{n\times n},\cdot)$中矩阵乘法“·”的单位元素。

- 逆元素：如果逆存在（$\cdot_{A}$是可逆的），则$A^{-1}$是矩阵$\pmb{A}\in\mathbb{R}^{n\times n}$的逆元素，且在恰好这种情况下，$(\mathbb{R}^{n\times n},\cdot)$是一个群，称为一般线性群。

### 定义 2.8（一般线性群）。
$\pmb{A}\,\in\,\mathbb{R}^{n\times n}$的可逆矩阵集合关于在（2.13）中定义的矩阵运算形成一个群，并称为一般线性群$GL(n,\mathbb{R})$。然而，由于矩阵乘法不是可交换的，因此这个群不是阿贝尔群。

### 2.4.2 向量空间

在之前的讨论中，我们关注了集合$\mathcal{G}$及其内部运算，即仅在$\mathcal{G}$内操作的映射$\mathcal{G} \times \mathcal{G} \rightarrow \mathcal{G}$。接下来，我们将考虑除了内部运算$+$之外还包含额外运算$\cdot$的集合，即向量$\pmb{x} \in \mathcal{G}$与标量$\lambda \in \mathbb{R}$的乘法。我们可以将内部运算视为一种形式的加法，而外部运算视为一种形式的缩放。请注意，内部/外部运算与内部/外部乘积无关。

### 定义 2.9（向量空间）。
实数值向量空间$V=(\mathcal{V},+,\cdot)$是一个带有两个操作的集合：

$$
\begin{align}
+:\,\mathcal{V}\times\mathcal{V}&\to\mathcal{V}\tag{2.62}\\
\cdot:\,\mathbb{R}\times\mathcal{V}&\to\mathcal{V}\tag{2.63}
\end{align}
$$

其中：

1. $(\mathcal{V},+)$是一个阿贝尔群

2. 分配律：
$$\begin{array}{r l}
&{1.\ \forall\lambda\in\mathbb{R},\pmb{x},\pmb{y}\in\mathcal{V}:\lambda\cdot(\pmb{x}+\pmb{y})=\lambda\cdot\pmb{x}+\lambda\cdot\pmb{y}}\\
&{2.\ \forall\lambda,\psi\in\mathbb{R},\pmb{x}\in\mathcal{V}:(\lambda+\psi)\cdot\pmb{x}=\lambda\cdot\pmb{x}+\psi\cdot\pmb{x}}
\end{array}$$

3. 外部运算的结合律：$\forall\lambda,\psi\in\mathbb{R},\mathbf{\xi}x\in\mathcal{V}:\lambda\!\cdot\!(\psi\!\cdot\!\mathbf{x})=(\lambda\psi)\!\cdot\!\mathbf{x}$

4. 外部运算的单位元素：$\forall{\pmb{x}}\in\mathcal{V}:1\!\cdot\!{\pmb{x}}={\pmb{x}}$

集合中的元素$\pmb{x} \in V$称为向量。$(\mathcal{V},+)$的单位元素是零向量$\mathbf0=[0,\ldots,0]^{\top}$，内部运算$+$称为向量加法。元素$\lambda \in \mathbb{R}$称为标量，外部运算$\cdot$称为标量乘法。请注意，标量乘积是不同的乘法，我们将在第3.2节中讨论这一点。标量

**注释** 向量之间的“乘法”$\mathbf{a} \mathbf{b}$，$\mathbf{a},\mathbf{b} \in \mathbb{R}^{n}$，没有定义。理论上，我们可以定义元素级乘法，使得$\pmb{c}=\pmb{a}\pmb{b}$，其中$c_{j}=a_{j}b_{j}$。这种“数组乘法”在许多编程语言中常见，但在标准矩阵乘法规则下使用时意义有限：通过将向量视为$n \times 1$矩阵，我们可以使用在（2.13）中定义的矩阵乘法。然而，这样向量的维度就不匹配了。只有向量之间的乘法被定义：$\mathbf{a}\mathbf{b}^{\top} \in \mathbb{R}^{n \times n}$（外积），$\mathbf{a}^{\scriptscriptstyle\top}\mathbf{b} \in \mathbb{R}$（内积/标量/点积）。

### 示例 2.11（向量空间）

让我们看看一些重要的例子：

$\mathcal{V}=\mathbb{R}^{n}$，$n\in\mathbb{N}$是一个向量空间，其操作定义如下：

- 加法：$x+y=(x_{1},\dots,x_{n})+(y_{1},\dots,y_{n})=(x_{1}+y_{1},.\dots,x_{n}+y_{n})$ 对于所有$\pmb{x},\pmb{y}\in\mathbb{R}^{n}$

- 标量乘法：$\lambda\pmb{x}\,=\,\lambda(x_{1},\cdots,x_{n})\,=\,(\lambda x_{1},\cdots,\lambda x_{n})$ 对于所有$\pmb{\lambda}\in\mathbb{R},\pmb{x}\in\mathbb{R}^{n}$

$\mathcal{V}=\mathbb{R}^{m\times n},m,n\in\mathbb{N}$是一个向量空间，其操作定义如下：

- 加法：$\pmb{A}+\pmb{B} = \begin{bmatrix}a_{11} + b_{11} & \cdots & a_{1n} + b_{1n} \\ \vdots &  & \vdots \\a_{m1} + b_{m1} & \cdots & a_{mn} + b_{mn}  \end{bmatrix}$ 元素级定义对于所有$A,B\in{\mathcal{V}}$

- 标量乘法：$\lambda\pmb{A}\,=\,\left[\begin{array}{c c c}{{\lambda a_{11}}}&{{\cdots}}&{{\lambda a_{1n}}}\\ {{\vdots}}&{{}}&{{\vdots}}\\ {{\lambda a_{m1}}}&{{\cdots}}&{{\lambda a_{m n}}}\end{array}\right]$ 如第2.2节所述定义。记住$\mathbb{R}^{m\times n}$等同于$\mathbb{R}^{mn}$。

$\mathcal{V}=\mathbb{C}$，使用复数的标准加法定义。

**注释**。在接下来的内容中，我们将使用$(\mathcal{V},+,\cdot)$表示向量空间$V$，其中$+$和·是标准的向量加法和标量乘法。此外，我们将使用$\pmb{x}\in{V}$表示$V$中的向量，以简化符号。 

**注释**。$\mathbb{R}^{n}$，$\mathbb{R}^{n\times1}$，$\mathbb{R}^{1\times n}$这些向量空间仅在我们书写向量的方式上有所不同。在接下来的内容中，我们不会区分$\textstyle\mathbb{R}^{n}$和$\mathbb{R}^{n\times1}$，这允许我们将$n$元组表示为列向量：

$$
\pmb{x}=\left[\begin{array}{c}{x_{1}}\\ {\vdots}\\ {x_{n}}\end{array}\right].
$$
这简化了向量空间操作的符号表示。然而，我们区分$\mathbb{R}^{n\times1}$和$\mathbb{R}^{1\times n}$（行向量）以避免与矩阵乘法混淆。默认情况下，我们写$\pmb{x}$表示列向量，行向量表示为$\mathbf{x}^{\top}$，即$x$的转置。$\diamondsuit$


### 2.4.3 向量子空间

接下来，我们将介绍向量子空间。直观上，它们是包含在原始向量空间中的集合，具有在子空间内执行向量空间操作时永远不会离开该子空间的特性。从这个意义上说，它们是“封闭”的。向量子空间是机器学习中的一个核心概念。例如，第10章展示了如何使用向量子空间进行维数降低。

### 定义 2.10 
设$V=(\mathcal{V},+,\cdot)$是一个向量空间，$U \subseteq V$，且$U \neq \emptyset$。如果$U=(\mathcal{U},+,\cdot)$是一个向量空间，其向量空间操作$+$和$\cdot$仅限于$\mathcal{U} \times \mathcal{U}$和$\mathbb{R} \times \mathcal{U}$，则称$U$为$V$（向量子空间线性子空间）的子空间。我们用$U \subseteq V$表示$V$中的子空间$U$。

如果$\mathcal{U} \subseteq \mathcal{V}$且$V$是一个向量空间，则$U$可以继承$V$的许多性质，因为这些性质对所有$\pmb{x} \in \mathcal{V}$成立，特别是对所有$\mathcal{U} \subseteq \mathcal{V}$。这包括阿贝尔群性质、分配律、结合律和单位元素。为了确定$(\mathcal{U},+,\cdot)$是否是$V$的子空间，我们仍然需要证明：

1. $\mathcal{U} \neq \emptyset$，特别是：$\mathbf{0} \in \mathcal{U}$

2. $U$的封闭性：

a. 与外部操作有关：$\forall \lambda \in \mathbb{R} \, \forall \pmb{x} \in \mathcal{U}:\lambda\pmb{x} \in \mathcal{U}$。

b. 与内部操作有关：$\forall \pmb{x},\pmb{y} \in \mathcal{U}:\pmb{x}+\pmb{y} \in \mathcal{U}$。

### 示例 2.12（向量子空间）

让我们看看一些例子：

对于每一个向量空间$V$，它的平凡子空间是$V$本身和$\{\mathbf{0}\}$。

只有图2.6中的例子D是$\mathbb{R}^{2}$（带有通常的内/外运算）的子空间。在A和C中，封闭性被违反；B中不包含0。线性方程组的解集$\pmb{A x}=\mathbf{0}$，其中$n$个未知数$\pmb{x}=[x_{1},\cdots,x_{n}]^{\top}$是一个$\textstyle\mathbb{R}^{n}$的子空间。非齐次线性方程组的解$\pmb{A x}=\pmb{b}$，$\pmb{b}\neq\mathbf{0}$不是一个$\textstyle\mathbb{R}^{n}$的子空间。任意多个子空间的交集本身也是一个子空间。

![](images/9b607189744245d997eb8bf33fc6447fbefa373983ab7f6d8a11905cbbfa4c4b.jpg)
图 2.6 并非所有 R2 的子集都是子空间。在 A 和 C 中，封闭性被违反了；B 不包含 0 向量。只有 D 是子空间。

**注释**。对于线性方程组$A x=\mathbf{0}$的每一个子空间$U\subseteq(\mathbb{R}^{n},+,\cdot)$，它是$\pmb{x}\in\mathbb{R}^{n}$的子空间。


## 2.5 线性独立性

接下来，我们将仔细研究向量（向量空间的元素）可以做什么。特别是，我们可以将向量相加并用标量乘以它们。封闭性保证了我们得到的是同一向量空间中的另一个向量。有可能找到一组向量，通过将它们相加并缩放，我们可以表示向量空间中的每一个向量。这组向量称为**基**，我们将在第2.6.1节中讨论它们。在到达这个部分之前，我们需要引入线性组合和线性独立性的概念。

### 定义 2.11（线性组合）。
考虑一个向量空间 $V$ 和有限个向量 $\pmb{x}_{1},\dots,\pmb{x}_{k}\in V$。则，形式为

$$
{\pmb v}=\lambda_{1}{\pmb{x}}_{1}+\cdots+\lambda_{k}{\pmb{x}}_{k}=\sum_{i=1}^{k}\lambda_{i}{\pmb{x}}_{i}\in V
$$
的线性组合 $\lambda_{1},\ldots,\lambda_{k}\in\mathbb{R}$ 是向量 $\pmb{x}_{1},\dots,\pmb{x}_{k}$ 的线性组合。

零向量总是可以表示为 $k$ 个向量 $\pmb{x}_{1},\ldots,\pmb{x}_{k}$ 的线性组合，因为 $\textstyle\mathbf{\dot{0}}\ =\ \sum_{i=1}^{k}0\pmb{x}_{i}$ 总是成立。接下来，我们对一组向量的非平凡线性组合感兴趣，即在等式（2.65）中，不是所有系数 $\lambda_{i}$ 都为零的向量 $\pmb{x}_{1},\ldots,\pmb{x}_{k}$ 的线性组合。

线性相关 线性无关

### 定义 2.12（线性（不）相关）。
考虑一个向量空间 $V$ 以及 $k\in\mathbb{N}$ 和 $\pmb{x}_{1},\dots,\pmb{x}_{k}\,\in\,V$。如果存在一个非零线性组合

 $$\begin{array}{r}{\mathbf{0}\,=\,\sum_{i=1}^{k}\lambda_{i}\pmb{x}_{i}}\end{array}$$

使得至少有一个 $\lambda_{i}\neq0$，则向量 $\pmb{x}_{1},\dots,\pmb{x}_{k}$ 是**线性相关**的。如果仅存在平凡解，即 $\lambda_{1}=.\,.\,.=\lambda_{k}=0$，则向量 $\pmb{x}_{1},\dots,\pmb{x}_{k}$ 是**线性无关**的。

线性独立性是线性代数中最重要概念之一。直观上，一组线性无关的向量由没有冗余的向量组成，即如果从集合中移除任何一个向量，我们将失去一些东西。在接下来的章节中，我们将对这个直观概念进行更正式的定义。


### 示例 2.13（线性相关向量）

地理上的例子有助于澄清线性独立的概念。在肯尼亚的内罗毕描述基加利（卢旺达）的位置时，可能会说：“你可以通过先向西北方向走 $506\,\mathrm{km}$ 到乌干达的坎帕拉，然后向西南方向走 $374\,\mathrm{km}$ 来到达基加利。” 这些信息足以描述基加利的位置，因为地理坐标系统可以被视为二维向量空间（忽略海拔和地球的曲面）。这个人可能会补充说：“它大约向西走 $751\,\mathrm{km}$。” 尽管这个最后的陈述是真的，但在给定先前信息的情况下，找到基加利并不需要它（参见图 2.7 的插图）。在这个例子中，$^{\alpha}\!506\,\mathrm{km}$ 西北”向量（蓝色）和 $\mathrm{^{4}374\,k m}$ 西南”向量（紫色）是线性独立的。这意味着西南向量不能用西北向量来描述，反之亦然。然而，第三个 $^{\alpha}\!751\,\mathrm{km}$ 西”向量（黑色）是其他两个向量的线性组合，这使得向量组线性相关。等价地，给定 $^{\alpha}\!751\,\mathrm{km}$ 西”和 $\mathrm{^{4}374\,k m}$ 西南”可以线性组合来获得 $^{\alpha}\!506\,\mathrm{km}$ 西北”。

![](images/43ae75dc0555c0421f6909bce64c394156b4e16c106a218ac4be966b27a66a2b.jpg)  
图 2.7 二维空间（平面）中线性相关向量的地理示例（对基本方向的粗略近似）。

注释。以下属性有助于判断向量是否线性独立：

$k$ 个向量要么线性相关，要么线性独立。没有第三种选择。如果至少有一个向量 $\pmb{x}_{1},\ldots,\pmb{x}_{k}$ 为零，则它们线性相关。同样的规则适用于两个向量完全相同的情况。对于 $\{\pmb{x}_{1},\cdots,\pmb{x}_{k}\,:\,\pmb{x}_{i}\,\neq\,\mathbf{0},i\,=\,1,\cdots,k\}$，$k\,\geqslant\,2$，如果且仅如果（至少）其中一个向量是其他向量的线性组合，则这些向量线性相关。特别是，如果一个向量是另一个向量的倍数，即 $\mathbf{\mathit{x}}_{i}=\lambda\mathbf{\mathit{x}}_{j}$，$\lambda\in\mathbb{R}$，则集合 $\{\pmb{x}_{1},\cdots,\pmb{x}_{k}:\pmb{x}_{i}\neq\mathbf{0},i=1,\cdots,k\}$ 线性相关。检查向量 $\pmb{x}_{1},\dots,\pmb{x}_{k}\in V$ 是否线性独立的一种实用方法是使用高斯消元法：将所有向量写成矩阵 $\pmb{A}$ 的列，并执行高斯消元直到矩阵呈行阶梯形式（这里不需要行阶梯形式的简化）：

- 拐点列指示与左侧向量线性独立的向量。注意构建矩阵时向量的顺序。

- 非拐点列可以表示为左侧拐点列的线性组合。例如，行阶梯形式

$$
\left[\begin{array}{l l l}{1}&{3}&{0}\\ {0}&{0}&{2}\end{array}\right]
$$
告诉我们第一和第三列是拐点列。第二列是非拐点列，因为它等于第一列的三倍。

所有列向量都是线性独立的当且仅当所有列都是拐点列。如果有至少一个非拐点列，则列（因此，相应的向量）线性相关。

### 示例 2.14  

考虑 $\mathbb{R}^{4}$ 与  

$$
\pmb{x}_{1}=\left[\begin{array}{l}{1}\\ {2}\\ {-3}\\ {4}\end{array}\right],\quad\pmb{x}_{2}=\left[\begin{array}{l}{1}\\ {1}\\ {0}\\ {2}\end{array}\right],\quad\pmb{x}_{3}=\left[\begin{array}{l}{-1}\\ {-2}\\ {1}\\ {1}\end{array}\right].
$$
为了检查它们是否线性相关，我们遵循一般方法并求解  

$$
\lambda_{1}{\pmb{x}}_{1}+\lambda_{2}{\pmb{x}}_{2}+\lambda_{3}{\pmb{x}}_{3}=\lambda_{1}\left[\begin{array}{l}{1}\\ {2}\\ {-3}\\ {4}\end{array}\right]+\lambda_{2}\left[\begin{array}{l}{1}\\ {1}\\ {0}\\ {2}\end{array}\right]+\lambda_{3}\left[\begin{array}{l}{-1}\\ {-2}\\ {1}\\ {1}\end{array}\right]={\bf0}
$$
对于 $\lambda_{1},\dots,\lambda_{3}$ 。我们将向量 ${\bf{\Delta}}{\bf{x}}_{i}$ 写成矩阵的列，并应用基本行操作直到我们识别出拐点列：  

$$
\begin{array}{r}{\left[\begin{array}{l l l}{1}&{1}&{-1}\\ {2}&{1}&{-2}\\ {-3}&{0}&{1}\\ {4}&{2}&{1}\end{array}\right]\quad\sim\dots\dots\sim}\end{array}\quad\left[\begin{array}{l l l}{1}&{1}&{-1}\\ {0}&{1}&{0}\\ {0}&{0}&{1}\\ {0}&{0}&{0}\end{array}\right]\mathrm{~.~}
$$
在这里，矩阵的每一列都是拐点列。因此，没有非平凡解，我们需要 $\lambda_{1}\,=\,0,\lambda_{2}\,=\,0,\lambda_{3}\,=\,0$ 来解方程系统。因此，向量 ${\pmb{x}}_{1},{\pmb{x}}_{2},{\pmb{x}}_{3}$ 是线性独立的。  

注释。考虑一个向量空间 $V$ 与 $k$ 个线性独立向量 $b_{1},\ldots,b_{k}$ 和 $m$ 个线性组合  

$$
\begin{array}{c}{{\displaystyle{\boldsymbol x}_{1}=\sum_{i=1}^{k}\lambda_{i1}{\boldsymbol b}_{i}\,,}}\\ {{\vdots}}\\ {{{\boldsymbol x}_{m}=\sum_{i=1}^{k}\lambda_{i m}{\boldsymbol b}_{i}\,.}}\end{array}
$$
定义 $B\,=\,[b_{1},\cdots,b_{k}]$ 作为列由线性独立向量 $b_{1},\ldots,b_{k}$ 组成的矩阵，我们可以写  

$$
\pmb{x}_{j}=\pmb{B}\pmb{\lambda}_{j}\,,\quad\pmb{\lambda}_{j}=\left[\begin{array}{c}{\lambda_{1j}}\\ {\vdots}\\ {\lambda_{k j}}\end{array}\right]\,,\quad j=1,\ldots,m\,,
$$
以更紧凑的形式。  

我们想要测试 $\pmb{x}_{1},\dots,\pmb{x}_{m}$ 是否线性独立。为此目的，我们遵循测试一般方法的步骤，当 $\textstyle\sum_{j=1}^{m}\psi_{j}{\pmb{x}}_{j}={\bf0}$ 。使用（2.71），我们得到  

$$
\sum_{j=1}^{m}\psi_{j}{\pmb{x}}_{j}=\sum_{j=1}^{m}\psi_{j}{\pmb{B}}{\pmb\lambda}_{j}={\pmb{b}}\sum_{j=1}^{m}\psi_{j}{\pmb\lambda}_{j}\;.
$$
这意味着 $\{\pmb{x}_{1},\dots,\pmb{x}_{m}\}$ 是线性独立的当且仅当列向量 $\{\lambda_{1},\cdots,\lambda_{m}\}$ 是线性独立的。  

注释。在向量空间 $V$ 中，$m$ 个 $k$ 个向量 $\pmb{x}_{1},\ldots,\pmb{x}_{k}$ 的线性组合是线性相关的，如果 $m>k$ 。$\diamondsuit$


### 示例 2.15  

考虑一组线性独立的向量 $b_{\mathrm{1}},b_{\mathrm{2}},b_{\mathrm{3}},b_{\mathrm{4}}\in\mathbb{R}^{n}$ 和  

$$
\begin{array}{l c l c l c l c l}
{{\pmb{x}_{1}}}&{{=}}&{{\pmb{b}_{1}}}&{{-}}&{2{\pmb{b}_{2}}}&{{+}}&{{\pmb{b}_{3}}}&{{-}}&{{\pmb{b}_{4}}}\\
{{\pmb{x}_{2}}}&{{=}}&{{-4{\pmb{b}_{1}}}}&{{-}}&{{2{\pmb{b}_{2}}}}& & &{{+}}&{{4{\pmb{b}_{4}}}}\\
{{\pmb{x}_{3}}}&{{=}}&{{2{\pmb{b}_{1}}}}&{{+}}&{{3{\pmb{b}}_{2}}}&{{-}}&{{\pmb{b}_{3}}}&{{-}}&{{3{\pmb{b}}_{4}}}\\
{{\pmb{x}_{4}}}&{{=}}&{{17{\pmb{b}_{1}}}}&{{-}}&{{10{\pmb{b}_{2}}}}&{{+}}&{{11{\pmb{b}_{3}}}}&{{+}}&{{\pmb{b}_{4}}}
\end{array}.
$$

向量 $\pmb{x}_{1},\cdots,\pmb{x}_{4}\ \in \mathbb{R}^{n}$ 是否线性独立？为了回答这个问题，我们研究了列向量  

$$
\left\{  
\begin{bmatrix}  
1 \\
-2 \\
1 \\
-1  
\end{bmatrix},  
\begin{bmatrix}  
-4 \\
2 \\
3 \\
-1  
\end{bmatrix},  
\begin{bmatrix}  
-2 \\
0 \\
4 \\
-3  
\end{bmatrix},  
\begin{bmatrix}  
17 \\
-10 \\
11 \\
1  
\end{bmatrix}  
\right\}.
$$  

是否线性独立。对应线性方程系统的系数矩阵 $A={\left[\begin{array}{l l l l}{1}&{-4}&{2}&{17}\\ {-2}&{-2}&{3}&{-10}\\ {1}&{0}&{-1}&{11}\\ {-1}&{4}&{-3}&{1}\end{array}\right]}$ 的简化行阶梯形式给出为  

$$
\left[{\begin{array}{r r r r}{1}&{0}&{0}&{-7}\\ {0}&{1}&{0}&{-15}\\ {0}&{0}&{1}&{-18}\\ {0}&{0}&{0}&{0}\end{array}}\right]\ .
$$
我们看到对应的线性方程系统是非平凡可解的：最后一列不是主列，且 $\pmb{x}_{4}=-7\pmb{x}_{1}\!-\!15\pmb{x}_{2}\!-\!18\pmb{x}_{3}$ 。因此，$\pmb{x}_{1},\dots,\pmb{x}_{4}$ 是线性相关的，因为 $\pmb{x}_{4}$ 可以表示为 $\pmb{x}_{1},\dots,\pmb{x}_{3}$ 的线性组合。


## 2.6 基础与秩  

在向量空间 $V$ 中，我们特别关注具有以下性质的向量集合 $\mathcal{A}$：任何向量 $v\,\in\,V$ 都可以通过 $\pmb{x}_{1},\dots,\pmb{x}_{k}$ 的线性组合来表示，其中 $\pmb{x}_{1},\dots,\pmb{x}_{k}\,\subseteq\,{\mathcal{V}}$。这些向量具有特殊性质，在以下内容中，我们将对其进行描述。

### 2.6.1 生成集与基  

### 定义 2.13（生成集与跨度）。
考虑一个向量空间 $V=(\mathcal{V},+,\cdot)$ 和向量集合 ${\mathcal{A}}=\{{\pmb{x}}_{1},\cdots,{\pmb{x}}_{k}\}\,\subseteq\,{\mathcal{V}}$，向量 $\pmb{ v}\in$ V 可以表示为 $\pmb{x}_{1},\dots,\pmb{x}_{k}$ 的线性组合，集合 $\mathcal{A}$ 称为生成集。所有向量在 $\mathcal{A}$ 中的线性组合的集合称为 $\mathcal{A}$ 的跨度。如果 $\mathcal{A}$ 能够覆盖向量空间 $V$，我们写作 $V=\operatorname{span}[\mathcal{A}]$ 或 $V=\operatorname{span}[\pmb{x}_{1},\cdots,\pmb{x}_{k}]$。

生成集是能够覆盖向量（子）空间的向量集合，即每个向量都可以表示为生成集中的向量的线性组合。现在，我们将更加具体地描述能够覆盖向量（子）空间的最小生成集。
  

### 定义 2.14（基）。
考虑一个向量空间 $V=(\mathcal{V},+,\cdot)$ 和集合 ${\mathcal{A}}\subseteq$ $\nu$，集合 $\mathcal{A}$ 称为最小集，如果存在一个更小的集合 $\tilde{\mathcal{A}}\subsetneq\bar{\mathcal{A}}\subsetneq\mathcal{V}$，使得 $\tilde{\mathcal{A}}$ 也能够覆盖 $V$。每一个线性独立的生成集都是最小集，并且称为向量空间 $V$ 的基础。

设 $V\;=\;(\mathcal{V},+,\cdot)$ 为一个向量空间，且 ${\mathcal{B}}\,\subseteq\,{\mathcal{V}},{\mathcal{B}}\,\neq\,\emptyset$。则以下陈述等价：

1. $\mathcal{B}$ 是向量空间 $V$ 的最小生成集。
2. $\mathcal{B}$ 是向量空间 $V$ 中的最大线性独立向量集，即向量空间 $V$ 中任何向量添加到这个集合中都会使其线性相关。
3. 每个向量 $\pmb{x}\in V$ 都可以通过 $\mathcal{B}$ 中的向量的线性组合来表示，并且这种线性组合是唯一的，即对于任意的 $\lambda_{i},\psi_{i}\in\mathbb{R},\mathbf{\mathit{b}}_{i}\in\mathcal{B}$，有 $\lambda_{i}=\psi_{i},\;i=1,\cdots,k$。

基础是一个最小生成集和向量空间中最大的线性独立向量集。


### 示例 2.16  

在 $\mathbb{R}^{3}$ 中，标准基是标准基  

$$
\mathcal{B}=\left\{\left[\begin{array}{c}{{1}}\\ {{0}}\\ {{0}}\end{array}\right],\left[\begin{array}{c}{{0}}\\ {{1}}\\ {{0}}\end{array}\right],\left[\begin{array}{c}{{0}}\\ {{0}}\\ {{1}}\end{array}\right]\right\}.
$$
$\mathbb{R}^{3}$ 中的不同基是  

$$
\mathcal{B}_{1}=\left\{\left[\begin{array}{c}{1}\\ {0}\\ {0}\end{array}\right],\left[\begin{array}{c}{1}\\ {1}\\ {0}\end{array}\right],\left[\begin{array}{c}{1}\\ {1}\\ {1}\end{array}\right]\right\},\,\mathcal{B}_{2}=\left\{\left[\begin{array}{c}{0.5}\\ {0.8}\\ {0.4}\end{array}\right],\left[\begin{array}{c}{1.8}\\ {0.3}\\ {0.3}\end{array}\right],\left[\begin{array}{c}{-2.2}\\ {-1.3}\\ {3.5}\end{array}\right]\right\}.
$$
集合  

$$
A=\left\{{\left[{\begin{array}{l}{1}\\ {2}\\ {3}\\ {4}\end{array}}\right]}\,,{\left[{\begin{array}{l}{2}\\ {-1}\\ {0}\\ {2}\end{array}}\right]}\,,{\left[{\begin{array}{l}{1}\\ {1}\\ {0}\\ {-4}\end{array}}\right]}\right\}
$$
是线性独立的，但不是 $\mathbb{R}^{4}$ 的生成集（也不是基础）：例如，向量 $[1,0,0,0]^{\top}$ 无法通过 $A$ 中元素的线性组合获得。

注释。每个向量空间 $V$ 都有一个基础 $\mathcal{B}$。前面的例子表明一个向量空间 $V$ 可以有多个基，即没有唯一的基。然而，所有基都具有相同的元素数量，即基础向量。$\diamondsuit$ 基础向量  

我们只考虑有限维向量空间 $V$。在这种情况下，$V$ 的维数是 $V$ 的基础向量的数量，并写作 $\dim(V)$。如果 $U\subseteq V$ 是 $V$ 的子空间，则 $\dim(U)\leqslant\dim(V)$ 并且 $\dim(U)=\dim(V)$ 当且仅当 $U=V$。直观上，向量空间的维数可以被认为是该向量空间中的独立方向的数量。  

注释。向量空间的维数不一定是向量中的元素数量。例如，向量空间 $V=\operatorname{span}[{\binom{0}{1}}]$ 是一维的，尽管基础向量包含两个元素。$\diamondsuit$ 注释。子空间 $U=\operatorname{span}[\pmb{x}_{1},.\,.\,,\pmb{x}_{m}]\subseteq\mathbb{R}^{n}$ 的基础可以通过执行以下步骤找到：  

1. 将生成向量写成矩阵 $\pmb{A}$ 的列。  

2. 确定 $\pmb{A}$ 的行阶梯形式。  

3. 与主列关联的生成向量是 $U$ 的基础。

### 示例 2.17（确定基）

对于一个向量子空间 $U\subseteq\mathbb{R}^{5}$，由向量

$$
\pmb{x}_{1}=\left[\begin{array}{c}{1}\\ {2}\\ {-1}\\ {-1}\\ {-1}\end{array}\right],\quad\pmb{x}_{2}=\left[\!\begin{array}{c}{2}\\ {-1}\\ {1}\\ {2}\\ {-2}\end{array}\right],\quad\pmb{x}_{3}=\left[\!\begin{array}{c}{3}\\ {-4}\\ {3}\\ {5}\\ {-3}\end{array}\right],\quad\pmb{x}_{4}=\left[\!\begin{array}{c}{-1}\\ {8}\\ {-5}\\ {-6}\\ {1}\end{array}\right]\in\mathbb{R}^{5},
$$
我们感兴趣的是找出哪些向量 $\pmb{x}_{1},\dots,\pmb{x}_{4}$ 是 $U$ 的基础。为此，我们需要检查 $\pmb{x}_{1},\dots,\pmb{x}_{4}$ 是否线性独立。因此，我们需要解

$$
\sum_{i=1}^{4}\lambda_{i}{\pmb{x}}_{i}={\bf0}\,，
$$
这导致一个齐次线性方程组，其系数矩阵为

$$
\left[{\bf x}_{1},{\bf x}_{2},{\bf x}_{3},{\bf x}_{4}\right]=\left[\begin{array}{l l l l}{1}&{2}&{3}&{-1}\\ {2}&{-1}&{-4}&{8}\\ {-1}&{1}&{3}&{-5}\\ {-1}&{2}&{5}&{-6}\\ {-1}&{-2}&{-3}&{1}\end{array}\right].
$$
通过线性方程组的基本变换规则，我们得到行阶梯形式

$$
\begin{array}{r}{\left[\begin{array}{r r r r}{1}&{2}&{3}&{-1}\\ {2}&{-1}&{-4}&{8}\\ {-1}&{1}&{3}&{-5}\\ {-1}&{2}&{5}&{-6}\\ {-1}&{-2}&{-3}&{1}\end{array}\right]\quad\rightsquigarrow\cdots\rightsquigarrow}\end{array}\quad\left[\begin{array}{r r r r}{1}&{2}&{3}&{-1}\\ {0}&{1}&{2}&{-2}\\ {0}&{0}&{0}&{1}\\ {0}&{0}&{0}&{0}\\ {0}&{0}&{0}&{0}\end{array}\right]\,。
$$
由于主列指示哪些集合的向量是线性独立的，我们从行阶梯形式中看到 ${\pmb{x}}_{1},{\pmb{x}}_{2},{\pmb{x}}_{4}$ 是线性独立的（因为线性方程组 $\lambda_{1}{\pmb{x}}_{1}+\lambda_{2}{\pmb{x}}_{2}+\lambda_{4}{\pmb{x}}_{4}={\bf0}$ 只能通过 $\lambda_{1}=\lambda_{2}=\lambda_{4}=0$ 解）。因此，$\{\pmb{x}_{1},\pmb{x}_{2},\pmb{x}_{4}\}$ 是 $U$ 的基础。


### 2.6.2 矩阵的秩

矩阵 $\pmb{A}~\in~\mathbb{R}^{m\times n}$ 的线性独立列的数量等于线性独立行的数量，并称为矩阵 $\pmb{A}$ 的秩，用符号 $\operatorname{rk}(A)$ 表示。

注释。矩阵的秩具有以下重要性质：

$\operatorname{rk}(A)=\operatorname{rk}(A^{\top})$，即列秩等于行秩。

矩阵 $\pmb{A}\,\in\,\mathbb{R}^{m\times n}$ 的列集生成一个子空间 $U\subseteq\mathbb{R}^{m}$，其维数为 $\dim(U)=$ $\operatorname{rk}(A)$。稍后我们将称这个子空间为像集或范围。$U$ 的基础可以通过对 $\pmb{A}$ 应用高斯消元法来识别主列找到。

矩阵 $\pmb{A}\,\in\,\mathbb{R}^{m\times n}$ 的行集生成一个子空间 $W\,\subseteq\,\mathbb{R}^{n}$，其维数为 $\mathrm{dim}(W)\;=$ $\operatorname{rk}(A)$。$W$ 的基础可以通过对 $A^{\top}$ 应用高斯消元法找到。

对于矩阵 $\pmb{A}\,\in\,\mathbb{R}^{n\times n}$，如果且仅如果 $\operatorname{rk}(A)=n$，则矩阵 $\pmb{A}$ 是可逆的（即正则的）。对于所有 $\pmb{A}\,\in\,\mathbb{R}^{m\times n}$ 和所有 $\pmb{b}\,\in\,\mathbb{R}^{m}$，线性方程组 $A x=b$ 可以求解的条件是 $\operatorname{rk}(A)=\operatorname{rk}(A|b)$，其中 $A|b$ 是增广系统。对于 $\pmb{A}\in\mathbb{R}^{m\times n}$，解方程组 $\mathbf{\nabla}A\mathbf{x}=\mathbf{0}$ 的解空间的维数为 $n\mathrm{~-~}\operatorname{rk}(\pmb{A})$。稍后，我们将称这个子空间为核或零空间。

矩阵 $\pmb{A}\in\mathbb{R}^{m\times n}$ 具有满秩，如果其秩等于相同维度矩阵的最大可能满秩。这意味着满秩矩阵的秩是行数和列数中的较小值，即 $\operatorname{rk}(A)=\operatorname*{min}(m,n)$。如果矩阵的秩不等于其最大可能的秩，则称该矩阵为秩不足矩阵。

### 示例 2.18（秩）

$$
\mid A=\left[\begin{array}{c c c}{{1}}&{{0}}&{{1}}\\ {{0}}&{{1}}&{{1}}\\ {{0}}&{{0}}&{{0}}\end{array}\right].
$$
$$
A=\left[{\begin{array}{r r r}{1}&{2}&{1}\\ {-2}&{-3}&{1}\\ {3}&{5}&{0}\end{array}}\right].
$$
我们使用高斯消元法来确定秩：

$$
\begin{array}{r l}{\left[\begin{array}{l l l}{1}&{2}&{1}\\ {-2}&{-3}&{1}\\ {3}&{5}&{0}\end{array}\right]}&{{}\rightsquigarrow\cdots\rightsquigarrow}\end{array}\quad\left[\begin{array}{l l l}{1}&{2}&{1}\\ {0}&{1}&{3}\\ {0}&{0}&{0}\end{array}\right]\ .
$$
在这里，我们看到线性独立的行和列的数量为 2，因此 $\operatorname{rk}(A)=2$。

## 2.7 线性映射  

接下来，我们将研究保持向量空间结构的映射，这将使我们能够定义坐标的概念。在本章的开始，我们说向量是可以通过相加和乘以标量的对象，结果仍然是一个向量。我们希望在应用映射时保持这一性质：考虑两个实向量空间 $V,W$ 。映射 $\Phi:V\to W$ 保持向量空间结构，如果对于所有 ${\pmb{x}},{\pmb{ y}}\in{\pmb{ V}}$ 和 $\lambda\,\in\,\mathbb{R}$ 满足：

$$
\begin{array}{c}{{\Phi({\pmb{x}}+{\pmb y})=\Phi({\pmb{x}})+\Phi({\pmb y})}}\\ {{\Phi(\lambda{\pmb{x}})=\lambda\Phi({\pmb{x}})}}\end{array}
$$
我们可以通过以下定义来总结这一点：

线性映射 向量空间同态 线性变换

### 定义 2.15 (线性映射) 。

对于向量空间 $V,W$ ，映射 $\Phi:V\to W$ 被称为线性映射（或向量空间同态/线性变换）如果对于所有 $x,y\in V\,\forall\lambda,\psi\in\mathbb{R}:\Phi(\lambda\pmb{x}+\psi\pmb{y})=\lambda\Phi(\pmb{x})+\psi\Phi(\pmb{y})\,.$

实际上，我们可以将线性映射表示为矩阵（第 2.7.1 节）。回想一下，我们也可以将一组向量收集为矩阵的列。在处理矩阵时，我们必须记住矩阵代表的是线性映射还是向量集合。在第 4 章中，我们将更多地了解线性映射。在继续之前，我们简要介绍一些特殊映射。

### 定义 2.16 (单射、满射、双射) 。

考虑映射 $\Phi$ : $\mathcal{V}\rightarrow\mathcal{W}$ ，其中 $\mathcal{V},\mathcal{W}$ 可以是任意集合。然后 $\Phi$ 被称为

单射 如果 $\forall{\pmb{x}},{\pmb y}\in\mathcal{V}:\Phi({\pmb{x}})=\Phi({\pmb y})\implies{\pmb{x}}={\pmb y}.$

满射 如果 $\Phi(\mathcal{V})=\mathcal{W}$ 。双射 如果它是单射和满射。

如果 $\Phi$ 是满射，那么 $\mathcal{W}$ 中的每个元素都可以通过 $\Phi$ 从 $\mathcal{V}$ “到达”。双射 $\Phi$ 可以“撤销”，即存在一个映射 $\Psi : \mathcal{W}\to\mathcal{V}$ 使得 $\Psi\circ\Phi({\pmb{x}})={\pmb{x}}$ 。映射 $\Psi$ 然后称为 $\Phi$ 的逆映射，并通常表示为 $\Phi^{-1}$ 。

有了这些定义，我们引入以下向量空间 $V$ 和 $W$ 之间的特殊线性映射情况：

同构：$\Phi:V\to W$ 线性和双射 自同态：$\Phi:V\rightarrow V$ 线性 自同构 $\Phi:V\rightarrow V$ 线性和双射 我们定义 $\operatorname{id}_{V}\,:\,V\,\rightarrow\,V$ 作为向量空间 $V$ 中的  自我映射 或 自同构 。

同构 自同态 自同构 自我映射 自同构

### 示例 2.19（同态）

映射 $\Phi:\mathbb{R}^{2}\rightarrow\mathbb{C}$，$\Phi({\pmb{x}})=x_{1}+i x_{2}$，是一个同态：

$$
\Phi\left(\begin{bmatrix}  
x_1 \\
x_2  
\end{bmatrix} + \begin{bmatrix}  
y_1 \\
y_2  
\end{bmatrix}\right) = (x_1 + y_1) + i(x_2 + y_2) = x_1 + i x_2 + (y_1 + i y_2) \\
 
=\Phi\left(\begin{bmatrix}  
x_1 \\
x_2  
\end{bmatrix}\right) + \Phi\left(\begin{bmatrix}  
y_1 \\
y_2  
\end{bmatrix}\right)
$$

$$
\Phi\left(\lambda \begin{bmatrix}  
x_1 \\
x_2  
\end{bmatrix}\right) = \lambda x_1 + \lambda i x_2 = \lambda (x_1 + i x_2) = \lambda \Phi\left(\begin{bmatrix}  
x_1 \\
x_2  
\end{bmatrix}\right).  
$$

这也证明了为什么复数可以用 $\mathbb{R}^{2}$ 中的元组表示：存在一个双射线性映射，将 $\mathbb{R}^{2}$ 中元组的元素级加转换为具有相应加法的复数集。请注意，我们只展示了线性，但没有证明双射。

### 定理 2.17（Axler (2015) 中的定理 3.59） 。

有限维向量空间 $V$ 和 $W$ 是同构的当且仅当 $\mathrm{dim}(V)=\mathrm{dim}(W)$。

定理 2.17 表明存在两个相同维度的向量空间之间的线性、双射映射。直觉上，这意味着相同维度的向量空间在某种程度上是同一种事物，因为它们可以相互转换而不会损失任何信息。

定理 2.17 还为我们提供了处理 $\mathbb{R}^{m\times n}$（$m\times n$ 矩阵的向量空间）和 $\mathbb{R}^{mn}$（长度为 $mn$ 的向量的向量空间）的方式，因为它们的维度都是 $mn$，并且存在一个线性、双射映射将一个转换为另一个。

注释。考虑向量空间 $V,W,X$。则：

对于映射 $\Phi\,:\,V\,\rightarrow\,W$ 和 $\Psi\,:\,W\,\rightarrow\,X$，映射 $\Psi\circ\Phi:V\to X$ 也是线性的。如果 $\Phi:V\to W$ 是一个同构，那么 $\Phi^{-1}:W\to V$ 也是一个同构。

线性代数

![](images/ccbe96eac5eecd8aede9bfe22d7f18460d581515640a97004a38a4ec2224b7e0.jpg)  
图 2.8 由两组基向量定义的两个不同的坐标系统。向量 $x$ 根据所选择的坐标系统具有不同的坐标表示。

如果 $\Phi:V\to W.$，$\Psi:V\to W$ 是线性的，则 $\Phi+\Psi$ 和 $\lambda\Phi$，$\lambda\in\mathbb{R}$，也是线性的。

### 2.7.1 线性映射的矩阵表示

任何 $n$ 维向量空间都与 $\textstyle\mathbb{R}^{n}$ 同构（定理 2.17）。我们考虑一个 $n$ 维向量空间 $V$ 的基 $\{b_{1},\ldots,b_{n}\}$。在接下来的内容中，基向量的顺序很重要。因此，我们写成

$$
B=(b_{1},\cdots,b_{n})
$$
有序基，并称这个 $n$ 元组为 $V$ 的有序基。

注释（符号） 。现在到了符号使用有点棘手的地方。因此，我们在这里总结一些部分。$B=(b_{1},\cdots,b_{n})$ 是有序基，$\mathcal{B}=\{b_{1},\cdots,b_{n}\}$ 是（无序的）基，而 $B=[b_{1},\cdots,b_{n}]$ 是一个矩阵，其列是向量 $b_{1},\ldots,b_{n}$。$\diamondsuit$

定义 2.18（坐标） 。考虑一个向量空间 $V$ 和一个有序基 $B=(b_{1},\cdots,b_{n})$。对于任何 $\pmb{x}\in V$，我们得到一个唯一的表示（线性组合）

$$
{\pmb{x}}=\alpha_{1}{\pmb{b}}_{1}+.\,.+\alpha_{n}{\pmb{b}}_{n}
$$
坐标  

坐标向量 坐标表示  

相对于 $B$ 的 $x$。然后 $\alpha_{1},\ldots,\alpha_{n}$ 是 $\pmb{x}$ 相对于 $B$ 的坐标，而向量

$$
\alpha={\left[\begin{array}{l}{\alpha_{1}}\\ {\vdots}\\ {\alpha_{n}}\end{array}\right]}\in\mathbb{R}^{n}
$$
是 $\pmb{x}$ 相对于有序基 $B$ 的坐标向量/坐标表示。

 。  

基实际上定义了一个坐标系统。我们熟悉二维笛卡尔坐标系统，它由标准基向量 $e_{1},e_{2}$ 覆盖。在这个坐标系统中，向量 $\pmb{x}\in\mathbb{R}^{2}$ 的表示告诉我们如何线性组合 $e_{1}$ 和 $e_{2}$ 来得到 $\pmb{x}$。然而，$\mathbb{R}^{2}$ 中的任何基都定义了一个有效的坐标系统，之前相同的向量 $x$ 可能在 $(\pmb{b}_{1},\pmb{b}_{2})$ 基中有不同的坐标表示。在图 2.8 中，相对于标准基 $(e_{1},e_{2})$，向量 $x$ 的坐标是 $[2,2]^{\top}$。然而，相对于基 $(b_{1},b_{2})$，相同的向量 $x$ 以 $[1.09,0.72]^{\top}$ 表示，即 $\pmb{x}=1.09\pmb{b}_{1}+0.72\pmb{b}_{2}$。在接下来的章节中，我们将发现如何获得这种表示。


### 示例 2.20 

让我们看看一个几何上的 $\pmb{x}\in\mathbb{R}^{2}$，其坐标为 $[2,3]^{\top}$ 相对于 $\mathbb{R}^{2}$ 的标准基 $(e_{1},e_{2})$。这意味着，我们可以写出 $\pmb{x}=2\pmb{e}_{1}+3\pmb{e}_{2}$。然而，我们不必选择标准基来表示这个向量。如果我们使用基向量 $\pmb{b}_{1}=[1,-1]^{\top},\pmb{b}_{2}=[1,1]^{\top}$，我们将获得坐标 $\frac{1}{2}[-1,5]^{\top}$ 来表示相对于 $(\pmb{b}_{1},\pmb{b}_{2})$ 同样的向量（见图 2.9）。

注释。对于 $n$ 维向量空间 $V$ 和 $V$ 的有序基 $B$，映射 $\Phi\,:\,\mathbb{R}^{n}\,\rightarrow\,V$，$\Phi(e_{i})\;=\;b_{i}$，$i\;=\;1,\ldots,n$，是线性的（由于定理 2.17，它是同构的），其中 $(e_{1},\ldots,e_{n})$ 是 $\mathbb{R}^{n}$ 的标准基。

现在我们准备好明确地将矩阵与有限维向量空间之间的线性映射联系起来。

图 2.9 不同基的选择下向量 $x$ 的坐标表示。

![](images/3fa76c3a815bd94d45b89f1fb7ad483cf094534a57cefa37f1069127dc182b8c.jpg)  

定义 2.19（变换矩阵） 。考虑向量空间 $V,W$，它们对应于有序基 $B=(b_{1},\cdots,b_{n})$ 和 $C=(c_{1},\cdots,c_{m})$。此外，我们考虑一个线性映射 $\Phi:V\to W$。对于 $j\in\{1,\ldots,n\}$，

$$
\Phi({\pmb{b}}_{j})=\alpha_{1j}{\pmb c}_{1}+\cdot\cdot\cdot+\alpha_{m j}{\pmb c}_{m}=\sum_{i=1}^{m}\alpha_{i j}{\pmb c}_{i}
$$
是 $\Phi(\pmb{b}_{j})$ 相对于 $C$ 的唯一表示。然后，我们称 $m\times n$ 矩阵 $\pmb{A}_{\Phi}$，其中元素由给出，

$$
A_{\Phi}(i,j)=\alpha_{i j}\,，
$$
是 $\Phi$ 的变换矩阵（相对于有序基 $B$ 的 $V$ 和 $C$ 的 $W$）。  

变换矩阵  

$\Phi(\pmb{b}_{j})$ 相对于 $W$ 的有序基 $C$ 的坐标是 $\pmb{A}_{\Phi}$ 的第 $j$ 列。考虑有限维向量空间 $V,W$，有序基 $B,C$ 和线性映射 $\Phi:V\to W$ 以及变换矩阵 $\pmb{A}_{\Phi}$。如果 $\hat{\pmb{x}}$ 是坐标向量 $\pmb{x}\in{\pmb{ V}}$ 相对于 $B$ 的坐标向量，$\hat{\pmb{y}}$ 是坐标向量 ${\pmb y}=\Phi({\pmb{x}})\in W$ 相对于 $C$ 的坐标向量，那么  

$$
\begin{array}{r}{\hat{\pmb{y}}=\pmb{A}_{\Phi}\hat{\pmb{x}}\,.}\end{array}
$$
这意味着变换矩阵可以用来将相对于有序基在 $V$ 中的坐标映射到相对于有序基在 $W$ 中的坐标。


### 示例 2.21（变换矩阵）

考虑同态 $\Phi\ :\ V\ \rightarrow\ W$ 和有序基 $\textit{B}=(\pmb{b}_{1},\dots,\pmb{b}_{3})$ 为 $V$ 的基，以及 $C=(c_{1},\cdots,c_{4})$ 为 $W$ 的基。有：

$$
\begin{array}{r l}
&{\Phi(\pmb{b}_{1})=\pmb{c}_{1}-\pmb{c}_{2}+3\pmb{c}_{3}-\pmb{c}_{4}}\\
&{\Phi(\pmb{b}_{2})=2\pmb{c}_{1}+\pmb{c}_{2}+7\pmb{c}_{3}+2\pmb{c}_{4}}\\
&{\Phi(\pmb{b}_{3})=3\pmb{c}_{2}+\pmb{c}_{3}+4\pmb{c}_{4}}
\end{array}
$$

根据基 $B$ 和 $C$，变换矩阵 $\pmb{A}_{\Phi}$ 满足 $\Phi(\pmb{b}_{k})=\textstyle\sum_{i=1}^{4}\alpha_{i k}{\pmb c}_{i}$ 对于 $k=1,\cdots,3$，并给出为：

$$
\begin{array}{r}
A_{\Phi}=[\alpha_{1},\alpha_{2},\alpha_{3}]=\left[\begin{array}{l l l}
1 & 2 & 0 \\
-1 & 1 & 3 \\
3 & 7 & 1 \\
-1 & 2 & 4
\end{array}\right]\,,
\end{array}
$$

其中 $\alpha_{j},\;j=1,2,3$ 是相对于基 $C$ 的 $\Phi(\pmb{b}_{j})$ 的坐标向量。

### 示例 2.22（向量的线性变换）

![](images/cd28858341e2f432da9f5156036d631f2b7f63ec68acb7b8da581eb51ae0d7b3.jpg)  
图 2.10 一组向量的三个线性变换示例。图 2.10(a) 显示了 $\mathbb{R}^{2}$ 中的 400 个向量，每个向量由对应于 $(x_{1},x_{2})$ 坐标的点表示。这些向量排列成一个正方形。当我们使用矩阵 $A_{1}$ 在（2.97）中对这些每个向量进行线性变换时，我们得到图 2.10(b) 中的旋转正方形。如果我们应用由 $A_{2}$ 表示的线性映射，我们得到图 2.10(c) 中的矩形，其中每个 $x_{1}$ 坐标被拉伸了 2 倍。图 2.10(d) 显示了使用 $A_{3}$ 进行线性变换后的原始正方形图 2.10(a)，其中 $A_{3}$ 是反射、旋转和拉伸的组合。

### 2.7.2 基变换

接下来，我们将更仔细地研究如果我们改变 $V$ 和 $W$ 中的基，线性映射 $\Phi:V\to W$ 的变换矩阵如何变化。考虑两个有序基：

$$
B=(\pmb{b}_{1},\cdots,\pmb{b}_{n}),\quad\tilde{B}=(\tilde{\pmb{b}}_{1},\cdots,\tilde{\pmb{b}}_{n})
$$
为 $V$，以及两个有序基：

$$
C=(\pmb{c}_{1},\cdots,\pmb{c}_{m}),\quad\tilde{C}=(\tilde{\pmb{c}}_{1},\cdots,\tilde{\pmb{c}}_{m})
$$
为 $W$。矩阵 $\pmb{A}_{\Phi}\,\in\,\mathbb{R}^{m\times n}$ 是将 $\Phi:V\to W$ 映射到基 $B$ 和 $C$ 的变换矩阵，而 $\tilde{\pmb{A}}_{\Phi}\in\mathbb{R}^{m\times n}$ 是相对于基 $\tilde{B}$ 和 $\tilde{C}$ 的对应变换映射。接下来，我们将研究 $\pmb{A}$ 和 $\tilde{\pmb{A}}$ 之间的关系，即如果我们选择从 $B,C$ 到 $\tilde{B},\tilde{C}$ 进行基变换，如何将 $\pmb{A}_{\Phi}$ 转换为 $\tilde{\pmb{A}}_{\Phi}$。

注释。实际上，我们得到了身份映射 $\operatorname{id}_{V}$ 的不同坐标表示。在图 2.9 的上下文中，这意味着将相对于 $(e_{1},e_{2})$ 的坐标映射到相对于 $(\pmb{b}_{1},\pmb{b}_{2})$ 的坐标，同时不改变向量 $x$。通过改变基和相应的向量表示，相对于这个新基的变换矩阵可以具有特别简单的形式，这使得计算变得直接。$\diamondsuit$

### 示例 2.23（基变换）

考虑一个变换矩阵

$$
\pmb{A}=\left[{\begin{array}{l l}{2}&{1}\\ {1}&{2}\end{array}}\right]
$$
相对于 $\mathbb{R}^{2}$ 的标准基。如果我们定义一个新的基

$$
B=(\left[1\atop1\right],\left[1\atop-1\right])
$$
我们得到一个对角变换矩阵

$$
\tilde{\pmb{A}}=\left[\begin{array}{c c}{{3}}&{{0}}\\ {{0}}&{{1}}\end{array}\right]
$$
相对于 $B$，这比 $\pmb{A}$ 更容易处理。

接下来，我们将研究将相对于一个基的坐标向量映射到相对于不同基的坐标向量的映射。我们首先陈述主要结果，然后提供解释。

### 定理 2.20（基变换）。

对于线性映射 $\Phi:V\to W$，有序基

$$
B=(\pmb{b}_{1},\cdots,\pmb{b}_{n}),\quad\tilde{B}=(\tilde{\pmb{b}}_{1},\cdots,\tilde{\pmb{b}}_{n})
$$
为 $V$ 和

$$
C=(\pmb{c}_{1},\cdots,\pmb{c}_{m}),\quad\tilde{C}=(\tilde{\pmb{c}}_{1},\cdots\tilde{\pmb{c}}_{m})
$$
为 $W$，以及 $\Phi$ 相对于基 $B$ 和 $C$ 的变换矩阵 $\pmb{A}_{\Phi}$，相对于基 $\tilde{B}$ 和 $\tilde{C}$ 的对应变换矩阵 $\tilde{\cal A}_{\Phi}$ 给出为

$$
\tilde{\cal A}_{\Phi}={\pmb T}^{-1}{\pmb A}_{\Phi}{\pmb S}\,.
$$
这里，$\pmb{S}\in\mathbb{R}^{n\times n}$ 是映射矩阵，它将相对于 $\tilde{B}$ 的坐标映射到相对于 $B$ 的坐标，而 $\pmb{T}\in\mathbb{R}^{m\times m}$ 是映射矩阵，它将相对于 $\tilde{C}$ 的坐标映射到相对于 $C$ 的坐标。

证明。遵循 Drumm 和 Weil (2001) 的方法，我们可以将新基 $\tilde{B}$ 中的向量表示为相对于基 $B$ 的基向量的线性组合，这样有

$$
\tilde{\pmb{b}}_{j}=s_{1j}\pmb{b}_{1}+\cdot\cdot\cdot+s_{n j}\pmb{b}_{n}=\sum_{i=1}^{n}s_{i j}\pmb{b}_{i}\,,\quad j=1,\dots,n\,.
$$
同样，我们将新基向量 $\tilde{C}$ 表示为相对于基 $C$ 的基向量的线性组合，得到

$$
\tilde{\pmb{c}}_{k}=t_{1k}\pmb{c}_{1}+\cdot\cdot\cdot+t_{m k}\pmb{c}_{m}=\sum_{l=1}^{m}t_{l k}\pmb{c}_{l}\,,\quad k=1,\dots,m\,.
$$
我们定义 $\pmb{S}\,=\,\bigl(\bigl(\mathfrak{s}_{i j}\bigr)\bigr)\,\in\,\mathbb{R}^{n\times n}$ 为映射矩阵，它将相对于 $\tilde{B}$ 的坐标映射到相对于 $B$ 的坐标，而 $\pmb{T}=\left(\left(t_{l k}\right)\right)\in\mathbb{R}_{\sim}^{m\times m}$ 为映射矩阵，它将相对于 $\tilde{C}$ 的坐标映射到相对于 $C$ 的坐标。特别是，$S$ 的第 $j$ 列是相对于 $B$ 的 $\tilde{\pmb{b}}_{j}$ 的坐标表示，而 $T$ 的第 $k$ 列是相对于 $C$ 的 $\tilde{\pmb{c}}_{k}$ 的坐标表示。请注意，$S$ 和 $T$ 都是正规矩阵。


我们将从两个视角来探讨 $\Phi(\tilde{\pmb{b}}_{j})$。首先，通过应用映射 $\Phi$，我们得到对于所有 $j=1,\dots,n$$

$$
\Phi(\tilde{\pmb{b}}_{j})=\sum_{k=1}^{m}\underbrace{\tilde{a}_{k j}\tilde{\mathbf{c}}_{k}}_{\in W}\,\overset{(2.107)}{=}\sum_{k=1}^{m}\tilde{a}_{k j}\sum_{l=1}^{m}t_{l k}\mathbf{c}_{l}=\sum_{l=1}^{m}\left(\sum_{k=1}^{m}t_{l k}\tilde{a}_{k j}\right)\mathbf{c}_{l}\,，
$$

这里，我们首先将新的基向量 $\tilde{\pmb{c}}_{k}\,\in\,{\cal W}$ 表示为基向量 $\pmb{c}_{l}~\in~W$ 的线性组合，然后交换了求和的顺序。

其次，当我们表示 $\tilde{\pmb{b}}_{j}\,\in\,V$ 为 $b_{j}\in V$ 的线性组合时，我们得到

$$
\begin{array}{c l c r}
{\Phi(\tilde{\pmb{b}}_{j})\stackrel{(2.106)}{=}\Phi\left(\displaystyle\sum_{i=1}^{n}s_{i j}\pmb{b}_{i}\right)=\displaystyle\sum_{i=1}^{n}s_{i j}\Phi(b_{i})=\displaystyle\sum_{i=1}^{n}s_{i j}\displaystyle\sum_{l=1}^{m}a_{l i}\pmb{c}_{l}}\\
{=\displaystyle\sum_{l=1}^{m}\left(\displaystyle\sum_{i=1}^{n}a_{l i}s_{i j}\right)\pmb{c}_{l}\,,} & {j=1,\ldots,n\,,}
\end{array}
$$

这里我们利用了 $\Phi$ 的线性性质。比较 (2.108) 和 (2.109b)，对于所有 $j=1,\dotsc,n$ 和 $l=1,\ldots,m$，我们得到

$$
\sum_{k=1}^{m}t_{l k}\tilde{a}_{k j}=\sum_{i=1}^{n}a_{l i}s_{i j}
$$
因此，

$$
T\tilde{\boldsymbol{A}}_{\Phi}=\boldsymbol{A}_{\Phi}\boldsymbol{S}\in\mathbb{R}^{m\times n}\,，
$$
从而，

$$
\tilde{\cal A}_{\Phi}={\cal T}^{-1}{\cal A}_{\Phi}S\,，
$$
这证明了定理 2.20。

定理 2.20 告诉我们，通过在 $V$（$B$ 替换为 $\tilde{B}$）和 $W$（$C$ 替换为 $\tilde{C}$）中进行基变换，线性映射 $\Phi:V\to W$ 的变换矩阵 $\pmb{A}_{\Phi}$ 被等效的矩阵 ${\tilde{\pmb{A}}}_{\Phi}$ 替换，其中

$$
\tilde{\boldsymbol{A}}_{\Phi}=\boldsymbol{\cal T}^{-1}\boldsymbol{A}_{\Phi}\boldsymbol{S}。
$$
图 2.11 描述了同态 $\Phi:V\rightarrow W$ 和有序基 $B,{\tilde{B}}$ 为 $V$ 的基，以及 $C,{\tilde{C}}$ 为 $W$ 的基（用蓝色标记）。我们可以将基变换后的映射 $\Phi_{\tilde{C}\tilde{B}}$ 用基 $\tilde{B},\tilde{C}$ 表示为同态的复合：

$$
\Phi_{\tilde{C}\tilde{B}}=\Xi_{\tilde{C}C}\circ\Phi_{C B}\circ\Psi_{B\tilde{B}}
$$
其中 $\Xi_{\tilde{C}C}$ 和 $\Psi_{B\tilde{B}}$ 分别对应于基变换的线性映射。对应的变换矩阵用红色表示。

![](images/b9c158311e4c3c336f3b8a528eec3ab43d6f1d91a939d7d08385808af161bbae.jpg)

图 2.11 描述了同态 $\Phi:V\rightarrow W$ 和有序基 $B,{\tilde{B}}$ 为 $V$ 的基，以及 $C,{\tilde{C}}$ 为 $W$ 的基（用蓝色标记）。映射 $\Phi_{C B}$ 是 $\Phi$ 的实例，它将基向量 $B$ 映射到基向量 $C$ 的线性组合。假设我们知道了基变换前的映射 $\Phi_{C B}$ 的变换矩阵 $\pmb{A}_{\Phi}$，相对于有序基 $B,C$。当我们从 $B$ 进行基变换到 $\tilde{B}$ 在 $V$ 中，以及从 $C$ 进行基变换到 $\tilde{C}$ 在 $W$ 中时，我们可以确定对应的变换矩阵 ${\tilde{\cal A}}_{\Phi}$ 如下：首先，我们找到线性映射 $\Psi_{B\tilde{B}}:V\to V$ 的矩阵表示，它将相对于新基 $\tilde{B}$ 的坐标映射到（唯一的）相对于“旧”基 $B$ 的坐标（在 $V$ 中）。然后，我们使用变换矩阵 $\pmb{A}_{\Phi}$ 来将这些坐标映射到相对于 $C$ 在 $W$ 中。最后，我们使用线性映射 $\Xi_{{\tilde{C}}C}:W\to W$ 来将相对于 $C$ 的坐标映射到相对于 $\tilde{C}$ 的坐标。因此，我们可以将线性映射 $\Phi_{\tilde{C}\tilde{B}}$ 表示为涉及“旧”基的线性映射的复合：

$$
\Phi_{\tilde{C}\tilde{B}}=\Xi_{\tilde{C}C}\circ\Phi_{C B}\circ\Psi_{B\tilde{B}}=\Xi_{C\tilde{C}}^{-1}\circ\Phi_{C B}\circ\Psi_{B\tilde{B}}\,。
$$
具体来说，我们使用 $\Psi_{B\tilde{B}}=\mathrm{id}_{V}$ 和 $\Xi_{C\tilde{C}}=\mathrm{id}_{W}$，即映射向量到自身的身份映射，但使用不同的基。


### 定义 2.21（等价）. 

两个矩阵 $\pmb{A},\pmb{\tilde{A}}\in\mathbb{R}^{m\times n}$ 等价于正规矩阵 $\pmb{ S}\in\ \mathbb{R}^{n\times n}$ 和 $\pmb{T}\in\ \mathbb{R}^{m\times m}$，使得 $\tilde{\pmb{A}}=\pmb{T}^{-1}\pmb{A S}$。

### 定义 2.22（相似）. 

两个矩阵 $\pmb{A},\tilde{\pmb{A}}\,\in\,\mathbb{R}^{n\times n}$ 相似，如果存在正规矩阵 $\pmb{S}\in\mathbb{R}^{n\times n}$，使得 $\tilde{\pmb{A}}=\pmb{S}^{-1}\pmb{A}\pmb{S}$。

注释. 相似的矩阵总是等价的。然而，等价的矩阵不一定是相似的。$\diamondsuit$

注释. 考虑向量空间 $V,W,X$。根据定理的注释，我们知道映射 $\Phi:V\,\rightarrow\,W$ 和 $\Psi\,:\,W\,\rightarrow\,X$ 的复合映射 $\Psi\,\circ\,\Phi\,:\,V\,\rightarrow\,X$ 也是线性的。对于相应的映射，其整体变换矩阵为 $A_{\Psi\circ\Phi}=A_{\Psi}A_{\Phi}$。

基于这个注释，我们可以从复合线性映射的角度来看基变换：

$\pmb{A}_{\Phi}$ 是线性映射 $\Phi_{C B}:V\rightarrow W$ 相对于基 $B,C$ 的变换矩阵。$\tilde{\pmb{A}}_{\Phi}$ 是线性映射 $\Phi_{\tilde{C}\tilde{B}}:V\to W$ 相对于基 ${\tilde{B}},{\tilde{C}}$ 的变换矩阵。$\pmb{S}$ 是线性映射 $\Psi_{B\tilde{B}}\,:\,V\,\rightarrow\,V$（同构）的变换矩阵，它将 $\tilde{B}$ 表示为相对于 $B$ 的形式。通常情况下，$\Psi=\mathrm{id}_{V}$ 是向量空间 $V$ 中的单位映射。

$\pmb{T}$ 是线性映射 $\Xi_{C\tilde{C}}:W\,\to\,W$（同构）的变换矩阵，它将 $\tilde{C}$ 表示为相对于 $C$ 的形式。通常情况下，$\Xi=\operatorname{id}_{W}$ 是向量空间 $W$ 中的单位映射。

如果我们（非正式地）仅根据基来描述变换，$\tilde{B}\rightarrow\tilde{C}=\tilde{B}\rightarrow B\rightarrow C\rightarrow\tilde{C}$，$\tilde{A}_{\Phi}={\bf T}^{-1}A_{\Phi}S$，$\pmb{S}:\tilde{B}\to\overset{\cdot}{B}$，$\pmb{T}:\tilde{C}\to C$，$\pmb{T}^{-1}:C\rightarrow\tilde{C}$，并且

$$
\begin{array}{c}{{\tilde{B}\rightarrow\tilde{C}=\tilde{B}\rightarrow B\rightarrow C\rightarrow\tilde{C}}}\\ {{\tilde{A}_{\Phi}={\bf T}^{-1}A_{\Phi}S\,.}}\end{array}
$$
请注意，(2.116) 中的操作顺序是从右到左的，因为右侧的变换使得 $\pmb{x}\mapsto\pmb{S x}\mapsto\pmb{A}_{\Phi}(\pmb{S x})\mapsto$ ${\pmb T}^{-1}\big({\pmb A}_{\Phi}(S x)\big)=\tilde{{\pmb A}}_{\Phi}{\pmb{x}}$。


### 示例 2.24（基变换）

考虑一个线性映射 $\Phi:\mathbb{R}^{3}\to\mathbb{R}^{4}$，其变换矩阵为

$$
\pmb{A}_{\Phi}=\left[\begin{array}{l l l}{1}&{2}&{0}\\ {-1}&{1}&{3}\\ {3}&{7}&{1}\\ {-1}&{2}&{4}\end{array}\right]
$$
相对于标准基

$$
B=(\left[{\begin{array}{c}{1}\\ {0}\\ {0}\end{array}}\right],\left[{\begin{array}{c}{0}\\ {1}\\ {0}\end{array}}\right],\left[{\begin{array}{c}{0}\\ {0}\\ {1}\end{array}}\right]),\quad C=(\left[{\begin{array}{c}{1}\\ {0}\\ {0}\\ {0}\end{array}}\right],\quad\left[{\begin{array}{c}{0}\\ {1}\\ {0}\\ {0}\end{array}}\right],\left[{\begin{array}{c}{0}\\ {0}\\ {1}\\ {0}\end{array}}\right],\left[{\begin{array}{c}{0}\\ {0}\\ {0}\\ {1}\end{array}}\right]).
$$
我们寻求相对于新基的映射 $\Phi$ 的变换矩阵 $\tilde{\pmb{A}}_{\Phi}$，

$$
\tilde{B}=(\,{\small\left[\begin{array}{l}{1}\\ {1}\\ {0}\end{array}\right]}\,,\,{\small\left[\begin{array}{l}{0}\\ {1}\\ {1}\end{array}\right]}\,,\,{\small\left[\begin{array}{l}{1}\\ {0}\\ {1}\end{array}\right]}\,)\in\mathbb{R}^{3},\quad\tilde{C}=(\,{\small\left[\begin{array}{l}{1}\\ {1}\\ {0}\\ {0}\end{array}\right]}\,,\,{\small\left[\begin{array}{l}{1}\\ {0}\\ {1}\\ {0}\end{array}\right]}\,,\,{\small\left[\begin{array}{l}{0}\\ {1}\\ {1}\\ {0}\end{array}\right]}\,,\,{\small\left[\begin{array}{l}{1}\\ {0}\\ {0}\\ {1}\end{array}\right]}\,)\,。
$$
然后，

$$
\pmb{S}=\left[\begin{array}{c c c}{1}&{0}&{1}\\ {1}&{1}&{0}\\ {0}&{1}&{1}\end{array}\right],\qquad\pmb{T}=\left[\begin{array}{c c c c}{1}&{1}&{0}&{1}\\ {1}&{0}&{1}&{0}\\ {0}&{1}&{1}&{0}\\ {0}&{0}&{0}&{1}\end{array}\right]\,，
$$
其中 $s$ 的第 $i$ 列是 $\tilde{\pmb{b}}_{i}$ 相对于基向量 $B$ 的坐标表示。由于 $B$ 是标准基，坐标表示很容易找到。对于一般的基 $B$，我们需要解线性方程组来找到 $\lambda_{i}$，使得 $\begin{array}{r}{\sum_{i=1}^{3}\lambda_{i}\pmb{b}_{i}=\tilde{\pmb{b}}_{j}}\end{array}$，$j=1,\dots,3$。同样，$T$ 的第 $j$ 列是 $\tilde{\pmb{c}}_{j}$ 相对于基向量 $C$ 的坐标表示。因此，我们得到

$$
{\begin{array}{r l}&{{\tilde{\boldsymbol{A}}}_{\Phi}={\pmb{T}}^{-1}{\pmb{A}}_{\Phi}{\pmb{S}}={\frac{1}{2}}\left[\begin{array}{l l l l}{1}&{1}&{-1}&{-1}\\ {1}&{-1}&{1}&{-1}\\ {-1}&{1}&{1}&{1}\\ {0}&{0}&{0}&{2}\end{array}\right]\left[\begin{array}{l l l}{3}&{2}&{1}\\ {0}&{4}&{2}\\ {10}&{8}&{4}\\ {1}&{6}&{3}\end{array}\right]}\\ &{\quad=\left[\begin{array}{l l l}{-4}&{-4}&{-2}\\ {6}&{0}&{0}\\ {4}&{8}&{4}\\ {1}&{6}&{3}\end{array}\right].}\end{array}}
$$
在第 4 章中，我们将能够利用基变换的概念找到一个基，相对于该基，同态的变换矩阵具有特别简单的（对角）形式。在第 10 章中，我们将研究数据压缩问题，并找到一个方便的基，我们可以将数据投影到该基上，同时最小化压缩损失。


### 2.7.3 像与核

线性映射的像与核是具有特定重要性质的向量子空间。接下来，我们将更仔细地描述它们。

定义 2.23（像与核）。

核/零空间
像/范围

定义域/共域 对于 $\Phi:V\to W$，我们定义：

核/零空间
$\ker(\Phi):=\Phi^{-1}(\mathbf{0}_{W})=\{v\in V:\Phi(v)=\mathbf{0}_{W}\}$

像/范围
$\operatorname{Im}(\Phi):=\Phi(V)=\{\pmb{w}\in W|\exists\pmb{v}\in V:\Phi(\pmb{v})=\pmb{w}\}\,.$

我们也将 $V$ 和 $W$ 分别称为 $\Phi$ 的定义域和共域。

直观上，像由所有使得 $\Phi$ 映射到单位元素 $\mathbf{0}_{W}\,\in\,W$ 的向量 $v\in V$ 组成。像是一些向量 $\pmb{w}\,\in\,W$ 的集合，这些向量可以通过 $\Phi$ 从 $V$ 中的任何向量“达到”。图 2.12 给出了一个示例。

注释。考虑一个线性映射 $\Phi:V\to W$，其中 $V,W$ 是向量空间。

总是有 $\Phi({\bf0}_{V})\;=\;{\bf0}_{W}$，因此 ${\bf0}_{V}\ \in\ \ker(\Phi)$。特别地，零空间永远不会为空。$\operatorname{Im}(\Phi)\subseteq W$ 是 $W$ 的子空间，$\ker(\Phi)\subseteq V$ 是 $V$ 的子空间。

![](images/0dcaee848c7c5e1ebf64b5f2ef0d8cec76d3a75565ab9ffa52d5d48aa8409713.jpg)  
图 2.12 线性映射 $\Phi:V\to W$ 的核与像。

$\Phi$ 是单射（一一对应）当且仅当 $\ker(\Phi)=\{\mathbf{0}\}$。


**注释（零空间）**

让我们考虑 $\pmb{A}\in\mathbb{R}^{m\times n}$ 和一个线性映射 $\Phi:\mathbb{R}^{n}\rightarrow\mathbb{R}^{m}$，对于 $A=[\pmb{a}_{1},\cdots,\pmb{a}_{n}]$，其中 $\mathbf{a}_{i}$ 是 $\pmb{A}$ 的列，我们得到

$$
\begin{array}{l}{\mathrm{Im}(\Phi)=\{A{\pmb{x}}:{\pmb{x}}\in\mathbb{R}^{n}\}=\left\{\displaystyle\sum_{i=1}^{n}x_{i}{\pmb a}_{i}:x_{1},.\,.\,,x_{n}\in\mathbb{R}\right\}}\\ {\quad\quad\quad=\mathrm{span}[{\pmb a}_{1},.\,.\,,{\pmb a}_{n}]\subseteq\mathbb{R}^{m}\,,}\end{array}
$$
即像由 $\pmb{A}$ 的列的范围组成，也称为列空间。因此，列空间（像）是 $\mathbb{R}^{m}$ 的子空间，其中 $m$ 是矩阵的“高度”。

$\operatorname{rk}(\pmb{A})=\dim(\operatorname{Im}(\Phi))$。零空间（核）$\ker(\Phi)$ 是齐次线性方程组 $\mathbf{\mathit{A}x}\ =\ \mathbf{0}$ 的通解，捕捉所有可能的线性组合，这些组合在 $\textstyle\mathbb{R}^{n}$ 中产生 $\mathbf{0}\in\mathbb{R}^{m}$。零空间是 $\textstyle\mathbb{R}^{n}$ 的子空间，其中 $n$ 是矩阵的“宽度”。零空间关注列之间的关系，我们可以使用它来确定是否/如何可以将一列表示为其他列的线性组合。

### 示例 2.25（线性映射的像与核）映射

$$
\Phi:\mathbb{R}^{4}\rightarrow\mathbb{R}^{2},\quad\left[\begin{array}{c}{x_{1}}\\ {x_{2}}\\ {x_{3}}\\ {x_{4}}\end{array}\right]\mapsto\left[\begin{array}{c c c c}{1}&{2}&{-1}&{0}\\ {1}&{0}&{0}&{1}\end{array}\right]\left[\begin{array}{c}{x_{1}}\\ {x_{2}}\\ {x_{3}}\\ {x_{4}}\end{array}\right]=\left[\begin{array}{c}{x_{1}+2x_{2}-x_{3}}\\ {x_{1}+x_{4}}\\ {x_{2}+x_{3}}\end{array}\right]
$$
$$
=x_{1}\left[1\right]+x_{2}\left[2\right]+x_{3}\left[\!-1\right]+x_{4}\left[\!0\right]
$$
是线性的。为了确定 $\operatorname{Im}(\Phi)$，我们可以取变换矩阵的列的范围并得到

$$
\begin{array}{r}{\mathrm{Im}(\Phi)=\mathrm{span}[\!\left[1\!\right],\left[2\!\right],\left[\begin{array}{c}{-1}\\ {0}\end{array}\!\right],\left[\begin{array}{c}{0}\\ {1}\end{array}\right]]\,.}\end{array}
$$
为了计算 $\Phi$ 的核（零空间），我们需要解 $\mathbf{\nabla}A\mathbf{x}=\mathbf{0}$，即我们需要解一个齐次方程组。为此，我们使用高斯消元将 $\pmb{A}$ 转换为行阶梯形式：

$$
\begin{array}{r l}{\left[\begin{array}{c c c c}{1}&{2}&{-1}&{0}\\ {1}&{0}&{0}&{1}\end{array}\right]}&{{}\rightsquigarrow\cdot\cdot\cdot\rightsquigarrow}&{\left[\begin{array}{c c c c}{1}&{0}&{0}&{1}\\ {0}&{1}&{-\frac{1}{2}}&{-\frac{1}{2}}\end{array}\right]\;.}\end{array}
$$
这个矩阵处于行阶梯形式，我们可以使用减一技巧来计算核（零空间）的基础（见第 2.3.3 节）。或者，我们可以将非主元列（第 3 和 4 列）表示为主元列（第 1 和 2 列）的线性组合。第三列 $\mathbf{\delta}_{\mathbf{{3}}}$ 等于 $-\frac{1}{2}$ 倍的第二列 $\mathbf{\delta}_{\mathbf{{\bar{\rho}}}_{2}}$。因此，$\mathbf{0}=\mathbf{a}_{3}+\textstyle{\frac{1}{2}}\mathbf{a}_{2}$。同样地，我们看到 $\begin{array}{r}{{\pmb a}_{4}={\pmb a}_{1}\!-\!\frac{1}{2}{\pmb a}_{2}}\end{array}$，因此 $\mathbf{0}=\mathbf{a}_{1}\!-\!\textstyle{\frac{1}{2}}\bar{\mathbf{a}}_{2}\!-\!\mathbf{a}_{4}$。总体而言，这给出了核（零空间）为

$$
\begin{array}{r}{\ker(\Phi)=\operatorname{span}[\,\!\left[\begin{array}{c}{0}\\ {\frac{1}{2}}\\ {1}\\ {0}\end{array}\right]\,,\,\left[\begin{array}{c}{-1}\\ {\frac{1}{2}}\\ {0}\\ {1}\end{array}\right]\,]\,.}\end{array}
$$
秩-零空间定理

### 定理 2.24（秩-零空间定理）。

对于向量空间 $V,W$ 和线性映射 $\Phi:V\to W$，它满足

$$
\dim(\ker(\Phi))+\dim(\operatorname{Im}(\Phi))=\dim(V)\,。
$$
线性映射的基本定理

秩-零空间定理也被称为线性映射的基本定理（Axler, 2015, 定理 3.22）。定理 2.24 的以下结论是直接的：

如果 $\mathrm{dim(Im}(\Phi))\,<\,\mathrm{dim}(V)$，则 $\ker(\Phi)$ 非平凡，即零空间包含 $\mathbf{0}_{V}$ 之外的元素，$\dim(\ker(\Phi))\geqslant1$。如果 $\pmb{A}_{\Phi}$ 是 $\Phi$ 相对于有序基的变换矩阵，且 $\mathrm{dim(Im}(\Phi))<\mathrm{dim}(V)$，则线性方程组 $\pmb{A}_{\Phi}\pmb{x}=$ 0 有无限多个解。如果 $\dim(V)=\dim(W)$，则三个等价关系

成立，因为 $\operatorname{Im}(\Phi)\subseteq W$。

## 2.8 线性空间

接下来，我们将更仔细地研究那些相对于原点偏移的空问，即那些不再是向量子空间的空问。此外，我们将简要讨论这些线性空间之间的映射的性质，这些映射类似于线性映射。

注释。在机器学习文献中，线性和线性之间的区别有时并不清晰，因此我们可以找到将线性空间/映射作为线性空间/映射的参考。$\diamondsuit$

### 2.8.1 线性子空间

### 定义 2.25（线性子空间）。

设 $V$ 为向量空间，$\pmb{x}_{0}\in V$ 且 $U\subseteq V$ 为子空间。则子集

$$
\begin{array}{r l}&{L=\pmb{x}_{0}+U:=\{\pmb{x}_{0}+\pmb{u}:\pmb{u}\in U\}}\\ &{\quad=\{\pmb{v}\in V|\exists\pmb{u}\in U:\pmb{v}=\pmb{x}_{0}+\pmb{u}\}\subseteq V}\end{array}
$$
称为 $V$ 的线性子空间或线性流形。$U$ 称为方向或方向子空间，$\pmb{x}_{0}$ 称为支持点。在第 12 章中，我们将这样的子空间称为超平面。

注意，线性子空间的定义排除了当 $\pmb{x}_{0}\ \notin\ U$ 时的 0。因此，当 $\pmb{x}_{0}\notin U$ 时，线性子空间不是 $V$ 的（线性）子空间（向量子空间）。

线性子空间/线性流形/方向子空间/支持点/超平面

线性子空间的示例包括 $\mathbb{R}^{3}$ 中的点、线和平面，它们不一定通过原点。

注释。考虑两个线性子空间 $L=\pmb{x}_{0}+U$ 和 $\tilde{L}=\tilde{\pmb{x}}_{0}+\tilde{U}$，它们属于向量空间 $V$。则 $L\subseteq{\tilde{L}}$ 当且仅当 $U\subseteq{\tilde{U}}$ 且 $\pmb{x}_{0}-\tilde{\pmb{x}}_{0}\in\tilde{U}$。

线性子空间通常由参数描述：考虑一个 $k$ 维线性子空间 $L=\pmb{x}_{\mathrm{0}}+U$，如果 $(\pmb{b}_{1},\dots,\pmb{b}_{k})$ 是 $U$ 的有序基，则 $L$ 中的每个元素 $\pmb{x}$ 可以唯一地表示为

$$
{\pmb{x}}={\pmb{x}}_{0}+\lambda_{1}{\pmb{b}}_{1}+.\,.+\lambda_{k}{\pmb{b}}_{k}\,，
$$
其中 $\lambda_{1},\ldots,\lambda_{k}\,\in\,\mathbb{R}$。这种表示称为 $L$ 的参数方程，其中参数向量为 $b_{1},\ldots,b_{k}$，参数为 $\lambda_{1},\dots,\lambda_{k}$。 ♢ 参数

### 示例 2.26（线性子空间）

一维线性子空间称为**直线**，可以表示为 ${\pmb y}\,=\,{\pmb{x}}_{0}\,+\,\lambda\pmb{b}_{1}$，其中 $\lambda\,\in\,\mathbb{R}$ 且 $U\;=\;\operatorname{span}[b_{1}]\;\subseteq\;\mathbb{R}^{n}$ 是 $\mathbb{R}^{n}$ 中的一维子空间。这意味着一条直线由支持点 $\mathbf{\delta}x_{0}$ 和定义方向的向量 $b_{1}$ 定义。图 2.13 给出了一个示例。

**直线** **平面** **超平面**

二维线性子空间 $\textstyle\mathbb{R}^{n}$ 称为**平面**。参数化为 $\pmb{y}=\pmb{x}_{0}+\lambda_{1}\pmb{b}_{1}+\lambda_{2}\pmb{b}_{2}$，其中 $\lambda_{1},\lambda_{2}\in\mathbb{R}$ 且 $U\,=\,\operatorname{span}[b_{1},b_{2}]\,\subseteq\,\mathbb{R}^{n}$。这意味着一个平面由支持点 $\mathbf{\delta}x_{0}$ 和两个线性无关的向量 $b_{1},b_{2}$ 定义，这些向量定义了方向子空间。在 $\mathbb{R}^{n}$ 中，$(n-1)$ 维线性子空间称为**超平面**，其参数方程为 $\begin{array}{r}{\pmb{y}\,=\,\pmb{x}_{0}\,+\,\sum_{i=1}^{n-1}\lambda_{i}\pmb{b}_{i}.}\end{array}$，其中 $b_{1},\ldots,b_{n-1}$ 形成一个 $(n\mathrm{~-~}1)$ 维子空间 $U$ 的基。这意味着一个超平面由支持点 $\pmb{x}_{0}$ 和 $(n-1)$ 个线性无关的向量 $b_{1},\ldots,b_{n-1}$ 定义，这些向量定义了方向子空间。在 $\mathbb{R}^{2}$ 中，一条直线也是超平面。在 $\mathbb{R}^{3}$ 中，一个平面也是超平面。

![](images/5c598b20e6cbb35aa6019795a62672fc51efea4ef20278b405c7119cc0dc02cb.jpg)  
图 2.13 直线是线性子空间。直线上的向量 $\pmb{y}$ ${\pmb{x}}_{0}+\lambda b_{1}$ 位于具有支持点 $\mathbf{\delta}_{\mathbf{x}_{\mathrm{0}}}$ 和方向 ${\pmb{b}}_{1}$ 的线性子空间 $L$

注释（非齐次线性方程组与线性子空间）对于 $\pmb{A}\,\in\,\mathbb{R}^{m\times n}$ 和 $\pmb{x}\,\in\,\mathbb{R}^{m}$，线性方程组 $\pmb{A}\pmb{\lambda}\ =\ \pmb{x}$ 的解要么为空集，要么是 $\mathbb{R}^{n}$ 中的线性子空间，其维度为 $n\mathrm{~-~}\operatorname{rk}(\pmb{A})$。特别是，方程 $\lambda_{1}\pmb{b}_{1}+.\,.\,.+\lambda_{n}\pmb{b}_{n}=\pmb{x}\,.$ 的解是一个非齐次线性方程组的解，其中 $(\lambda_{1},\cdots,\lambda_{n})\neq(0,\cdots,0)$，是一个 $\mathbb{R}^{n}$ 中的超平面。

在 $\mathbb{R}^{n}$ 中，每个 $k$ 维线性子空间是非齐次线性方程组 $A x\,=\,b$ 的解，其中 $\pmb{A}\,\in\,\mathbb{R}^{m\times n},\pmb{b}\,\in\mathbb{R}^{m}$，且 $\operatorname{rk}(A)\,=\,n\,-\,k$。回想一下，对于齐次方程组 $A x=\mathbf{0}$ 的解是一个向量子空间，我们也可以将其视为具有支持点 $\mathbf{x}_{0}={\bf0}$ 的特殊线性子空间。$\diamondsuit$


### 2.8.2 线性映射

类似于我们在第 2.7 节讨论的向量空间之间的线性映射，我们可以在两个线性空间之间定义线性映射。线性和线性映射紧密相关。因此，我们已经从线性映射中了解的许多性质，例如线性映射的复合仍然是线性映射，这些性质同样适用于线性映射。

定义 2.26（线性映射）。对于两个向量空间 $V,W$，线性映射 $\Phi:V\to W$ 和 $a\in W$，映射

$$
\begin{array}{r l}{\phi:V\to W}\\ {{\pmb{x}}\mapsto{\pmb a}+\Phi({\pmb{x}})}\end{array}
$$
是从 $V$ 到 $W$ 的线性映射。向量 $^{a}$ 称为 $\phi$ 的**平移线性映射向量**。

每个线性映射 $\phi:V\to W$ 都可以表示为一个线性映射 $\Phi:V\to W$ 和一个平移映射 $\tau:W\to W$ 在 $W$ 中的组合，使得 $\phi=\tau\circ\Phi$。映射 $\Phi$ 和 $\tau$ 是唯一的。线性映射的复合 $\phi^{\prime}\circ\phi$ 也是线性映射。如果 $\phi$ 是双射，则线性映射保持几何结构不变。它们还保持维度和平行性。

## 2.9 进一步阅读

有许多资源可以学习线性代数，包括 Strang (2003)、Golan (2007)、Axler (2015) 和 Liesen 和 Mehrmann (2015) 的教科书。此外，我们还在本章的介绍中提到了几个在线资源。我们在这里只介绍了高斯消元法，但解决线性方程组的其他方法有很多，我们推荐 Stoer 和 Burlirsch (2002)、Golub 和 Van Loan (2012) 和 Horn 和 Johnson (2013) 的数值线性代数教科书进行深入讨论。

在本书中，我们区分了线性代数的主题（例如向量、矩阵、线性独立、基）和与向量空间几何相关的主题。在第 3 章中，我们将引入内积，它诱导出范数。这些概念允许我们定义角度、长度和距离，我们将使用这些概念进行正交投影。投影在许多机器学习算法中都是关键，例如线性回归和主成分分析，我们将在第 9 章和第 10 章分别介绍这两个算法。


## 练习题  

练习题 2.1 我们考虑   $(\mathbb{R}\backslash\{-1\},\star)$ ，其中  

$$
a\star b:=a b+a+b,\quad\quad a,b\in\mathbb{R}\backslash\{-1\}
$$
a. 证明   $(\mathbb{R}\backslash\{-1\},\star)$ 是阿贝尔群。  

b. 求解  

$$
3\star x\star x=15
$$
在阿贝尔群   $(\mathbb{R}\backslash\{-1\},\star)$ 中，其中 $\star$ 定义在（2.134）中。  

练习题 2.2 让  $n$ 为   $\mathbb{N}\backslash\{0\}$ 中的数。让 $k,x$ 为   $\mathbb{Z}$ 中的数。我们将整数 $k$ 的同余类   $\bar{k}$ 定义为集合  

$$
\begin{array}{r l}&{{\overline{{k}}}=\left\{x\in\mathbb{Z}\mid x-k=0\ \,(\mathrm{mod}n)\right\}}\\ &{\quad=\left\{x\in\mathbb{Z}\mid\exists a\in\mathbb{Z}\colon(x-k=n\cdot a)\right\}.}\end{array}
$$
我们现在定义   $\mathbb{Z}/n\mathbb{Z}$（有时写作   $\mathbb{Z}_{n}$）为模 $n$ 的所有同余类的集合。欧几里得除法表明这个集合是一个包含   $n$ 个元素的有限集合：  

$$
\mathbb{Z}_{n} = \{0, 1, \ldots, n-1\}
$$
对于所有   $\overline{{a}},\overline{{b}}\in\mathbb{Z}_{n}$，我们定义  

$$
{\overline{{a}}}\oplus{\overline{{b}}}:={\overline{{a+b}}}
$$
a. 证明   $\left(\mathbb{Z}_{n},\oplus\right)$ 是一个群。它是阿贝尔群吗？  

b. 我们现在定义另一个操作 $\otimes$ 对于所有 $\overline{{a}}$ 与   $\bar{b}$ 为   $\mathbb{Z}_{n}$ 中的数，定义为  

$$
\begin{array}{r}{\overline{{{a}}}\otimes\overline{{{b}}}=\overline{{{a}\times{b}}}\,,}\end{array}
$$
其中   $a\times b$ 代表在   $\mathbb{Z}$ 中的通常乘法。  

让 $n=5$ 。绘制元素   $\mathbb{Z}_{5}\backslash\{{\overline{{0}}}\}$ 在   $\otimes$ 下的时间表，即计算所有 $\overline{{a}}$ 与 $\bar{b}$ 为   $\mathbb{Z}_{5}\backslash\{\overline{{0}}\}$ 的乘积 ${\overline{{a}}}\otimes{\overline{{b}}}$。  

因此，证明   $\mathbb{Z}_{5}\backslash\{{\overline{{0}}}\}$ 在   $\otimes$ 下是封闭的，并且具有   $\otimes$ 的单位元素。展示所有元素在   $\mathbb{Z}_{5}\backslash\{{\overline{{0}}}\}$ 下的逆元素   $\otimes$ 。得出结论   $(\mathbb{Z}_{5}\backslash\{\overline{{0}}\},\otimes)$ 是阿贝尔群。  

c. 证明   $(\mathbb{Z}_{8}\backslash\{\overline{{0}}\},\otimes)$ 不是一个群。  

d. 我们回顾 B´ ezout 定理，它表明两个整数   $a$ 与   $b$ 是互质的（即   $g c d(a,b)=1$）当且仅当存在两个整数 $u$ 与   $v$ 使得   $a u+b v=1$。证明   $(\mathbb{Z}_{n}\backslash\{\overline{{0}}\},\otimes)$ 是一个群当且仅当 $n\in\mathbb{N}\backslash\{0\}$ 是素数。

练习题 2.3

考虑以下$3\times3$矩阵的集合$\mathcal{G}$：

$$
\mathcal{G}= \left\{   
\begin{bmatrix}  
1 & x & z \\
0 & 1 & y \\
0 & 0 & 1   
\end{bmatrix} \in \mathbb{R}^{3 \times 3} \, \bigg| \, x, y, z \in \mathbb{R}   
\right\} 
$$

我们定义$\cdot$为标准矩阵乘法。集合$\left(\mathcal{G},\cdot\right)$是否构成一个群？如果是，它是否为阿贝尔群？请给出理由。 练习题 2.4

计算以下矩阵乘积（如果可能）：
a.  

$$
\begin{array}{r}{\left[\begin{array}{r r r}{1}&{2}\\ {4}&{5}\\ {7}&{8}\end{array}\right]\left[\begin{array}{r r r}{1}&{1}&{0}\\ {0}&{1}&{1}\\ {1}&{0}&{1}\end{array}\right]}\end{array}
$$
b.  

$$
\begin{array}{r}{\left[\begin{array}{l l l}{1}&{2}&{3}\\ {4}&{5}&{6}\\ {7}&{8}&{9}\end{array}\right]\left[\begin{array}{l l}{1}&{1}\\ {0}&{1}\\ {1}&{0}\end{array}\right]}\end{array}
$$
c.  

$$
\begin{array}{r}{\left[\begin{array}{l l l}{1}&{1}&{0}\\ {0}&{1}&{1}\\ {1}&{0}&{1}\end{array}\right]\left[\begin{array}{l l l}{1}&{2}&{3}\\ {4}&{5}&{6}\\ {7}&{8}&{9}\end{array}\right]}\end{array}
$$
d.  

$$
\begin{array}{r}{\left[\begin{array}{l l l l}{1}&{2}&{1}&{2}\\ {4}&{1}&{-1}&{-4}\end{array}\right]\left[\begin{array}{l l}{0}&{3}\\ {1}&{-1}\\ {2}&{1}\\ {5}&{2}\end{array}\right]}\end{array}
$$
e.  

$$
\begin{array}{r}{\left[\begin{array}{r r r}{0}&{3}\\ {1}&{-1}\\ {2}&{1}\\ {5}&{2}\end{array}\right]\left[\begin{array}{r r r r}{1}&{2}&{1}&{2}\\ {4}&{1}&{-1}&{-4}\end{array}\right]}\end{array}
$$
练习题 2.5

找到以下非齐次线性系统的解集$s$，其中$\pmb{A}$和$\pmb{b}$定义如下：

a.  

$$
A={\left[\begin{array}{l l l l}{1}&{1}&{-1}&{-1}\\ {2}&{5}&{-7}&{-5}\\ {2}&{-1}&{1}&{3}\\ {5}&{2}&{-4}&{2}\end{array}\right]}\;,\quad\pmb{b}={\left[\begin{array}{l}{1}\\ {-2}\\ {4}\\ {6}\end{array}\right]}
$$
b.  

$$
A={\left[\begin{array}{l l l l l}{1}&{-1}&{0}&{0}&{1}\\ {1}&{1}&{0}&{-3}&{0}\\ {2}&{-1}&{0}&{1}&{-1}\\ {-1}&{2}&{0}&{-2}&{-1}\end{array}\right]}~,\quad\pmb{b}={\left[\begin{array}{l}{3}\\ {6}\\ {5}\\ {-1}\end{array}\right]}
$$

练习题 2.6

使用高斯消元法，求解非齐次方程组 $\pmb{A x}=\pmb{b}$，其中

$$
A=\left[0\quad1\quad0\quad0\quad1\quad0\right]\,,\quad\boldsymbol{b}=\left[{2\atop-1}\right]\,.
$$
练习题 2.7

求解方程组 ${\pmb{A x}}\,=\,12{\pmb{x}}$ 中的所有解 $\pmb{x}=\left[\begin{array}{c}x_{1}\\ x_{2}\\ x_{3} \end{array}\right] \in \mathbb{R}^{3}$，其中

$$
A={\left[\begin{array}{l l l}{6}&{4}&{3}\\ {6}&{0}&{9}\\ {0}&{8}&{0}\end{array}\right]}
$$
且 $\textstyle\sum_{i=1}^{3}x_{i}=1$。

练习题 2.8

确定以下矩阵的逆矩阵（如果可能）：

a.  

$$
A={\left[\begin{array}{l l l}{2}&{3}&{4}\\ {3}&{4}&{5}\\ {4}&{5}&{6}\end{array}\right]}
$$
b.  

$$
A={\left[\begin{array}{l l l l}{1}&{0}&{1}&{0}\\ {0}&{1}&{1}&{0}\\ {1}&{1}&{0}&{1}\\ {1}&{1}&{1}&{0}\end{array}\right]}
$$
练习题 2.9

以下哪些集合是 $\mathbb{R}^{3}$ 的子空间？

$$
\begin{array}{r l}&{C=\dot{\{(\xi_{1},\xi_{2},\xi_{3})\in\mathbb{R}^{3}\mid\xi_{1}-2\xi_{2}+3\xi_{3}=\gamma\}}}\\ &{D=\{(\xi_{1},\xi_{2},\xi_{3})\in\mathbb{R}^{3}\mid\xi_{2}\in\mathbb{Z}\}}\end{array}
$$
练习题 2.10

以下向量组是否线性独立？

a.  

$$
\mathbf{x}_{1}={\left[{\begin{array}{l}{2}\\ {-1}\\ {3}\end{array}}\right]}\;,\quad\mathbf{x}_{2}={\left[{\begin{array}{l}{1}\\ {1}\\ {-2}\end{array}}\right]}\;,\quad\mathbf{x}_{3}={\left[{\begin{array}{l}{3}\\ {-3}\\ {8}\end{array}}\right]}
$$
b.  

$$
\pmb{x}_{1}=\left[\begin{array}{l}{1}\\ {2}\\ {1}\\ {0}\\ {0}\end{array}\right],\quad\pmb{x}_{2}=\left[\begin{array}{l}{1}\\ {1}\\ {0}\\ {1}\\ {1}\end{array}\right],\quad\pmb{x}_{3}=\left[\begin{array}{l}{1}\\ {0}\\ {0}\\ {1}\\ {1}\end{array}\right]
$$
练习题 2.11

将  

$$
\pmb{y}=\left[\begin{array}{l}{1}\\ {-2}\\ {5}\end{array}\right]
$$
表示为向量组  

$$
\pmb{x}_{1}=\left[\begin{array}{c}{{1}}\\ {{1}}\\ {{1}}\end{array}\right]\,,\quad\pmb{x}_{2}=\left[\begin{array}{c}{{1}}\\ {{2}}\\ {{3}}\end{array}\right]\,,\quad\pmb{x}_{3}=\left[\begin{array}{c}{{2}}\\ {{-1}}\\ {{1}}\end{array}\right]
$$
的线性组合。

练习题 2.12

考虑 $\mathbb{R}^{4}$ 中的两个子空间：  

$$
U_{1}=\mathrm{span}[\left[\begin{array}{c}{1}\\ {1}\\ {-3}\\ {1}\end{array}\right]\,,\,\left[\begin{array}{c}{2}\\ {-1}\\ {0}\\ {-1}\end{array}\right]\,,\quad U_{2}=\mathrm{span}[\left[\begin{array}{c}{-1}\\ {-2}\\ {2}\\ {1}\end{array}\right]\,,\,\left[\begin{array}{c}{2}\\ {-2}\\ {0}\\ {0}\end{array}\right]\,,\,\left[\begin{array}{c}{-3}\\ {6}\\ {-2}\\ {-1}\end{array}\right]\,.
$$
确定 $U_{1}\cap U_{2}$ 的基。

练习题 2.13

考虑两个子空间 $U_{1}$ 和 $U_{2}$，其中 $U_{1}$ 是齐次方程系统 $A_{1}x=\mathbf{0}$ 的解空间，$U_{2}$ 是齐次方程系统 ${\pmb A}_{2}{\pmb{x}}={\bf0}$ 的解空间，其中

$$
A_{1}=\left[{\begin{array}{c c c}{1}&{0}&{1}\\ {1}&{-2}&{-1}\\ {2}&{1}&{3}\\ {1}&{0}&{1}\end{array}}\right]\,,\quad A_{2}=\left[{\begin{array}{c c c}{3}&{-3}&{0}\\ {1}&{2}&{3}\\ {7}&{-5}&{2}\\ {3}&{-1}&{2}\end{array}}\right]\,.
$$
a. 确定 $U_{1}$ 和 $U_{2}$ 的维数。

b. 确定 $U_{1}$ 和 $U_{2}$ 的基。

c. 确定 $U_{1}\cap U_{2}$ 的基。

练习题 2.14

考虑两个子空间 $U_{1}$ 和 $U_{2}$，其中 $U_{1}$ 是矩阵 $A_{1}$ 列向量的生成空间，$U_{2}$ 是矩阵 $A_{2}$ 列向量的生成空间，其中

$$
A_{1}=\left[{\begin{array}{c c c}{1}&{0}&{1}\\ {1}&{-2}&{-1}\\ {2}&{1}&{3}\\ {1}&{0}&{1}\end{array}}\right]\,,\quad A_{2}=\left[{\begin{array}{c c c}{3}&{-3}&{0}\\ {1}&{2}&{3}\\ {7}&{-5}&{2}\\ {3}&{-1}&{2}\end{array}}\right]\,.
$$
a. 确定 $U_{1}$ 和 $U_{2}$ 的维数。

b. 确定 $U_{1}$ 和 $U_{2}$ 的基。

c. 确定 $U_{1}\cap U_{2}$ 的基。

练习题 2.15

令 $F=\{(x,y,z)\in\mathbb{R}^{3}\mid x{+}y{-}z=0\}$ 和 $G=\{(a{-}b,a{+}b,a{-}3b)\mid a,b\in\mathbb{R}\}$。

a. 证明 $F$ 和 $G$ 是 $\mathbb{R}^{3}$ 的子空间。

b. 不借助任何基向量，计算 $F\cap G$。

c. 找到 $F$ 和 $G$ 的一个基，使用之前找到的基向量计算 $F\cap G$，并检查结果与上一问题是否一致。


练习题 2.16

以下映射是否线性？

a. 让 $a,b\in\mathbb{R}$。

$$
\begin{array}{l}{\Phi:L^{1}([a,b])\to\mathbb{R}}\\ {\qquad\qquad f\mapsto\Phi(f)=\int_{a}^{b}f(x)d x\,,}\end{array}
$$

其中 $L^{1}([a,b])$ 表示区间 $[a,b]$ 上的可积函数集合。

$$
\begin{array}{r}{\Phi:C^{1}\to C^{0}\qquad\qquad\quad}\\ {f\mapsto\Phi(f)=f^{\prime}\,,}\end{array}
$$

其中对于 $k\geqslant1$，$C^{k}$ 表示 $k$ 次连续可微函数的集合，而 $C^{0}$ 表示连续函数的集合。

c. 

$$
\begin{array}{r}{\Phi:\mathbb{R}\to\mathbb{R}}\\ {x\mapsto\Phi(x)=\cos(x)}\end{array}
$$

d. 

$$
\begin{array}{r l}{\Phi:\mathbb{R}^{3}\to\mathbb{R}^{2}}&{{}}\\ {\pmb{x}\mapsto\left[\begin{array}{c c c}{1}&{2}&{3}\\ {1}&{4}&{3}\end{array}\right]\pmb{x}}\end{array}
$$

e. 让 $\theta$ 在区间 $[0,2\pi]$ 内。

$$
\begin{array}{r l}{\Phi:\mathbb{R}^{2}\to\mathbb{R}^{2}}&{{}}\\ {\pmb{x}\mapsto\left[\begin{array}{c c}{\cos(\theta)}&{\sin(\theta)}\\ {-\sin(\theta)}&{\cos(\theta)}\end{array}\right]\pmb{x}}\end{array}
$$

练习题 2.17

考虑线性映射

$$
\begin{array}{r l}
&\Phi:\mathbb{R}^{3}\rightarrow\mathbb{R}^{4}\\
&\Phi\left(\begin{bmatrix}
x_{1}\\
x_{2}\\
x_{3}
\end{bmatrix}\right)=\begin{bmatrix}
3x_{1}+2x_{2}+x_{3}\\
x_{1}+x_{2}+x_{3}\\
x_{1}-3x_{2}\\
2x_{1}+3x_{2}+x_{3}
\end{bmatrix}
\end{array} 
$$

找到变换矩阵 $\pmb{A}_{\Phi}$。

确定 $\mathrm{rk}({\pmb{A}}_{\Phi})$。

计算 $\Phi$ 的核和像。$\mathrm{dim}(\mathrm{ker}(\Phi))$ 和 $\mathrm{dim(Im}(\Phi))$ 分别是多少？

练习题 2.18

设 $E$ 为向量空间。设 $f$ 和 $g$ 是 $E$ 上的两个同构映射，使得 $f\circ g=\mathrm{id}_{E}$（即 $f\circ g$ 是单位映射 $\mathrm{id}_{E}$）。证明 $\ker(f)=\ker(g\circ f)$，$\operatorname{Im}(g)=\operatorname{Im}(g\circ f)$，$\ker(f)\cap\operatorname{Im}(g)=\{\mathbf{0}_{E}\}$。

练习题 2.19

考虑一个映射 $\Phi:\mathbb{R}^{3}\rightarrow\mathbb{R}^{3}$，其相对于 $\mathbb{R}^{3}$ 的标准基的变换矩阵为

$$
A_{\Phi}=\left[\begin{array}{c c c}{{1}}&{{1}}&{{0}}\\ {{1}}&{{-1}}&{{0}}\\ {{1}}&{{1}}&{{1}}\end{array}\right]\,.
$$
a. 确定 $\ker(\Phi)$ 和 $\operatorname{Im}(\Phi)$。

b. 确定相对于基

$$
B=(\begin{array}{l}{{\displaystyle[1]}}\\ {{\displaystyle1}}\\ {{\displaystyle1}}\end{array},\begin{array}{l}{{\displaystyle[1]}}\\ {{\displaystyle2}}\\ {{\displaystyle1}}\end{array},\begin{array}{l}{{\displaystyle[1]}}\\ {{\displaystyle0}}\\ {{\displaystyle0}}\end{array})\,,
$$
的变换矩阵 ${\tilde{\cal A}}_{\Phi}$，即执行从基 $B$ 到新基的基变换。

练习题 2.20

考虑向量 ${b}_{1},{b}_{2},{b}_{1}^{\prime},{b}_{2}^{\prime}$，$\mathbb{R}^{2}$ 中的 4 个向量，分别以 $\mathbb{R}^{2}$ 的标准基表示为

$$
\pmb{b}_{1}=\left[^{2}_{1}\right],\quad\pmb{b}_{2}=\left[^{-1}_{-1}\right],\quad\pmb{b}_{1}^{\prime}=\left[^{2}_{-2}\right],\quad\pmb{b}_{2}^{\prime}=\left[^{1}_{1}\right]
$$
以及两个有序基 $B=(b_{1},b_{2})$ 和 $B^{\prime}=(b_{1}^{\prime},b_{2}^{\prime})$。

a. 证明 $B$ 和 $B^{\prime}$ 是 $\mathbb{R}^{2}$ 的两个基，并绘制这些基向量。

b. 计算从 $B^{\prime}$ 到 $B$ 的基变换矩阵 $P_{1}$。

c. 考虑 $\mathbb{R}^{3}$ 中的三个向量 $c_{1},c_{2},c_{3}$，分别以标准基表示为

$$
\mathbf{c}_{1}={\left[\begin{array}{l}{1}\\ {2}\\ {-1}\end{array}\right]}\,,\quad\mathbf{c}_{2}={\left[\begin{array}{l}{0}\\ {-1}\\ {2}\end{array}\right]}\,,\quad\mathbf{c}_{3}={\left[\begin{array}{l}{1}\\ {0}\\ {-1}\end{array}\right]}
$$
并定义 $C=(c_{1},c_{2},c_{3})$。

(i) 通过使用行列式（见第 4.1 节）证明 $C$ 是 $\mathbb{R}^{3}$ 的基。 (ii) 让 $C^{\prime}=(c_{1}^{\prime},c_{2}^{\prime},c_{3}^{\prime})$ 是 $\mathbb{R}^{3}$ 的标准基。确定从 $C$ 到 $C^{\prime}$ 的基变换矩阵 $P_{2}$。

d. 考虑一个同态 $\Phi:\mathbb{R}^{2}\rightarrow\mathbb{R}^{3}$，使得

$$
\begin{array}{c c c}{{\Phi(b_{1}+b_{2})}}&{{=}}&{{c_{2}+c_{3}}}\\ {{\Phi(b_{1}-b_{2})}}&{{=}}&{{2c_{1}-c_{2}+3c_{3}}}\end{array}
$$
其中 $B=(b_{1},b_{2})$ 和 $C=(c_{1},c_{2},c_{3})$ 分别是 $\mathbb{R}^{2}$ 和 $\mathbb{R}^{3}$ 的有序基。确定相对于基 $B$ 和 $C$ 的 $\Phi$ 的变换矩阵 $\pmb{A}_{\Phi}$。

e. 计算相对于基 $B^{\prime}$ 和 $C^{\prime}$ 的 $\Phi$ 的变换矩阵 $A^{\prime}$。

f. 考虑向量 $\pmb{x}\in\mathbb{R}^{2}$，其在 $B^{\prime}$ 中的坐标为 $\left[2,3\right]^{\top}$。换句话说，${\pmb{x}}=2b_{1}^{\prime}+3b_{2}^{\prime}$。

(i) 计算 $x$ 在 $B$ 中的坐标。 (ii) 根据此，计算 $\Phi({\pmb{x}})$ 在 $C$ 中的坐标。 (iii) 然后，用 $c_{1}^{\prime},c_{2}^{\prime},c_{3}^{\prime}$ 表示 $\Phi({\pmb{x}})$。 (iv) 使用 $x$ 在 $B^{\prime}$ 中的表示和矩阵 $A^{\prime}$ 直接找到此结果。