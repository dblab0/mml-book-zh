# 第七章：连续优化 

由于机器学习算法是在计算机上实现的，因此数学公式表现为数值优化方法。本章介绍训练机器学习模型的基本数值方法。训练机器学习模型通常归结为寻找一组好的参数。"好"的概念由目标函数或概率模型确定，我们将在本书第二部分看到这些例子。给定一个目标函数，寻找最佳值是通过使用优化算法完成的。 本章涵盖了连续优化的两个主要分支（图7.1）：无约束和有约束优化。在本章中，我们假设我们的目标函数是可微的（见第5章），因此我们可以在空间中的每个位置访问梯度以帮助我们找到最优值。按照惯例，大多数机器学习中的目标函数都是要最小化的，即最佳值是最小值。直观地找到最佳值就像找到目标函数的谷底，而梯度指向我们上山。我们的想法是向下坡（与梯度相反）移动，希望找到最深的点。对于无约束优化，这是我们需要的唯一概念，但是有几个设计选择，我们将在7.1节中讨论。对于有约束优化，我们需要引入其他概念来处理约束（第7.2节）。我们还将介绍一类特殊的问题（第7.3节的凸优化问题），我们可以对达到全局最优值做出陈述。 考虑图7.2中的函数。该函数在$x \approx -4.5$处有一个全局最小值，函数值约为$-47$。由于函数是"平滑"的，因此可以利用梯度帮助找到最小值，指示我们应该向左还是向右迈出一步。这假设我们处于正确的碗中，因为另一个局部最小值在$x=0.7$附近。回想一下，我们可以通过计算函数的导数并将其设置为零来求解函数的所有驻点。对于 $$ \ell(x)=x^{4}+7x^{3}+5x^{2}-17x+3\,, $$ 我们得到相应的梯度为 $$ {\frac{\mathrm{d}\ell(x)}{\mathrm{d}x}}=4x^{3}+21x^{2}+10x-17\,. $$ 考虑到我们在$\mathbb{R}^{D}$中的数据和模型，我们面临的优化问题是连续优化问题，而不是离散变量的组合优化问题。 全局最小值 局部最小值 驻点是导数的实根，即梯度为零的点。 ![图7.1](images/ee6eefa87ab7ee5f03d01c63e8b3655b24ff05303798eac590edc4a517e45fa2.jpg) 图7.1 本章中介绍的与优化相关的概念思维导图。有两个主要想法：梯度下降和凸优化。 由于这是一个三次方程，当它等于零时一般有三个解。在示例中，其中两个是最小值，一个是最大值（约$x\,=\,-1.4$）。为了检查一个驻点是最小值还是最大值，我们需要对导数进行二次求导，并检查在驻点处二阶导数是正是负。在我们的情况下，二阶导数是 $$ {\frac{\mathrm{d}^{2}\ell(x)}{\mathrm{d}x^{2}}}=12x^{2}+42x+10\,. $$ 通过代入我们视觉估计的$x\,=\,-4.5,-1.4,0.7$的值，我们将观察到如预期的那样，中间点是最大值$\begin{array}{r}{\left(\frac{\mathrm{d}^{2}\ell(x)}{\mathrm{d}x^{2}}<0\right)}\end{array}$，而其他两个驻点是最小值。 请注意，在前面的讨论中，我们已经避免了解析地求解$x$的值，尽管对于像前面的低阶多项式，我们本可以做到这一点。通常，我们无法找到解析解，因此需要从某个值开始，比如说$x_{0}=-6$，并沿着负梯度前进。负梯度指示我们应该向右走，但没有告诉我们走多远（这称为步长）。此外，如果我们从右侧开始（例如，$x_{0}=0$），负梯度会引导我们到达错误的最低点。图7.2说明了对于$x>-1$，负梯度指向图右侧的最小值，该最小值的客观值较大。 在第7.3节中，我们将了解一类称为凸函数的函数，它们不会表现出这种对优化算法起始点的棘手依赖。对于凸函数，所有局部最小值都是全局最小值。事实证明，许多机器学习目标函数都是凸的，我们将在第12章看到一个例子。 根据阿贝尔-鲁菲尼定理，一般来说，对于五次或更高次的多项式没有代数解（阿贝尔，1826年）。 对于凸函数，所有局部最小值都是全局最小值。 本章到目前为止的讨论是关于一维函数的，在那里我们可以直观地看到梯度、下降方向和最优值的概念。在本章的其余部分，我们将在高维中发展相同的想法。不幸的是，我们只能在一维中直观地了解这些概念，但有些概念并不直接推广到更高维度，因此在阅读时需要小心。 ![图7.2](images/08006f1da73e25381dc0281de587e5ee165d585b62c136a585238bac3f7094b4.jpg) 图7.2 示例目标函数。负梯度由箭头指示，全局最小值由虚线蓝线指示。 以下是本文中提到的图片和公式的翻译说明： - 所有图片（例如，图7.1和图7.2）均保持原状，不进行翻译。 - 所有LaTeX公式（例如，$\ell(x)=x^{4}+7x^{3}+5x^{2}-17x+3$ 和 ${\frac{\mathrm{d}\ell(x)}{\mathrm{d}x}}=4x^{3}+21x^{2}+10x-17$）均保持不变。

## 7.1 使用梯度下降进行优化 

我们现在考虑解决实值函数最小值的问题 $$ \operatorname*{min}_{\pmb{x}}f(\pmb{x})\,, $$ 其中 $f\,:\,\mathbb{R}^{d}\,\rightarrow\,\mathbb{R}$ 是一个目标函数，它描述了当前的机器学习问题。我们假设我们的函数 $f$ 是可微的，我们无法以封闭形式解析地找到解。 我们使用梯度的行向量约定。 梯度下降是一种一阶优化算法。为了找到函数的局部最小值，梯度下降算法在当前点沿着函数梯度的负方向进行等比例的步进。回想一下第5.1节，梯度指向最陡上升的方向。另一个有用的直觉是考虑函数在某个值 $(f(\pmb{x})=c$ 对于某个值 $c\in\mathbb{R}$) 的等高线集，它们被称为等高线。梯度指向一个与我们要优化的函数的等高线正交的方向。 让我们考虑多变量函数。想象一个表面（由函数 $f(\pmb{x})$ 描述）和一个从特定位置 $\pmb{x}_{0}$ 开始的球。当球被释放时，它将沿着最陡下降的方向向下移动。梯度下降利用了这样一个事实：$f(\pmb{x}_{0})$ 的减少速度最快，如果从 $\pmb{x}_{0}$ 沿着 $f$ 在 $\pmb{x}_{0}$ 处的负梯度方向 $-((\nabla f)(\pmb{x}_{0}))^{\top}$ 移动。我们假设本书中的函数是可微的，并参考7.4节中更一般的情况。那么，如果 $$ {\pmb x}_{1}={\pmb x}_{0}-\gamma((\nabla f)({\pmb x}_{0}))^{\top} $$ 对于小的步长 $\gamma\geqslant0$，那么 $f(\pmb{x}_{1})\,\leqslant\,f(\pmb{x}_{0})$。注意我们使用梯度的转置，否则维度将不匹配。 这个观察允许我们定义一个简单的梯度下降算法：要找到函数 $f:\mathbb{R}^{n}\to R$，$\pmb{x}\mapsto f(\pmb{x})$ 的局部最优值 $f(\pmb{x}_{*})$，我们从我们想要优化的参数的初始猜测 $\pmb{x}_{0}$ 开始，然后根据以下迭代 $$ \pmb{x}_{i+1}=\pmb{x}_{i}-\gamma_{i}\big((\nabla f)(\pmb{x}_{i})\big)^{\top}\,. $$ 对于合适的步长 $\gamma_{i}$，序列 $f(\pmb{x}_{0})\geqslant f(\pmb{x}_{1})\geqslant.\,.\,.$ 收敛到一个局部最小值。

### 示例 7.1 

考虑一个二维二次函数

 $$ f\left( \begin{bmatrix} x_{1} \\x_{2}  \end{bmatrix} \right) = \frac{1}{2} \begin{bmatrix} x_{1} \\x_{2}  \end{bmatrix}^{\top} \begin{bmatrix} 2 & 1 \\ 1 & 20 \end{bmatrix} \begin{bmatrix} x_{1} \\x_{2}  \end{bmatrix} - \begin{bmatrix} 5 \\ 3 \end{bmatrix}^{\top} \begin{bmatrix} x_{1} \\x_{2}  \end{bmatrix} \tag{7.7}$$ 其梯度为 
 $$ \nabla f\left(\begin{bmatrix} x_{1} \\x_{2}  \end{bmatrix} \right) = \begin{bmatrix} x_{1} \\x_{2}  \end{bmatrix}^{\top} \begin{bmatrix} 2 & 1 \\ 1 & 20 \end{bmatrix} - \begin{bmatrix} 5 \\ 3 \end{bmatrix}^{\top} \tag{7.8}$$
  从初始位置 $\pmb{x}_{0}=[-3,-1]^{\top}$ 开始，我们迭代应用(7.6)得到一系列估计值，这些估计值收敛到最小值（如图7.3所示）。![](images/4df4243396c8591a774e0a0d75a871d6c53cdac248280af303fce5f412cd1f9f.jpg) 图7.3 二维二次表面的梯度下降（显示为热图）。有关说明，请参阅示例7.1。 （如图7.3所示）。我们可以看到（既可以从图中看出，也可以将 $\mathbf{\delta}x_{0}$ 代入 (7.8) 中，其中 $\gamma=0.085)$ ），$\mathbf{\delta}x_{0}$ 处的负梯度指向北和东，导致 $\pmb{x}_{1}=[-1.98,1.21]^{\top}$。重复该论点得到 $\pmb{x}_{2}=[-1.32,-0.42]^{\top}$，依此类推。 备注。梯度下降在接近最小值时可能相对较慢：其渐近收敛速度不如许多其他方法。使用球滚下山的类比，当表面是一个长而薄的峡谷时，问题条件不佳（Trefethen and Bau III, 1997）。对于条件不佳的凸问题，梯度下降随着梯度几乎正交于到最小点的最短方向而越来越“之字形”；请参见图7.3。

### 7.1.1 步长 

如前所述，在梯度下降中选择一个好的步长很重要。如果步长太小，梯度下降可能会很慢。如果选择的步长太大，梯度下降可能会超出范围，无法收敛，甚至可能发散。我们将在下一节讨论动量的使用。动量是一种平滑梯度更新中的不规则行为并抑制振荡的方法。 自适应梯度方法根据函数的局部性质，在每次迭代时重新调整步长。有两个简单的启发式方法（Toussaint, 2012）： 1. 当梯度步之后函数值增加时，步长太大。撤销步进并减小步长。 2. 当函数值减小时，步长可能更大。尝试增加步长。 尽管“撤销”步骤看起来像是浪费资源，但使用这个启发式方法可以保证单调收敛。 ### 示例 7.2（解线性方程组） 当我们解形式为 $A x=b$ 的线性方程时，实际上我们通过找到最小化平方误差的 $\pmb{x}_{*}$ 来近似解 $\mathbf{\nabla}A\mathbf{x}-\mathbf{b}=\mathbf{0}$ $$ \|A{\boldsymbol{\mathbf{x}}}-b\|^{2}=(A{\boldsymbol{\mathbf{x}}}-b)^{\top}(A{\boldsymbol{\mathbf{x}}}-b) $$ 如果我们使用欧几里得范数。关于 $_{_{\pmb{x}}}$ 的(7.9)的梯度是 $$ \nabla_{\pmb{x}}=2({\pmb{A}}{\pmb{x}}-b)^{\top}{\pmb{A}}\,. $$ 我们可以在梯度下降算法中直接使用这个梯度。然而，对于这个特殊的特殊情况，实际上有一个解析解，可以通过将梯度设置为零来找到。我们将在第9章中看到更多关于解平方误差问题的内容。 备注。当应用于方程组 ${\pmb A x}=$ $^{b}$ 的解时，梯度下降可能收敛缓慢。梯度下降的收敛速度取决于条件数 $\begin{array}{r}{\kappa=\frac{\sigma(\breve{\cal A})_{\mathrm{max}}}{\sigma({\cal A})_{\mathrm{min}}}}\end{array}$，这是 $\pmb{A}$ 的最大奇异值与最小奇异值（第4.5节）的比率。条件数本质上是测量最弯曲方向与最不弯曲方向的比率，这对应于我们的图像，即条件不好的问题是长而细的山谷：它们在一个方向上非常弯曲，但在另一个方向上非常平坦。与其直接解 $A x=b$，不如解 $P^{-1}(A\mathbf{x}-\mathbf{b})=\mathbf{0}$，$P$ 称为预处理程序。目标是设计 $P^{-1}$ 使得 $P P^{-1}A$ 有更好的条件数，但与此同时 $P^{-1}$ 易于计算。有关梯度下降、预处理和收敛的更多信息，请参阅Boyd和Vandenberghe（2004，第9章）。

### 7.1.2 梯度下降与动量

Goh (2017) 在一篇直观的博客文章中写了关于带动量的梯度下降。 如图7.3所示，如果优化表面的曲率使得某些区域缩放不佳，梯度下降的收敛可能会非常慢。曲率是这样的，梯度下降步骤在山谷的墙壁之间跳跃，并以小步骤接近最优值。为了改善收敛，提出的调整是给梯度下降一些记忆。 带动量的梯度下降（Rumelhart et al., 1986）是一种引入额外项来记住上一次迭代中发生了什么的方法。这种记忆抑制了振荡并平滑了梯度更新。继续使用球类比，动量项模拟了一个不愿意改变方向的沉重球的现象。想法是有一个带有记忆的梯度更新来实现移动平均。基于动量的方法记住每次迭代 $i$ 的更新 $\Delta\pmb{x}_{i}$，并将下一次更新确定为当前和先前梯度的线性组合 $$ \begin{array}{r l} &{\pmb{x}_{i+1}=\pmb{x}_{i}-\gamma_{i}((\nabla f)(\pmb{x}_{i}))^{\top}+\alpha\Delta\pmb{x}_{i}}\\ &{\Delta\pmb{x}_{i}=\pmb{x}_{i}-\pmb{x}_{i-1}=\alpha\Delta\pmb{x}_{i-1}-\gamma_{i-1}((\nabla f)(\pmb{x}_{i-1}))^{\top}\,,} \end{array} $$ 其中 $\alpha\ \in\ [0,1]$。有时我们只知道梯度的近似值。在这种情况下，动量项很有用，因为它可以平均掉梯度的不同噪声估计。获得近似梯度的一种特别有用的方法是使用随机近似，我们接下来讨论这个方法。 
### 7.1.3 随机梯度下降

计算梯度可能非常耗时。然而，通常可以找到梯度的“便宜”近似。只要近似梯度大致指向与真实梯度相同的方向，近似梯度仍然有用。 随机梯度下降（通常缩写为SGD）是梯度下降方法的一种随机近似，用于最小化可以写成可微函数之和的目标函数。这里的“随机”是指我们承认我们不知道精确的梯度，而只知道对其的噪声近似。通过约束近似梯度的概率分布，我们理论上仍然可以保证SGD将收敛。 在机器学习中，给定 $n=1,\cdot\cdot\cdot,N$ 数据点，我们通常考虑由每个示例 $n$ 产生的损失 $L_{n}$ 之和构成的目标函数。在数学表示中，我们有以下形式 $$ L(\pmb\theta)=\sum_{n=1}^{N}L_{n}(\pmb\theta)\, $$ 其中 $\pmb{\theta}$ 是感兴趣的参数向量，即我们希望找到使 $L$ 最小的 $\pmb{\theta}$。来自回归（第9章）的一个例子是负对数似然，它表示为单个示例的对数似然之和，因此 $$ L(\pmb\theta)=-\sum_{n=1}^{N}\log p(y_{n}|\pmb x_{n},\pmb\theta)\, $$ 其中 $\pmb{x}_{n}\in\mathbb{R}^{D}$ 是训练输入，$y_{n}$ 是训练目标，$\pmb{\theta}$ 是回归模型的参数。 如前所述，标准的梯度下降是一种“批量”优化方法，即通过根据以下公式更新参数向量来进行优化 $$ \pmb{\theta}_{i+1}=\pmb{\theta}_{i}-\gamma_{i}(\nabla L(\pmb{\theta}_{i}))^{\top}=\pmb{\theta}_{i}-\gamma_{i}\sum_{n=1}^{N}(\nabla L_{n}(\pmb{\theta}_{i}))^{\top} $$ 对于合适的步长参数 $\gamma_{i}$。计算和梯度可能需要昂贵的所有单个函数 $L_{n}$ 的梯度评估。当训练集巨大且/或不存在简单公式时，梯度求和的计算变得非常昂贵。 考虑 (7.15) 中的项 $\begin{array}{r l}{\phantom{\sum_{n=1}^{N}}\sum_{n=1}^{\tilde{N}}(\nabla L_{n}(\pmb\theta_{i}))}&{{}}\end{array}$ ∇。我们可以通过取一个更小的 $L_{n}$ 和来减少计算量。与使用所有 $L_{n}$ 的批量梯度下降不同，我们为小批量梯度下降随机选择一个 $L_{n}$ 的子集。在极端情况下，我们只随机选择一个 $L_{n}$ 来估计梯度。为什么取数据子集是明智的关键洞察是，为了使梯度下降收敛，我们只需要梯度是对真实梯度的无偏估计。实际上，(7.15) 中的项 $\begin{array}{r l}{\phantom{\sum_{n=1}^{N}}\sum_{n=1}^{\check{N}}(\nabla L_{n}(\pmb{\theta}_{i}))}&{{}}\end{array}$ ∇ 是梯度的期望值的经验估计（第6.4.1节）。因此，任何其他无偏经验估计的期望值，例如使用数据的任何子样本，都足以使梯度下降收敛。 备注。当学习率以适当的速率下降，并且满足相对温和的假设时，随机梯度下降几乎可以肯定地收敛到局部最小值（Bottou, 1998）。 为什么应该考虑使用近似梯度？一个主要原因是实际实现的限制，例如CPU/GPU内存的大小或计算时间的限制。我们可以将用于估计梯度的子集大小视为与估计经验均值时样本大小（第6.4.1节）相同的方式。大的小批量大小将提供对梯度的准确估计，减少参数更新的方差。此外，大的小批量可以利用高度优化的矩阵操作在矢量化实现中的成本和梯度。方差减少导致更稳定的收敛，但每次梯度计算将更加昂贵。 相比之下，小的小批量快速估计。如果我们保持小批量大小较小，我们的梯度估计中的噪声将允许我们摆脱一些糟糕的局部最优，否则我们可能会陷入其中。在机器学习中，优化方法用于通过最小化训练数据上的目标函数来训练，但总体目标是提高泛化性能（第8章）。由于机器学习的目标不一定需要目标函数最小值的精确估计，因此使用小批量方法的近似梯度已被广泛使用。随机梯度下降在处理大规模机器学习问题（Bottou et al., 2018）方面非常有效， ![图7.4 约束优化示意图](images/21033fb32356feeb38c60fb644ed3d10db93354a3df9abd053a0be61b87cba21.jpg) 图7.4 约束优化示意图。无约束问题（由等高线表示）的最小值在右侧（由圆圈表示）。盒子约束（$-1\leqslant x\leqslant1$ 和 $-1\leqslant y\leqslant1$）要求最优解在盒子内，结果是最优值由星号表示。

## 7.2 约束优化与拉格朗日乘数 

在上一节中，我们考虑了求函数最小值的问题 $$ \operatorname*{min}_{\pmb{x}}f(\pmb{x})\,, $$ 其中 $f:\mathbb{R}^{D}\rightarrow\mathbb{R}$。 在本节中，我们有一些额外的约束条件。也就是说，对于实值函数 $g_{i}:\mathbb{R}^{D}\rightarrow\mathbb{R}$，其中 $i=1,...,m$，我们考虑约束优化问题（图 7.4 用图示形式展示了该问题）： $$ \begin{array}{r l} {\underset{\pmb{x}}{\mathrm{min}}}&{f(\pmb{x})}\\ {\mathrm{subject\ to}}&{g_{i}(\pmb{x})\leqslant0\quad\mathrm{for\,all}\quad i=1,...,m\,.} \end{array} $$ 值得注意的是，一般的函数 $f$ 和 $g_{i}$ 可能是非凸的，我们在下一节中将考虑凸情况。 将约束问题转化为无约束问题的一个显然但不实用的方法是使用指示函数 $$ J(\pmb{x})=f(\pmb{x})+\sum_{i=1}^{m}\mathbf{1}\big(g_{i}(\pmb{x})\big)\,, $$ 其中 $\mathbf{1}(z)$ 是一个无穷阶梯函数 $$ \mathbf{1}(z)=\left\{ \begin{array}{ll} 0 & \text{if } z\leqslant0 \end{array} \right. $$ 拉格朗日乘数 拉格朗日乘数 当不满足约束条件时，这个无穷阶梯函数给出无限的惩罚，因此会得到相同的解。然而，这种无穷阶梯函数也难以优化。我们可以通过引入拉格朗日乘数来克服这个困难。拉格朗日乘数的想法是用线性函数代替阶梯函数。 为问题 (7.17) 引入拉格朗日函数，其中对应的拉格朗日乘数 $\lambda_{i}\geqslant0$ 对应于每个不等式约束（Boyd 和 Vandenberghe, 2004, 第 4 章），使得 $$ \begin{array}{l} \mathfrak{L}(\pmb{x},\pmb{\lambda})=f(\pmb{x})+\sum_{i=1}^{m}\lambda_{i}g_{i}(\pmb{x})\\ \quad\quad\quad\quad=f(\pmb{x})+\pmb{\lambda}^{\top}\pmb{g}(\pmb{x})\,, \end{array} $$ 其中在最后一行中，我们将约束 $g_{i}(\pmb{x})$ 重组为向量 $\pmb{g}(\pmb{x})$，将所有拉格朗日乘数 $\pmb{\lambda}$ 重组为向量 $\pmb{\lambda}\in\mathbb{R}^{m}$。 接下来，我们引入拉格朗日对偶性的概念。优化中的对偶性是指将一个变量集 $\pmb{x}$（称为原始变量）的问题转化为另一个变量集 $\pmb{\lambda}$（称为对偶变量）的问题。我们介绍了两种不同的对偶方法：在本节中，我们讨论拉格朗日对偶性；在第 7.3.3 节中，我们讨论 Legendre-Fenchel 对偶性。 定义 7.1. 问题 (7.17) $$ \begin{array}{r l} {\underset{\pmb{x}}{\mathrm{min}}}&{f(\pmb{x})}\\ {\mathrm{subject\ to}}&{g_{i}(\pmb{x})\leqslant0\quad\mathrm{for\,all}\quad i=1,...,m} \end{array} $$ 被称为 **原始问题**，对应的原始变量为 $\pmb{x}$。相应的 **对偶问题** 为 $$ \begin{array}{r l} {\underset{\pmb{\lambda}\in\mathbb{R}^{m}}{\mathrm{max}}}&{\mathfrak{D}(\pmb{\lambda})}\\ {\mathrm{subject\ to}}&{\pmb{\lambda}\geqslant\mathbf{0}\,} \end{array} $$ 其中 $\pmb{\lambda}$ 是对偶变量，$\mathfrak{D}(\pmb{\lambda})=\operatorname*{min}_{\pmb{x}\in\mathbb{R}^{d}}\mathfrak{L}(\pmb{x},\pmb{\lambda})$。 注解。在定义 7.1 的讨论中，我们使用了两个独立重要的概念（Boyd 和 Vandenberghe, 2004）。 最大最小不等式 首先是一个 **最大最小不等式**，它说对于任何有两个参数的函数 $\varphi(\pmb{x},\pmb{y})$，最大最小值小于等于最小最大值，即 $$ \operatorname*{max}_{\pmb{y}}\operatorname*{min}_{\pmb{x}}\varphi(\pmb{x},\pmb{y})\leqslant\operatorname*{min}_{\pmb{x}}\operatorname*{max}_{\pmb{y}}\varphi(\pmb{x},\pmb{y})\,. $$ 这个不等式可以通过考虑不等式 $$ \mathrm{For~all~}x,y\qquad\operatorname*{min}_{x}\varphi(x,y)\leqslant\operatorname*{max}_{y}\varphi(x,y)\,. $$ 来证明。注意到在 (7.24) 的左边取对 $\pmb{y}$ 的最大值保持了不等式，因为对于所有的 $\pmb{y}$，这个不等式成立。同样，我们可以在 (7.24) 的右边对 $\pmb{x}$ 取最小值来获得 (7.23)。 第二个概念是 **弱对偶性**，它使用 (7.23) 表示原始值总是大于或等于对偶值。这些可以在 (7.27) 中更详细地描述。$\diamondsuit$ 回想一下，在 (7.18) 中 $J(\pmb{x})$ 与拉格朗日函数 (7.20b) 的区别在于我们放宽了指示函数到线性函数。因此，当 $\lambda\geqslant0$ 时，拉格朗日函数 $\mathfrak{L}(\pmb{x},\pmb{\lambda})$ 是 $J(\pmb{x})$ 的下界。因此， minmax 与 $\lambda$ 关于拉格朗日函数 $\mathfrak{L}(\pmb{x},\pmb{\lambda})$ 的最大值 $$ J({\pmb x})=\operatorname*{max}_{{\pmb\lambda}\geqslant{\bf0}}{\mathfrak{L}}({\pmb x},{\pmb\lambda}) $$ 通对最优化，原始问题是最小化 $J(\pmb{x})$，即 $$ \operatorname*{min}_{\pmb{x}\in\mathbb{R}^{d}}\operatorname*{max}_{\pmb{\lambda}\geqslant\mathbf{0}}\mathfrak{L}(\pmb{x},\pmb{\lambda}) $$ 通过最优化不等式 (7.23)，可以得出交换最优化和最大化操作仍然可以获得较小的值，即 $$ \operatorname*{min}_{\pmb{x}\in\mathbb{R}^{d}}\operatorname*{max}_{\pmb{\lambda}\geqslant\mathbf{0}}\mathfrak{L}(\pmb{x},\pmb{\lambda})\geqslant\operatorname*{max}_{\pmb{\lambda}\geqslant\mathbf{0}}\operatorname*{min}_{\pmb{x}\in\mathbb{R}^{d}}\mathfrak{L}(\pmb{x},\pmb{\lambda}) $$ 这被称为 **弱对偶性**。注意到右侧内层是 **对偶目标函数** $\mathfrak{D}(\lambda)$，定义也遵循上述公式。 与具有约束条件的原始优化问题不同，对于给定的 $\lambda$，$\mathrm{min}_{{\pmb x}\in\mathbb{R}^{d}}\,\mathfrak{L}({\pmb x},{\pmb\lambda})$ 是一个无约束优化问题。如果 $\mathrm{min}_{{\pmb x}\in\mathbb{R}^{d}}\,\mathfrak{L}({\pmb x},{\pmb\lambda})$ 的优化很容易，那么整体问题也易于解决。我们可以从 (7.20b) 看到，$\mathfrak{L}({\pmb x},{\pmb\lambda})$ 对于 $\lambda$ 是线性的。因此，$\mathrm{min}_{{\pmb x}\in\mathbb{R}^{d}}\,\mathfrak{L}({\pmb x},{\pmb\lambda})$ 是关于 $\lambda$ 的仿射函数的点最小值，因此即使 $f(\cdot)$ 和 $g_{i}(\cdot)$ 可能是非凸的，$\mathfrak{D}(\lambda)$ 也仍然是凹的。外部问题是最大化一个凹函数，可以高效地计算。 假设 $f(\cdot)$ 和 $g_{i}(\cdot)$ 可微，我们可以通过对拉格朗日函数关于 $\pmb{x}$ 求导，设置导数为零，解出最优值来找到拉格朗日对偶问题。我们在第 7.3.1 节和第 7.3.2 节中讨论了两个具体的例子，其中 $f(\cdot)$ 和 $g_{i}(\cdot)$ 是凸的。 注解 (等式约束)。考虑 (7.17) 有额外的等式约束 $$ \begin{array}{r l} {\underset{\pmb{x}}{\mathrm{min}}}&{f(\pmb{x})}\\ {\mathrm{subject\ to}}&{g_{i}(\pmb{x})\leqslant0\quad\mathrm{for\,all}\quad i=1,...,m}\\ &{h_{j}(\pmb{x})=0\quad\mathrm{for\,all}\quad j=1,...,n\,.} \end{array} $$ 可以通过将等式约束替换为两个不等式约束来建模等式约束，即每个等式约束 $h_{j}({\pmb x})=0$ 相当于替换为两个约束 $h_{j}({\pmb x})\leqslant0$ 和 $h_{j}({\pmb x})\geqslant0$。结果，相应的拉格朗日乘数将为无约束。因此，在 (7.28) 中，对不等式约束的拉格朗日乘数进行非负约束，而对等式约束的拉格朗日乘数保持无约束。

## 7.3 凸优化

凸优化问题 强对偶性 凸集 图 7.5 凸集的例子。 ![](images/23bdf000f56ef813848444f77360b6f15585b9d8966107f713361a4d3aff549c.jpg) 图 7.6 非凸集的例子。 ![](images/6f366a412734d4ccf9ecdc02b571b3689a2b4ccfb379d9b909d47557e5bea6aa.jpg) 凸函数 凹函数 上图 我们关注一类特别有用的优化问题，其中可以保证全局最优化。当 $f(\cdot)$ 为凸函数，且约束条件涉及 $g(\cdot)$ 和 $h(\cdot)$ 都为凸集时，称为 **凸优化问题**。在这个设定中，我们有 **强对偶性**：对偶问题的最优解与原始问题的最优解相同。在机器学习文献中，凸函数和凸集之间的区别通常没有严格呈现，但可以从上下文中推断出隐含的意义。 定义 一个集合 $\mathcal{C}$ 是 **凸集**，如果对于任意 $x,y \in \mathcal{C}$ 和任意标量 $\theta$ 使得 $0 \leqslant \theta \leqslant 1$ 时，我们有 $$ \theta x + (1 - \theta)y \in \mathcal{C}\,. $$ 凸集是这样的集合：连接集合中任意两点的直线都在集合内。图 7.5 和图 7.6 分别展示了凸集和非凸集的例子。 凸函数是这样的函数，其任意两点连线都位于该函数上方。图 7.2 展示了一个非凸函数，而图 7.3 展示了一个凸函数。图 7.7 中也有展示了一个凸函数。 定义 7.3. 让函数 $f: \mathbb{R}^{D} \rightarrow \mathbb{R}$ 的定义域是一个凸集。如果对于 $f$ 的定义域内的任意 $\pmb{x}, \pmb{y}$ 和任意标量 $\theta$ 使得 $0 \leqslant \theta \leqslant 1$，我们有 $$ f(\theta \pmb{x} + (1 - \theta) \pmb{y}) \leqslant \theta f(\pmb{x}) + (1 - \theta) f(\pmb{y})\,. $$ 注解。**凹函数**是凸函数的负函数。 在 (7.28) 中，涉及 $g(\cdot)$ 和 $h(\cdot)$ 的约束条件裁剪了函数，使其位于某个标量值处，从而形成了集合。另一个关于凸函数和凸集的联系是考虑“填充”一个凸函数所得到的集合。凸函数类似于一个碗状物体，我们可以设想将水倒进它的形状来填充它。这样形成的填充集合称为凸函数的 **上图**，这是一个凸集。 如果一个函数 $f: \mathbb{R}^{n} \to \mathbb{R}$ 在其定义域内是可微的，我们可以使用其梯度 $\nabla_{\pmb{x}}f(\pmb{x})$ （第 5 章第二节）来描述凸性。如果一个函数 $f(\pmb{x})$ 是凸的，则对于任意两点 $\pmb{x}, \pmb{y}$ 有 $$ f(\pmb{y}) \geqslant f(\pmb{x}) + \nabla_{\pmb{x}}f(\pmb{x})^{\top}(\pmb{y} - \pmb{x})\,. $$ 如果进一步知道函数 $f(\pmb{x})$ 是二阶可微的，即在 $f(\pmb{x})$ 的定义域内存在黑森矩阵（5.147），那么 $f(\pmb{x})$ 是凸的当且仅当 $\nabla_{\pmb{x}}^{2}f(\pmb{x})$ 半正定（Boyd 和 Vandenberghe, 2004）。
### 示例 7.3 

负熵 $f(x) = x \log_{2}x$ 在 $x > 0$ 时是凸函数。函数的可视化如图 7.8 所示，可以看出函数是凸的。为了说明前面关于凸性的定义，我们可以通过检查两点 $x = 2$ 和 $x = 4$ 的计算来展示。要证明 $f(x)$ 的凸性，我们实际上需要验证所有 $x \in \mathbb{R}$。 根据定义 7.3，考虑两点之间的中点（即 $\theta = 0.5$），左侧为 $f(0.5 \cdot 2 + 0.5 \cdot 4) = f(3) = 3 \log_{2}3 \approx 4.75$。右侧为 $0.5(2 \log_{2}2) + 0.5(4 \log_{2}4) = 1 + 4 = 5$。因此，满足定义。 由于 $f(x)$ 是可微的，我们可以用(7.31)的条件进行验证。计算 $f(x)$ 的导数，我们得到 $$ \nabla_{x}(x \log_{2}x) = 1 \cdot \log_{2}x + x \cdot \frac{1}{x \log_{e}2} = \log_{2}x + \frac{1}{\log_{e}2}. $$ 使用相同的两个测试点 $x = 2$ 和 $x = 4$，左侧为 $f(4) = 8$。右侧为 $$ \begin{array}{l} f(x) + \nabla_{x}^{\top}(y - x) = f(2) + \nabla f(2) \cdot (4 - 2) \\ \qquad\qquad\qquad = 2 + (1 + \frac{1}{\log_{e}2}) \cdot 2 \approx 6.9. \end{array} $$ ![](images/03290f0d6f10bf556a6ccc02cec0e610afcaa6a7e6d2c79fbe67244824161ed7.jpg) 图 7.8 负熵函数（是一个凸函数）及其在 $x=2$ 处的切线。 从第一原理出发，我们可以通过回顾定义来检查一个函数或集合是否是凸的。实际上，我们通常依靠保持凸性的操作来检查特定的函数或集合是否是凸的。尽管细节有所不同，这是我们在第 2 章中引入的闭包概念的又一次应用。 
### 示例 7.4 

非负加权凸函数的和是凸函数。观察到如果 $f$ 是一个凸函数，且 $\alpha \geq 0$ 是一个非负标量，那么函数 $\alpha f$ 是凸的。可以通过对定义 7.3 中的方程式两边同时乘以 $\alpha$，并结合非负数乘法不会改变不等式这一事实来验证这一点。 如果 $f_{1}$ 和 $f_{2}$ 是凸函数，根据定义我们有 $$ \begin{array}{r l} & f_{1}(\theta \pmb{x} + (1-\theta) \pmb{y}) \leq \theta f_{1}(\pmb{x}) + (1-\theta) f_{1}(\pmb{y}) \\ & f_{2}(\theta \pmb{x} + (1-\theta) \pmb{y}) \leq \theta f_{2}(\pmb{x}) + (1-\theta) f_{2}(\pmb{y}). \end{array} $$ 两边相加得到 $$ \begin{array}{r l} & f_{1}(\theta \pmb{x} + (1-\theta) \pmb{y}) + f_{2}(\theta \pmb{x} + (1-\theta) \pmb{y}) \\ & \leq \theta f_{1}(\pmb{x}) + (1-\theta) f_{1}(\pmb{y}) + \theta f_{2}(\pmb{x}) + (1-\theta) f_{2}(\pmb{y}), \end{array} $$ 右边可以重排为 $$ \theta (f_{1}(\pmb{x}) + f_{2}(\pmb{x})) + (1-\theta) (f_{1}(\pmb{y}) + f_{2}(\pmb{y})). $$ 完成证明，即凸函数的和是凸函数。 结合前面的事实，我们看到 $\alpha f_{1}(\pmb{x}) + \beta f_{2}(\pmb{x})$ 对于 $\alpha, \beta \geq 0$ 是凸的。这个封闭性质可以通过类似的方法扩展到更多个凸函数的非负加权和。 注解。不等式 (7.30) 有时被称为 **Jensen 不等式**。Jensen 不等式实际上是一类与凸函数的非负加权和相关的不等式的统称。$\diamondsuit$ 总结来说，一个具有约束条件的优化问题被称为 **凸优化问题**，如果问题 $$ \begin{array}{r l r} {\underset{\pmb{x}}{\mathrm{min}} f(\pmb{x})} & {} & \\ {\mathrm{subject~to}~g_{i}(\pmb{x}) \leq 0} & \mathrm{for~all} & {i=1,.\,.\,,m} \\ {h_{j}(\pmb{x}) = 0} & \mathrm{for~all} & {j=1,.\,.\,,n,} \end{array} $$ 其中所有函数 $f(\pmb{x})$ 和 $g_{i}(\pmb{x})$ 是凸函数，且所有 $h_{j}(\pmb{x}) = 0$ 是凸集。在接下来的部分，我们将描述两个广泛使用且理解良好的凸优化问题类别。

### 7.3.1 线性规划

考虑所有前面的函数都是线性的情况，即 $$ \begin{array}{r l} {\operatorname*{min}_{\pmb{x}\in\mathbb{R}^{d}}}&{\pmb{c}^{\top}\pmb{x}}\\ {{\mathrm{subject~to}}}&{\pmb{A}\pmb{x}\leqslant\pmb{b}\,,} \end{array} $$ 其中 $\pmb{A}\in\mathbb{R}^{m\times d}$ 和 $\pmb{b}\in\mathbb{R}^{m}$。这被称为 **线性规划**。它有 $d$ 个变量和 $m$ 个线性约束。拉格朗日函数为 $$ \mathfrak{L}(\pmb{x},\pmb{\lambda}) = \pmb{c}^{\top}\pmb{x} + \pmb{\lambda}^{\top}(\pmb{A}\pmb{x} - \pmb{b})\,, $$ 其中 $\pmb{\lambda}\in\mathbb{R}^{m}$ 是拉格朗日乘数向量。重新排列对应于 $\pmb{x}$ 的项后，我们得到 $$ \mathfrak{L}(\pmb{x},\pmb{\lambda}) = (\pmb{c} + \pmb{A}^{\top}\pmb{\lambda})^{\top}\pmb{x} - \pmb{\lambda}^{\top}\pmb{b}\,. $$ 对 $\mathfrak{L}(\pmb{x},\pmb{\lambda})$ 关于 $\pmb{x}$ 求导并令其等于零，我们得到 $$ \pmb{c} + \pmb{A}^{\top}\pmb{\lambda} = \pmb{0}\,. $$ 因此，对偶拉格朗日函数为 $\mathfrak{D}(\pmb{\lambda}) = -\pmb{\lambda}^{\top}\pmb{b}$。我们希望最大化 $\mathfrak{D}(\pmb{\lambda})$。除了导数为零的约束外，我们还必须满足 $\pmb{\lambda} \geqslant \pmb{0}$ 的约束，从而得到以下对偶优化问题 $$ \begin{array}{r l} {\underset{\pmb{\lambda}\in\mathbb{R}^{m}}{\mathrm{max}}}&{\,-\pmb{b}^{\top}\pmb{\lambda}}\\ {\mathrm{subject\ to}}&{\,\pmb{c} + \pmb{A}^{\top}\pmb{\lambda} = \pmb{0}}\\ &{\,\pmb{\lambda} \geqslant \pmb{0}\,.} \end{array} $$ 通常的做法是将原始问题最小化，并将对偶问题最大化。这是一个具有 $m$ 个变量的线性规划问题。我们可以根据 $m$ 和 $d$ 的大小选择求解原始问题（7.39）或对偶问题（7.43）。 
### 示例 7.5（线性规划） 

考虑线性规划问题 $$ \begin{array}{r l} {\underset{\pmb{x}\in\mathbb{R}^{2}}{\mathrm{min}}}&{-\left[\frac{5}{3}\right]^{\top}\left[\begin{array}{l}x_{1}\\ x_{2}\end{array}\right]}\\ {\mathrm{subject~to~}}&{\left[\begin{array}{ll}2 & 2\\ 2 & -4\\ -2 & 1\\ 0 & -1\end{array}\right]\left[\begin{array}{l}x_{1}\\ x_{2}\end{array}\right] \leqslant \left[\begin{array}{l}33\\ 8\\ 5\\ -1\\ 8\end{array}\right]} \end{array} $$ 有两个变量。这个程序如图 7.9 所示。目标函数是线性的，因此等高线是线性的。标准形式的约束集合转化为图例。最优值必须位于阴影（可行）区域内，如星号所示。 ![](images/1d61c37d65fc0510e87a3280b2768992b894ab73da47e9a58ba94b3267055176.jpg) 图 7.9 线性规划的示例图。未加约束的问题（用等高线表示）的最小值在右边。给定约束条件的最佳值由星号表示。 # 7.3.2 二次规划 考虑目标函数是凸二次函数的情况，其中约束为仿射的，即 $$ \begin{array}{r l} {\underset{\pmb{x}\in\mathbb{R}^{d}}{\mathrm{min}}}&{\frac{1}{2}\pmb{x}^{\top}\pmb{Q}\pmb{x} + \pmb{c}^{\top}\pmb{x}}\\ {\mathrm{subject~to}}&{\pmb{A}\pmb{x} \leqslant \pmb{b}\,,} \end{array} $$ 其中 $\pmb{A}\in\mathbb{R}^{m\times d}$，$\pmb{b}\in\mathbb{R}^{m}$，$\pmb{c}\in\mathbb{R}^{d}$。对称正定矩阵 $\pmb{Q}\in\mathbb{R}^{d\times d}$ 使得目标函数是凸的。这被称为 **二次规划**。注意它有 $d$ 个变量和 $m$ 个线性约束。

### 7.3.2 二次规划
考虑目标函数为凸二次函数且约束为仿射形的一般二次规划问题，即 $$ \begin{array}{r l} {\underset{\pmb{x}\in\mathbb{R}^{2}}{\mathrm{min}}}&{\frac{1}{2}\left[\begin{array}{l}x_{1}\\ x_{2}\end{array}\right]^{\top}\left[\begin{array}{ll}2 & 1\\ 1 & 4\end{array}\right]\left[\begin{array}{l}x_{1}\\ x_{2}\end{array}\right] + \left[\begin{array}{l}5\\ 3\end{array}\right]^{\top}\left[\begin{array}{l}x_{1}\\ x_{2}\end{array}\right]}\\ {{\mathrm{subject~to}}}&{\left[\begin{array}{ll}1 & 0\\ -1 & 0\\ 0 & 1\\ 0 & -1\end{array}\right]\left[\begin{array}{l}x_{1}\\ x_{2}\end{array}\right] \leqslant \left[\begin{array}{l}1\\ 1\\ 1\\ 1\end{array}\right]} \end{array} $$ 均为两个变量的问题。这个程序也在图 7.4 中进行了图示。目标函数是凸二次函数，矩阵 $Q$ 为半正定矩阵，因此等高线是椭圆形。最优值必须位于阴影（可行）区域内，如图中星号所示。 拉格朗日函数为 $$ \begin{array}{l} \mathfrak{L}(\pmb{x},\pmb{\lambda}) = \frac{1}{2}\pmb{x}^{\top}\pmb{Q}\pmb{x} + \pmb{c}^{\top}\pmb{x} + \pmb{\lambda}^{\top}(\pmb{A}\pmb{x} - \pmb{b})\\ \qquad\qquad = \frac{1}{2}\pmb{x}^{\top}\pmb{Q}\pmb{x} + (\pmb{c} + \pmb{A}^{\top}\pmb{\lambda})^{\top}\pmb{x} - \pmb{\lambda}^{\top}\pmb{b} \end{array} $$ 再次重新排列了项。对 $\mathfrak{L}(\pmb{x},\pmb{\lambda})$ 关于 $\pmb{x}$ 求导并令其等于零，我们得到 $$ \pmb{Qx} + (\pmb{c} + \pmb{A}^{\top}\pmb{\lambda}) = \pmb{0} $$ 由于 $Q$ 是正定的，因此可逆，我们获得 $$ \pmb{x} = -\pmb{Q}^{-1}(\pmb{c} + \pmb{A}^{\top}\pmb{\lambda}) $$ 将式 (7.50) 代入原始拉格朗日函数 $\mathfrak{L}(\pmb{x},\pmb{\lambda})$，得到对偶拉格朗日函数 $$ \mathfrak{D}(\pmb{\lambda}) = -\frac{1}{2}(\pmb{c} + \pmb{A}^{\top}\pmb{\lambda})^{\top}\pmb{Q}^{-1}(\pmb{c} + \pmb{A}^{\top}\pmb{\lambda}) - \pmb{\lambda}^{\top}\pmb{b} $$ 因此，对偶优化问题为 $$ \begin{array}{r l} {\underset{\pmb{\lambda} \in \mathbb{R}^{m}}{\mathrm{max}}}&{ -\frac{1}{2}(\pmb{c} + \pmb{A}^{\top}\pmb{\lambda})^{\top}\pmb{Q}^{-1}(\pmb{c} + \pmb{A}^{\top}\pmb{\lambda}) - \pmb{\lambda}^{\top}\pmb{b}}\\ {\mathrm{subject\ to}}&{\pmb{\lambda} \geqslant \pmb{0}} \end{array} $$ 二次规划将在第 12 章中应用于机器学习中的一个应用。 
### 7.3.3 Legendre–Fenchel 转换和凸共轭 

支持超平面 Legendre 转换 我们回顾第 7.2 节中的对偶性概念，不考虑约束条件时，一个凸集可以等效地描述为其支撑超平面。一个超平面称为一个凸集的支撑超平面，如果它与凸集相交，并且凸集仅包含在其一侧。回想一下，我们可以通过填充一个凸函数来获得其上图，这是一个凸集。因此，我们可以用其支撑超平面来描述凸函数。此外，观察到支撑超平面恰好接触凸函数，并且实际上就是在该点的切线。并记得一个函数 $f(x)$ 在给定点 $x_0$ 的切线是该函数在该点处的梯度的评估 $\left.\frac{\mathrm{d} f(x)}{\mathrm{d} x}\right|_{x=x_0}$。简而言之，因为凸集可以等效地用支撑超平面来描述，因此凸函数可以等效地用其梯度的函数来描述。Legendre 转换正式化了这一概念。 我们从最为一般的定义入手，其中定义看起来可能反直观，然后通过特殊情形来理解这一定义与先前阐述的直观概念之间的联系。Legendre–Fenchel 转换是一种从凸且可微函数 $f(\pmb{x})$ 到基于切线 $\pmb{s}(\pmb{x}) = \nabla_{\pmb{x}}f(\pmb{x})$ 的函数的变换（类似于傅里叶变换）。重要的是要强调这是对函数 $f(\cdot)$ 的变换，并不是变量 $\pmb{x}$ 或函数的评估的变换。Legendre–Fenchel 转换也被称为 **凸共轭**，并密切关系到对偶性（Hiriart-Urruty 和 Lemaréchal, 2001, 第 5 章）。 
### 定义 7.4. 

函数 $f:\mathbb{R}^{D} \rightarrow \mathbb{R}$ 的 **凸共轭** 是一个由 $$ f^{*}(s) = \operatorname*{sup}_{\pmb{x} \in \mathbb{R}^{D}} (\langle \pmb{s}, \pmb{x} \rangle - f(\pmb{x}))\, $$ 定义的函数。注意，前面的凸共轭定义并不需要函数 $f$ 是凸的或可微的。在定义 7.4 中我们使用了广义内积（第 3 章第二节），但在本节中我们将考虑有限维向量的标准点积 $(\langle \pmb{s}, \pmb{x} \rangle = \pmb{s}^{\top} \pmb{x})$ 以避免过多的技术细节。 为了从几何角度理解定义 7.4，考虑一个简单的凸且可微的一维函数，例如 $f(x) = x^2$。由于我们处理的是一个一维问题，超平面退化为直线。考虑一条直线 $y = s x + c$。我们可以通过其支撑超平面来描述凸函数，所以让我们试着用这条直线来描述函数 $f(x)$。对其中的每个点 $(x_0, f(x_0))$，固定直线的斜率 $s \in \mathbb{R}$，找到使得该直线通过点 $(x_0, f(x_0))$ 的 $c$ 的最小值。注意到 $c$ 的最小值是斜率为 $s$ 的直线恰好接触函数 $f(x) = x^2$ 的地方的截距。通过 $(x_0, f(x_0))$ 且斜率为 $s$ 的直线可以表示为 $$ y - f(x_0) = s(x - x_0) $$ 该直线的 $y$ 截距为 $-s x_0 + f(x_0)$。使得 $y = s x + c$ 与函数 $f$ 的图相交的 $c$ 的最小值为 $$ \operatorname*{inf}_{x_0} \; -s x_0 + f(x_0) $$ 前述的凸共轭定义默认为这个值的负值。这一推理过程在选择的是一维凸且可微的函数时并不依赖，对所有 $\pmb{f} : \mathbb{R}^{D} \rightarrow \mathbb{R}$ 都适用，这些函数可能是非凸且不可微的。 注记：对于比如 $f(x) = x^2$ 这样的凸且可微函数，其凸共轭为一个特殊案例，其中不存在上确界，并且函数与其 Legendre 转换之间存在一一对应关系。我们可以从理论出发推导这一结果。对于一个凸且可微的函数，我们知道在 $x_0$ 处切线与 $f(x_0)$ 相切 $$ f(x_0) = s x_0 + c $$ 我们需要用梯度 $\nabla_{\pmb{x}} f(\pmb{x}_0)$ 来描述凸函数，其中 $s = \nabla_{\pmb{x}} f(\pmb{x}_0)$。按此可重排得 $-c$ 的表达式 $$ -c = s x_0 - f(x_0) $$ 注意到 $-c$ 随 $x_0$ （即 $s$）改变，因此将其视为 $s$ 的函数 $$ f^*(s) := s x_0 - f(x_0) $$ 将 (7.58) 与定义 7.4 对比，我们可以看到 (7.58) 是定义 7.4 的特例（去掉上确界）$\diamondsuit$ 对偶函数有一些有用的性质；特别是，对于凸函数，再次应用 Legendre 转换会得到原来的函数。与函数的梯度 $\nabla_{x} f(x)$ 相比，$f^*(s)$ 的梯度是 $x$。下面两个示例展示了凸共轭在机器学习中的常用应用。

### 7.3.4 凸共轭应用

为了说明凸共轭的应用，考虑基于正定矩阵 $\mathcal{K} \in \mathbb{R}^{n \times n}$ 的二次函数 $$ f(\pmb{y}) = \frac{\lambda}{2} \pmb{y}^{\top} \pmb{K}^{-1} \pmb{y} $$ 设原始变量为 $\pmb{y} \in \mathbb{R}^{n}$，对偶变量为 $\pmb{\alpha} \in \mathbb{R}^{n}$。 应用定义 7.4，我们得到函数 $$ f^{*}(\pmb{\alpha}) = \operatorname*{sup}_{\pmb{y} \in \mathbb{R}^{n}} \langle \pmb{y}, \pmb{\alpha} \rangle - \frac{\lambda}{2} \pmb{y}^{\top} \pmb{K}^{-1} \pmb{y} $$ 由于函数是可微的，我们可以通过取导数找到最大的值， $$ \frac{\partial \left[ \langle \pmb{y}, \pmb{\alpha} \rangle - \frac{\lambda}{2} \pmb{y}^{\top} \pmb{K}^{-1} \pmb{y} \right]}{\partial \pmb{y}} = (\pmb{\alpha} - \lambda \pmb{K}^{-1} \pmb{y})^{\top} $$ 因此当梯度为零时，我们有 $\pmb{y} = \frac{1}{\lambda} \pmb{K} \pmb{\alpha}$。代入上述表达式可得 $$ f^{*}(\pmb{\alpha}) = \frac{1}{\lambda} \pmb{\alpha}^{\top} \pmb{K} \pmb{\alpha} - \frac{\lambda}{2} \left( \frac{1}{\lambda} \pmb{K} \pmb{\alpha} \right)^{\top} \pmb{K}^{-1} \left( \frac{1}{\lambda} \pmb{K} \pmb{\alpha} \right) = \frac{1}{2\lambda} \pmb{\alpha}^{\top} \pmb{K} \pmb{\alpha} $$
### 示例 7.8
在机器学习中，我们经常使用函数的和；例如，训练集的目标函数包括每个训练样本损失之和。接下来，我们推导损失函数 $\ell(t)$ 的凸共轭，其中 $\ell: \mathbb{R} \rightarrow \mathbb{R}$。这还展示了如何将凸共轭应用到向量情形。设 $$ \mathcal{L}(\pmb{t}) = \sum_{i=1}^{n} \ell_i(t_i) $$ 则有 $$ \begin{aligned} \mathcal{L}^*(\pmb{z}) &= \operatorname*{sup}_{\pmb{t} \in \mathbb{R}^n} \langle \pmb{z}, \pmb{t} \rangle - \sum_{i=1}^{n} \ell_i(t_i) \\ &= \operatorname*{sup}_{\pmb{t} \in \mathbb{R}^n} \sum_{i=1}^{n} z_i t_i - \ell_i(t_i) \\ &= \sum_{i=1}^{n} \operatorname*{sup}_{t_i \in \mathbb{R}} z_i t_i - \ell_i(t_i) \end{aligned} $$ $$ = \sum_{i=1}^{n} \ell_i^*(z_i) $$定义共轭 回忆第 7.2 节中我们使用拉格朗日乘数推导了对偶优化问题。此外，对于凸优化问题，我们有 **强对偶性**，即原始问题和对偶问题的解相同。这里的 Legendre-Fenchel 转换也能够用于推导对偶优化问题。特别是，当函数是凸且可微的时，上确界是唯一的。为了进一步探讨两种方法的关系，考虑一个线性约束下的凸优化问题。 
### 示例 7.9 

设 $f(\pmb{y})$ 和 $g(\pmb{x})$ 是凸函数，$\pmb{A}$ 是一个适当维度的实矩阵，满足 $\pmb{A x} = \pmb{y}$。则 $$ \operatorname*{min}_{\pmb{x}} f(\pmb{A} \pmb{x}) + g(\pmb{x}) = \operatorname*{min}_{\pmb{A x} = \pmb{y}} f(\pmb{y}) + g(\pmb{x}) $$ 引入拉格朗日乘数 $\pmb{u}$ 对约束 $\pmb{A x} = \pmb{y}$，我们有 $$ \begin{array}{r l l} &{}& \operatorname*{min}_{\pmb{A x} = \pmb{y}} f(\pmb{y}) + g(\pmb{x}) = \operatorname*{min}_{\pmb{x}, \pmb{y}} \operatorname*{max}_{\pmb{u}} f(\pmb{y}) + g(\pmb{x}) + (A \pmb{x} - \pmb{y})^{\top} \pmb{u} \\ &{}& = \operatorname*{max}_{\pmb{u}} \operatorname*{min}_{\pmb{x}, \pmb{y}} f(\pmb{y}) + g(\pmb{x}) + (A \pmb{x} - \pmb{y})^{\top} \pmb{u} \end{array} $$ 其中，通过函数与 $f(\pmb{y})$ 和 $g(\pmb{x})$ 是凸函数的事实，我们交换最大和最小。利用点积对称性 $$ \begin{array}{r l} & \quad \underset{\pmb{u}}{\operatorname*{max}} \left[ \underset{\pmb{x}, \pmb{y}}{\operatorname*{min}} \, f(\pmb{y}) + g(\pmb{x}) + (A \pmb{x} - \pmb{y})^{\top} \pmb{u} \right] \\ & = \underset{\pmb{u}}{\operatorname*{max}} \left[ \underset{\pmb{y}}{\operatorname*{min}} - \pmb{y}^{\top} \pmb{u} + f(\pmb{y}) \right] + \left[ \underset{\pmb{x}}{\operatorname*{min}} (\pmb{A} \pmb{x})^{\top} \pmb{u} + g(\pmb{x}) \right] \\ & = \underset{\pmb{u}}{\operatorname*{max}} \left[ \underset{\pmb{y}}{\operatorname*{min}} - \pmb{y}^{\top} \pmb{u} + f(\pmb{y}) \right] + \left[ \underset{\pmb{x}}{\operatorname*{min}} \pmb{x}^{\top} \pmb{A}^{\top} \pmb{u} + g(\pmb{x}) \right] \end{array} $$ 因此我们得到 $$ \underset{\pmb{u}}{\operatorname*{max}} \left[ \underset{\pmb{y}}{\operatorname*{min}} - \pmb{y}^{\top} \pmb{u} + f(\pmb{y}) \right] + \left[ \underset{\pmb{x}}{\operatorname*{min}} \pmb{x}^{\top} \pmb{A}^{\top} \pmb{u} + g(\pmb{x}) \right] = \underset{\pmb{u}}{\operatorname*{max}} - f^{*}(\pmb{u}) - g^{*}(-\pmb{A}^{\top} \pmb{u}) $$ 对于一般的内积，$A^{\top}$ 替换为共轭算子 $A^{*}$。 因此我们证明了 $$ \operatorname*{min}_{\pmb{x}} f(\pmb{A} \pmb{x}) + g(\pmb{x}) = \operatorname*{max}_{\pmb{u}} - f^{*}(\pmb{u}) - g^{*}(-\pmb{A}^{\top} \pmb{u}) $$ Legendre-Fenchel共轭对于可以表示为凸优化问题的机器学习问题非常有用，特别是对独立应用于每个样本的凸损失函数，共轭损失提供了一个方便的方式来推导对偶问题。

## 7.4 进一步阅读

连续优化是一个活跃的研究领域，我们并不试图提供最近进展的全面总结。 从梯度下降的角度来看，有两个主要的薄弱环节，每一个都有其各自的文献。第一个挑战是梯度下降是一种一阶算法，不使用有关表面曲率的信息。当存在很长的山谷时，梯度指向的往往是不感兴趣的维度。动量的概念可以推广为一类加速方法（Nesterov, 2018）。共轭梯度方法通过考虑先前的方向来避免梯度下降所面临的问题（Shewchuk, 1994）。二阶方法如牛顿方法使用黑塞矩阵来提供有关曲率的信息。阶梯选择和动量等想法多来源于考虑目标函数的曲率（Goh, 2017; Bottou et al., 2018）。拟牛顿方法如 L-BFGS 试图使用更廉价的计算方法来逼近黑塞矩阵（Nocedal and Wright, 2006）。最近，计算下降方向的其他度量引起了兴趣，产生了如镜像梯度（Beck and Teboulle, 2003）和自然梯度（Toussaint, 2012）等方法。 胡戈·贡卡尔维斯的博客也是一个介绍 Legendre–Fenchel 转换的好资源：https://tinyurl.com/ydaal7hj 第二个挑战是如何处理非可微函数。当函数中有突变时，梯度方法不适用。在这种情况下，可以使用子梯度方法（Shor, 1985）。对于进一步的信息和优化非可微函数的算法，我们推荐伯特塞卡斯（Bertsekas, 1999）的书。在数值求解连续优化问题方面，包括约束优化问题在内的许多方法的研究都有大量的文献。了解这段文献的良好起点是卢内伯格（Luenberger, 1969）和邦纳斯等人（Bonnans et al., 2006）的书籍。Bubeck（2015）提供了一个关于连续优化的最新综述。 现代机器学习的应用往往意味着数据集的大小使批量梯度下降变得不可行，因此随机梯度下降是大规模机器学习方法的首选工具。相关文献的最新综述包括 Hazan (2015) 和 Bottou et al. (2018)。 对于对偶性和凸优化，博伊德和范登伯格（Boyd and Vandenberghe, 2004）提供了在线的讲义和幻灯片。伯特塞卡斯（Bertsekas, 2009）提供了更为数学化的处理方法，而优化领域的关键研究人员之一写的最近的书籍是 Nesterov (2018)。凸优化的基础是凸分析，对感兴趣于此领域的读者，我们推荐 Rochafellar (1970)，Hiriart-Urruty 和 Lemaréchal (2001)，以及 Borwein 和 Lewis (2006) 的著作。凸分析相关的书籍中也涵盖了 Legendre–Fenchel 转换，而 Zia 等人 (2009) 提供了更易于初学者理解的介绍。关于 Legendre–Fenchel 转换在凸优化算法分析中的角色，Polyak (2016) 提供了一个综述。

## 练习

7.1 考虑一元函数

$$
f(x)=x^{3}+6x^{2}-3x-5.
$$

求其驻点并指出它们是极大值点、极小值点还是鞍点。

7.2 考虑随机梯度下降的更新方程（方程 (7.15)）。写出当使用批量大小为 1时的更新方程。

7.3 考虑以下陈述是否正确：

a. 任意两个凸集的交集仍然是凸集。
b. 任意两个凸集的并集仍然是凸集。
c. 凸集 $A$ 从另一个凸集 $B$ 中减去的结果是凸集。

7.4 考虑以下陈述是否正确：

a. 任意两个凸函数之和仍然是凸函数。
b. 任意两个凸函数之差仍然是凸函数。
c. 任意两个凸函数之积仍然是凸函数。
d. 任意两个凸函数的最大值仍然是凸函数。

7.5 将以下优化问题表示为矩阵形式的标准线性规划问题

$$
\operatorname*{max}_{\pmb{x}\in\mathbb{R}^{2},\;\xi\in\mathbb{R}}\pmb{p}^{\top}\pmb{x}+\xi
$$

并满足约束条件 $\xi\geqslant0$，$x_{0}\leqslant0$ 和 $x_{1}\leqslant3$。

7.6 考虑图 7.9 所示的线性规划问题，

$$
\operatorname*{min}_{{\pmb{x}}\in\mathbb{R}^{2}}-\left[3\right]^{\top}\left[x_{1}\right]
$$

$$
{\mathrm{subject~to}}\;\;{\left[\begin{array}{l l}{2}&{2}\\ {2}&{-4}\\ {-2}&{1}\\ {0}&{-1}\\ {0}&{1}\end{array}\right]}\;{\left[\begin{array}{l}{x_{1}}\\ {x_{2}}\end{array}\right]}\leqslant{\left[\begin{array}{l}{33}\\ {8}\\ {5}\\ {-1}\\ {8}\end{array}\right]}
$$

利用拉格朗日对偶性推导其对偶线性规划问题。

7.7 考虑图 7.4 所示的二次规划问题，

$$
\begin{array}{r l}&{\underset{x\in\mathbb{R}^{2}}{\operatorname*{min}}\frac{1}{2}\left[\begin{array}{l}{x_{1}}\\ {x_{2}}\end{array}\right]^{\top}\left[\begin{array}{l l}{2}&{1}\\ {1}&{4}\end{array}\right]\left[\begin{array}{l}{x_{1}}\\ {x_{2}}\end{array}\right]+\left[\begin{array}{l}{5}\\ {3}\end{array}\right]^{\top}\left[\begin{array}{l}{x_{1}}\\ {x_{2}}\end{array}\right]}\\ &{\mathrm{subject~to}\,\,\left[\begin{array}{l l}{1}&{0}\\ {-1}&{0}\\ {0}&{1}\\ {0}&{-1}\end{array}\right]\left[\begin{array}{l}{x_{1}}\\ {x_{2}}\end{array}\right]\leqslant\left[\begin{array}{l}{1}\\ {1}\\ {1}\\ {1}\end{array}\right]}\end{array}
$$

利用拉格朗日对偶性推导其对偶二次规划问题。

7.8 考虑以下凸优化问题

$$
\begin{array}{r l}{\underset{\pmb{w}\in\mathbb{R}^{D}}{\mathrm{min}}}&{\frac{1}{2}\pmb{w^{\top}}\pmb{w}}\\ {\mathrm{subject~to~}}&{\pmb{w^{\top}}\pmb{x}\geqslant1\,.}\end{array}
$$

通过引入拉格朗日乘子 $\lambda$ 推导拉格朗日对偶问题。

7.9 考虑向量 $\pmb{x}\in\mathbb{R}^{D}$ 的负熵，

$$
f(\pmb{x})=\sum_{d=1}^{D}x_{d}\log x_{d}\,.
$$

假设标准内积，推导其共轭函数 $f^{*}(s)$。

提示：取适当函数的梯度并将其设为零。

7.10 考虑函数

$$
\boldsymbol{f}(\mathbf{\mathit{x}})=\frac{1}{2}\mathbf{\mathit{x}}^{\top}\mathbf{\mathit{A}}\mathbf{\mathit{x}}+\mathbf{\mathit{b}}^{\top}\mathbf{\mathit{x}}+\boldsymbol{c}\,,
$$

其中 $\pmb{A}$ 是严格正定的，这意味着它是可逆的。推导该函数的共轭函数 $f(\pmb{x})$。

提示：取适当函数的梯度并将其设为零。

7.11 边缘损失（支持向量机使用的损失函数）由下式给出

$$
L(\alpha)=\operatorname*{max}\{0,1-\alpha\}\,,
$$

如果我们希望应用诸如 L-BFGS 等梯度方法而不诉诸次梯度方法，我们需要平滑边缘损失中的折点。计算边缘损失的共轭函数 $L^{*}(\beta)$，其中 $\beta$ 是对偶变量。加入 $\ell_{2}$ 近端项，并计算所得函数的共轭

$$
L^{*}(\beta)+\frac{\gamma}{2}\beta^{2}\,,
$$

其中 $\gamma$ 是给定的超参数。