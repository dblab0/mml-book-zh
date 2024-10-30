# 第十一章 高斯混合模型的密度估计  

在之前的章节中，我们已经讨论了机器学习中的两个基本问题：回归（第9章）和降维（第10章）。在本章中，我们将探讨机器学习的第三个支柱：密度估计。在我们的旅程中，我们将介绍重要的概念，如期望最大化（EM）算法以及混合模型中密度估计的潜在变量视角。

当我们将机器学习应用于数据时，我们通常希望以某种方式表示数据。一种直接的方法是将数据点本身作为数据的表示；见图11.1为例。然而，如果数据集非常庞大，或者我们对数据的特征感兴趣，这种方法可能不太有用。在密度估计中，我们使用参数族中的密度来紧凑地表示数据，例如高斯分布或Beta分布。例如，我们可能在寻找数据集的均值和方差，以便使用高斯分布紧凑地表示数据。均值和方差可以通过我们在第8.3节中讨论的工具：最大似然估计或最大后验估计来找到。然后，我们可以使用这个高斯的均值和方差来表示数据背后的分布，即我们认为数据集是从该分布中抽样的一次典型实现。

![](images/589fe546652033b693213b8709c57baa063900627621cd4e3144a21c9e718d38.jpg)  

实际上，高斯分布（或类似地，我们迄今为止遇到的所有其他分布）具有有限的建模能力。例如，图11.1中数据的高斯近似是一个糟糕的近似。接下来，我们将探讨一种更具有表现力的分布族，我们可以使用它们来进行密度估计：混合模型。

混合模型可以通过凸组合$K$个简单的（基础）分布来描述分布$p(\pmb{x})$：

$$
\begin{array}{l}
p(\pmb{x})=\sum_{k=1}^{K}\pi_{k}p_{k}(\pmb{x}) \\
0\leqslant\pi_{k}\leqslant1\,,\quad\sum_{k=1}^{K}\pi_{k}=1\,,
\end{array}
$$

其中，$p_{k}$是基础分布族的成员，例如高斯分布、伯努利分布或伽马分布，$\pi_{k}$是混合权重。混合模型比相应的基础分布更具表现力，因为它们允许多模态数据表示，即它们可以描述具有多个“簇”的数据集，例如图11.1中的示例。

我们将专注于高斯混合模型（GMMs），其中基础分布是高斯分布。对于给定的数据集，我们旨在最大化模型参数的似然性来训练GMM。为此，我们将使用第5章、第6章和第7.2节的结果。然而，与我们之前讨论的其他应用（线性回归或PCA）不同，我们不会找到闭式最大似然解。相反，我们将得到一组依赖的联立方程，只能通过迭代求解。

## 11.1 高斯混合模型  

高斯混合模型 是一种密度模型，其中我们将有限的高斯分布数量 $K$ 个 $\mathcal{N}(\pmb{x}\,|\,\pmb{\mu}_{k},\,\pmb{\Sigma}_{k})$ 结合在一起，使得模型为  

$$
\begin{array}{l}
p(\pmb{x}\,|\,\pmb{\theta})=\sum_{k=1}^{K}\pi_{k}\mathcal{N}\big(\pmb{x}\,|\,\pmb{\mu}_{k},\,\pmb{\Sigma}_{k}\big) \\
0\leqslant\pi_{k}\leqslant1\,,\quad\sum_{k=1}^{K}\pi_{k}=1\,,
\end{array}
$$

其中我们定义 $\pmb\theta:=\{\pmb\mu_{k},\pmb\Sigma_{k},\pi_{k}:\ k=1,.\,.\,.\,,K\}$ 为模型的所有参数集合。这种高斯分布的凸组合比简单的高斯分布（从 (11.3) 中恢复，当 $K=1$ 时）提供了显著更多的灵活性来建模复杂的密度。图 11.2 给出了一个示例，显示了加权的  

![](images/ce5f254051f92ba52b6aecccf397f2baa71019247e86134f7dfa91f1481690fa.jpg)  
图 11.2 高斯混合模型。高斯混合分布（黑色）是由高斯分布的凸组合构成的，并且比任何一个单独的成分更具表达力。虚线表示加权的高斯成分。  

成分和混合密度，其表达式为  

$$
p(x\,|\,\theta)=0.5\mathcal{N}\big(x\,|\,-2,\,\frac{1}{2}\big)+0.2\mathcal{N}\big(x\,|\,1,\,2\big)+0.3\mathcal{N}\big(x\,|\,4,\,1\big)\,.
$$
## 11.2 参数学习通过最大似然法  

我们给定一个数据集 ${\mathbfit{X}}\ =\ \{{\pmb{x}}_{1},.\,.\,.\,,{\pmb{x}}_{N}\}$ ，其中 ${\pmb x}_{n}$ ， $n=$ q,\cdot\cdot\cdot,N,$ ，是从一个未知分布 $p(\pmb{x})$ 中独立同分布抽取的。我们的目标是通过一个具有 $K$ 个混合成分的高斯混合模型（GMM）来找到这个未知分布 $p(\pmb{x})$ 的一个良好近似/表示。GMM的参数包括 $K$ 个均值 $\pmb{\mu}_{k}$ ，协方差 $\pmb{\Sigma}_{k}$ ，以及混合权重 $\pi_{k}$ 。我们将所有这些自由参数汇总为 $\pmb{\theta}:=$ $\{\pi_{k},\pmb{\mu}_{k},\pmb{\Sigma}_{k}:\ k=1,.\,.\,.\,,K\}$ 。

### 示例 11.1 (初始设置)  

![](images/d1e6d396e742198db0f077289ad99449ebe53ecfe23a34417362a40b89f39439.jpg)  
图 11.3 初始设置：具有三个混合成分（虚线）和七个数据点（圆点）的 GMM（黑色）。  

在本章中，我们将有一个简单的运行示例，帮助我们说明和可视化重要的概念。  

我们考虑一个一维数据集 $\mathcal{X}\,=\,\{-3,-2.5,-1,0,2,4,5\}$ ，包含七个数据点，并希望找到一个具有 $K\,=\,3$ 个成分的 GMM，以建模数据的密度。我们初始化混合成分如下：

$$
\begin{array}{r l}
&{p_{1}(x)=\mathcal{N}\big(x\,|\,-4,\,1\big)}\\
&{p_{2}(x)=\mathcal{N}\big(x\,|\,0,\,0.2\big)}\\
&{p_{3}(x)=\mathcal{N}\big(x\,|\,8,\,3\big)}
\end{array}
$$

并赋予它们相等的权重 $\pi_{1}\,=\,\pi_{2}\,=\,\pi_{3}\,=\,{\textstyle{\frac{1}{3}}}$ 。相应的模型（以及数据点）如图 11.3 所示。  

接下来，我们将详细说明如何获得模型参数 $\pmb{\theta}$ 的最大似然估计 $\pmb{\theta}_{\mathrm{ML}}$ 。我们首先写出似然函数，即给定参数的训练数据的预测分布。我们利用独立同分布的假设，得到因子化的似然函数

$$
p(\mathcal{X}\,|\,\pmb\theta)=\prod_{n=1}^{N}p(\pmb x_{n}\,|\,\pmb\theta)\,,\quad p(\pmb x_{n}\,|\,\pmb\theta)=\sum_{k=1}^{K}\pi_{k}\mathcal{N}\big(\pmb x_{n}\,|\,\pmb\mu_{k},\,\pmb\Sigma_{k}\big)\,,
$$
其中每个单独的似然项 $p(\pmb{x}_{n}\mid\pmb{\theta})$ 是一个高斯混合密度。然后我们得到对数似然函数

$$
\log p(\mathcal{X}\,|\,\pmb{\theta})=\sum_{n=1}^{N}\log p(\pmb{x}_{n}\,|\,\pmb{\theta})=\underbrace{\sum_{n=1}^{N}\log\sum_{k=1}^{K}\pi_{k}\mathcal{N}\big(\pmb{x}_{n}\,|\,\pmb{\mu}_{k},\,\pmb{\Sigma}_{k}\big)}_{=:\mathcal{L}}.
$$
我们希望找到使对数似然函数 $\mathcal{L}$（定义在 11.10 中）最大化的参数 $\pmb{\theta}_{\mathrm{ML}}^{\ast}$ 。我们的“常规”方法是计算对数似然函数 $\mathcal{L}$ 关于模型参数 $\pmb{\theta}$ 的梯度 $\mathrm{d}\mathcal{L}/\mathrm{d}\theta$ ，将其设为 0 ，并求解 $\theta$ 。然而，与我们在第 9.2 节讨论的线性回归中的最大似然估计不同，我们不能得到一个闭式解。然而，我们可以利用迭代方案找到良好的模型参数 $\pmb{\theta}_{\mathrm{ML}}$ ，这将是 GMM 的 EM 算法。关键思想是每次更新一个模型参数，同时保持其他参数不变。  

注。 如果我们考虑单个高斯作为所需的密度，则 (11.10) 中的 $k$ 的总和消失，我们可以直接对高斯成分应用对数，从而得到

$$
\begin{array}{r}
\log\mathcal{N}\big(\pmb{x}\,|\,\pmb{\mu},\,\pmb{\Sigma}\big)=-\frac{D}{2}\log(2\pi)-\frac{1}{2}\log\operatorname*{det}(\pmb{\Sigma})-\frac{1}{2}(\pmb{x}-\pmb{\mu})^{\top}\pmb{\Sigma}^{-1}(\pmb{x}-\pmb{\mu}).
\end{array}
$$

这种简单形式允许我们找到 $\pmb{\mu}$ 和 $\pmb{\Sigma}$ 的闭式最大似然估计，如第 8 章所述。在 (11.10) 中，我们不能将对数移入 $k$ 的总和中，因此我们无法获得一个简单的闭式最大似然解。$\diamondsuit$  

函数的局部极值表现出其梯度关于参数必须消失（必要条件）的性质；参见第 7 章。在我们的情况下，当我们优化 (11.10) 中的对数似然函数关于 GMM 参数 $\pmb{\mu}_{k},\pmb{\Sigma}_{k},\pi_{k}$ 时，我们得到以下必要条件：

$$
\begin{array}{l}
\displaystyle\frac{\partial\mathcal{L}}{\partial\boldsymbol{\mu}_{k}}=\mathbf{0}^{\intercal}\,\Longleftrightarrow\,\sum_{n=1}^{N}\frac{\partial\log p(\boldsymbol{x}_{n}\,|\,\boldsymbol{\theta})}{\partial\boldsymbol{\mu}_{k}}=\mathbf{0}^{\intercal}\,,\\
\displaystyle\frac{\partial\mathcal{L}}{\partial\boldsymbol{\Sigma}_{k}}=\mathbf{0}\,\Longleftrightarrow\,\sum_{n=1}^{N}\frac{\partial\log p(\boldsymbol{x}_{n}\,|\,\boldsymbol{\theta})}{\partial\boldsymbol{\Sigma}_{k}}=\mathbf{0}\,,\\
\displaystyle\frac{\partial\mathcal{L}}{\partial\boldsymbol{\pi}_{k}}=0\,\Longleftrightarrow\,\sum_{n=1}^{N}\frac{\partial\log p(\boldsymbol{x}_{n}\,|\,\boldsymbol{\theta})}{\partial\boldsymbol{\pi}_{k}}=0\,.
\end{array}
$$

对于这三个必要条件，通过应用链式法则（参见第 5.2.2 节），我们要求形式为

$$
\frac{\partial\log p(\pmb{x}_{n}\,|\,\pmb{\theta})}{\partial\pmb{\theta}}=\frac{1}{p(\pmb{x}_{n}\,|\,\pmb{\theta})}\frac{\partial p(\pmb{x}_{n}\,|\,\pmb{\theta})}{\partial\pmb{\theta}}\,,
$$
其中 $\pmb\theta=\{\pmb\mu_{k},\pmb\Sigma_{k},\pi_{k},k=1,.\,.\,.\,,K\}$ 是模型参数，且

$$
\frac{1}{p(\boldsymbol{x}_{n}\,|\,\theta)}=\frac{1}{\sum_{j=1}^{K}\pi_{j}\mathcal{N}\big(\boldsymbol{x}_{n}\,|\,\boldsymbol{\mu}_{j},\,\Sigma_{j}\big)}\,.
$$
接下来，我们将通过 (11.12) 到 (11.14) 计算这些偏导数。但在进行这些计算之前，我们引入一个将在本章余下部分中起重要作用的数量：责任度。

## 11.2.1 责任度 

我们定义量  
$$
r_{n k}:=\frac{\pi_{k}\mathcal{N}\big(\pmb{x}_{n}\,|\,\pmb{\mu}_{k},\,\pmb{\Sigma}_{k}\big)}{\sum_{j=1}^{K}\pi_{j}\mathcal{N}\big(\pmb{x}_{n}\,|\,\pmb{\mu}_{j},\,\pmb{\Sigma}_{j}\big)}
$$  
为第 $k$ 个混合成分对第 $n$ 个数据点的责任度。第 $k$ 个混合成分对数据点 $\pmb{x}_{n}$ 的责任 $r_{n k}$ 与似然性  
$$
p(\pmb{x}_{n}\,|\,\pi_{k},\pmb{\mu}_{k},\pmb{\Sigma}_{k})=\pi_{k}\mathcal{N}\big(\pmb{x}_{n}\,|\,\pmb{\mu}_{k},\,\pmb{\Sigma}_{k}\big)
$$  
成比例。向量 $\pmb{r}_{n}$ 遵循玻尔兹曼/吉布斯分布。  
给定数据点时，混合成分的责任度与该数据点可能是该混合成分的合理样本有关。注意，$\pmb{r}_{n}\ :=\ [r_{n1},.\,.\,.\,,r_{n K}]^{\top}\ \in\ \mathbb{R}^{K}$ 是一个（归一化的）概率向量，即 $\textstyle\sum_{k}r_{n k}\,=\,1$ 且 $r_{n k}\;\geqslant\;0$。这个概率向量在 $K$ 个混合成分之间分配概率质量，我们可以将 $\pmb{r}_{n}$ 看作是 $\pmb{x}_{n}$ 对 $K$ 个混合成分的“软分配”。因此，公式 (11.17) 中的责任 $r_{n k}$ 代表 $\pmb{x}_{n}$ 由第 $k$ 个混合成分生成的概率。  

### 示例 11.2 (责任度)  

对于图 11.3 中的例子，我们计算责任 $r_{n k}$  
责任 $r_{n k}$ 是第 $k$ 个混合成分生成第 $n$ 个数据点的概率。  
$$
\left[\begin{array}{ccc}
1.0 & 0.0 & 0.0 \\
1.0 & 0.0 & 0.0 \\
0.057 & 0.943 & 0.0 \\
0.001 & 0.999 & 0.0 \\
0.0 & 0.066 & 0.934 \\
0.0 & 0.0 & 1.0 \\
0.0 & 0.0 & 1.0
\end{array}\right]\in\mathbb{R}^{N\times K}\,.
$$  
第 $n$ 行告诉我们所有混合成分对 $x_{n}$ 的责任度。每个数据点的所有责任之和（每行的和）为 1。第 $k$ 列给出了第 $k$ 个混合成分的责任度概览。我们可以看到，第三个混合成分（第三列）对前四个数据点没有任何责任，但对剩余数据点的责任度很大。每列所有条目的和给出了值 $N_{k}$，即第 $k$ 个混合成分的总责任。在我们的例子中，我们得到 $N_{1}\,=\,2.058$，$N_{2}\,=\,2.008$，$N_{3}=2.934$。  

接下来，我们确定给定责任度时模型参数 $\pmb{\mu}_{k},\pmb{\Sigma}_{k},\pi_{k}$ 的更新。我们将看到更新方程都依赖于责任，这使得最大似然估计问题的闭式解不可能。然而，对于给定的责任，我们将一次更新一个模型参数，同时固定其他参数。之后，我们将重新计算责任。迭代这两个步骤最终会收敛到一个局部最优解，并且是 EM 算法的具体实例。我们将在第 11.3 节中对此进行更详细的讨论。

### 11.2.2 更新均值  

### 定理 11.1  (GMM 均值更新) .  

GMM 的均值参数 $\pmb{\mu}_{k},\,k=1,.\,.\,.\,,K,$ 的更新公式为  

$$
\pmb{\mu}_{k}^{\text{new}}=\frac{\sum_{n=1}^{N}r_{n k}\pmb{x}_{n}}{\sum_{n=1}^{N}r_{n k}}\,,
$$
其中责任值 $r_{n k}$ 定义在 (11.17) 中。  

注释.  (11.20) 中个体混合成分的均值 $\pmb{\mu}_{k}$ 的更新依赖于所有均值、协方差矩阵 $\pmb{\Sigma}_{k}$ 和混合权重 $\pi_{k}$ 通过 $r_{n k}$ 给出 (11.17) 中。因此，我们无法同时获得所有 $\pmb{\mu}_{k}$ 的闭式解。$\diamondsuit$  

证明 从 (11.15) 可知，对均值参数 $\pmb{\mu}_{k},\,k=1,.\,.\,.\,,K,$ 的对数似然函数梯度需要计算偏导数  

$$
\begin{array}{l}{\displaystyle\frac{\partial p(\boldsymbol{\mathbf{\mathit{x}}}_{n}\,|\,\boldsymbol{\mathbf{\theta}})}{\partial\boldsymbol{\mu}_{k}}=\sum_{j=1}^{K}\pi_{j}\frac{\partial\mathcal{N}\big(\boldsymbol{\mathbf{\mathit{x}}}_{n}\,|\,\boldsymbol{\mathbf{\mu}}_{j},\,\boldsymbol{\mathbf{\Sigma}}_{j}\big)}{\partial\boldsymbol{\mu}_{k}}=\pi_{k}\frac{\partial\mathcal{N}\big(\boldsymbol{\mathbf{\mathit{x}}}_{n}\,|\,\boldsymbol{\mathbf{\mu}}_{k},\,\boldsymbol{\mathbf{\Sigma}}_{k}\big)}{\partial\boldsymbol{\mu}_{k}}}\\ {\displaystyle\quad\quad=\pi_{k}(\boldsymbol{\mathbf{\mathit{x}}}_{n}-\boldsymbol{\mathbf{\mu}}_{k})^{\intercal}\boldsymbol{\mathbf{\Sigma}}_{k}^{-1}\mathcal{N}\big(\boldsymbol{\mathbf{\mathit{x}}}_{n}\,|\,\boldsymbol{\mathbf{\mu}}_{k},\,\boldsymbol{\mathbf{\Sigma}}_{k}\big)\,,}\end{array}
$$
其中我们利用了只有第 $k$ 个混合成分依赖于 $\pmb{\mu}_{k}$。我们使用 (11.21b) 中的结果代入 (11.15)，并将所有项合并，使得 $\mathcal{L}$ 关于 $\pmb{\mu}_{k}$ 的所需偏导数为  

$$
\begin{array}{l}
\displaystyle{\frac{\partial{\mathcal{L}}}{\partial\pmb{\mu}_{k}}}
= \sum_{n=1}^{N} \frac{\partial\log p(\pmb{x}_{n} \mid \theta)}{\partial\pmb{\mu}_{k}}
= \sum_{n=1}^{N} \frac{1}{p(\pmb{x}_{n} \mid \theta)} \frac{\partial p(\pmb{x}_{n} \mid \theta)}{\partial\pmb{\mu}_{k}} \\
\quad = \sum_{n=1}^{N} (\pmb{x}_{n} - \pmb{\mu}_{k})^{\top} \Sigma_{k}^{-1} \underbrace{\overline{\left\{\frac{\pi_{k} \mathcal{N}(\pmb{x}_{n} \mid \pmb{\mu}_{k}, \pmb{\Sigma}_{k})}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}(x_{n} \mid \mu_{j}, \Sigma_{j}) = r_{n k}}\right\}}}_{=\pmb{x}_{n}} \\
\quad = \sum_{n=1}^{N} r_{n k} (\pmb{x}_{n} - \pmb{\mu}_{k})^{\top} \Sigma_{k}^{-1}.
\end{array}
$$
我们利用 (11.16) 中的恒等式和 (11.21b) 中偏导数的结果得到了 (11.22b)。值 $r_{n k}$ 是我们在 (11.17) 中定义的责任值。  

我们现在解 (11.22c) 以求 $\pmb{\mu}_{k}^{\text{new}}$ 使得 $\begin{array}{r}{\frac{\partial\mathcal{L}(\pmb{\mu}_{k}^{\text{new}})}{\partial\pmb{\mu}_{k}}=\mathbf{0}^{\top}}\end{array}$ 并获得  

$$
\sum_{n=1}^{N}r_{n k}{\pmb x}_{n}=\sum_{n=1}^{N}r_{n k}{\pmb\mu}_{k}^{\text{new}}\iff{\pmb\mu}_{k}^{\text{new}}=\frac{\sum_{n=1}^{N}r_{n k}{\pmb x}_{n}}{\left[\sum_{n=1}^{N}r_{n k}\right]}=\frac{1}{\left[\overline{{N_{k}}}\right]}\sum_{n=1}^{N}r_{n k}{\pmb x}_{n}\,,
$$
其中我们定义  

$$
N_{k}:=\sum_{n=1}^{N}r_{n k}
$$
为第 $k$ 个混合成分在整个数据集中的总责任值。这完成了定理 11.1 的证明。  

直观上，(11.20) 可以被解释为重要性加权的蒙特卡洛估计均值，其中数据点 $\pmb{x}_{n}$ 的重要性权重是第 $k$ 个聚类对 $\pmb{x}_{n}$ 的责任值 $r_{n k}$，$k=1,\ldots,K$。  

因此，均值 $\pmb{\mu}_{k}$ 会受到强度为 $r_{n k}$ 的数据点 $\pmb{x}_{n}$ 的拉力。均值会更强烈地向那些对应混合成分责任值高（即似然性高）的数据点拉近。图 11.4 说明了这一点。我们也可以将 (11.20) 中的均值更新解释为在由  

$$
\pmb{r}_{k}:=[r_{1k},.\,.\,.\,,r_{N k}]^{\top}/N_{k}\,,
$$
给出的分布下的期望值，即  

$$
\pmb{\mu}_{k}\leftarrow\mathbb{E}_{\pmb{r}_{k}}[\mathcal{X}]\,.
$$
图 11.4  GMM 中混合成分均值参数的更新。均值 $\pmb{\mu}$ 会受到对应责任值的权重拉向个别数据点。  

![](images/4da3c1e15aa53a26612fda5d83379454cea1ee2c93ae160f1050d9316409fb33.jpg)  

![](images/d045c112d2d6b9149ede1498da39ac2a8283cba61579143d780406cdb22aea7b.jpg)  
图 11.5  GMM 中更新均值值的影响。 (a) 更新均值值之前的 GMM；(b) 更新均值值 $\mu_{k}$ 之后的 GMM，保留方差。  

在图 11.3 的示例中，均值值更新如下：  

$$
\begin{array}{l}{\mu_{1}:-4\to-2.7}\\ {\mu_{2}:0\to-0.4}\\ {\mu_{3}:8\to3.7}\end{array}
$$
在这里，我们看到第一个和第三个混合成分的均值向数据的区域移动，而第二个成分的均值变化不大。图 11.5 说明了这种变化，其中图 11.5(a) 显示了更新均值值之前的 GMM 密度，图 11.5(b) 显示了更新均值值 $\mu_{k}$ 之后的 GMM 密度。  

(11.20) 中的均值参数更新看起来相当直接。然而，请注意责任值 $r_{n k}$ 是 $\pi_{j},\pmb{\mu}_{j},\pmb{\Sigma}_{j}$ 的函数，对于所有 $j=1,\cdot\cdot\cdot,K$，因此 (11.20) 中的更新依赖于 GMM 的所有参数，而我们如第 9.2 节中的线性回归或第 10 章中的主成分分析中获得的闭式解无法获得。

### 11.2.3 更新协方差

定理 11.2  (GMM 协方差的更新) .  GMM 中协方差参数 $\pmb{\Sigma}_{k}$ 的更新公式为

$$
{\pmb{\Sigma}}_{k}^{\text{new}}=\frac{1}{N_{k}}\sum_{n=1}^{N}r_{n k}({\pmb{x}}_{n}-{\pmb{\mu}}_{k})({\pmb{x}}_{n}-{\pmb{\mu}}_{k})^{\top}\,,
$$
其中 $r_{n k}$ 和 $N_{k}$ 分别由 (11.17) 和 (11.24) 定义。

证明 为了证明定理 11.2，我们通过计算对数似然 $\mathcal{L}$ 关于协方差 $\pmb{\Sigma}_{k}$ 的偏导数，将其设为 0，然后求解 $\pmb{\Sigma}_{k}$。我们从一般方法开始

$$
\frac{\partial{\mathcal{L}}}{\partial\Sigma_{k}}=\sum_{n=1}^{N}\frac{\partial\log p(\pmb{x}_{n}\,|\,\pmb{\theta})}{\partial\Sigma_{k}}=\sum_{n=1}^{N}\frac{1}{p(x_{n}\,|\,\pmb{\theta})}\frac{\partial p(\pmb{x}_{n}\,|\,\pmb{\theta})}{\partial\Sigma_{k}}\,.
$$
我们已经从 (11.16) 知道了 $p(x_{n}\,|\,\pmb{\theta})$。为了获得剩余的偏导数 $\frac{\partial p(\pmb{x}_{n}\,|\,\pmb{\theta})}{\partial\pmb{\Sigma}_{k}}$，我们将高斯分布 $p(\pmb{x}_{n}\mid\pmb{\theta})$ 的定义 (见 (11.9)) 写下来，并去掉所有项，只保留第 $k$ 项。我们得到

$$
\begin{array}{r l}
& \frac{\partial p(x_{n}\,|\,\theta)}{\partial\Sigma_{k}} \\
& = \frac{\partial}{\partial\Sigma_{k}}\left(\pi_{k}(2\pi)^{-\frac{D}{2}}\operatorname{det}(\Sigma_{k})^{-\frac{1}{2}}\exp\left(-\frac{1}{2}(x_{n}-\mu_{k})^{\top}\Sigma_{k}^{-1}(x_{n}-\mu_{k})\right)\right) \\
& = \pi_{k}(2\pi)^{-\frac{D}{2}}\left[\frac{\partial}{\partial\Sigma_{k}}\operatorname{det}(\Sigma_{k})^{-\frac{1}{2}}\exp\left(-\frac{1}{2}(x_{n}-\mu_{k})^{\top}\Sigma_{k}^{-1}(x_{n}-\mu_{k})\right)\right. \\
& \quad \left. + \operatorname{det}(\Sigma_{k})^{-\frac{1}{2}}\frac{\partial}{\partial\Sigma_{k}}\exp\left(-\frac{1}{2}(x_{n}-\mu_{k})^{\top}\Sigma_{k}^{-1}(x_{n}-\mu_{k})\right)\right]\,.
\end{array}
$$

我们使用以下恒等式

$$
\begin{array}{r l r}
& & \frac{\partial}{\partial\Sigma_{k}}\operatorname{det}(\Sigma_{k})^{-\frac{1}{2}} = -\frac{1}{2}\operatorname{det}(\Sigma_{k})^{-\frac{1}{2}}\Sigma_{k}^{-1}\,, \\
& & \frac{\partial}{\partial\Sigma_{k}}\left((\pmb{x}_{n}-\pmb{\mu}_{k})^{\top}\Sigma_{k}^{-1}(\pmb{x}_{n}-\pmb{\mu}_{k})\right) = -\Sigma_{k}^{-1}(\pmb{x}_{n}-\pmb{\mu}_{k})(\pmb{x}_{n}-\pmb{\mu}_{k})^{\top}\Sigma_{k}^{-1}
\end{array}
$$

并得到 (经过一些整理) 在 (11.31) 中所需的偏导数为

$$
\begin{array}{l}
\frac{\partial p(\pmb{x}_{n}\,|\,\pmb{\theta})}{\partial\pmb{\Sigma}_{k}} = \pi_{k}\,\mathcal{N}\big(\pmb{x}_{n}\,|\,\pmb{\mu}_{k},\,\pmb{\Sigma}_{k}\big) \\
\quad \cdot \left[-\frac{1}{2}\left(\pmb{\Sigma}_{k}^{-1} - \pmb{\Sigma}_{k}^{-1}(\pmb{x}_{n}-\pmb{\mu}_{k})(\pmb{x}_{n}-\pmb{\mu}_{k})^{\top}\pmb{\Sigma}_{k}^{-1}\right)\right].
\end{array}
$$

将所有内容放在一起，对数似然关于 $\pmb{\Sigma}_{k}$ 的偏导数为

$$
\begin{align}
\frac{\partial{\mathcal{L}}}{\partial\Sigma_{k}} 
&= \sum_{n=1}^{N} \frac{\partial p(x_{n} \mid \theta)}{\partial\Sigma_{k}}
= \sum_{n=1}^{N} \frac{1}{p(x_{n} \mid \theta)} \frac{\partial p(x_{n} \mid \theta)}{\partial\Sigma_{k}} & \quad \scriptstyle(11.36a) \\
&= \sum_{n=1}^{N} \frac{\pi_{k} \mathcal{N}(x_{n} \mid \mu_{k}, \Sigma_{k})}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}(x_{n} \mid \mu_{j}, \Sigma_{j})} \\
&\quad - \frac{1}{2} \left[\Sigma_{k}^{-1} - \Sigma_{k}^{-1}(x_{n} - \mu_{k})(x_{n} - \mu_{k})^{\top} \Sigma_{k}^{-1}\right] & \quad \scriptstyle(11.36b) \\
&= -\frac{1}{2} \sum_{n=1}^{N} r_{n k} \left(\Sigma_{k}^{-1} - \Sigma_{k}^{-1}(x_{n} - \mu_{k})(x_{n} - \mu_{k})^{\top} \Sigma_{k}^{-1}\right) \\
&= -\frac{1}{2} \Sigma_{k}^{-1} \sum_{n=1}^{N} r_{n k} + \frac{1}{2} \Sigma_{k}^{-1} \left(\sum_{n=1}^{N} r_{n k} (x_{n} - \mu_{k})(x_{n} - \mu_{k})^{\top}\right) \Sigma_{k}^{-1}.
\end{align}
$$

我们看到责任值 $r_{n k}$ 也出现在这个偏导数中。将其设为 0，我们得到必要的最优条件

$$
\begin{array}{r l r}
& N_{k}\pmb{\Sigma}_{k}^{-1} = \pmb{\Sigma}_{k}^{-1}\left(\sum_{n=1}^{N}r_{n k}(\pmb{x}_{n}-\pmb{\mu}_{k})(\pmb{x}_{n}-\pmb{\mu}_{k})^{\top}\right)\pmb{\Sigma}_{k}^{-1} \\
& \iff N_{k}\pmb{I} = \left(\sum_{n=1}^{N}r_{n k}(\pmb{x}_{n}-\pmb{\mu}_{k})(\pmb{x}_{n}-\pmb{\mu}_{k})^{\top}\right)\pmb{\Sigma}_{k}^{-1}\,.
\end{array}
$$

通过求解 $\pmb{\Sigma}_{k}$，我们得到

$$
{\pmb{\Sigma}}_{k}^{\text{new}}=\frac{1}{N_{k}}\sum_{n=1}^{N}r_{n k}\big({\pmb{x}}_{n}-{\pmb{\mu}}_{k}\big)\big({\pmb{x}}_{n}-{\pmb{\mu}}_{k}\big)^{\top}\,,
$$
其中 $\pmb{r}_{k}$ 是由 (11.25) 定义的概率向量。这给出了 $\pmb{\Sigma}_{k}$ 的简单更新规则，适用于 $k=1,\ldots,K$，并证明了定理 11.2。

类似于 (11.20) 中 $\pmb{\mu}_{k}$ 的更新，我们可以将 (11.30) 中协方差的更新解释为重要性加权的中心化数据 $\tilde{\mathcal{X}}_{k}:=\{\pmb{x}_{1}-\pmb{\mu}_{k},\dots,\pmb{x}_{N}-\pmb{\mu}_{k}\}$ 的平方的期望值。

### 例 11.4（方差更新）

在图 11.3 的例子中，方差更新如下：

$$
\begin{array}{l}{\sigma_{1}^{2}:1\to0.14}\\ {\sigma_{2}^{2}:0.2\to0.44}\\ {\sigma_{3}^{2}:3\to1.53}\end{array}
$$

在这里，我们看到第一和第三成分的方差显著收缩，而第二成分的方差略有增加。

图 11.6 说明了这一情况。图 11.6(a) 与图 11.5(b) 相同（但放大了），显示了在更新方差之前的 GMM 密度及其各个成分。图 11.6(b) 显示了更新方差之后的 GMM 密度。

![](images/9d22f4119ec39a93994c2b601e0729d5d10a69d6c73cca5760aab0cc03c97070.jpg)  
(a) 更新方差之前的 GMM 密度及其各个成分。

![](images/f6675a0a806d8bd2ef2a831d3a9989c105f16e5b6836660bbd741f610152e8a5.jpg)  
(b) 更新方差之后的 GMM 密度及其各个成分。

类似于均值参数的更新，我们可以将式 (11.30) 解释为数据点 ${\pmb x}_{n}$ 与第 $k$ 个混合成分关联的加权协方差的蒙特卡洛估计，其中权重是责任值 $r_{n k}$。与均值参数的更新类似，这个更新依赖于所有 $\pi_{j},\pmb{\mu}_{j},\pmb{\Sigma}_{j}$，$j=1,\dots,K_{z}$，通过责任值 $r_{n k}$，这禁止了闭式解的存在。

### 11.2.4 更新混合权重

定理 11.3  (GMM 混合权重的更新) . GMM 的混合权重更新为

$$
\pi_{k}^{\text{new}} = \frac{N_{k}}{N}\,,\quad k=1,.\,.\,.\,,K\,,
$$

其中 $N$ 是数据点的数量，$N_{k}$ 的定义见 (11.24)。

证明 为了找到对数似然函数关于权重参数 $\pi_{k}$ 的偏导数，$k=1,\ldots,K$，我们通过拉格朗日乘数法考虑约束 $\sum_{k}\pi_{k} = 1$（参见第 7.2 节）。拉格朗日函数为

$$
\mathfrak{L} = \mathcal{L} + \lambda \left(\sum_{k=1}^{K} \pi_{k} - 1\right)
$$

$$
= \sum_{n=1}^{N} \log \sum_{k=1}^{K} \pi_{k} \mathcal{N} \left(\pmb{x}_{n} \mid \pmb{\mu}_{k}, \pmb{\Sigma}_{k}\right) + \lambda \left(\sum_{k=1}^{K} \pi_{k} - 1\right)\,,
$$

其中 $\mathcal{L}$ 是 (11.10) 中的对数似然函数，第二项编码了所有混合权重之和为 1 的等式约束。我们关于 $\pi_{k}$ 求偏导数得到

$$
\begin{array}{r l r}
\frac{\partial \mathfrak{L}}{\partial \pi_{k}} &= \sum_{n=1}^{N} \frac{\mathcal{N} \left(\pmb{x}_{n} \mid \pmb{\mu}_{k}, \Sigma_{k}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N} \left(\pmb{x}_{n} \mid \pmb{\mu}_{j}, \Sigma_{j}\right)} + \lambda \\
&= \frac{1}{\pi_{k}} \underbrace{\sum_{n=1}^{N} \frac{\pi_{k} \mathcal{N} \left(\pmb{x}_{n} \mid \pmb{\mu}_{k}, \Sigma_{k}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N} \left(\pmb{x}_{n} \mid \pmb{\mu}_{j}, \Sigma_{j}\right)}}_{=N_{k}} + \lambda = \frac{N_{k}}{\pi_{k}} + \lambda\,,
\end{array}
$$

以及关于拉格朗日乘数 $\lambda$ 的偏导数为

$$
\frac{\partial \mathfrak{L}}{\partial \lambda} = \sum_{k=1}^{K} \pi_{k} - 1\,.
$$

将两个偏导数设置为 0（最优条件的必要条件）得到方程组

$$
\begin{array}{c}
\pi_{k} = -\frac{N_{k}}{\lambda}\,, \\
1 = \sum_{k=1}^{K} \pi_{k}\,.
\end{array}
$$

使用 (11.46) 在 (11.47) 中，并解出 $\pi_{k}$，我们得到

$$
\sum_{k=1}^{K} \pi_{k} = 1 \iff -\sum_{k=1}^{K} \frac{N_{k}}{\lambda} = 1 \iff -\frac{N}{\lambda} = 1 \iff \lambda = -N\,.
$$

这允许我们用 $-N$ 替换 (11.46) 中的 $\lambda$，得到

$$
\pi_{k}^{\text{new}} = \frac{N_{k}}{N}\,,
$$

这给出了权重参数 $\pi_{k}$ 的更新，并证明了定理 11.3。

在 (11.42) 中，我们可以将混合权重识别为第 $k$ 个聚类的总责任与数据点数量的比值。由于 $N = \sum_{k} N_{k}$，数据点的数量也可以解释为所有混合成分的总责任，使得 $\pi_{k}$ 是第 $k$ 个混合成分在数据集中的相对重要性。

注释。由于 $N_{k} = \sum_{i=1}^{N} r_{nk}$，混合权重 (11.42) 的更新方程也依赖于所有 $\pi_{j}, \mu_{j}, \Sigma_{j}, j = 1,.\,.\,.\,.\,K$ 通过责任 $r_{nk}$。$\diamondsuit$

![](images/dabe2ccc808696c1fe02abe732de19340750f5376ccf39db1c999612ab91409.jpg)  
(a) 更新混合权重之前的 GMM 密度和个体成分。

![](images/f6f69a38f1baa78d6f70528a61ac335c9473ea76a1f19aa7a7c45cc3f4efd2f0.jpg)  
(b) 更新混合权重之后的 GMM 密度和个体成分。

在我们的运行示例中，从图 11.3 中，混合权重更新如下：

$$
\begin{array}{l}
\pi_{1} : \frac{1}{3} \to 0.29 \\
\pi_{2} : \frac{1}{3} \to 0.29 \\
\pi_{3} : \frac{1}{3} \to 0.42
\end{array}
$$

在这里，我们看到第三个成分获得了更多的权重/重要性，而其他成分变得稍微不那么重要。图 11.7 说明了更新混合权重的效果。图 11.7(a) 与图 11.6(b) 相同，显示了更新混合权重之前的 GMM 密度及其个体成分。图 11.7(b) 显示了更新混合权重之后的 GMM 密度。

总体而言，更新均值、方差和权重一次后，我们得到了图 11.7(b) 中所示的 GMM。与图 11.3 中的初始化相比，我们可以看到参数更新使 GMM 密度向数据点转移了一部分质量。

在更新均值、方差和权重一次后，图 11.7(b) 中的 GMM 拟合已经比图 11.3 中的初始化显著更好。这也体现在对数似然值上，它们从 $-28.3$（初始化）增加到 $-14.4$，经过一个完整的更新周期。

## 11.3 EM 算法

EM 算法不幸的是，在公式 (11.20)、(11.30) 和 (11.42) 中，参数 $\pmb{\mu}_{k},\pmb{\Sigma}_{k},\pi_{k}$ 的更新并不构成混合模型参数更新的闭式解，因为责任值 $r_{n k}$ 依赖于这些参数的方式相当复杂。然而，这些结果建议了一种简单的迭代方案，通过最大似然法来解决参数估计问题。期望最大化算法（EM 算法）由 Dempster 等人（1977）提出，是一种用于学习混合模型（最大似然或 MAP）参数的一般迭代方案，更广泛地说，是用于潜在变量模型的迭代方案。  

在我们关于高斯混合模型的例子中，我们选择初始值 $\pmb{\mu}_{k},\pmb{\Sigma}_{k},\pi_{k}$ 并交替进行直到收敛：  

$E$ 步：评估责任值 $r_{n k}$（数据点 $n$ 属于混合成分 $k$ 的后验概率）。$M$ 步：使用更新的责任值重新估计参数 $\pmb{\mu}_{k},\pmb{\Sigma}_{k},\pi_{k}$。  

EM 算法中的每一步都会增加对数似然函数（Neal 和 Hinton, 1999）。为了检查收敛性，我们可以检查对数似然函数或参数本身。高斯混合模型参数的 EM 算法的具体实现如下：  

1. 初始化 $\pmb{\mu}_{k},\pmb{\Sigma}_{k},\pi_{k}$。

2. $E$ 步：使用当前参数 $\pi_{k},\pmb{\mu}_{k},\pmb{\Sigma}_{k}$ 为每个数据点 ${\bf x}_{n}$ 评估责任值 $r_{n k}$：  

$$
r_{n k}=\frac{\pi_{k}\mathcal{N}\big(\pmb{x}_{n}\,|\,\pmb{\mu}_{k},\,\pmb{\Sigma}_{k}\big)}{\sum_{j}\pi_{j}\mathcal{N}\big(\pmb{x}_{n}\,|\,\pmb{\mu}_{j},\,\pmb{\Sigma}_{j}\big)}\,.
$$
3. $M$ 步：使用当前责任值 $r_{n k}$（来自 $E$ 步）重新估计参数 $\pi_{k},\pmb{\mu}_{k},\pmb{\Sigma}_{k}$：  

$$
\begin{array}{l}{\displaystyle\boldsymbol{\mu}_{k}=\frac{1}{N_{k}}\sum_{n=1}^{N}r_{n k}\boldsymbol{x}_{n}\,,}\\ {\displaystyle\boldsymbol{\Sigma}_{k}=\frac{1}{N_{k}}\sum_{n=1}^{N}r_{n k}(\boldsymbol{\mathbf{\mathit{x}}}_{n}-\boldsymbol{\mathbf{\mu}}_{k})(\boldsymbol{\mathbf{\mathit{x}}}_{n}-\boldsymbol{\mathbf{\mu}}_{k})^{\intercal}\,,}\\ {\displaystyle\boldsymbol{\pi}_{k}=\frac{N_{k}}{N}\,.}\end{array}
$$
在更新均值 $\pmb{\mu}_{k}$（公式 11.54）后，它们随后用于更新相应的协方差（公式 11.55）。  

### 例 11.6 (高斯混合模型拟合)  

![](images/b46e79035f4eedf60ac0fb2910b67b57be8c6ce2afa8f761de80a14d5265b58e.jpg)  
(b) 作为 EM 迭代函数的负对数似然。  

![](images/6082aaa768e423fa293da13d1532190dac48d3b84fbce8e5155eff6fb03b7002.jpg)  
图 11.9 说明了使用 EM 算法拟合具有三个成分的高斯混合模型到二维数据集的过程。 (a) 数据集； (b) 作为 EM 迭代函数的负对数似然（越低越好）。红色点表示 EM 迭代中对应的高斯混合模型成分的迭代次数，图 (c) 至 (f) 显示了这些迭代次数对应的高斯混合模型成分。黄色圆圈表示高斯混合模型成分的均值。图 11.10(a) 显示了最终的高斯混合模型拟合。  

当我们对图 11.3 中的例子运行 EM 算法时，我们经过五次迭代后得到了图 11.8(a) 中的最终结果，并且图 11.8(b) 显示了负对数似然随 EM 迭代次数的变化情况。最终的高斯混合模型为  

$$
\begin{array}{l}{p(x)=0.29\mathcal{N}\big(x\,|\,-2.75,\,0.06\big)+0.28\mathcal{N}\big(x\,|\,-0.50,\,0.2}\\ {\phantom{0.29}+0.43\mathcal{N}\big(x\,|\,3.64,\,1.63\big)\,.}\end{array}
$$
我们对图 11.1 中的二维数据集应用了 EM 算法，其中包含 $K=3$ 个混合成分。图 11.9 说明了 EM 算法的一些步骤，并显示了负对数似然随 EM 迭代次数的变化情况（图 11.9(b)）。图 11.10(a) 显示了  

![](images/2f26a6fba09da8d333602cfe2b42b9cbeeb6fef2f935c79b49a11f50e61e0b69.jpg)  

![](images/d550069f52e827b4c35aef9913b4fd3865a46c07ba1ee0bf441c14491d8bc92b.jpg)  

相应的最终高斯混合模型拟合。图 11.10(b) 显示了数据点的最终混合成分责任值。当 EM 收敛时，数据集根据混合成分的责任值着色。虽然左侧的数据点明显由单一混合成分负责，但右侧两个数据簇的重叠可能由两个混合成分生成。很明显，存在一些数据点无法唯一归属于单一成分（蓝色或黄色），因此这些簇对这些点的责任值约为 0.5。

## 11.4 隐变量视角  

我们可以从隐变量模型的角度来看GMM，即隐变量 $_z$ 只能取有限个值。这与PCA不同，在PCA中，隐变量是 $\mathbb{R}^{M}$ 中的连续值。

概率视角的优点在于：（i）它将为我们在前几节中做出的一些经验性决定提供正当理由；（ii）它允许将责任具体解释为后验概率；（iii）更新模型参数的迭代算法可以以一种原则性的方式推导出来，即隐变量模型中的最大似然参数估计的EM算法。

### 11.4.1 生成过程与概率模型  

为了推导GMM的概率模型，考虑生成数据的生成过程是有用的，即使用概率模型来生成数据的过程。

我们假设有一个包含 $K$ 个成分的混合模型，并且一个数据点 $_{_{\pmb{x}}}$ 只能由一个混合成分生成。引入一个二元指示变量 $z_{k}\in\{0,1\}$，有两个状态（见第6.2节），表示第 $k$ 个混合成分是否生成了该数据点，使得

$$
p(\pmb{x}\,|\,z_{k}=1)=\mathcal{N}\big(\pmb{x}\,|\,\pmb{\mu}_{k},\,\pmb{\Sigma}_{k}\big)\:.
$$
将 $\boldsymbol{z}:=\,[z_{1},.\,.\,.\,,z_{K}]^{\intercal}\,\in\,\mathbb{R}^{K}$ 视为一个包含 $K-1$ 个概率的向量。例如，对于 $K=3$，一个有效的 $\boldsymbol{z}$ 为 $\boldsymbol{z}\,=\,[z_{1},z_{2},z_{3}]^{\top}\,=\,[0,1,0]^{\top}$，这会选择第二个混合成分，因为 $z_{2}=1$。

注释。有时这种概率分布被称为“多项分布”，它是二项分布的一般化（Murphy, 2012）。$\diamondsuit$ 一位热编码 1-of-$K$ 表示

$_z$ 的性质意味着 $\textstyle\sum_{k=1}^{K}z_{k}=1$。因此，$_z$ 是一位热编码（也称为 1-of-$K$ 表示）。

到目前为止，我们假设指示变量 $z_{k}$ 是已知的。然而，在实践中，这并不是情况，我们对隐变量 $_z$ 放置一个先验分布

$$
p(z)=\pmb{\pi}=[\pi_{1},\dots,\pi_{K}]^{\top}\,,\quad\sum_{k=1}^{K}\pi_{k}=1\,,
$$
然后第 $k$ 个条目

$$
\pi_{k}=p(z_{k}=1)
$$
图 11.11 GMM 的图形模型，包含一个数据点。

![](images/b3713cc45490bacf86a17428801221b1b76a5d33a5d5f212cc8418ebd7c62c6f.jpg)

祖先采样

这个概率向量的性质描述了第 $k$ 个混合成分生成数据点 $\pmb{x}$ 的概率。

注释（从GMM采样）。这种隐变量模型的构建（见图11.11中的相应图形模型）使其非常适合一种非常简单的采样过程（生成过程）来生成数据：在第一步中，我们根据 $p(z)=\pi$ 随机选择一个混合成分 $i$（通过一位热编码 $z$）；在第二步中，我们从相应的混合成分中抽取样本。当我们丢弃隐变量的样本，只留下 $\mathbf{\Delta}\mathbf{x}^{(i)}$ 时，我们得到了GMM的有效样本。这种采样方式，其中随机变量的样本依赖于图形模型中该变量的父变量的样本，称为祖先采样。$\diamondsuit$

一般来说，一个概率模型由数据和隐变量的联合分布定义（见第8.4节）。在先验 $p(z)$ 定义为（19）和（11.60）以及条件 $p(\pmb{x}\,|\,\pmb{z})$ 从（11.58）中，我们通过

$$
p(\pmb{x},z_{k}=1)=p(\pmb{x}\,|\,z_{k}=1)p(z_{k}=1)=\pi_{k}\mathcal{N}\big(\pmb{x}\,|\,\pmb{\mu}_{k},\,\pmb{\Sigma}_{k}\big)
$$
对于 $k=1,\dots,K$，得到这个联合分布的所有 $K$ 个成分，因此

$$
p(\pmb{x},z)=\left[\begin{array}{c}{p(\pmb{x},z_{1}=1)}\\ {\vdots}\\ {p(\pmb{x},z_{K}=1)}\end{array}\right]=\left[\begin{array}{c}{\pi_{1}\mathcal{N}\big(\pmb{x}\,|\,\pmb{\mu}_{1},\,\Sigma_{1}\big)}\\ {\vdots}\\ {\pi_{K}\mathcal{N}\big(\pmb{x}\,|\,\pmb{\mu}_{K},\,\Sigma_{K}\big)}\end{array}\right],
$$
这完全指定了概率模型。

### 11.4.2 似然性

为了在潜在变量模型中获得似然性 $p(\pmb{x}\,|\,\pmb{\theta})$，我们需要消除潜在变量（参见第 8.4.3 节）。在我们的情况下，可以通过对联合分布 $p(\pmb{x},\pmb{z})$（见 11.62 式）中的所有潜在变量求和来实现，因此有

$$
p(\pmb{x}\,|\,\pmb{\theta})=\sum_{z}p(\pmb{x}\,|\,\pmb{\theta},z)p(z\,|\,\pmb{\theta})\,,\quad\pmb{\theta}:=\{\pmb{\mu}_{k},\pmb{\Sigma}_{k},\pi_{k}:\;k=1,\ldots,K\}\,.
$$

我们现在明确地对概率模型的参数 $\pmb{\theta}$ 进行条件化，这是我们在之前忽略的。在 11.63 式中，我们对所有可能的一热编码 $z$ 进行求和，这由 $\sum_{z}$ 表示。由于每个 $z$ 只有一个非零项，因此 $z$ 的所有可能配置/设置只有 $K$ 种。例如，如果 $K=3$，那么 $z$ 可以有以下配置

$$
\left[\begin{array}{l}{1}\\ {0}\\ {0}\end{array}\right],\ \left[\begin{array}{l}{0}\\ {1}\\ {0}\end{array}\right],\ \left[\begin{array}{l}{0}\\ {0}\\ {1}\end{array}\right]\,.
$$

在 11.63 式中对所有可能的 $z$ 配置求和等价于查看 $z$ 向量的非零项并写出

$$
\begin{aligned}
p(\pmb{x}\,|\,\pmb{\theta}) &= \sum_{z}p(\pmb{x}\,|\,\pmb{\theta},z)p(z\,|\,\pmb{\theta}) \\
&= \sum_{k=1}^{K}p(\pmb{x}\,|\,\pmb{\theta},z_{k}=1)p(z_{k}=1\,|\,\pmb{\theta})
\end{aligned}
$$

因此，所需的边缘分布为

$$
\begin{aligned}
p(\pmb{x}\,|\,\pmb{\theta}) &= \sum_{k=1}^{K}p(\pmb{x}\,|\,\pmb{\theta},z_{k}=1)p(z_{k}=1|\pmb{\theta}) \\
&= \sum_{k=1}^{K}\pi_{k}\mathcal{N}\big(\pmb{x}\,|\,\pmb{\mu}_{k},\,\pmb{\Sigma}_{k}\big)\,,
\end{aligned}
$$

我们将其识别为来自 11.3 式的 GMM 模型。给定数据集 $\mathcal{X}$，我们立即得到似然性

$$
p(\mathcal{X}\,|\,\pmb{\theta}) = \prod_{n=1}^{N}p(\pmb{x}_{n}\,|\,\pmb{\theta}) \overset{(11.66b)}{=} \prod_{n=1}^{N}\sum_{k=1}^{K}\pi_{k}\mathcal{N}\big(\pmb{x}_{n}\,|\,\pmb{\mu}_{k},\,\pmb{\Sigma}_{k}\big)\,,
$$

这正是 11.9 式中的 GMM 似然性。因此，带有潜在指示符 $z_{k}$ 的潜在变量模型是关于高斯混合模型的一种等价思考方式。

![](images/9c0ca67e3bd658c4fbd875e31d0b00cc9e3606b296c3b562f1c53e06e77958df.jpg)

图 11.12 高斯混合模型的图形模型，包含 $N$ 个数据点。

### 11.4.3 后验分布  

让我们简要了解一下潜在变量 $_z$ 的后验分布。根据贝叶斯定理，第 $k$ 个分量生成数据点 $\pmb{x}$ 的后验概率为

$$
p(z_{k}=1\,|\,\pmb{x})=\frac{p(z_{k}=1)p(\pmb{x}\,|\,z_{k}=1)}{p(\pmb{x})}\,,
$$

其中边缘概率 $p(\pmb{x})$ 如 (11.66b) 所示。这给出了第 $k$ 个指示变量 $z_{k}$ 的后验分布

$$
p(z_{k}=1\,|\,\pmb{x})=\frac{p(z_{k}=1)p(\pmb{x}\,|\,z_{k}=1)}{\sum_{j=1}^{K}p(z_{j}=1)p(\pmb{x}\,|\,z_{j}=1)}=\frac{\pi_{k}\mathcal{N}(\pmb{x}\,|\,\pmb{\mu_{k}},\,\pmb{\Sigma_{k}})}{\sum_{j=1}^{K}\pi_{j}\mathcal{N}(\pmb{x}\,|\,\pmb{\mu_{j}},\,\pmb{\Sigma_{j}})}\,,
$$

我们将其识别为第 $k$ 个混合分量对数据点 $\pmb{x}$ 的责任。请注意，我们省略了对 GMM 参数 $\pi_{k},\pmb{\mu}_{k},\pmb{\Sigma}_{k}$ 的显式条件，其中 $k=1,\ldots,K$。

### 11.4.4 扩展到完整数据集  

迄今为止，我们仅讨论了数据集仅包含一个数据点 $\pmb{x}$ 的情况。然而，先验和后验的概念可以直接扩展到包含 $N$ 个数据点 $\mathcal{X}:=\{\pmb{x}_{1},\cdot\cdot\cdot,\pmb{x}_{N}\}$ 的情况。在 GMM 的概率解释中，每个数据点 $\pmb{x}_{n}$ 都有自己的潜在变量

$$
\boldsymbol{z}_{n}=\left[z_{n1},.\,.\,.\,,z_{n K}\right]^{\intercal}\in\mathbb{R}^{K}\,.
$$

在仅考虑一个数据点 $\pmb{x}$ 的情况下，我们省略了索引 $n$，但现在这变得重要了。

我们对所有潜在变量 $z_{n}$ 共享相同的先验分布 $\pi$。相应的图形模型如图 11.12 所示，我们使用了盘子记号。

条件分布 $p(\pmb{x}_{1},.\,.\,.\,,\pmb{x}_{N}\,\vert\,z_{1},.\,.\,.\,,z_{N})$ 在数据点之间因子分解，并给出为

$$
p(\pmb{x}_{1},.\,.\,,\pmb{x}_{N}\,|\,z_{1},.\,.\,.\,,z_{N})=\prod_{n=1}^{N}p(\pmb{x}_{n}\,|\,z_{n})\,.
$$

为了获得后验分布 $p(z_{n k}\,=\,1\,|\,\pmb{x}_{n})$，我们遵循第 11.4.3 节中的相同推理，并应用贝叶斯定理得到

$$
\begin{array}{l}
p(z_{n k}=1\,|\,\pmb{x}_{n})=\frac{p(\pmb{x}_{n}\,|\,z_{n k}=1)p(z_{n k}=1)}{\sum_{j=1}^{K}p(\pmb{x}_{n}\,|\,z_{n j}=1)p(z_{n j}=1)} \\
\qquad\qquad=\frac{\pi_{k}\mathcal{N}\big(\pmb{x}_{n}\,|\,\pmb{\mu}_{k},\,\pmb{\Sigma}_{k}\big)}{\sum_{j=1}^{K}\pi_{j}\mathcal{N}\big(\pmb{x}_{n}\,|\,\pmb{\mu}_{j},\,\pmb{\Sigma}_{j}\big)}=r_{n k}\,.
\end{array}
$$

这意味着 $p(z_{k}=1\,|\,\pmb{x}_{n})$ 是第 $k$ 个混合分量生成数据点 $\pmb{x}_{n}$ 的（后验）概率，并对应于我们在 (11.17) 中引入的责任 $r_{n k}$。现在，责任不仅具有直观的解释，还具有数学上的解释作为后验概率。

### 11.4.5 重新审视 EM 算法  

我们介绍的用于最大似然估计的迭代方案 EM 算法可以从潜在变量的角度以一种原则性的方式推导出来。给定当前的模型参数设置 $\pmb\theta^{(t)}$，E 步计算期望对数似然

$$
\begin{array}{l}
Q(\pmb{\theta}\,|\,\pmb{\theta}^{(t)})=\mathbb{E}_{z\,|\,\pmb{x},\pmb{\theta}^{(t)}}[\log p(\pmb{x},z\,|\,\pmb{\theta})] \\
\quad\quad\quad\quad\quad\quad=\int\log p(\pmb{x},z\,|\,\pmb{\theta})p(z\,|\,\pmb{x},\pmb{\theta}^{(t)})\mathrm{d}z\,,
\end{array}
$$

其中 $\log p(\pmb{x},\pmb{z}\,|\,\pmb{\theta})$ 的期望是相对于潜在变量的后验 $p(z\mid\mathbf{\boldsymbol{x}},\mathbf{\boldsymbol{\theta}}^{(t)})$ 取的。M 步通过最大化 (11.73b) 来选择更新的模型参数 $\pmb{\theta}^{(t+1)}$。

尽管 EM 迭代确实增加了对数似然，但没有保证 EM 收敛到最大似然解。EM 算法有可能收敛到对数似然的局部最大值。可以通过多次运行 EM 并使用不同的参数 $\pmb{\theta}$ 的初始设置来减少陷入不良局部最优的风险。我们不在此处进一步详细讨论，但可以参考 Rogers 和 Girolami (2016) 以及 Bishop (2006) 的优秀解释。

## 11.5 进一步阅读  

GMM 可以被视为生成模型的一种，因为它很容易通过祖先采样生成新的数据（Bishop, 2006）。对于给定的 GMM 参数 $\pi_{k},\pmb{\mu}_{k},\pmb{\Sigma}_{k}$，$k=1,\ldots,K$，我们从概率向量 $[\pi_{1},.\,.\,.,\pi_{K}]^{\top}$ 中采样一个索引 $k$，然后从 $\mathcal{N}(\pmb{\mu}_{k},\,\pmb{\Sigma}_{k})$ 中采样一个数据点 $\pmb{x}$。如果重复这个过程 $N$ 次，我们就可以得到由 GMM 生成的数据集。图 11.1 就是通过这种方法生成的。

在本章中，我们假设组件的数量 $K$ 是已知的。但在实践中，这通常不是情况。然而，我们可以使用第 8.6.1 节中讨论的嵌套交叉验证来找到好的模型。

Gaussian 混合模型与 $K$-means 聚类算法密切相关。$K$-means 也使用 EM 算法将数据点分配到聚类中。如果我们把 GMM 中的均值视为聚类中心，并忽略协方差（或设为 $\mathcal{I}$），我们就可以得到 $K$-means。正如 MacKay (2003) 所很好地描述的那样，$K$-means 对数据点进行“硬”分配到聚类中心 $\pmb{\mu}_{k}$，而 GMM 通过责任进行“软”分配。

我们仅简要介绍了 GMM 和 EM 算法的潜在变量视角。注意，EM 可以用于一般潜在变量模型中的参数学习，例如非线性状态空间模型（Ghahramani 和 Roweis, 1999; Roweis 和 Ghahramani, 1999）和强化学习（Barber, 2012）。因此，GMM 的潜在变量视角有助于以一种原则性的方式推导相应的 EM 算法（Bishop, 2006; Barber, 2012; Murphy, 2012）。

我们仅讨论了通过 EM 算法进行最大似然估计来寻找 GMM 参数。最大似然估计的标准批评也适用于这里：

与线性回归类似，最大似然估计可能会遭受严重的过拟合。在 GMM 情况下，当混合成分的均值与数据点相同时，协方差趋向于 0 时，这种情况会发生。然后，似然性会接近无穷大。Bishop (2006) 和 Barber (2012) 对此问题进行了详细讨论。我们仅获得参数 $\pi_{k},\pmb{\mu}_{k},\pmb{\Sigma}_{k}$ 的点估计，$k=q,\cdot\cdot\cdot,K$，这并不能表明参数值的不确定性。贝叶斯方法会在参数上放置先验，从而可以得到参数的后验分布。这个后验分布允许我们计算模型证据（边际似然），这可以用于模型比较，从而以一种原则性的方式确定混合成分的数量。不幸的是，在这种情况下，闭式推断是不可能的，因为没有这种模型的共轭先验。然而，近似方法，如变分推断，可以用来获得近似后验（Bishop, 2006）。

![](images/54bddd1722f6d096354d2bd3ba94e786a7623fe00abfdc33be1d2b36968b9b0e.jpg)  
图 11.13 频率直方图（橙色条形）和核密度估计（蓝色曲线）。核密度估计器产生的是底层密度的平滑估计，而频率直方图则是数据点（黑色）落入每个区间的未平滑计数。

有许多密度估计技术可供选择。在实践中，我们通常使用频率直方图和核密度估计。

频率直方图提供了一种非参数化的方法来表示连续密度，并由 Pearson (1895) 提出。频率直方图通过“分箱”数据空间并计算每个区间的数据点数量来构建。然后在每个区间的中心绘制一个条形，条形的高度与该区间内的数据点数量成正比。区间的大小是一个关键的超参数，不恰当的选择会导致过拟合和欠拟合。如第 8.2.4 节中所述，交叉验证可以用来确定一个良好的区间的大小。

核密度估计由 Rosenblatt (1956) 和 Parzen (1962) 独立提出，是一种非参数化的密度估计方法。给定 $N$ 个独立同分布的样本，核密度估计器将底层分布表示为

$$
p(\pmb{x})=\frac{1}{N h}\sum_{n=1}^{N}k\left(\frac{\pmb{x}-\pmb{x}_{n}}{h}\right)\,,
$$
其中 $k$ 是一个核函数，即一个非负函数，其积分值为 1，$h>0$ 是一个平滑/带宽参数，其作用类似于直方图中的区间的大小。注意，我们在数据集中的每个单个数据点 $\pmb{x}_{n}$ 上放置了一个核。常用的核函数包括均匀分布和高斯分布。核密度估计与直方图密切相关，但通过选择合适的核，我们可以保证密度估计的平滑性。图 11.13 展示了对于给定的 250 个数据点集，直方图和具有高斯形状的核密度估计器之间的差异。