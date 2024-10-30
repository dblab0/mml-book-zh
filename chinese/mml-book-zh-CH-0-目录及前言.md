# 机器学习的数学

Marc Peter Deisenroth A. Aldo Faisal Cheng Soon Ong  

# 目录  

# 第一部分 数学基础 9  

1 引言与动机 11

 1.1 寻找直觉的词汇 12

 1.2 阅读本书的两种方式 13

 1.3 练习与反馈 16  

2.1 线性方程组 19

 2.2 矩阵 22

 2.3 解线性方程组 27

 2.4 向量空间 35

2.5 线性独立 40

 2.6 基与秩 44

 2.7 线性映射 48

 2.8 仿射空间 61

 2.9 进一步阅读 63 
 
 练习 64  

3 解析几何 70

 3.1 范数 71

 3.2 内积 72

 3.3 长度与距离 75

 3.4 角度与正交性 76

 3.5 正交归一基 78

 3.6 正交补 79

 3.7 函数的内积 80

 3.8 正交投影 81

 3.9 旋转 91

 3.10 进一步阅读 94 
 
 练习 96  

4 矩阵分解 98 

4.1 行列式与迹 99 

4.2 特征值与特征向量 105

 4.3 乔里斯基分解 114

 4.4 特征分解与对角化 115

 4.5 奇异值分解 119

 4.6 矩阵逼近 129

 4.7 矩阵谱系 134

 4.8 进一步阅读 135 练习 137  

5 向量微积分 139

 5.1 单变量函数的微分 141

 5.2 偏微分与梯度 146

 5.3 向量值函数的梯度 149

 5.4 矩阵的梯度 155

 5.5 计算梯度的有用恒等式 158

 5.6 反向传播与自动微分 159

 5.7 高阶导数 164

 5.8 线性化与多元泰勒级数 165

 5.9 进一步阅读 170 
 
 练习 170  

6 概率与分布

6.1 概率空间的构建 172

 6.2 离散与连续概率 178

 6.3 求和规则、乘积规则与贝叶斯定理 183

 6.4 总结性统计与独立性 186

 6.5 高斯分布 197

 6.6 共轭与指数族 205

 6.7 变量变换/逆变换 214

 6.8 进一步阅读 221 练习 222  

7.1 使用梯度下降进行优化 227

 7.2 带约束的优化与拉格朗日乘子 233

 7.3 凸优化 236

 7.4 进一步阅读 246 
 
 练习 247  

# 第二部分 中心机器学习问题 249

8 当模型遇到数据 251

 8.1 数据、模型与学习 251

 8.2 经验风险最小化 258

 8.3 参数估计 265

 8.4 概率建模与推断 272

 8.5 有向图模型 278

 8.6 模型选择 283

9 线性回归 289

 9.1 问题表述 291

 9.2 参数估计 292

 9.3 贝叶斯线性回归 303

 9.4 最大似然作为正交投影 313

 9.5 进一步阅读 315

 10 主成分分析进行维度缩减 317

 10.1 问题设定 318

 10.2 最大方差视角 320

 10.3 投影视角 325

 10.4 特征向量计算与低秩近似 333

 10.5 高维PCA 335

 10.6 PCA实践中的关键步骤 336

 10.7 潜在变量视角 339

 10.8 进一步阅读 343

11 高斯混合模型进行密度估计 348

 11.1 高斯混合模型 349

 11.2 最大似然参数学习 350

 11.3 EM算法 360

 11.4 潜在变量视角 363

 11.5 进一步阅读 368

 12 支持向量机进行分类 370

 12.1 分割超平面 372

 12.2 原始支持向量机 374

 12.3 对偶支持向量机 383

 12.4 核函数 388

 12.5 数值解法 390

 12.6 进一步阅读 392

参考文献 395 

索引 407


# 前言

机器学习是人类知识与推理提炼成适合构建机器和工程自动化系统的形式的最新尝试之一。随着机器学习变得无处不在，其软件包变得易于使用，将低层次的技术细节抽象并隐藏起来对于从业者来说是自然且可取的。然而，这也带来了从业者可能对设计决策和机器学习算法限制变得不敏感的风险。

对成功机器学习算法背后魔法感兴趣的热情实践者目前面临着一系列令人望而却步的先决知识要求：

编程语言与数据分析工具 大规模计算及其相关框架 数学与统计学以及机器学习如何建立在其之上

在大学中，机器学习的入门课程往往在课程早期部分覆盖这些先决条件中的一些。由于历史原因，机器学习的课程往往在计算机科学系教授，学生在那里通常接受前两个知识领域方面的训练，但在数学和统计学方面则不那么深入。

当前的机器学习教科书主要专注于机器学习算法和方法论，并假设读者在数学和统计学方面具有能力。因此，这些书籍仅在一或两章中介绍背景数学，要么在书的开头，要么作为附录。我们发现许多想要深入研究基本机器学习方法基础的人，在阅读机器学习教科书所需的数学知识方面遇到了困难。在大学教授本科和研究生课程后，我们发现高中数学与阅读标准机器学习教科书所需的数学水平之间的差距对于许多人来说太大。

本书将基本机器学习概念的数学基础置于首位，并将信息收集在单一位置，以缩小或甚至消除这一技能差距。


## 为什么还需要一本关于机器学习的书？

机器学习建立在数学语言之上，用以表达那些直觉上看似明显但形式化起来却出奇困难的概念。一旦正确形式化，我们就能深入理解我们想要解决的任务。全球范围内，数学学习者的一个常见抱怨是，所涵盖的主题似乎与实际问题几乎没有关联。我们相信，机器学习是对人们学习数学的一个明显且直接的激励。

“数学在大众心中与恐惧和焦虑相连。你可能会认为我们在讨论蜘蛛。”（Strogatz，2014，第281页）

本书旨在成为现代机器学习基础数学文献的指南。我们通过直接指出数学概念在基本机器学习问题中的应用来激发学习数学的需求。为了保持书籍的简短，许多细节和更高级的概念被省略了。具备本书中介绍的基本概念，以及它们如何适应机器学习的更大背景，读者可以找到大量资源进行进一步学习，我们在各章末尾提供了这些资源。对于具有数学背景的读者，本书提供了一个简短但精确的机器学习概览。与那些专注于机器学习方法与模型（MacKay，2003；Bishop，2006；Alpaydin，2010；Barber，2012；Murphy，2012；Shalev-Shwartz和Ben-David，2014；Rogers和Girolami，2016）或机器学习的编程方面（Müller和Guido，2016；Raschka和Mirjalili，2017；Chollet和Allaire，2018）的书籍不同，我们只提供了四个代表性的机器学习算法示例。相反，我们专注于模型背后的数学概念。我们希望读者能够深入理解机器学习的基本问题，并将使用机器学习时出现的实际问题与数学模型的基本选择联系起来。

我们的目标不是编写一本经典的机器学习书籍。相反，我们的意图是为四个中心机器学习问题提供数学背景，以使阅读其他机器学习教科书变得更加容易。


## 目标读者是谁？

随着机器学习在社会中的应用日益广泛，我们相信每个人都应该对机器学习的底层原理有所了解。本书以学术数学风格编写，这使我们能够精确地阐述机器学习背后的概念。我们鼓励不熟悉这种看似简练风格的读者坚持下去，并牢记每个主题的目标。我们在文本中穿插了评论和备注，希望能为理解整体图景提供有用的指导。

本书假设读者具备高中数学和物理课程中通常涵盖的数学知识。例如，读者应该在之前见过导数和积分，以及二维或三维的几何向量。从这些基础出发，我们对这些概念进行了泛化。因此，本书的目标读者包括本科生、夜校学生和参加在线机器学习课程的学习者。

类比音乐，人们与机器学习的互动可以分为三种类型：

敏锐的听众 机器学习的普及，通过提供开源软件、在线教程和基于云的工具，使得用户不必担心管道的具体细节。用户可以专注于使用现成的工具从数据中提取见解。这使得非技术领域专家也能从机器学习中受益。这类似于听音乐；用户能够选择并辨别不同类型的机器学习，并从中受益。更有经验的用户像是音乐评论家，询问关于机器学习在社会中的应用的重要问题，如伦理、公平性和个人隐私。我们希望本书能为思考机器学习系统的认证和风险管理提供基础，并让他们利用自己的领域专业知识构建更好的机器学习系统。

经验丰富的艺术家 机器学习的熟练实践者可以将不同的工具和库插入分析管道中。典型的实践者可能是数据科学家或工程师，他们理解机器学习接口及其用例，并能够从数据中做出出色的预测。这类似于演奏家演奏音乐，高度熟练的实践者可以使现有乐器焕发活力，并为观众带来愉悦。使用本书中呈现的数学作为入门，实践者将能够理解他们最喜欢的方法的优点和局限性，并扩展和泛化现有的机器学习算法。我们希望本书能激励对机器学习方法进行更严格和原则性的开发。

初出茅庐的作曲家 随着机器学习应用于新领域，机器学习方法的开发者需要开发新方法并扩展现有算法。他们通常是研究人员，需要理解机器学习的数学基础并揭示不同任务之间的关系。这类似于音乐作曲家，在音乐理论和结构的规则内，创作新的和令人惊叹的作品。我们希望这本书能为那些想成为机器学习作曲家的人提供一个对其他技术书籍的高层次概览。社会迫切需要新的研究人员，他们能够提出并探索新颖的方法来解决从数据中学习的众多挑战。


## 致谢  

我们对许多审阅了书稿初稿并忍受了概念痛苦阐述的人表示衷心的感谢。我们尝试采纳了我们并不强烈反对的他们的想法。特别要感谢Christfried Weber，他仔细阅读了书中的许多部分，并对结构和呈现方式提出了详细建议。许多朋友和同事也慷慨地在每一章的不同版本上提供了他们的时间和精力。我们很幸运能够从在线社区的慷慨中受益，他们通过https://github.com 提出了改进意见，极大地提高了这本书的质量。

以下人员发现了错误，提出了澄清，并建议了相关文献，他们通过https://github.com 或个人沟通进行了贡献。他们的名字按字母顺序排列。

Abdul-Ganiy Usman Ellen Broad Adam Gaier Fengkuangtian Zhu Adele Jackson Fiona Condon Aditya Menon Georgios Theodorou Alasdair Tran He Xin Aleksandar Krnjaic Irene Raissa Kameni Alexander Makrigiorgos Jakub Nabaglo Alfredo Canziani James Hensman Ali Shafti Jamie Liu Amr Khalifa Jean Kaddour Andrew Tanggara Jean-Paul Ebejer Angus Gruen Jerry Qiang Antal A. Buss Jitesh Sindhare Antoine Toisoul Le Cann John Lloyd Areg Sarvazyan Jonas Ngnawe Artem Artemev Jon Martin Artyom Stepanov Justin Hsi Bill Kromydas Kai Arulkumaran Bob Williamson Kamil Dreczkowski Boon Ping Lim Lily Wang Chao Qu Lionel Tondji Ngoupeyou Cheng Li Lydia Knüfing Chris Sherlock Mahmoud Aslan Christopher Gray Mark Hartenstein Daniel McNamara Mark van der Wilk Daniel Wood Markus Hegland Darren Siegel Martin Hewing David Johnston Matthew Alger Dawei Chen Matthew Lee  

Maximus McCann Shakir Mohamed Mengyan Zhang Shawn Berry Michael Bennett Sheikh Abdul Raheem Ali Michael Pedersen Sheng Xue Minjeong Shin Sridhar Thiagarajan Mohammad Malekzadeh Syed Nouman Hasany Naveen Kumar Szymon Brych Nico Montali Thomas Bühler Oscar Armas Timur Sharapov Patrick Henriksen Tom Melamed Patrick Wieschollek Vincent Adam Pattarawat Chormai Vincent Dutordoir Paul Kelly Vu Minh Petros Christodoulou Wasim Aftab Piotr Januszewski Wen Zhi Pranav Subramani Wojciech Stokowiec Quyu Kong Xiaonan Chong Ragib Zaman Xiaowei Zhang Rui Zhang Yazhou Hao Ryan-Rhys Griffiths Yicheng Luo Salomon Kabongo Young Lee Samuel Ogunmola Sandeep Mavadia Yu Lu Sarvesh Nikumbh Yun Cheng Sebastian Raschka Yuxiao Huang Senanayak Sesh Kumar Karri Zac Cranko Seung-Heon Baek Zijian Cao Shahbaz Chaudhary Zoe Nolan  

通过GitHub做出贡献，但其真实姓名未在GitHub个人资料中列出的人员包括：

SamDataMad insad empet bumptiousmonkey HorizonP victorBigand idoamihai cs-maillist 17SKYE deepakiim kudo23 jessjing1995  

我们同样非常感谢Parameswaran Raman和剑桥大学出版社组织的许多匿名审阅者，他们阅读了早期版本手稿的一章或多章，并提供了建设性的批评，导致了显著的改进。特别要提到Dinesh Singh Negi，他是我们的L T X支持者，对与L T X相关的问题提供了详细和及时的建议。最后但同样重要的是，我们非常感谢我们的编辑Lauren Cowles，她耐心地引导我们完成了这本书的孕育过程。


符号表
![](images/58c4b4470c8ba7df60b41eef0086ca8eb0b7f01929bfa2a310d742301f120134.jpg)

缩写和首字母缩略词表
![](images/e36a810ed91e457bcc020a4f9c0dd424e88147b055e521534c39c196e04ac975.jpg)

![](images/5aa55cc8da6080046a6add321787ae2592c171b68f51165990bb79b2176e6633.jpg)
