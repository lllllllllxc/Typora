### Prior Based Pyramid Residual Clique Network for Human Body Image Super-Resolution

#### 摘要

本文旨在通过学习有效的特征表示和利用有用的人体先验知识，将微小的人图像超分辨率到高分辨率的对应图像。

首先，我们提出残差团块(RCB)来充分利用图像超分辨率(SR)的紧凑特征表示。其次，将一系列RCBs以粗到精的方式级联，构建金字塔残差团网络(PRCN)，该网络在一个前馈通道中同时重构多个SR结果(如2×、4×和8×)。第三，利用人体解析图作为形状先验，均匀离散曲线变换(UDCT)的高频子带作为纹理先验，增强重构人体图像的细节。

#### 介绍

近年来，基于卷积神经网络(CNN)的方法在图像SR方面表现出了优异的性能。

通常，基于CNN的SR方法由两个关键组件组成:特征提取模块和上采样模块，这两个模块对SR性能都有很大的影响。特征提取模块通常包含一组相同的特征提取块。一些先前的研究[5,6]表明，深度cnn导致更好的性能;然而，它们消耗了太多的内存和计算时间。

在本文中，我们的目标是以更低的计算成本达到更好的性能。

我们的工作主要有以下贡献:

(1)我们提出了一种**新的特征提取块，即RCB**，它充分利用了LR图像的分层特征。它将团块与残余连接相结合，允许丰富的信息在低层和高层之间流动，并通过将各层递归到一个团中来减少参数。我们采用反卷积层构建上采样模块，并基于一组级联的RCBs构建了一个高效的金字塔网络，我们构建了一个有效的PRCN，该PRCN对人体进行了先验估计，并以粗到精的方式重建了HR图像。

(2)我们的模型估计了两种类型的人体先验:人体解析图作为形状先验，UDCT的高频子带作为纹理先验。

(3)实验结果表明，我们的方法在PSNR和SSIM方面优于最先进的方法。此外，我们通过姿态估计和人工解析任务验证了SR图像的质量，在这些任务中我们观察到一致的改进。

#### 本方法

![image-20230901090219055](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901090219055.png)

总结：我们提出了一种基于CNN和先验知识的人体SR方法。具体来说，我们的方法由图像重建和先验估计分支组成，如图1所示。图像重建分支以LR人体图像为输入，逐步重建不同金字塔层次的SR结果。人体先验估计分支对人体先验进行估计，并将其作为有用的线索注入图像重建分支。我们的模型可以端到端的方式进行训练，并且上述两个分支彼此紧密协作。

![image-20230901090920168](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901090920168.png)

式中，**α为超参数，θ为待优化的模型参数，L<sub>Image</sub>和L<sub>Prior</sub>分别为图像重构和先验估计的损失函数**。

##### Image Reconstruction Branch

图像重建分支为S级PRCN结构(即S=3)。

设**X为输入图像，R (s)和Y (s)分别为第s层的预测残差图像和重构图像**。

假设在第s层有N个RCB，则第N个RCB的输出可以写成:

![image-20230901092941929](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901092941929.png)

其中**H <sup>s</sup><sub>n</sub> (n=1,2,...,N)表示第N个RCB在第s层的运行功能**。在第一个金字塔层，我**们使用双三次插值操作将X的分辨率增加一个比例因子T，从而获得上采样图像X<SUP>(1)</SUP>**。然后，我们执行元素求和，将X<SUP>(1)</SUP>和R<sup>(1)</sup>组合如下:

![image-20230901093741593](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901093741593.png)

其中D1表示第一级的转置层(反卷积层)，用于将特征映射的大小增加2倍。

值得注意的是，上采样尺度因子X<sup> (s)</sup>与金字塔水平s之间的关系是:**T= 2<sup>s</sup>**

我们首先使用拼接块从人类先验图像和前一个金字塔级别的残差图像中提取特征。

然后，我们可以得到第二层和第三层的重构图像Y(2)和Y(3)，如下所示:

![image-20230901095242939](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901095242939.png)

其中，P<sup>(1)</sup>和P<sup>(2)</sup>为先验估计分支估计出的先验知识，C<sup>2</sup>和C<sup>3</sup>分别表示第二层和第三层拼接块的操作。

最小化损失函数L<sub>Image</sub>的组合如下:

![image-20230901095708580](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901095708580.png)

其中，K为训练样本个数，Y <sup>(s)</sup>和Y <sup>(s)</sup>_hat分别为第s层金字塔层的重建图像和ground-truth HR图像。

由于形状和纹理特征的分布相似，所以除了每个金字塔层的最后两层之外，所有特征都在两个任务之间共享。具体来说，我们使用两个1×1卷积层分别预测两种先验知识。假设先验估计分支中有J个RCB，则预测的纹理先验P <sup>(s)</sup><sub> t</sub>和形状先验P <sup>(s)</sup><sub> s</sub>可以表示为:

![image-20230901102751321](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901102751321.png)

其中H<sup>P</sup> <sub>J</sub> (j = 1,2,...,J)表示先验估计分支中第J个特征提取块的操作函数，O<SUB>t</sub>和O<sub>s</sub>分别表示1×1卷积层对纹理和形状先验估计的操作。根据前面的工作[13]，我们通过逐元素乘法将P <sup>(s)</sup><sub> t</sub>和P<sup> (s)</sup> <sub>s</sub>组合为如下所示:

![image-20230901103142259](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901103142259.png)

式中，M为通道数(即p的语义类别数)。我们的先验估计分支能够同时产生尺度因子为2×和4×的纹理和形状先验知识。先验估计分支的损失函数L<sub>Prior</sub>可表示为:

![image-20230901104050708](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901104050708.png)

式中P <sup>(s)</sup>和P<sup> (s)_hat</sup>分别表示第s金字塔层估计的先验映射和groundtruth标签。

##### RCB

![image-20230901101108598](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901101108598.png)

(a) RCB的结构，其中Fin和Fout表示RCB的输入和输出，**不同阶段颜色相似的3×3卷积层共享模型参数。**在每个RCB中，我们允许使用红色和黑色箭头将信息从低级层传递到高级层，并使用绿色箭头将信息从高级层传递到低级层。(b) RCB的信息流，箭头为卷积运算，F为卷积层提取的特征映射。

**与Clique块不同的是**，我们在第一阶段的每个3×3卷积层之前都采用1×1卷积层，使得输入特征映射的数量与第二阶段对应的3×3卷积层相同。这种设计允许在不同阶段共享相应3×3卷积层(图2a中具有不同颜色和标签的块)的权重，以一种新颖的循环方式更新RCB中的卷积层。

我们还在RCB结构中加入了局部特征融合和局部残差学习。**局部特征融合**是指在RCB的第二阶段使用1×1卷积层自适应融合来自卷积层的精细信息。

然后，我们应用**局部残差学习**进一步提高网络的表示能力。

在第二阶段，一个特征映射的输出作为另一个特征映射的输入，促进信息流的最大化。设Fin和Fout分别表示RCB的输入和输出，F <sup>1×1</sup> <sub>i</sub> (i = 1,...,4)表示第i个1×1卷积层提取的特征图，F (1) i和F (2) i分别表示第i个3×3卷积层在第一阶段和第二阶段提取的特征图。在第一阶段，特征映射的前馈传递可表示为:![image-20230901143752981](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901143752981.png)

其中*为卷积运算，W表示对应卷积层的权值，σ表示激活函数。在第二阶段，特征映射的前馈传递可以写成:![image-20230901144246354](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901144246354.png)

![image-20230901144322053](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901144322053.png)

**(14)-(17)揭示阶段一数据的流向。**

##### 人体先验

解析图包含不同人体的不同组件，如帽子、头发、脸等，这些组件有助于用于重建的语义布局信息。

UDCT将图像分解为几个高频子带，每个子带捕获图像特定方向的详细特征。我们使用元素求和来融合所有高频子带，创建一个代表人体图像局部纹理细节的全向纹理先验。

#### 实验

##### 数据集

我们在**HumanSR数据集**上进行了实验来评估性能[13]。人体图像来自三个公共数据集:**ATR**[47]、**CIHP**[48]和**LIP**[49]数据集。HumanSR数据集的训练集由3万张图像组成，其中6500张来自CIHP, 7500张来自LIP，其余来自ATR。测试集由600张图像组成，其中ATR、CIHP和LIP三个数据集各有200张图像。

**实现细节**：我们将人体图像调整为320×160作为地面真实HR图像，并通过对HR图像进行双三次降采样来生成LR输入图像。

为了先验地生成纹理真值，我们对HR图像应用1级UDCT，生成6个高频子带。此外，我们采用HIPN模型[3]使用19个语义类估计ground truth shape prior。在图像重建和先验估计分支中，每个金字塔层都包含2个RCB。

Eq. 1中的**α**设为0.5。反卷积层对所有比例因子采用**17×17滤波器**，RCB中的每个卷积层后面都有一个Leakly Rectified Linear Unit (**LReLU**)，除了局部融合层，其负范围设置为0.05。所有层的**学习率最初设置为0.0001**，每100个epoch减半。总训练周期设置为240。我们使用**Adam**作为我们的优化器。

**评价指标**：按照标准方案，我们采用峰值信噪比(PSNR)和结构相似度(SSIM)[50]作为评价指标。此外，我们利用特征相似度(FSIM)[51]、视觉信息保真度(VIF)[52]和学习感知图像补丁相似度(LPIPS)[53]来进一步评估不同方法的重建性能。在人工解析和姿态估计方面，我们遵循先前的工作[49,13]，分别采用相交-超并度(IoU)和正确关键点相对于头部的百分比(PCKh)作为评价指标。值得注意的是，LPIPS指标值越低，表示重建图像的感知相似性越高，而其他指标值越高，表示性能越好。

#### 消融实验

**Evaluation on Key Settings of RCB.**

我们首先研究了生长速率(即，表示为G的3×3滤波器的数量)和每个RCB的卷积层数(表示为N)的影响。我们使用N和G的不同组合构建了图像重建分支。图5中，我们在HumanSR测试集上评估了相对于PSNR的SR性能。

![image-20230901164504514](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901164504514.png)

我们探讨了RCB连接策略对SR性能的影响。

![image-20230901164849470](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901164849470.png)

**Effectiveness of RCB.**

五种网络的上采样模块是相同的，而五种网络的特征提取模块包含不同的特征提取块，每个网络只包含一个块。

![image-20230901165111333](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901165111333.png)

**Impact of Prior Knowledge.**

(1)在图7中，我们首先比较了不同先验对人体图像4倍SR的有效性。

![image-20230901165348940](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901165348940.png)

值得注意的是，对比方法[22,25,13]需要分别训练三种不同的模型来处理2x、4x和8x的SR，而我们的PCRN能够通过单个PCRN模型处理多个上采样尺度。从表2中可以看出，使用我们的先验知识，所有的方法都有明显的改进，我们的PCRN优于其他参数更少、速度更快的方法。

![image-20230901165549899](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901165549899.png)

为了公平比较，我们使用上述方法发布的代码，在同一个HumanSR训练集上重新训练所有模型。

![image-20230901170049210](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901170049210.png)

![image-20230901170107538](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230901170107538.png)
