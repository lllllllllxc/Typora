### Source-free Video Domain Adaptation by Learning Temporal Consistency for Action Recognition

#### 摘要

**基于视频的无监督域适应（VUDA）**：这些方法需要在适应过程中不断访问源数据。然而，在许多实际应用中，源视频域中的主题和场景应该与目标视频域中的主题和场景无关。并且，源视频数据属于隐私，难以获取。

在本文中，我们提出了一种新的关注**时间一致性网络(ATCoN)**，通过学习时间一致性来解决SFVDA（Source-Free Video-based Domain Adaptation）问题，该网络由跨局部时间特征执行的两个新的一致性目标(即特征一致性和源预测一致性)保证。

#### 介绍

为了避免昂贵的数据注释，各种基于视频的无监督域自适应(VUDA)方法被引入，通过减少源视频域和目标视频域之间的差异，将知识从标记的源视频域转移到未标记的目标视频域。

该方案只提供训练良好的源视频模型和未标记的目标领域数据进行自适应。

因此，整体时间特征可能包含模糊的语义信息，并且不会具有区别性。相反，我们假设对于源视频，提取的局部时间特征不仅具有区别性，而且彼此之间一致，具有相似的特征分布模式，这意味着相似的语义信息。这种假说被称为**跨时间假说**。我们的方法被设计成局部时间特征在特征表示上是一致的，这将导致相应的整体时间特征是有效的和有区别的。

由于只有具有源分类器的源模型可用于自适应，因此目标数据与源数据分布的相关性与目标数据在源分类器上的预测高度相关。

因此，为了更好地使目标时间特征适应源分类器，相应的局部时间特征与源数据分布的相关性也应该保持一致。这种一致性可以解释为局部时间特征相对于固定源分类器的源预测一致性。此外，为了提高视频特征的可分辨性，需要将局部时间特征精心组合，构建整体时间特征。

**ATCoN的目的是通过学习由特征一致性和源预测一致性组成的时间一致性，获得满足跨时间假设的有效的、有判别性的整体时间特征。ATCoN通过关注具有高源预测置信度的局部时间特征，进一步将目标数据与源数据分布对齐，而无需访问源数据。**

#### 结论

制定了具有挑战性但现实的无源视频域适应(SFVDA)问题，该问题解决了视频中的数据隐私问题。我们提出了一种新的ATCoN来有效地解决SFVDA。最后证明了优越性

#### 相关工作

尽管图像的SFDA研究取得了进展，但SFVDA尚未得到解决。

#### 本方法

在无源视频域自适应(SFVDA)场景中，我们只得到一个由空间特征提取器G<sub>S、sp</sub>、时间特征提取器G<sub>S,t</sub>和分类器H<sub>S</sub>，以及一个未标记的目标域D<sub>T</sub> = {V<sub>iT</sub>} <sup>nT</sup> <sub>i=1</sub>，具有n<sub>T</sub>以p<sub>T</sub>的概率分布为特征的 i.i.d视频。源模型通过训练其参数θ<sub>S</sub>、s<sub>p</sub>、θ<sub>S</sub>、t、θ<sub>H</sub>生成，标记的源域D<sub>S</sub> = {(V<sub>iS</sub>, y<sub>iS</sub>)} <sup>nS</sup><sub> i=1</sub>，包含n<sub>S</sub>个视频。我们假设标记的源域视频和未标记的目标域视频共享相同的C类，但在将源模型调整为D<sub>T</sub>时，D<sub>S</sub>是不可访问的。

![image-20230904144416189](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230904144416189.png)

(V <sup>(r)</sup> <sub>iS</sub>)<sub>m</sub> = {f<sup> (a)</sup><sub> iS</sub>, f<sup>(b)</sup> <sub>iS</sub>，…}<sub>m</sub>是具有r个时序帧的第m个clip。a和b是帧序号。 积分函数g<sup>(r)</sup><SUB>S</sub>

最终的整体时间特征t<sub>iS</sub>是应用于所有局部时间特征的简单平均聚合,

![image-20230904145507918](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230904145507918.png)

通过在t<sub>iS</sub>上应用源分类器H<sub>S</sub>进一步计算源预测。源模型以标准交叉熵损失作为目标函数进行训练，其公式为:

![image-20230904145913379](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230904145913379.png)

其中σ是softmax函数，其C -th元素定义为σ<sub>c</sub>(x) =

![image-20230904150155996](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230904150155996.png)

为了使源模型更具判别性和可转移性，从而更好地对目标数据进行对齐，我们进一步采用了标签平滑技术[35]，使提取的特征分布在均匀分离的紧密聚类中。进一步表示为：

![image-20230904150410003](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230904150410003.png)

y ' <sub>iS</sub>是平滑的标签

![image-20230904150528522](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230904150528522.png)

ϵ为平滑参数，值设为0.1.
