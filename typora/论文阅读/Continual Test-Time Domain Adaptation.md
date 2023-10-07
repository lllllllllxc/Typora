### Continual Test-Time Domain Adaptation

https://qin.ee/cotta.

#### 摘要

![image-20230920121632997](C:\Users\35106\AppData\Roaming\Typora\typora-user-images\image-20230920121632997.png)

造成这一现象的**原因**有两个方面:首先，在不断变化的环境下，由于分布的移位，伪标签变得更杂讯和误校准[13]。因此，早期的预测错误更容易导致误差积累[4]。其次，由于模型在很长一段时间内不断适应新的分布，来自源领域的知识很难保存。

由于平均教师预测通常比标准模型具有更高的质量[55]，我们使用**加权平均教师模型来提供更准确的预测**。另一方面，**对于具有较大域间隙的测试数据，我们使用增强平均预测**来进一步提高伪标签的质量。

#### Introduction

测试-时域自适应的目的是通过在推理期间学习未标记的测试(目标)数据来适应源预训练模型。

![image-20230920141102414](C:\Users\35106\AppData\Roaming\Typora\typora-user-images\image-20230920141102414.png)

目标数据是按顺序提供的，并且来自不断变化的环境。**使用现成的源预训练网络来初始化目标网络**。该模型基于当前目标数据在线更新，并以在线方式给出预测结果。目标网络的自适应不依赖于任何源数据。

**预测和更新是在线执行的，这意味着模型只能访问当前的数据流，而不能访问完整的测试数据或任何源数据。所提出的设置与现实世界的机器感知系统非常相关。**例如，对于自动驾驶系统来说，周围环境是不断变化的(例如，天气从晴天变为多云，然后变为雨天)。它们甚至可以突然改变(例如，当一辆汽车驶出隧道时，相机突然过度曝光)。感知模型需要在这些非平稳的域位移下进行自我调整和在线决策。

<u>值得指出的是，我们的方法很容易实现。权重加增平均策略和随机恢复可以很容易地结合到任何现成的预训练模型中，而无需在源数据上重新训练它。</u>

**推理期间指的是模型进行预测或生成输出时的阶段**。在无监督域适应中，源数据是可以在推理期间使用的，因为模型在学习阶段可以完全访问和利用源数据。然而，在隐私问题或法律限制下，源数据通常在推理期间不可用，意味着模型不能直接访问或使用源数据来进行预测或生成输出。

#### 本方法

##### 问题定义

我们在表1中列出了在线连续测试时间适应设置与现有适应设置之间的主要区别。 

![image-20230920150552389](C:\Users\35106\AppData\Roaming\Typora\typora-user-images\image-20230920150552389.png)

在源数据(X<SUP>S</sup>, Y<sup> S</sup>)**上训练参数为**θ的已有预训练模型f<sub>θ<sub>0</sub></sub>(x)，

未标记的目标域数据X<sup>T</sup>按顺序提供，模型只能访问当前时间步长的数据。在时间步长t处，提供目标数据X<SUP>T</SUP><sub>t</sub>作为输入，模型f<sub>θ<sub>t</sub></sub>需要做出预测f<sub>θ<sub>t</sub></sub>(X<SUP>T</SUP><sub>t</sub>)，根据未来输入θ<sub>t</sub>−→θ<sub>t+1</sub>进行自我调整。X<SUP>T</SUP><sub>t</sub>的数据分布是不断变化的。基于在线预测对模型进行评估。

##### 方法论

![image-20230920155729032](C:\Users\35106\AppData\Roaming\Typora\typora-user-images\image-20230920155729032.png)

![image-20230920143917649](C:\Users\35106\AppData\Roaming\Typora\typora-user-images\image-20230920143917649.png)

![IMG_20230921_084203](D:\githubre\Typora\typora\图片\IMG_20230921_084203.jpg)

**源模型**：这需要对源数据进行重新训练，并且不可能重用现有的预训练模型。在我们提出的测试时间适应方法中，我们解除了这个负担，并且不需要修改体系结构或额外的源培训过程。因此，任何现有的预训练模型都可以使用，而无需对源进行再训练。

**权重平均伪标签**：给定目标数据X<SUP>T</SUP><sub>t</sub>和模型f<sub>θ<sub>t</sub></sub>，自训练框架下常见的测试时间目标是最小化预测y^<sup>T</sup><sub>t</sub> = f<sub>θ<sub>t</sub></sub> (X<SUP>T</SUP><sub>t</sub>)与伪标签之间的交叉熵一致性。

我们使用**加权平均教师模型f<sub>θ'</sub>来生成伪标签**。在时间步长t = 0时，将教师网络初始化为与源预训练网络相同。在时间步长t处，伪标签首先由教师生成y<sup>ˆ′</sup><sup>T</sup><sub>t</sub> =f<sub>θ'</sub>(X<SUP>T</SUP><sub>t</sub>)

学生f<sub>θ<sub>t</sub></sub>通过学生和教师预测之间的交叉熵损失来更新:

![image-20230920153625648](C:\Users\35106\AppData\Roaming\Typora\typora-user-images\image-20230920153625648.png)

其中，y<sup>ˆ′</sup><sup>T</sup><sub>tc</sub>为教师模型软伪标签预测中c类出现的概率，y<sup>ˆ</sup><sup>T</sup><sub>tc</sub>为主模型(学生)的预测。这种损失加强了教师和学生预测之间的一致性。

在使用公式1更新学生模型θ<sub>t</sub>−→θ<sub>t+1</sub>后，我们使用学生权重通过指数移动平均更新教师模型的权重:

![image-20230920153956127](C:\Users\35106\AppData\Roaming\Typora\typora-user-images\image-20230920153956127.png)

其中α是平滑因子。我们对输入数据X<SUP>T</SUP><sub>t</sub>的最终预测是y<sup>ˆ′</sup><sup>T</sup><sub>t</sub>中概率最高的类别。

**增广平均伪标签**：在不断变化的环境下，测试分布可能发生巨大变化，这可能使增强策略无效。在这里，我们考虑了测试时间的域移，并通过预测置信度近似域差。仅在域差较大时进行增广，以减少误差累积。

![image-20230920154543272](C:\Users\35106\AppData\Roaming\Typora\typora-user-images\image-20230920154543272.png)

其中，y<sup>˜′T</sup><sub>t</sub>是来自教师模型的增强平均预测，y<sup>ˆ′</sup><sup>T</sup><sub>t</sub>是来自教师模型的直接预测，conf(f<sub>θ<sub>0</sub></sub>(X<SUP>T</SUP><sub>t</sub>))是源预训练模型对当前输入X<SUP>T</SUP><sub>t</sub>的预测置信度，p<sub>th</sub>是置信度阈值。通过使用公式4中的预训练模型f<sub>θ<sub>0</sub></sub>计算当前输入X<SUP>T</SUP><sub>t</sub>上的预测置信度，我们试图近似源和当前域之间的域差。

**我们假设较低的置信度表示较大的领域差距，较高的置信度表示较小的领域差距**。因此，当置信度较高且大于阈值时，我们直接使用y<sup>ˆ′</sup><sup>T</sup><sub>t</sub>作为伪标签，不使用任何增值。当置信度较低时，我们额外应用N个随机增广来进一步提高伪标签质量。

在具有小域间隙的自信样本上随机扩增，有时会降低模型的性能。所以过滤至关重要。

通过改进的伪标签更新学生:

![image-20230920155638107](C:\Users\35106\AppData\Roaming\Typora\typora-user-images\image-20230920155638107.png)

**随机修复**：长期自我训练的持续适应不可避免地会引入错误并导致遗忘。强烈的分布移位会导致校准错误甚至错误的预测。在这种情况下，自我训练可能只会强化错误的预测。即使新数据没有严重偏移，模型也可能因为不断的适应而无法恢复。

考虑在时间步长为t时，基于方程1的**梯度更新**后的学生模型f<sub>θ</sub>内的卷积层:

![image-20230920160147360](C:\Users\35106\AppData\Roaming\Typora\typora-user-images\image-20230920160147360.png)

其中*表示卷积运算，x<sub>l</sub>和x<sub>l+1</sub>表示该层的输入和输出，W<sub>t+1</sub>表示**可训练的卷积滤波器**。

本文提出的随机恢复方法通过以下方式对权值W进行更新:

![image-20230920160533254](C:\Users\35106\AppData\Roaming\Typora\typora-user-images\image-20230920160533254.png)

其中⊙表示逐元素的乘法。p是一个小的恢复概率，M是与W<sub>t+1</sub>形状相同的掩模张量，Bernoulli表示伯努利分布。掩码张量决定W<sub>t+1</sub>中的哪个元素要恢复到源权重W<sub>0</sub>。如果M中对应位置的值是1，那么W<sub>t+1</sub>中的对应元素将从源权重W<sub>0</sub>中恢复；如果M中对应位置的值是0，那么W<sub>t+1</sub>中的对应元素将保持不变。

通过随机地将可训练权值中的少量张量元素恢复到初始权值，避免了网络偏离初始源模型太远，从而避免了灾难性遗忘。此外，通过保留源模型的信息，我们能够训练所有可训练的参数，而不会遭受模型崩溃的痛苦。

<<<<<<< Updated upstream


数据集
=======
#### 实验
>>>>>>> Stashed changes

1.配置环境

environment.yml里面的括号后面的编号删除。

**`RobustBench`**手动安装。

segment也得下载和配置环境（# Example logs are included in ./example_logs/base.log, tent.log, and cotta.log.）。

模型动物园：

要使用模型库中的模型，您可以在下表中找到所有可用的模型 ID。

download.sh：下载并解压

2.内容

<<<<<<< Updated upstream
=======
**目的**：

夜间语义分割任务

>>>>>>> Stashed changes
上面比较

**代码**

robustbench：标准化的对抗性稳健性基准，对标auto Attack，使用攻击来检测。

data

jpeg格式的图片和对应的标签

leader board

输出的结果是一个包含排行榜信息的 HTML 表格的字符串。

model zoo

调用模型，直接从命令行从模型库重现模型评估。

**文件类型**

运行.sh：配置环境、调用

.yaml:超参

.py:结构体和函数

我的实验

DA里，源域cityscapes，目标域Zurich（属于cityscapes）

预训练模型（骨干网络）：refinenet

一般都是拿这5000张精细标注(gt fine)的样本集来进行训练和评估的。当然，还有一个策略就是，先对粗糙标注的样本集进行一个简单的训练，然后再基于精细标注的数据集进行final training。 这里我们只谈gt fine样本集的训练。

[语义分割学习系列（三）cityscapes数据集介绍_cityscape19个类转为34个类-CSDN博客](https://blog.csdn.net/avideointerfaces/article/details/104139298?ops_request_misc=&request_id=&biz_id=102&utm_term=cityscapes数据集&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-104139298.nonecase&spm=1018.2226.3001.4187)
