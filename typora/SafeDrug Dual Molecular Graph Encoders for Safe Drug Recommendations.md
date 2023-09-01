### SafeDrug: Dual Molecular Graph Encoders for Safe Drug Recommendations

缩写和名词

**proportional-integral-derivative (PID)**：PID是一种控制系统，它根据误差信号的大小和变化率来计算输出信号，以使系统的控制变量达到期望值。PID控制器由三个部分组成：比例，积分和微分。比例部分根据误差信号的大小产生输出信号，积分部分根据误差信号的积累产生输出信号，微分部分根据误差信号的变化率产生输出信号。这三个部分的输出信号相加形成最终的PID控制器输出信号，用于调整控制变量。

**Multi-Hot**编码中，向量的每个元素仍然表示一个特征是否存在，但每个向量可以包含多个非零元素，表示该向量包含多个特征。

#### Abstract

（1）仅靠EHR是有限制的，首先很多重要数据没有考虑其中，其次DDI被隐性考虑，导致结果不是最优。

（2）SafeDrug采用**全局消息传递神经网络(MPNN)**模块和**局部双向学习模块**，对药物分子的连通性和功能性进行全面编码。SafeDrug还具有**可控损失函数**，可有效控制推荐用药组合中的DDI水平。

（3）SafeDrug需要的参数少，运行快，指标好。

#### Introduction

##### 不足：

**Inadequate Medication Encoding**：每个药物被认为是一个(二进制)单元，忽略了药物在其有意义的分子图表示中编码重要的药物特性，如疗效和安全性。此外，分子的亚结构与功能相关。

**Implicit and Non-controllable DDI Modeling**：现有的一些研究通过软约束或间接约束来模拟药物-药物相互作用

##### 贡献：

safeddrug模型首先学习患者表征，并将其输入**双分子编码器**，以捕获药物的全局药理学特性和局部亚结构模式。全局上，通过MPNN进行层层传递信息，便于获取功能信息。局部上，将药物分成亚结构。在这项工作中，亚结构表示被馈送到一个有效的掩模神经网络中。模型的最后输出是通过对全局和局部编码嵌入进行逐元素集成得到的。

在训练过程中，如果单个样本的**DDI率**超过一定的阈值/目标，则负DDI信号将被强调并反向传播。在实验中，自适应梯度下降可以平衡模型精度和最终DDI水平。safedrug模型具有预设的 target，能够提供可靠的药物组合，满足不同水平的DDI要求。

####  Related Works

药物推荐分为两大类： **instance-based and longitudinal recommendation**
**methods。**前者注重病人目前的健康状况，如LEAP。后者注重病史中的时间依赖，要么将最终的药物组合建模为多标签二元分类，要么通过顺序决策。如RETAIN和GAMENet。

早期，分子描述符和药物指纹常用来表示药物分子。深度学习模型是近年来发展起来的，用于学习**分子表示**和模拟分子子结构(一组连接的原子)。本文提出了在推荐过程中同时捕获分子全局和局部信息的safedrug。

#### Problem Formulation

一个病人j的**EHR**可以表示为一个序列

![image-20230819161321042](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230819161321042.png)

![image-20230819161516047](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230819161516047.png)

Vj是病人数量，分别表示multi-hot diagnosis, procedure and medication vectors。

我们使用对称二进制邻接矩阵表示**DDI关系**。

![image-20230819163402203](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230819163402203.png)

Dij = 1表示已经报道了药物i和药物j之间的相互作用，而Dij = 0表示存在安全的共处方。目标函数由两部分组成:(i)提取真实的药物组合m(t)，作为惩罚m^(t)(药物推荐)的监督;(ii)利用矩阵D导出m^ (t)上的无监督DDI约束。

#### The SafeDrug Model

![image-20230819164108302](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230819164108302.png)

**SafeDrug 模型**由四个部分组成：患者表示模块、全局消息传递神经网络 (MPNN) 编码器、二分编码器和药物表示模块。 -患者表现模块从 EHR 数据中学习患者表现，而 MPNN 编码器则量化患者表现与药物表现之间的相似性。双部分编码器对药物的分子亚结构功能进行编码，药物表示模块结合了全球和局部药物表示载体，以获得最终的药物产出。

**Diagnosis Embedding**

-**嵌入表**以 Ed 表示，这是一个维度为 r|d|×dim 的矩阵，其中 |D| 表示唯一诊断的数量，dim 表示嵌入空间的维度。 -嵌入表的每一行都存储用于特定诊断的嵌入向量，该向量本质上是诊断的数字表示。 -为了将多热诊断向量 d (t) {0, 1} |D| 投影到嵌入空间，在诊断向量和嵌入表之间执行向量矩阵点积。此操作对每个诊断嵌入进行求和，从而为整个诊断向量生成一个嵌入向量。 -<u>在训练期间，嵌入表 Ed 是可以学习的，并且可以在每次就诊和每位患者之间共享。这意味着所有患者和就诊都使用相同的嵌入表，并且在训练期间会更新该表以提高模型的准确性。</u> -使用嵌入表允许模型捕获不同诊断之间的关系，并以模型可以轻松处理的数字格式表示它们。 

![image-20230819193149945](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230819193149945.png)

**Procedure Embedding**

这是一种使用共享过程表 Ep 对关联的过程向量 p (t) 进行编码的方法。 -手术向量 p (t) 是一个多热向量，代表患者当前的健康状况。 -程序表 Ep 是一个大小为 r|p|×dim 的矩阵，其中 dim 是嵌入维度，|P| 是过程集的基数。 -嵌入过程包括将程序向量 p (t) 与程序表 Ep 相乘以获得嵌入向量 p (t) e。 -嵌入向量 d (t) e 和 p (t) e 编码患者当前的健康状况，但健康快照（简单印象）可能不足以做出治疗决策。 -为了对动态患者病史进行建模，使用两个单独的 RNN 分别获得隐藏的诊断和手术向量 d (t) h 和 p (t) h。 -如果其中一个序列在实践中可能无法访问，则 RNN 模型将当前嵌入向量和之前的隐藏向量作为输入来生成当前隐藏向量。 -通过将诊断嵌入 d (t) h 和嵌入 p (t) h 的程序串联到双长向量中，然后应用带有参数 W1 的前馈神经网络 NN1 (·) 来降低维度，从而获得患者表征 h (t)。 -生成的患者表现形式 h (t) 是一个大小为 Rdim 的紧凑向量，它全面编码了患者当前的健康状况和动态病史。 -然后，通过对药物分子数据库进行全面建模，使用该患者表述来生成安全药物推荐。

![image-20230819193222350](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230819193222350.png)

**Patient Representation**

 -患者表述是对患者的诊断和手术嵌入的简洁表示。 -诊断嵌入是指以矢量形式表示患者的诊断的过程，该过程是使用自然语言处理技术完成的。 -手术嵌入是指以矢量形式表示患者的医疗程序的过程，这也是使用自然语言处理技术完成的。 -使用串联操作将诊断和过程嵌入串联成一个双长向量。 -然后将连接的向量通过前馈神经网络 NN1 传递，该网络的参数为 W1。 -神经网络的输出是患者表示法 h (t)，它是 Rdim 中的向量。 -然后，通过对药物分子数据库进行全面建模，使用患者代表来生成安全药物推荐。 -药物分子数据库建模过程包括使用全局消息传递神经网络 (MPNN) 模块和局部双部分学习模块对药物分子的连通性和功能进行编码。 -SafeDrug模型还具有可控损失功能，可以有效控制推荐药物组合中的药物相互作用（DDI）。

![image-20230819193227365](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230819193227365.png)

**Dual Molecular Graph Encoders**

(I) Global MPNN Encoder

我们使用具有可学习指纹的消息传递神经网络(MPNN)算子对药物分子数据进行编码，旨在将单分子图上的原子信息卷积并池化为向量表示。

-收集所有出现的原子的集合，并设计了一个可学习的原子嵌入表 Ea。表格的每一行都会查找特定原子的初始嵌入/指纹。 -<u>**SafeDrug中使用的MPNN模型的设计与以前的作品不同，前者主要使用原子描述符作为初始特征。后者对于分子结构建模，原子与原子的连通性比单个原子更为重要。**</u> -药物分子图由原子表示为顶点，原子-原子键作为边缘，邻接矩阵A和来自Ea的初始原子指纹。 -制定了在图上传递的分层消息，其中使用消息函数 messaGel (·) 和顶点更新函数 updaTel (·) 以及分层参数矩阵 W (l) 更新来自每个原子邻居的编码消息。 -对L层应用消息传递后，通过读出函数汇集药物分子的全球表示形式，该函数计算所有原子指纹的平均值。 -对每个药物分子应用具有共享参数的相同的 MPNN 编码器，并将所有药物分子的 MPNN 嵌入收集到药物存储器中。

(II) Local Bipartite Encoder

使用MPNN编码器将具有相似结构的分子映射到附近的空间。然而，一对在结构域上有显著重叠的药物可能在DDI相互作用或其他功能活动方面表现不同。

Molecule Segmentation：使用分子分割方法来捕捉药物分子与其亚结构之间的关系，从而可以构建一个双部分结构来指示哪些药物含有特定的亚结构。Hij = 1表示药物j含有子结构i。

-**双向学习**是SafeDrug模型中使用的一种技术，用于根据其功能信息得出药物表征。 -该技术的输入是患者表达 h (t)，使用前馈神经网络 (NN3) 和sigmoid函数 (σ2) 将其转换为局部功能向量。 -由此产生的载体 m (t) f 量化了每种功能的重要性，可以看作是当前患者的抗疾病功能组合。 -下一步是制定涵盖所有这些抗病功能的药物建议，同时还要防止药物相互作用（DDI）。 -<u>为了实现这一目标</u>，SafeDrug模型执行网络修剪并设计了一个屏蔽的1层神经网络 (NN4)，该网络将功能向量 m (t) f 映射到相应的局部药物表示形式。 -NN4 的参数矩阵被二分架构 H 所掩盖，该架构本质上使用参数矩阵执行逐元素乘积。 -在训练过程中，NN4学会了将功能向量映射到药物表示中，同时考虑了双部分架构 H。 -由于 H 的稀疏性，掩码 H 使模型的参数少得多，从而降低了模型的计算复杂性。 -此外，maskH通过避免共同处方相互作用药物来帮助预防DDI，这在论文的附录中得到了证实

**Medication Representation**

![image-20230820152649831](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230820152649831.png)

按元素乘积用于组合全局和局部表示。这意味着将全局向量的每个元素与局部表示的相应元素相乘。 -生成的输出是一个介于 0 和 1 之间的值的向量，可以解释为每种药物被包含在推荐药物组合中的可能性。 -使用阈值来确定推荐组合中包含哪些药物。

**Model Training and Inference**

safeddrug是端到端的训练。我们同时学习了Ed、Ep、RNNd和RNNp中的参数、网络参数W1、Ea、MPNN fW(i)g中的分层参数、层归一化(LN)中的参数以及W2、W3和W4。

-<u>本文中的推荐任务被制定为药物推荐的多标签二元分类</u>。 -使用二元交叉熵 (BCE) 和多标签铰链损失（multi）作为**损失函数**，以预测药物代表性并确保稳健性。  -基于药物相互作用（DDI）邻接矩阵 D，该模型还设计了相对于输出实值药物表示的不良DDI损失，这会惩罚具有不良相互作用的药物对的预测值。 -<u>不良的 DDI 损失函数(Lddi)定义为两种药物的预测值与 DDI 邻接矩阵中相应条目的乘积之和。 -值得注意的是，上述损失函数是为单次就诊定义的，在训练期间，损失反向传播是在患者层面进行的，方法是平均所有就诊的损失。</u>

-**可控损失函数（L）**是不同损失测量项的加权和，其中 α 和 β 是预定义的超参数。 -两个预测损失项 Lbce 和 Lmulti 是兼容的，并且从验证集中选择 α。 -该模型还考虑了数据集中可能存在的不良药物相互作用 (DDI)，这些不良相互作用 (DDI) 可能会随着正确的学习而增加。 -为了平衡预测损失和 DDI 损失，模型在训练过程中使用比例积分微分 (PID) 控制器调整 β。 -当推荐药物的DDI率高于一定阈值时，比例误差信号用作负反馈。 -模型为损失函数设定了 DDI 接受率 γ，如果患者级别的 DDI 低于阈值，则只能最大限度地提高预测精度。 -可控因子 β 遵循分段策略，即如果 DDI 小于或等于 γ，则将其设置为 1，否则，对其进行自适应调整以减少 DDI。 -通过预设不同的 γ 值，该模型可以满足不同级别的 DDI 要求。 -在推理阶段，模型对输出药物表示应用阈值δ = 0.5并选择值大于阈值的药物作为最终建议。

#### Experiments

数据库和指标

数据库：**MIMIC-III**

我们使用**DDI率、Jaccard相似度、F1评分、PRAUC和药物数量**这5个疗效指标来评估推荐效果，使用**参数数量、训练时间和推理时间**这3个指标来评估模型的复杂性。

**Baselines**：我们将safeddrug与以下基线进行了不同角度的比较:标准逻辑回归(LR)、多标签分类方法:集成分类器链(ECC)、基于rnn的方法:RETAIN 、基于实例的方法:LEAP 、基于纵向记忆的方法:DMNC 和GAMENet等。我们还比较了SafeDrug和它的两个模型变体。我们使用SafeDrugL来表示只有二部编码器(局部，L)的模型，使用SafeDrugG来表示只有MPNN编码器(全局，G)的模型。

![image-20230820195503671](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230820195503671.png)

![image-20230820195714985](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230820195714985.png)

我们测试了一系列范围从0到0:08的靶向DDI γ。对于每个γ，我们训练一个单独的模型。实验共进行5次。收敛之后，我们收集并显示表4中的平均指标。

从结果来看，<u>**DDI速率在大多数情况下受到γ的控制和上限，这与设计一致。当γ变大时，在一种组合中允许使用更多的药物，safeddrug更加准确。当γ过小(< 0:02)时，模型精度将急剧下降。**</u>

![image-20230820200039463](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230820200039463.png)

一个训练周期算法

![image-20230825100851872](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230825100851872.png)

代码实现

（1）调配环境

"C:\Users\lxc\Desktop\typora\Anaconda.md"

仔细看GitHub里面的信息，包括环境条件，数据来源，这些是代码跑动的关键。

（2）processing

搞清楚各个模块的功能，并不是所有模块都要跑，都能跑。这个模块处理后的数据已经有了且原始数据比较大，处理效率低。

代码理解：

（1）名词

**mapping file**（映射文件）是一种文件格式，用于将一个数据源中的数据映射到另一个数据源中的数据。通常用于将数据从一种格式转换为另一种格式，或者将数据从一个系统移植到另一个系统。映射文件通常包含源数据和目标数据之间的映射规则，以及转换过程中需要执行的操作。常见的映射文件格式包括XML、JSON、CSV等。

**drug SMILES string dic**t是一个包含药物SMILES字符串的字典。SMILES（简化分子输入线性表示）是一种用于表示分子结构的字符串表示法，可以通过一系列规则将分子结构映射到一个字符串上。

**ICD-9编码**是国际疾病分类第九版的编码系统，用于对医学诊断和手术操作进行编码。它由世界卫生组织（WHO）制定，包含一系列的数字和字母，用于描述各种疾病、症状、伤害和手术操作。

**ATC三级代码**是药物分类体系中的一种分类方法，由世界卫生组织（WHO）制定。它将药物分为不同的类别和子类别，以便更好地管理和使用药物。ATC代码由七个字符组成，其中第一级表示药物的主要治疗作用，第二级表示药物的治疗用途，第三级表示药物的化学结构和药理作用。

**SMM**全称为Structured Multi-label Model，它将多标签分类问题转化为一个序列标注问题，可以根据患者历史就诊记录预测出患者可能需要的药物。

（2）数据

- - drug-atc.csv: drug to atc code mapping file
  - ndc2atc_level4.csv: NDC code to ATC-4 code mapping file
  - ndc2xnorm_mapping.txt: NDC to xnorm mapping file
  - id2drug.pkl: drug ID to drug SMILES string dict
- other files that generated from mapping files and MIMIC dataset (we attach these files here, user could use our provided scriots to generate)
  - data_final.pkl: intermediate result
  - ddi_A_final.pkl: ddi matrix
  - ddi_matrix.pkl: H mask structure
  - ehr_adj_final.pkl: used in GAMENet baseline
  - idx2ndc.pkl: idx2ndc mapping file
  - ndc2drug.pkl: ndc2drug mapping file
  - voc_final.pkl: diag/prod/med dictionary

（3）模型

- processing.py: is used to process the MIMIC original dataset，所以只要有处理后数据，可以不运行并且不需要原始数据。

  诊断、手术和药物信息最初记录在单独的文件中，即 “DIAGNOSES ICD.csv”、“程序 ICD.csv” 和 “PRESCRIPTIONS.csv”。 -论文分别提取这些文件，然后按患者身份证和就诊证合并。 -合并后，诊断和程序采用 ICD-9 编码，它们将在训练前转换为多热向量。 -本文从TWOSIDES中提取了按TOP-40严重程度类型划分的药物相互作用（DDI）信息，这些信息由ATC三级代码报告。 -然后，该论文将NDC药物编码转换为相同的ATC级别代码以进行集成。 -在实现中，本文还计算了药物级MPNN嵌入物的平均值，并汇总了相应ATC三级代码的子结构。 -有关分子连接信息，该论文首先从 drugbank.com 获取了分子的 SMILES 字符串。 -然后，本文使用RDKit.chem软件通过将原子视为节点，将化学键视为边缘来提取分子邻接矩阵。

- ddi_mask_H.py: is used to get ddi_mask_H.pkl

- get_SMILES.py: is our crawler, used to transform atc-4 code to SMILES string. It generates idx2drug.pkl.

（4）训练

![image-20230824135926903](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230824135926903.png)