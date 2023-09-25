https://www.cnblogs.com/nickchen121/p/15105048.html

[Transformer和BERT的前世今生](https://www.bilibili.com/video/BV11v4y137sN/?spm_id_from=333.999.0.0&vd_source=c98fad59c69f91a794e1744235745aa0)

### 目录

[TOC]



#### 01+02发展史

![image-20230619095726008](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230619095726008.png)

#### 03什么是预训练

##### 预训练有什么用

机器学习：偏数学

深度学习：大数据支持

我们首先介绍下卷积神经网络（CNN），CNN 一般用于图片分类任务，并且CNN 由多个层级结构组成，不同层学到的图像特征也不同，**越浅的层学到的特征越通用（横竖撇捺），越深的层学到的特征和具体任务的关联性越强（人脸-人脸轮廓、汽车-汽车轮廓）**

 ![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/%E5%9B%BE%E5%83%8F%E9%A2%84%E8%AE%AD%E7%BB%83%E7%A4%BA%E4%BE%8B.jpg)

如果只有小数据和其他大数据。利用相同的浅层和训练后的深层。

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/%E9%A2%84%E8%AE%AD%E7%BB%83%E7%9A%84%E5%BA%94%E7%94%A8.jpg)

1. 通过 ImageNet 数据集我们训练出一个模型 A
2. 由于上面提到 CNN 的浅层学到的特征通用性特别强，我们可以对模型 A 做出一部分改进得到模型 B（两种方法）：
   1. 冻结：浅层参数使用模型 A 的参数，高层参数随机初始化，**浅层参数一直不变**，然后利用领导给出的 30 张图片训练参数
   2. 微调：浅层参数使用模型 A 的参数，高层参数随机初始化，然后利用领导给出的 30 张图片训练参数，**但是在这里浅层参数会随着任务的训练不断发生变化**

##### 预训练是什么

用大模型A的浅层去完成小数据量的任务B。

任务A和任务B极其相似

##### 预训练怎么用

transforms库

#### 04统计语言模型

##### 语言模型

语言模型通俗点讲就是**计算一个句子的概率。**

![image-20230619103755387](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230619103755387.png)

1. 假设给定两句话 “判断这个词的磁性” 和 “判断这个词的词性”，语言模型会认为后者更自然。转化成数学语言也就是：P(判断，这个，词，的，词性)>P(判断，这个，词，的，磁性)（**判断**）
2. 假设给定一句话做填空 “判断这个词的____”，则问题就变成了给定前面的词，找出后面的一个词是什么，转化成数学语言就是：P(词性|判断，这个，词，的)>P(磁性|判断，这个，词，的)（**预测**）

##### 统计语言模型

统计语言模型的基本思想就是**计算条件概率**。

链式法则：

![image-20230619104511075](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230619104511075.png)

![image-20230619104642987](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230619104642987.png)

![image-20230619104714585](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230619104714585.png)

##### 多元统计语言模型

马尔科夫链的思想（减小计算量）：

假设 Wnext 只和它之前的 **k 个词有相关性**，k=1 时作一个单元语言模型，k=2 时称为二元语言模型。

##### 平滑策略

然而对于绝大多数具有现实意义的文本，会出现数据稀疏的情况，例如**训练时未出现，测试时出现了的未登录单词**。

由于数据稀疏问题，则会出现概率值为 0 的情况（填空题将无法从词典中选择一个词填入），为了避免 0 值的出现，会使用一种平滑的策略——分子和分母都加入一个非 0 正数。

![image-20230619105447736](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230619105447736.png)

count为出现次数。

##### 神经网络语言模型

神经网络语言模型则引入神经网络架构来估计单词的分布，**并且通过词向量的距离衡量单词之间的相似度，因此，对于未登录单词，也可以通过相似词进行估计，进而避免出现数据稀疏问题**。

它的学习任务是输入某个句中单词 wt=bert 前的 t−1 个单词，要求网络正确预测单词 “bert”

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.jpg)

![image-20230619112756952](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230619112756952.png)

（1）这个x就表示一句话，最后输出是1*V的向量，里面代表每个单词的概率，最大值就是最有可能的词。

###### 余弦相似度

计算的是两个向量之间的夹角余弦值，值越接近1表示两个向量越相似，值越接近0则表示两个向量越不相似。

![image-20230619112420388](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230619112420388.png)

###### 词向量

**这个 C(wi) 其实就是单词对应的 Word Embedding 值，也就是我们这节的核心——词向量。**

![image-20230619112641298](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230619112641298.png)

Q越小，可以**控制词向量的维度**，减少存储空间。

训练Q，让词向量可以**更精准**地表示一个词。在精准的基础上，判断任务就变成了分类任务。

#### 06Wordtovec模型

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/word2vec.jpg)

NNLM 和 Word2Vec 基本一致（一模一样），不考虑细节，网络架构就是一模一样。

##### NNLM 和 Word2Vec 的区别

NNLM --》 重点是预测下一词，双层感知机softmax(w2(tanh(（w1(xQ)+b1）))+b2)

Word2Vec --》 CBOW 和 Skip-gram 的两种架构的重点都是得到一个 Q 矩阵，softmax(w1 (xQ) +b1)

1. CBOW：一个老师告诉多个学生，Q 矩阵怎么变
2. Skip：多个老师告诉一个学生，Q 矩阵怎么变

原因：（1）最重要的是input中的Q矩阵。

​              (2)   NNLM是要准确预测，所以需要tanh提高精准度。

##### Word2Vec的缺点

词向量不能进行多意 ---》 ELMO

#### 07预训练的语言模型下游任务改造

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/we%E6%A8%A1%E5%BC%8F%E4%B8%8B%E7%9A%84%E9%A2%84%E8%AE%AD%E7%BB%83.jpg)

预训练语言模型终于出来（给出一句话，我们先使用独热编码（一一对应的一种表查询），再使用Word2Vec 预训练好的 Q 矩阵直接得到词向量，然后进行接下来的任务）

1. 冻结：可以不改变 Q 矩阵
2. 微调：随着任务的改变，改变 Q 矩阵

#### 08ELMO模型

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/%E5%9F%BA%E4%BA%8E%E4%B8%8A%E4%B8%8B%E6%96%87%E7%9A%84emedding.jpg)

不只是训练一个 Q 矩阵，我还可以把这个次的上下文信息融入到这个 Q 矩阵中

左边的 LSTM 获取 E2 的上文信息，右边就是下文信息。LSTM之间的信息是有联系的，而word2vec之间的词没有建立联系。

x1,x2, x4,x5 --> Word2Vec x1+x2+x4+x5 ---> 预测那一个词

获取上下文信息后，把三层的信息进行一个叠加

E1+E2+E3 = K1 一个新的词向量 ≈ E1

K1 包含了第一个词的词向量包含单词特征、句法特征、语义特征

怎么用

##### E2，E3 不同，E1+E2+E3 不同

apple --》 我吃了一个 苹果 -- 》 [1,20,10]

apple --》我在用苹果手机 --》[1,10,20]

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/elmo%E8%AE%AD%E7%BB%83%E5%90%8E%E7%9A%84%E4%BD%BF%E7%94%A8.jpg)

LSTM 无法并行，长期依赖

#### 09什么是注意力机制

注意力机制：我们会把我们的焦点聚焦在比较重要的事物上

##### 怎么做注意力

我（查询对象 Q），这张图（被查询对象 V）。

重要度计算，其实是不是就是相似度计算（更接近），点乘其实是求内积（不要关心为什么可以）

Q，K=k1,k2,⋯,kn ，我们一般使用点乘的方式，注意力机制中的QKV指的是三个矩阵，即查询矩阵（Q）、键矩阵（K）和值矩阵（V）。

通过点乘的方法计算Q 和 K 里的每一个事物的相似度，就可以拿到 Q 和k1的相似值s1，Q 和k2的相似值s2，Q 和kn的相似值 sn

做一层 softmax(s1,s2,⋯,sn) 就可以得到概率(a1,a2,⋯,an)

进而就可以找出哪个对Q 而言更重要了

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/attention-%E8%AE%A1%E7%AE%97%E5%9B%BE.png)

我们还得进行一个汇总，当你使用 Q 查询结束了后，Q 已经失去了它的使用价值了，我们最终还是要拿到这张图片的，只不过现在的这张图片，它多了一些信息（多了于我而言更重要，更不重要的信息在这里）

V = (v1,v2,⋯,vn)

(a1,a2,⋯,an)∗+(v1,v2,⋯,vn)=(a1∗v1+a2∗v2+⋯+an∗vn)= V'

这样的话，就得到了一个新的 V'，这个新的 V' 就包含了，哪些更重要，哪些不重要的信息在里面，然后用 V' 代替 V。

#### 10自注意力机制

##### 注意力机制

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/self-attention.jpg)

QK 相乘求相似度，做一个 scale（未来做 softmax 的时候避免出现极端情况）

然后做 Softmax 得到概率

新的向量表示了K 和 V（K==V），然后这种表示还暗含了 Q 的信息（于 Q 而言，K 里面重要的信息），也就是说，挑出了 K 里面的关键点

##### 自-注意力机制（Self-Attention）（向量）

Self-Attention 的关键点再于，不仅仅是 K≈≈V≈≈Q 来源于同一个 X，这三者是同源的

通过 X 找到 X 里面的关键点

并不是 K=V=Q=X，而是通过三个参数 WQ,WK,WV与X获得K、V、Q。

1.Q、K、V的获取

输入一句话thinking machines,转化为x1和x2。

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/qkv.jpg)

2.Matmul：

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/Q-K%E4%B9%98%E7%A7%AF.jpg)

3.Scale+Softmax：

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/qk-scale.jpg)

4.Matmul：

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/qk-softmax.jpg)

z1表示的就是 thinking 的新的向量表示

对于 thinking，初始词向量为x1

现在我通过 thinking machines 这句话去查询这句话里的每一个单词和 thinking 之间的相似度(即thinking和thinking之间求相似度，thinking和machines之间求相似度)。

新的z1依然是 thinking 的词向量表示，只不过**这个词向量的表示蕴含了 thinking machines 这句话对于 thinking 而言哪个更重要的信息。**

z2同理。

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/self-attention-%E5%A5%BD%E5%A4%842.jpg)

也就是说 its 有 law 这层意思，而通过自注意力机制得到新的 its 的词向量，则会包含一定的 laws 和 application 的信息。

##### 自注意力机制（矩阵

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E7%9F%A9%E9%98%B5%E5%9B%BE.jpg)

内部就表示，句子内部之间的联系。

##### 注意力机制和自注意力机制的区别

（1）自注意力机制，特别狭隘，属于注意力机制的，注意力机制包括自注意力机制的。

（2）自注意力机制规定了 QKV 同源，而且固定了 QKV 的做法。

#### 11 Self-Attention相比较 RNN和LSTM的优缺点 

###### RNN

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/RNN-unrolled.png)

每往后面传导，就会带着前面x的部分信息，过长的话，上文信息就会很少，如何几乎消失。

###### LSTM

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/LSTM%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84.jpg)

LSTM 通过各种门，遗忘门，选择性的可以记忆之前的信息（200 词）

（1）RNNs 长序列依赖问题，无法做并行。Self-Attention可以并行。

（2）Self-Attention 得到的新的词向量具有句法特征和语义特征（词向量的表征更完善）。

（3）Self-Attention计算量过于大。

#### 12 Masked Self-Attention（掩码自注意力机制

（1）为什么要做这个改进：生成模型，生成单词，一个一个生成的。当我们做生成任务的时候，我们也想对生成的这个单词做注意力计算的时候，看不到后面的内容。

（2）自注意力机制明确的知道这句话有多少个单词，并且一次性给足，而掩码是分批次给，最后一次才给足。

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/mask-attention-map.jpg)

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/mask-attention-map-softmax.jpg)

#### 13 Multi-Head Self-Attention（从空间角度解释为什么做多头）

如何多头 

对于 X，我们不是说，直接拿 X 去得到 Z，而是把 X 分成了 8 块（8 头），得到 Z0-Z7。

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/8-head-attention.jpg)

然后把 Z0-Z7 拼接起来，再做一次线性变换（改变维度）得到新的 Z。

**![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/multi-head-attention.png)**

有什么作用？

机器学习的本质是什么：y=σ(wx+b)，在做一件什么事情，非线性变换（把一个看起来不合理的东西，通过某个手段（训练模型），让这个东西变得合理）

非线性变换的本质又是什么？改变空间上的位置坐标，任何一个点都可以在维度空间上找到，通过某个手段，让一个不合理的点（位置不合理），变得合理

这就是词向量的本质

one-hot 编码（0101010）

word2vec（11，222，33）

emlo（15，3，2）

attention（124，2，32）

multi-head attention（1231，23，3），把 X 切分成 8 块（8 个子空间），这样一个原先在一个位置上的 X，去了空间上 8 个位置，通过对 8 个点进行寻找，找到更合适的位置。

#### 14 Positional Encoding （为什么 Self-Attention 需要位置编码）

##### 为什么需要位置编码

既然可以并行，也就是说，词与词之间不存在顺序关系（打乱一句话，这句话里的每个词的词向量依然不会变），即无位置关系（既然没有，我就加一个，通过位置编码的形式加）

##### 位置编码怎么做的

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/%E4%BD%8D%E7%BD%AE%E5%90%91%E9%87%8F.jpg)

##### 具体做法

1.d代表模型词向量的维度，利用公式把位置编码。

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81%E5%85%AC%E5%BC%8F.png)

2.把位置编码和向量叠加在一起，转化成新的。

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81%E5%92%8C%E8%AF%8D%E5%90%91%E9%87%8F%E4%B9%8B%E5%92%8C.png)

#### 15 Transformer 框架概述 

##### 整体框架

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/tf-%E6%95%B4%E4%BD%93%E6%A1%86%E6%9E%B6.jpg)

seq2seq

一句话，一个视频就是一个序列

序列（编码器）到序列（解码器）

分成两部分，编码器和解码器（生成任务和翻译任务）

##### 机器翻译流程（Transformer）

流程 1

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/tf-%E6%A1%86%E6%9E%B6%E7%AE%80%E5%8C%96.jpg)

给一个输入，给出一个输出（输出是输入的翻译的结果）

“我是一个学生” --》（通过 Transformer） I am a student

流程 2

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/tf-ed-%E6%A1%86%E6%9E%B6.jpg)

编码器和解码器

编码器：把输入变成一个词向量（Self-Attetion）

解码器：得到编码器输出的词向量后，生成翻译的结果

流程 3

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/tf-ed-%E5%A4%8D%E6%9D%82.jpg)

Nx 的意思是，编码器里面又有 N 个小编码器（默认 N=6）

通过 6 个编码器，对词向量一步又一步的强化（增强），不断更新Z。

流程 4

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/ed-%E7%BB%86%E5%88%86.jpg)

了解 Transformer 就是了解 Transformer 里的小的编码器（Encoder）和小的解码器（Decoder）

FFN（Feed Forward）：w2(（w1x+b1）)+b2，两个线性变化。

#### 16 Transformer 的编码器（Encodes）——我在做更优秀的词向量

##### 编码器概略图

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/ed-%E7%BB%86%E5%88%86.jpg)

编码器包括两个子层，Self-Attention、Feed Forward

每一个子层的传输过程中都会有一个（残差网络+归一化），也就是Self-Attention要经过残差网络+归一化之后，才传输到Feed Forward。

##### 编码器详细图

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/encoder-%E8%AF%A6%E7%BB%86%E5%9B%BE.png)

Thinking

--》得到绿色的 x1（词向量，可以通过 one-hot、word2vec 得到）+ 叠加位置编码（给 x1 赋予位置属性）得到黄色的 x1

--》输入到 Self-Attention 子层中，做注意力机制（x1、x2 拼接起来的一句话做），得到 z1（x1 与 x1，x2拼接起来的句子做了自注意力机制的词向量，表征的仍然是 thinking），也就是说 z1 拥有了位置特征、句法特征、语义特征的词向量

--》残差网络（避免梯度消失，w3(w2(w1x+b1)+b2)+b3，如果 w1，w2，w3 特别小，0.0000000000000000……1，x 就没了，【w3(w2(w1x+b1)+b2)+b3+x】），归一化（LayerNorm），做标准化（避免梯度爆炸），得到了深粉色的 z1

--》Feed Forward，Relu（w2(w1x+b1)+b2），（前面每一步都在做线性变换，wx+b，线性变化的叠加永远都是线性变化（线性变化就是空间中平移和扩大缩小），通过 Feed Forward中的 Relu 做一次非线性变换，这样的空间变换可以无限拟合任何一种状态了），得到 r1（是 thinking 的新的表征）

#### 17 Transformer 的解码器（Decoders）——我要生成一个又一个单词 _

解码器会接收编码器生成的词向量，然后通过这个词向量去生成翻译的结果

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/ed-%E7%BB%86%E5%88%86.jpg)

解码器的 Self-Attention 在编码已经生成的单词

假如目标词“我是一个学生”---》masked Self-Attention

训练阶段：目标词“我是一个学生”是已知的，然后 Self-Attention 是对“我是一个学生” 做计算

如果不做 masked，每次训练阶段，都会获得全部的信息

如果做 masked，Self-Attention 第一次对“我”做计算

Self-Attention 第二次对“我是”做计算

……

测试阶段：

1. 目标词未知，假设目标词是“我是一个学生”（未知），Self-Attention 第一次对“我”做计算
2. 第二次对“我是”做计算
3. ……

而测试阶段，没生成一点，获得一点

![image-20230627103638114](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230627103638114.png)

Linear 层转换成词表的维度

softmax 得到最大词的概率

#### 18 Transformer 的动态流程

Encoder层生成K和V。

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/tf-%E5%8A%A8%E6%80%81%E7%94%9F%E6%88%90.gif)

decoder层进行预测。

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/tf-%E5%8A%A8%E6%80%81%E7%BB%93%E6%9E%9C-2.gif)

#### 19 Transformer 解码器的两个为什么（为什么做掩码、为什么用编码器-解码器注意力）

##### 问题一：为什么 Decoder 需要做 Mask

机器翻译：源语句（我爱中国），目标语句（I love China）

为了解决训练阶段和测试阶段的 gap（不匹配）

训练阶段：解码器会有输入，这个输入是目标语句，就是 I love China，通过已经生成的词，去让解码器更好的生成（每一次都会把所有信息告诉解码器）

测试阶段：解码器也会有输入，但是此时，测试的时候是不知道目标语句是什么的，这个时候，你每生成一个词，就会有多一个词放入目标语句中，每次生成的时候，都是已经生成的词（测试阶段只会把已经生成的词告诉解码器）

##### 问题二：为什么 Encoder 给予 Decoders 的是 K、V 矩阵

Q来源解码器，K=V来源于编码器

Q是查询变量，Q 是已经生成的词

K=V 是源语句

当我们生成这个词的时候，通过已经生成的词和源语句做自注意力，就是确定源语句中哪些词对接下来的词的生成更有作用，首先他就能找到当前生成词

### 实践部分

#### 00 通过 Pytorch 实现 Transformer 框架完整代码

#### 02 Transformer 中 Add&Norm （残差和标准化）代码实现 

![img](https://imgmd.oss-cn-shanghai.aliyuncs.com/BERT_IMG/encoder-%E8%AF%A6%E7%BB%86%E5%9B%BE.png)

![image-20230701120802256](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230701120802256.png)



```
class LayerNorm(nn.Module):

    def __init__(self, feature, eps=1e-6):
        """
        :param feature: self-attention 的 x 的大小
        :param eps:
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

实现标准化公式。（先残差，后标准化）

```
class SublayerConnection(nn.Module):
    """
    这不仅仅做了残差，这是把残差和 layernorm 一起给做了

    """
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        # 第一步做 layernorm
        self.layer_norm = LayerNorm(size)
        # 第二步做 dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """
        :param x: 就是self-attention的输入
        :param sublayer: self-attention层，即z
        :return:
        """
        return self.dropout(self.layer_norm(x + sublayer(x)))
```

这段代码会把绿色方框里面的内容全做完。

#### 0201 为什么 Pytorch 定义模型要有一个 init 和一个 forward，两者怎么区分 

```
class LayerNorm(nn.Module):

    def __init__(self, feature, eps=1e-6):
        """
        :param feature: self-attention 的 x 的大小
        :param eps:
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    
# python 面向对象
# 当你不做高拓展的时候，下面这种写法被你给写死了
# 一个 512 维的向量，还有一个 256 维的向量
l1 = LayerNorm(512)
l2 = LayerNorm(256)

l1()#代表512维的
l2()#代表256维的

'''类似调用，改动很简单。只需要修改上面的维度即可。'''
```

```python
class LayerNorm1(nn.Module):

    def __init__(self):
        """
        :param feature: self-attention 的 x 的大小
        :param eps:
        """
        super(LayerNorm1, self).__init__()

    def forward(self, feature, x, eps=1e-6):
        self.a_2 = nn.Parameter(torch.ones(feature))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
l3 = LayerNorm1()
l3(1)

l3(1)
l3(1)


l3(1)
l3(1)
```

#### 03 Transformer 中的多头注意力（Multi-Head Attention）Pytorch代码实现 

一个准则：输入放在forward里面，要初始化的放init。

##### 自注意力计算

```
def self_attention(query, key, value, dropout=None, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    #QK计算
    # mask的操作在QK之后，softmax之前
    '''掩码'''
    if mask is not None:
        mask.cuda()
        scores = scores.masked_fill(mask == 0, -1e9)
    self_attn = F.softmax(scores, dim=-1)
    '''防止过拟合'''
    if dropout is not None:
        self_attn = dropout(self_attn)
    return torch.matmul(self_attn, value), self_attn
```

##### 多头注意力计算

```
# PYthon/PYtorch/你看的这个模型的理论
class MultiHeadAttention(nn.Module):

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)
        self.d_k = d_model // head
        self.head = head
        self.d_model = d_model

        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)

        # 在Transformer模型中，这种线性变换被用来将输入向量投影到query, key, 和value表示空间中，以便进行注意力计算。

        self.linear_out = nn.Linear(d_model, d_model)
        #只有多头注意力输出需要线性变换
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None

    def forward(self,  head, d_model, query, key, value, dropout=0.1,mask=None):
        """

        :param head: 头数，默认 8
        :param d_model: 输入的维度 512
        :param query: Q
        :param key: K
        :param value: V
        :param dropout:
        :param mask:
        :return:
        """
        
        # if mask is not None:
        #     # 多头注意力机制的线性变换层是4维，是把query[batch, frame_num, d_model]变成[batch, -1, head, d_k]
        #     # 再1，2维交换变成[batch, head, -1, d_k], 所以mask要在第一维添加一维，与后面的self attention计算维度一样
        #     mask = mask.unsqueeze(1)

        n_batch = query.size(0)

        # 多头需要对这个 X 切分成多头

        # query==key==value
        # [b,1,512]
        # [b,8,1,64]

        # [b,32,512]
        # [b,8,32,64]
'''线性变换后值不相等'''
        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 32, 64]
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 32, 64]
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 32, 64]

        x, self.attn = self_attention(query, key, value, dropout=self.dropout, mask=mask)
        # [b,8,32,64]
        # [b,32,512]
        # 变为三维， 或者说是concat head
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)

        return self.linear_out(x)
```

#### 04 Transformer 中的位置编码的 Pytorch 实现 

```
class PositionalEncoding(nn.Module):

    def __init__(self, dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
#位置编码必须成对出现。
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))

        """
        构建位置编码pe
        pe公式为：
        PE(pos,2i/2i+1) = sin/cos(pos/10000^{2i/d_{model}})
        """
        pe = torch.zeros(max_len, dim)  
        # max_len 是解码器生成句子的最长的长度，假设是 10.矩阵化不是单一的词。
        
        '''生成公式'''
        position = torch.arange(0, max_len).unsqueeze(1)
        
        #torch.arange(0, max_len) 用于生成一个从 0 到 max_len-1 的整数序列，而 unsqueeze(1) 则用于将这个一维序列转换为一个列向量，即在第二个维度上添加了一个维度，最终得到了一个形状为 (max_len, 1) 的张量。
        
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *-(math.log(10000.0) / dim)))


        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        self.drop_out = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):

        emb = emb * math.sqrt(self.dim)

        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.drop_out(emb)
        return emb
'''位置编码与x的内容无关，所以可以提前创建好位置编码，然后根据需要选用位置编码。'''
```

#### 06 Transformer 中的Linear+Softmax 的

```
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)
```

#### 07 Transformer 中的掩码多头注意力机制（Masked Multi-head Attention）的实现

```
def src_trg_mask(src, r2l_trg, trg, pad_idx):
    """
    :param src: 编码器的输入
    :param r2l_trg: r2l方向解码器的输入
    :param trg: l2r方向解码器的输入
    :param pad_idx: pad的索引
    :return: trg为None，返回编码器输入的掩码；trg存在，返回编码器和解码器输入的掩码
    """

    # TODO: enc_src_mask是元组，是否可以改成list，然后修改这种冗余代码
    # 通过src的长短，即视频特征向量提取的模式，判断有多少种特征向量需要进行mask
    if isinstance(src, tuple) and len(src) == 4:
        # 不同模式的视频特征向量的mask
        #在src[0][:, :, 0]中，第一个:代表取所有的行，第二个:代表取所有的列，而0表示只取每行的第一个元素。
        src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)  # 二维特征向量
        src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)  # 三维特征向量
        src_object_mask = (src[2][:, :, 0] != pad_idx).unsqueeze(1)  # 目标检测特征向量
        src_rel_mask = (src[3][:, :, 0] != pad_idx).unsqueeze(1)  # 目标关系特征向量

        # 视频所有特征向量mask的拼接
        enc_src_mask = (src_image_mask, src_motion_mask, src_object_mask, src_rel_mask)
        dec_src_mask = src_image_mask & src_motion_mask  # 视频二维和三维特征向量mask的拼接
        src_mask = (enc_src_mask, dec_src_mask)  # 视频最终的mask
    elif isinstance(src, tuple) and len(src) == 3:
        src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
        src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)
        src_object_mask = (src[2][:, :, 0] != pad_idx).unsqueeze(1)

        enc_src_mask = (src_image_mask, src_motion_mask, src_object_mask)
        dec_src_mask = src_image_mask & src_motion_mask
        src_mask = (enc_src_mask, dec_src_mask)
    elif isinstance(src, tuple) and len(src) == 2:
        src_image_mask = (src[0][:, :, 0] != pad_idx).unsqueeze(1)
        src_motion_mask = (src[1][:, :, 0] != pad_idx).unsqueeze(1)

        enc_src_mask = (src_image_mask, src_motion_mask)
        dec_src_mask = src_image_mask & src_motion_mask
        src_mask = (enc_src_mask, dec_src_mask)
    else:
        # 即只有src_image_mask，即二维特征的mask
        src_mask = src_image_mask = (src[:, :, 0] != pad_idx).unsqueeze(1)

    # 判断是否需要对trg，也就是解码器的输入进行掩码
    if trg and r2l_trg:
        """
        trg_mask是填充掩码和序列掩码，&前是填充掩码，&后是通过subsequent_mask函数得到的序列掩码
        其中type_as，是为了让序列掩码和填充掩码的维度一致
        """
        trg_mask = (trg != pad_idx).unsqueeze(1) & sequence_mask(trg.size(1)).type_as(src_image_mask.data)
        # r2l_trg的填充掩码
        r2l_pad_mask = (r2l_trg != pad_idx).unsqueeze(1).type_as(src_image_mask.data)
        # r2l_trg的填充掩码和序列掩码
        r2l_trg_mask = r2l_pad_mask & sequence_mask(r2l_trg.size(1)).type_as(src_image_mask.data)
        # src_mask[batch, 1, lens]  trg_mask[batch, 1, lens]
        return src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask
    else:
        return src_mask
```

1.使用`unsqueeze`函数将这个一维布尔张量变成二维张量，并在第二维上增加一个维度。这样就得到了一个形状为`(batch_size, 1, seq_len)`的二维张量，其中`batch_size`表示批次大小，`seq_len`表示特征向量的长度。这个二维张量就是编码器的掩码，用于在模型中忽略填充部分。

2.这个函数首先通过判断`src`的长度，即视频特征向量提取的模式，来判断有多少种特征向量需要进行mask。然后根据不同的模式，生成不同的特征向量的mask，并将所有特征向量mask拼接起来，得到编码器输入的掩码。

如果`trg`不为None，那么还需要对解码器的输入进行掩码。这里使用了`sequence_mask`函数生成序列掩码，然后使用逻辑与运算符`&`将填充掩码和序列掩码拼接起来，得到解码器输入的掩码。最后将编码器输入的掩码和解码器输入的掩码返回。

#### 08 Transformer 中的编码器（Encoder）的Pytorch实现

![image-20230704151425039](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230704151425039.png)

```
class EncoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayer_connection[1](x, self.feed_forward)


class EncoderLayerNoAttention(nn.Module):
    def __init__(self, size, attn, feed_forward, dropout=0.1):
        super(EncoderLayerNoAttention, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        return self.sublayer_connection[1](x, self.feed_forward)
class Encoder(nn.Module):

    def __init__(self, n, encoder_layer):
        super(Encoder, self).__init__()
        self.encoder_layer = clones(encoder_layer, n)
#n代表编码器的个数。
    def forward(self, x, src_mask):
        for layer in self.encoder_layer:
            x = layer(x, src_mask)
        return x
```

#### 09 自然而然就能听懂的Transformer 中[双向]解码器（Decoder）的Pytorch实现

![image-20230704152018072](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230704152018072.png)

```
class DecoderLayer(nn.Module):

    def __init__(self, size, attn, feed_forward, sublayer_num, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(size, dropout), sublayer_num)

#memory是编码层的输入。

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory=None, r2l_trg_mask=None):
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, trg_mask))
        x = self.sublayer_connection[1](x, lambda x: self.attn(x, memory, memory, src_mask))
'''双向解码器'''
        if r2l_memory is not None:
            x = self.sublayer_connection[-2](x, lambda x: self.attn(x, r2l_memory, r2l_memory, r2l_trg_mask))

        return self.sublayer_connection[-1](x, self.feed_forward)
        
#正向解码器        
class R2L_Decoder(nn.Module):

    def __init__(self, n, decoder_layer):
        super(R2L_Decoder, self).__init__()
        self.decoder_layer = clones(decoder_layer, n)

    def forward(self, x, memory, src_mask, r2l_trg_mask):
        for layer in self.decoder_layer:
            x = layer(x, memory, src_mask, r2l_trg_mask)
        return x

#反向解码器
class L2R_Decoder(nn.Module):

    def __init__(self, n, decoder_layer):
        super(L2R_Decoder, self).__init__()
        self.decoder_layer = clones(decoder_layer, n)

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask):
        for layer in self.decoder_layer:
            x = layer(x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask)
        return x
```

#### 10 Transformer 框架搭建之 init 在干吗？（万事具备，开始调包

初始化一些小结构。

```
class ABDTransformer(nn.Module):

#定义一些需要改变的参数。

    def __init__(self, vocab, d_feat, d_model, d_ff, n_heads, n_layers, dropout, feature_mode,
                 device='cuda', n_heads_big=128):
        super(ABDTransformer, self).__init__()
        self.vocab = vocab
        self.device = device
        #self.feature_mode = feature_mode（多模态，也就是多个特征提取）

        c = copy.deepcopy

        # attn_no_heads = MultiHeadAttention(0, d_model, dropout)

        attn = MultiHeadAttention(n_heads, d_model, dropout)

        attn_big = MultiHeadAttention(n_heads_big, d_model, dropout)

        # attn_big2 = MultiHeadAttention(10, d_model, dropout)

        feed_forward = PositionWiseFeedForward(d_model, d_ff)

        if feature_mode == 'one':
            self.src_embed = FeatEmbedding(d_feat, d_model, dropout)
 '''以下是多模态才需要的
        elif feature_mode == 'two':
            self.image_src_embed = FeatEmbedding(d_feat[0], d_model, dropout)
            self.motion_src_embed = FeatEmbedding(d_feat[1], d_model, dropout)
        elif feature_mode == 'three':
            self.image_src_embed = FeatEmbedding(d_feat[0], d_model, dropout)
            self.motion_src_embed = FeatEmbedding(d_feat[1], d_model, dropout)
            self.object_src_embed = FeatEmbedding(d_feat[2], d_model, dropout)
        elif feature_mode == 'four':
            self.image_src_embed = FeatEmbedding(d_feat[0], d_model, dropout)
            self.motion_src_embed = FeatEmbedding(d_feat[1], d_model, dropout)
            self.object_src_embed = FeatEmbedding(d_feat[2], d_model, dropout)
            self.rel_src_embed = FeatEmbedding(d_feat[3], d_model, dropout)
  '''
        self.trg_embed = TextEmbedding(vocab.n_vocabs, d_model)
        self.pos_embed = PositionalEncoding(d_model, dropout)

        # self.encoder_no_heads = Encoder(n_layers, EncoderLayer(d_model, c(attn_no_heads), c(feed_forward), dropout))

        self.encoder = Encoder(n_layers, EncoderLayer(d_model, c(attn), c(feed_forward), dropout))

        #self.encoder_big = Encoder(n_layers, EncoderLayer(d_model, c(attn_big), c(feed_forward), dropout))

        # self.encoder_big2 = Encoder(n_layers, EncoderLayer(d_model, c(attn_big2), c(feed_forward), dropout))

        self.encoder_no_attention = Encoder(n_layers,
                                            EncoderLayerNoAttention(d_model, c(attn), c(feed_forward), dropout))

        self.r2l_decoder = R2L_Decoder(n_layers, DecoderLayer(d_model, c(attn), c(feed_forward),
                                                              sublayer_num=3, dropout=dropout))
        self.l2r_decoder = L2R_Decoder(n_layers, DecoderLayer(d_model, c(attn), c(feed_forward),
                                                              sublayer_num=4, dropout=dropout))

        self.generator = Generator(d_model, vocab.n_vocabs)
```

