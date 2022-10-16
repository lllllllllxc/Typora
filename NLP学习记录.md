## Markdown

**加粗**   * * ——  * * 

*斜体*  * —— *

1. a
2. b
   1. b1（Tab）
   2. b2
   3. (无内容按回车则跳回上一级)

表格

| 1    | 2    | 3    |
| ---- | ---- | ---- |
|      |      |      |

**|1|2|3|**

数学公式

行内公式 $\sum_i^k$  嵌入在文本内

行外公式 单独占据一行


$$
\frac{a}{b}  行外公式
\begin{aligned}
x & = \sum_{i=1}^k=\frac{x_1}{h_2}\\
&=y+1
\end{aligned}
$$

 行外公式中包含多个行使用开始结束符 \begin{aligned} \end{aligned}，使用\\\进行换行，使用&选择对齐方式,&在谁前面就以谁对齐
$$
\begin{aligned} &KPI=(N+S)W \\ &PI=N+S \\ &I=W \end{aligned} 
$$




`单个反引号（左上角）是 代码片段`

```
​```三个单引号是代码块，可以选择语言
```

分栏符 triple - 

---

引用 >

>参考文献
>
>1. 1
>2. 2
>3. 
>
>

## git 常用语句

###### push流程

1.git add .（.为all）

2.git commit -m"提交描述"

3.git push -u origin master(master为branch的名称)

## python语法

\_\_name\__在当前.py文件中为\_\_main\_\_，在其他.py文件中是具体包名+文件名

故使用**if __ name __ == "__ main __"**来判断是否在当前文件中执行，否则import进去的文件所执行的操作将会迭代进当前文件中，使用该方法有利于每个文件进行单独测试

类中的**self**即当前类的实体

实例化对象的属性可以在\_\_init\_\_里面初始化，也能在函数中初始化，也能在实例化后调用该对象进行添加属性

python中**继承**的表示即将父类写在类名后的()内，如FeatEmbedding(nn.Moudle)

在子类的def __ init __ （self）方法中需调用super().__ init __(self)初始化父类从而调用父类中的方法  

**enumerate**(data,[start=0])

指定一个strat，返回一个从该索引开始的枚举对象

```python
data = [1,2,3]
for num,value in enumerate(data):
    print(num,value)
out:
0 1
1 2
2 3
```

**行内函数**

 ```python
#行内函数的定义
net,loss = lambda X: d2l.linreg(X,w,b),d2l.squared_loss
#调用
l = loss(net(X),y)
 ```

sorted+enumera+ lambda

```python
token = ['traveller', 'for', 'so', 'it']
counter = count_corpus(token)
sorted(enumerate(token),key=lambda x:x[1])
out:[(1, 'for'), (3, 'it'), (2, 'so'), (0, 'traveller')]
sorted(enumerate(token),key=lambda x:x[0])
out:[(0, 'traveller'), (1, 'for'), (2, 'so'), (3, 'it')]
```

sorted+lambda

````python
token = ['traveller', 'for', 'so', 'it','it','it','for']
counter = collection.Counter(token)
print(counter.items())
out:dict_items([('traveller', 1), ('for', 2), ('so', 1), ('it', 3)])
sorted(counter.items(),key=lambda x:x[1],reverse=True)
out:[('it', 3), ('for', 2), ('traveller', 1), ('so', 1)]
````



**f'{}' 等同于format**

```python
print('a的值为{value}'.format(value=xx.value))
print(f'a的值为{xx.value}')
torch.device(f'cuda:{i}')for i in range(torch.cuda.device_count())]
```

**数据类型判断**：

```python
token = [[1],[2]]
isinstance(token[0],list)
out:True
```



**列表生成式**(双层for循环): 

```python
tokens = [[1,2,3],[2,3,4],[3,4,5]]
token = [token for line in tokens for token in line]
print(token)
out:[1, 2, 3, 2, 3, 4, 3, 4, 5]
```

**三种主要数据类型**

字典：dic = {'a':1,'b':2}

列表：list = [1,2,3]

元组：tuple = (1,2,3)

列表和元组都是序列，都可以存储任何数据类型，都可以通过索引访问。

但tuple一旦建立其内容不可改变，相反list可以改变。因此tuple可以当作dict中的key，而list则不行

当元组内嵌套的元素是可变类型比如list时，其不可改变的特性将失去。

###### pycharm快捷键

alt+1隐藏/显示左侧栏

ctrl+tab切换文件

shift+enter不改变当前行并换行

###### jupyter

ctrl+D删除代码内整行

pytorch(Conda+App)默认路径修改:

https://zhuanlan.zhihu.com/p/48962153

## 深度学习术语

###### 错误率

Top-1 error：模型预测某个东西类别，并输出1个预测结果，则该结果正确的概率为Top-1 accuracy，反之为top-1 error

Top-5 error同理，且Top-5 error的数值会比Top-1 error低

###### 监督

有无监督即数据集有无标签

 监督学习是最常见的一种机器学习，它的训练数据是有标签的，训练目标是能够给新数据（测试数据）以正确的标签。 

通常无监督学习是指不需要人为注释的样本中抽取信息。例如word2vec。

半监督学习介于两者之间。算法上，包括一些对常用监督式学习算法的延伸，这些算法首先试图对未标识数据进行建模，在此基础上再对标识的数据进行预测

SOTA：state-of-the-art当前最先进的水平

## NLP学习记录

1.使用conda安装pytorch时，换源后要将

`conda install pytorch torchvision torchaudio cpuonly -c pytorch`

中-c pytorch去掉，否则仍使用默认源

2.张量tensor与矩阵matrix区别

tensor定义：A rank-n tensor in m-dimensions is a mathematical object that has n indices and m^n components and obeys certain transformation rules.

rank of Tensor: number of basic vectors you needed to fully specify a component of the tensor. 描述一个component分量所需要的向量个数

a rank of 2 tensor named stress tensor 应力张量

矩阵只是一列数字，而张量还有独特的变换性质，它遵守着特定的变换规则，并具有物理上的含义，可以用矩阵来表示张量，但是张量本身具有更深的物理含义

张量的变换规则transformation rules: a tensor is an object that is invariant under a change of coordinate systems, with components that change according to a special set of mathematical rules.（坐标系统变化情况下，张量是恒定不变的）

爱因斯坦标记法：free index只能出现一次，dummy index 只能出现两次

In PyTorch, Tensor is the important component in constructing dynamic computational graph. tensor是构建动态计算图的重要组成部分

It contains data and grad, which storage the value of node and gradient w.r.t loss respectively. 其中分别存储了数据w本身和损失loss对w的导数即梯度grad。而tensor中的grad本身也是一个tensor。故在pytorch中取grad时需要w.grad.data，并且在更新w时也要使用w.data = w.data - 0.01 * w.grad.data，使用data来更新防止使用tensor在更新的过程中构建出无用的计算图从而吃掉内存。对w进行新的梯度更新之前需要将其梯度值清零，否则将会默认累加。

pytorch中loss tensor在进行.backward()反向传播后，会将计算出的grad存入指定requires_grad=True的tensor变量中，并且会释放掉前馈过程构建的计算图。

#### 3.梯度下降

流程：1）定义代价函数(损失函数) 2）选择起始点 3）计算梯度 4）按学习率前进

##### 代价函数/损失函数

cost/lost function 用来评价模型的准确程度，当损失降到最低时，其中所训练出的wi也就越准确。通过对损失函数求梯度再*LearningRate得到每次wi所需要更新的幅度

线性回归 Linear Regression即线性的预测函数

wx + b —> y

为找到合适的预测函数，高尔顿提出了最小二乘法Least Square Method

<img src="NLP学习记录.assets\1652276207809.png" alt="1652276207809" style="zoom:50%;" />

<img src="NLP学习记录.assets\1652276248901.png" alt="1652276248901" style="zoom:50%;" />

<img src="NLP学习记录.assets\1652276315366.png" alt="1652276315366" style="zoom:50%;" />

由此得出了代价函数cost/loss function：$e = a * w^2+b * w+c$

该损失最小时，预测函数则最为精准

<img src="NLP学习记录.assets\1657787646042.png" alt="1657787646042" style="zoom: 50%;" />

mean为计算出的平均损失，即为平均平方误差(Mean Square Error) 

当损失函数中y_hat的因变量只有一个时可以使用穷举法选出最优w

当有多个时y_hat=x*w+b w、b都为因变量，都需要找到最优值，此时进行穷举相当于多了一个维度，变为在二维区域中进行搜索。以此类推，n个变量则需要在n维空间中进行搜索，穷举的数量以幂函数递增至k的n次方

深度学习中所要解决的最大问题不是局部最优，而是鞍点问题，因为陷入鞍点将导致梯度无法继续更新

**优化算法**：

当模型没有解析解时需要通过优化算法近似地迭代来得到数值解。

BGD(Batch Gradient Descent)批量梯度下降 用全部训练样本参与计算

SGD(stochastic Gradient Descent)随机梯度下降 每次只用一个样本进行计算

MBGD(Mini-Batch Gradient Descent)小批量梯度下降 每次选用一小批样本进行计算

**BGD与SGD对比**

代码对比：

BGD在计算损失和计算梯度时每次训练都是使用整个批量数据计算后取平均值

<img src="NLP学习记录.assets\1658134083074.png" alt="1658134083074" style="zoom:50%;" />

<img src="NLP学习记录.assets\1658134069957.png" alt="1658134069957" style="zoom:50%;" />

BGD中的gradient可以并行计算(分别计算再求和)，而SGD由于w的更新具有线性传递性故不能并行

<img src="C:\Users\Dust\AppData\Roaming\Typora\typora-user-images\1657859757127.png" alt="1657859757127" style="zoom:50%;" />

二者折中后为Mini-Batch SGD

MBSGD(Mini-Batch Stochastic Gradient Descent)小批量随机梯度下降 是深度学习默认的求解算法

优化算法中的学习率、批量大小不是由模型训练来的，而是人为定义的，因此称其为**超参数**





Logistic回归问题中常见的sigmoid（S型）函数如下

<img src="NLP学习记录.assets\1658121294646.png" alt="1658121294646" style="zoom:50%;" />

其将输出值映射到[-1,1] ，最为常用的是映射到[0,1]的 
$$
1/1+e^{-x}
$$
 概率分布使用交叉熵来衡量损失 (BCE，Mini-Batch Cross-Entropy Loss)
$$
loss = -(ylogy_{hat} + (1-y)log(1-y_{hat}))
$$
y=1时**{loss=-logy_hat}**  y_hat越接近于1则损失越小

y=0时**loss = -log(1-y_hat)**y_hat越接近于0则损失越小

<img src="NLP学习记录.assets\1658122834821.png" alt="1658122834821" style="zoom: 67%;" />

#### 4.反向传播

当神经网络层数变多，则需要更新的wi就会变多，故对其偏导的解析式难以挨个求出，故需要如下图所示

正向进行运算、求偏导并存储计算结果，反向将结果逆向带入偏导公式，逐一求出偏导结果。

<img src="NLP学习记录.assets\1657862234298.png" alt="1657862234298" style="zoom:50%;" />

<img src="NLP学习记录.assets/1664432694298.png" alt="1664432694298" style="zoom:50%;" />

##### mini-batch的读取

`#Training cycle`

`for epoch in range(training_epochs):`

​	`#loop over all baches`

​	`for i in range(total_batch):`

每进行一个epoch则将所有数据都训练了一遍即训练了一个Batch的数据，由于mini-Batch将整个Batch又进行了分割，故需要在epoch下循环将每个mini-Batch都训练一次，Batch-Size为每个mini-batch的数据量，iteration为训练batch-size大小的数据的次数。batch-size*iteration=Batch的总大小

使用dataset、dataloader对整个batch的数据进行存储和读取，dataset中存储的数据需要能进行按索引读取，即实现__getItem__() ,取数据的过程又根据文件的大小分为全部读入，和使用时按索引读入。

#### 5.神经网络

#####  卷积  

卷积用极简的数学形式描述了一个动态过程

<img src="NLP学习记录.assets/1665125789428.png" alt="1665125789428" style="zoom:50%;" />

<img src="NLP学习记录.assets/1665125778882.png" alt="1665125778882" style="zoom:50%;" />

一个卷积核计算出一个通道的输出结果，m个卷积核则输出的通道为m个。卷积核的通道数跟输入数据的通道数相同(输入的每一个通道都要对应一个卷积核)。故上图中对应输入(n,width_in,height_in),输出(m,width_out,height_out)的卷积核为(m,n,kernel_size_width,kernel_size_height) ,个卷积块，每个卷积块中有对应输出通道数的单通道卷积。

<img src="NLP学习记录.assets/1665125766315.png" alt="1665125766315" style="zoom:50%;" />

1X1卷积可以减少运算量，实际效果如何？

##### 循环

增加了时间的概念，纵向扩宽了神经网络 

RNNCell的本质是一个线性层

<img src="NLP学习记录.assets/1665125644055.png" alt="1665125644055" style="zoom:50%;" />

图左实际作用为图右，即同一个线性层反复使用（权重w不变）

RNN中为什么用tanh比较多？

RNN公式：(i为inputsize，h为hiddensize，h为某一时刻的hidden)

<img src="NLP学习记录.assets/1665125654119.png" alt="1665125654119" style="zoom:50%;" />



one-hot 向量存在的问题:1.高维 2.稀疏 3.硬编码

应改进为1.低维 2.稠密 3.通过学习得来

解决问题为使用embedding，即进行如下映射（降维）

<img src="NLP学习记录.assets/1665125670428.png" alt="1665125670428" style="zoom:50%;" />

Pytorch中使用RNNCell与RNN的区别为个体与整体的区别

```python
dataset = torch.randn(seq_len,batch_size,input_size)

hidden = torch.zeros(batch_size,hidden_size)

rnn_cell = torch.nn.RNNCell(input_size,hidden_size)

for index,input in enumerate(dataset):`
​	print(' index',index)`		
​	print('inputsize',inputsize)`
​	hidden = rnn_cell(input,hidden)`
input.shape=(batchSize,inputSize)

hidden.shape=(batchSize,hiddenSize)
```



`dataset = torch.randn(seq_len,batch_size,input_size)`

`hidden = torch.zeros(batch_size,hidden_size)`

`rnn_cell = torch.nn.RNNCell(input_size,hidden_size``)`

`for index,input in enumerate(dataset):`

​	`print(' index',index)`		

​	`print('inputsize',inputsize)`

​	`hidden = rnn_cell(input,hidden)`

input.shape=(batchSize,inputSize)

hidden.shape=(batchSize,hiddenSize)

rnncell中构建模型时batchSize只有在初始化h0时才用得到

def init_hidden(self):

​	return torch.zeros(self.batch_size,self.hidden_size	)



`rnn = torch.nn.RNN(input_size,hidden_size,num_layers)`

`#inputs为整个输入序列，out为h1~hn，hidden为hn，所需参数hidden即为h0`

`inputs = torch.randn(seq_len,batchSize,inputSize)`#由于输入是整个序列故新添加了seq_len

`hidden = torch.zeros(numlayers,batchSize,hiddenSize)`

`out, hidden = rnn(inputs,hidden)`

**input.shape=(seqlen,batchSize,inputSize)**

**h_0.shape(numlayers,batchSize,hiddenSize)**

**output.shape=(seqlen,batchSize,hiddenSize)**

**h_n.shape =(numlayers,batchSize,hiddenSize)**

#### 6.感知机

o=σ(<w,x>+b)   σ(x)={1 if x>0, -1 otherwise}

即为一个二分类问题。

对比线性回归其输出的是一个实数，而SVM输出的是一个类

对比Softmax回归其输出的是n个类的概率，而SVM只输出单一类

###### 感知机训练

​	

```
initialize w=0 and b=0
repeat
	//符合if条件即预测错误
	if yi[<w,xi> + b]<=0 then
		w <- w + yixi and b <- b + yi
	end if
until all classified correctly
#相当于loss函数为
l(y,x,w) = max(0,-y<w,x>)
```

**多层感知机**（MLP，Multilayer perception)/(FNN，Fully-connected Neural Network)/前馈神经网络

使用隐藏层和激活函数来得到非线性模型

$h = σ(W_1x+b_1) $

$o=w_{2}^{T}h+b_2$

若σ为线性函数，则整体作用等价于一个单层感知机

即将h带入后$o=aw_{2}^{T}W_1x+b^{'}$

常用的激活函数：sigmoid、tanh、ReLU

##### 训练误差、泛化误差

训练误差：模型在训练数据上的误差

泛化误差：模型在新数据上的误差

计算误差的数据集：验证数据集(不能跟训练数据有交叉)、测试数据集(真正的测试数据集是只用一次的)

当超参数是在验证数据集上调出时，并不代表对于新的数据的泛化能力

###### K-则交叉验证

当没有足够多的数据时，将训练数据分为k块

| k     | 1        | 2        | 3        |
| ----- | -------- | -------- | -------- |
| step1 | validate | train    | train    |
| step2 | train    | validate | train    |
| step3 | train    | train    | validate |

常用k=5或10

一般将整个数据集划分为**训练+验证**（可以使用K-则交叉验证对这一部分进一步划分）、**测试**

##### 过拟合、欠拟合

过拟合即模型对于训练数据过于精细化地捕捉即捕捉了过多的不具有普适性的噪音，相当于考试背答案，数据少模型大时容易出现该问题

| -          | 数据简单 | 数据复杂 |
| ---------- | -------- | -------- |
| 模型容量低 | 正常     | 欠拟合   |
| 容量高     | 过拟合   | 正常     |

<img src="NLP学习记录.assets/1664518431221.png" alt="1664518431221" style="zoom:50%;" />

泛化误差会随着模型变大后过于关注训练数据的细节而变大

给定一个模型种类，影响其的两个因素：参数个数、参数值的选择范围

###### 权重衰退

最常见的处理过拟合的方法，是通过约束参数的取值范围来实现的。

比如使用均方范数作为柔性限制$min \space l(w,b)+\frac{λ}{2}||w||^2$ 该柔性限制影响效果如下，将原本的最优解往原点方向拉扯了。新增加的项称为惩罚项

<img src="NLP学习记录.assets/1664634364577.png" alt="1664634364577" style="zoom:33%;" />

加入惩罚项后梯度更新法则如下：

<img src="NLP学习记录.assets/1664634774212.png" alt="1664634774212" style="zoom:50%;" />

通常$λ*η<1$，即每次将$w_t$缩小后再进行梯度更新，也因而被称为权重衰退

###### 丢弃法

对x加入噪音得到x', $E[x']=p*0+(1-p)\frac{x}{1-p}=x$

dropout是正则项，只在训练中使用(类似于MLM)，每次只随机采样一部分子cell进行传递

<img src="NLP学习记录.assets/1664681184533.png" alt="1664681184533" style="zoom:50%;" />

###### 数值稳定性

常见的两个问题：梯度爆炸、梯度消失

梯度爆炸会导致值超出值域、对学习率敏感

梯度消失会使得梯度值变为0，从而训练不会有进展，对回传到的底部层影响尤为严重

解决方法：

1.将乘法变加法，比如ResNet、LSTM

2.归一化，梯度归一化、梯度裁剪

3.合理的权重初始化和激活函数：

将每层的输出和梯度看作是随机变量，并让它们的均值和方差都保持一致

###### 序列模型

马尔可夫假设：当前数据只跟τ个过去数据点相关

$p(x_t|x_1,……,x_{t-1})=p(x_t|x_{t-τ},……x_{t-1})=p(x_t|f(x_{t-τ}，……，x_{t-1})$

可以使用MLP对已知的τ个数据进行自回归

潜变量模型：引入潜变量h_t来表示过去信息$h_t = f(x_1,……,x_{t-1})$,即$x_t=p(x_t|h_t)$

<img src="NLP学习记录.assets/1665108668531.png" alt="1665108668531" style="zoom: 50%;" />

可以分为两部分：1）通过前一时刻的$h_{t-1}$和x算当前时刻的h 2)根据算出的当前时刻h, 以及前一时刻的x算出当前时刻x

###### 语言模型

估计文本序列的联合概率给定文本序列x1，……，Xt，语言模型的目标是估计联合概率p(X1,……,Xt)

应用于1）预训练模型(e.g. bert,gpt-3) 2)生成文本，给定前面几个词，不断使用Xt~p(Xt|X1,……,Xt)生成后续文本 3）判断多个序列中哪个更常见(e.g. “to recognize speech”和“to wreck a nice beach” )

使用计数来建模 其中n为corpus的数量

seq_len=2时$p(x,x')=p(x)p(x'|x)=\frac{n(x)}{n}\frac{n(x,x')}{n(x)}$  n(x,x')为x与x'连续出现的次数

seq_len=3时$p(x,x',x'')=p(x)p(x'|x)p(x''|x,x')=\frac{n(x)}{n}\frac{n(x,x')}{n(x)}\frac{n(x,x',x'')}{n(x,x')}$

当序列很长时，可以用马尔可夫假设：

一元语法即τ=0 每个token在序列中出现概率都独立  $p(x_1,x_2,x_3,x_4)=p(x_1)p(x_2)p(x_3)p(x_4)$

二元语法 τ=1 每个token只与其前一个相关  $p(x_1,x_2,x_3,x_4)=p(x_1)p(x_2|x_1)p(x_3|x_2)p(x_4|x_3)=\frac{n(x_1)}{n}\frac{n(x_1,x_2)}{n(x_1)}\frac{n(x_2,x_3)}{n(x_2)}\frac{n(x_3,x_4)}{n(x_3)}$

三元语法 τ=2 每个token至多与其前两个相关 $p(x_1,x_2,x_3,x_4)=p(x_1)p(x_2|x_1)p(x_3|x_1,x_2)p(x_4|x_2,x_3)$

语言模型的输出就是判断下一个token，若字典大小为m，则为m分类问题，可以使用CrossEntropyLoss来衡量损失

衡量一个语言模型好坏可以使用**平均交叉熵**：

$π=\frac{1}{n}\sum_{i=1}^n-logp(x_t|x_{t-1},……)$

p(x_t|x_{t-1},……)即真实词的softmax输出

NLP使用**困惑度(perplexity**)，exp(π）来衡量，困惑度=1则说明只有一种可能，=k则k个token都有可能

RNN应用：

<img src="NLP学习记录.assets/1665294650745.png" alt="1665294650745" style="zoom:67%;" />

###### 梯度裁剪

如果梯度长度超过θ，则将其重置为θ。梯度裁剪可以有效预防梯度爆炸

$g<—min(1,\frac{θ}{||g||})g$

#### 从预训练到bert发展历程 

##### 预训练

预训练（最早在图像领域使用）是通过一个已经训练好的模型A，去完成一个小数据量的任务B（使用A的**浅层参数**

前提为任务A和B极其相似

预训练分为Frozen、Fine-Tuning两种，前者为与训练好的浅层参数不改变，后者将随下游任务进行改变。

预训练用法

fairseq、transformers库

##### 统计语言模型

###### 语言模型

是完成类似以下任务的工具

1.<u>**比较任务**</u> 判断句子出现的概率大小 P（“判断这个词的词性”），P（“判断这个词的雌性”）

2.**<u>预测任务</u>**token的推理 “判断这个词的`_____`”

**统计语言模型**则为使用统计的方法解决以上问题

以下使用了条件概率的链式法则

首先进行分词 “判断这个词的词性”=”判断“+”这个“+”词“+”的“+”词性“

再根据公式求出整句话出现的概率

<img src="NLP学习记录.assets/1665125692295.png" alt="1665125692295" style="zoom:50%;" />

第二个问题则为P(w_next|"判断"，"这个"，"词"，"的")

将词库中的词装入集合V, 把集合中的每一个词，都进行以上计算求max

当存在P(w_next|"判断"，"这个"，"词"，"的"，……)后缀很长时，计算量则增大，为了减少计算量，提出了n元计算模型。

###### n元统计语言模型

P(词性|"这个"，"词"，"的")

P(磁性|"这个"，"词"，"的")

P(词性|"词"，"的")

P(磁性|"词"，"的")

即把n个词，取2个词(2元)，3个词(3元)

如何去计算

```basic
“词性是动词”
“判断单词的词性”
“磁性很强的磁铁”
“北京的词性是名词”
```

P(词性|的)=$\frac{cout(词性，的)}{count(的)}$=$\frac{2}{3}$

P(策略|平滑)=$\frac{cout(平滑，策略)}{count(平滑))}$$\frac{0}{0}$

不存在的改用 *|V|*为词库大小，从而避免分子分母都为0，该方法称为平滑策略

<img src="NLP学习记录.assets/1665125704272.png" alt="1665125704272" style="zoom:50%;" />

##### 神经网络语言模型

使用神经网络的方法去完成以上两个任务

完成该任务的前提是让计算机认识单词，所以有以下方式

###### 独热编码(one-hot)



<img src="NLP学习记录.assets/1665125714439.png" alt="1665125714439" style="zoom:50%;" />

假设词典V中有8个单词，则one-hot给出一个8*8的矩阵，其中

“banana”对应00000001

“fruit”对应      01000000

该向量关系显然无法用余弦相似度计算二者相似度

<img src="C:\Users\Dust\AppData\Roaming\Typora\typora-user-images\1662081470367.png" alt="1662081470367" style="zoom:50%;" />

NNLM主要用来==预测下一个词== 公式为

$softmax(W_1(tanh((W_2,C)+b1)+b2)$

```basic
w1*Q=c1
w2*Q=c2
w3*Q=c3
w4*Q=c4
C=[c1,c2,c3,c4]
//Q是一个随机矩阵，是可学习的参数
```

这个替代one-hot的c，即为判断该词的**词向量**

###### 词向量

解决了one-hot占用存储空间过大，以及无法计算余弦相似度的问题

通过Q矩阵可以控制词向量的维度

并且乘出的新向量可以进行余弦相似度的计算

![1662083402317](C:\Users\Dust\AppData\Roaming\Typora\typora-user-images\1662083402317.png)

其中**Q矩阵的精准度合理度非常重要**，也推动着后期的发展

有了词向量后，也同时具备了完成**任务一**的条件。(下游任务？训练好Q以后任何子token即具有了词向量，从而可以处理更为细节的任务)

###### Word2Vec

也是神经网络语言模型->专门用来生成词向量

<img src="NLP学习记录.assets/1665125856851.png" alt="1665125856851" style="zoom: 67%;" />

word2Vec是一类模型分为

**CBOW**

给出一个词的==上下文==，预测该词

**Skip-gram**

给出一个词，预测这个词的上下文

二者都是为了得到Q矩阵，而NNLM则为了预测下一个词

故二者是为了训练好input端的Q矩阵

1. CBOW一个老师告诉多个学生，Q矩阵如何变
2. Skip-gram为多个老师告诉一个学生Q矩阵如何变

###### NNLM与Word2Vec区别

NNLM重在预测准确性，使用双层感知机$softmax(W_1(tanh((W_2,(x*Q))+b1)+b2)$

Word2Vec不在意预测准确性，故去掉激活函数，提高了计算效率

###### Word2Vec缺点

1. 无法表示多义词(由ELMO解决)

Word2Vec模型是否属于预训练模型?  `属于`



![1665125875682](NLP学习记录.assets/1665125875682.png)

如上图从words到embedding可以使用Word2Vec模型预训练好的Q矩阵直接将one-hot转换为词向量，故其属于预训练模型，从而也有两种使用方式1.Frozen冻结2.Fine-Tuning微调

###### ELMo模型

不只是训练一个Q矩阵，并且将上下文信息融入到Q矩阵中

![1665125890669](NLP学习记录.assets/1665125890669.png)

左边的LSTM获取En的上文信息，右边获取En的下文信息

通过上下文信息得到了词向量Tn，该词向量中包含了(多层特征：语义特征、句法特征、单词特征)

由于句法特征、语义特征在不同句子环境汇总不同，故同一个单词在不同句子中的词向量会有区分

###### Attention

对于一个模型(CNN、LSTM)很难决定什么数据重要或不重要

<img src="NLP学习记录.assets/1665125901411.png" alt="1665125901411" style="zoom:50%;" />

而人会根据图片中成分的重要性去聚焦

注意力机制中则是通过计算查询对象Q和被查询对象V之间的相似度来衡量重要性的

<img src="NLP学习记录.assets/1665125917860.png" alt="1665125917860" style="zoom:67%;" />

$(K_1,K_2,\cdots,K_n)*(Q_1,Q_2,\cdots,Q_n)=(s_1,s_2,\cdots,s_n)$

$softmax(s_1,s_2,\cdots,s_n)=(a_1,a_2,\cdots,a_n)$

$(a_1,a_2,\cdots,a_n)*+(V_1,V_2,\cdots,V_n)=(a_1*V_1+a_2*V_2+\cdots+a_n*V_n)=V'$

一般K=V，在Transformer中，K可以！=V但K和V之间一定有某种联系，这样QK计算来的相似度对于V的重要性才有意义

<img src="NLP学习记录.assets/1665125933456.png" alt="1665125933456" style="zoom:50%;" />

QK相乘求相似度后进行缩放，防止softmax后差距过大(指数函数增量大)，softmax后得到概率，V相乘后得到的$V^`$中隐含了V中信息对于Q而言的重要程度

###### self-attention

关键在于Q$\approx$K$\approx$V来源于同一个X(对X用不同的矩阵$W^Q、W^K、W^V$做空间变换得来)，即在注意力机制的基础上添加了约束条件

以下为处理第一个单词“Thinking”时self-attention的流程

1.通过学习来的$W^Q、W^K、W^V$以及词向量乘得q、k、v

<img src="NLP学习记录.assets/1665125943426.png" alt="1665125943426" style="zoom:50%;" />

2.分别计算q1 * k1、q2 * k2得到句子中所有成分对“Thinking”的注意分数

<img src="NLP学习记录.assets/1665125956411.png" alt="1665125956411" style="zoom:50%;" />

3.对score进行缩小，即除以$\sqrt{d_k}$，假设key的维度为64，则除以8（*为什们是除以k的维度？*）

4.对scale之后的分数进行softmax

<img src="NLP学习记录.assets/1665125969046.png" alt="1665125969046" style="zoom:50%;" />

5.将每个value向量乘softmax后的得分并求和得到新的$z_1$

<img src="NLP学习记录.assets/1665125969046.png" alt="1665125969046" style="zoom:67%;" />

$z_1=0.88*v_1 + 0.12*v_2$

self-attention相较于RNN、LSTM解决了长序列依赖问题，并且得到的新的**词向量**中包含了句法特征和语义特征（预训练语言模型优化的目的就是为了使词向量中内容更为准确，更有用）

句法特征

<img src="NLP学习记录.assets/1665125986872.png" alt="1665125986872" style="zoom: 67%;" />

语义特征

<img src="NLP学习记录.assets/1665125997416.png" alt="1665125997416" style="zoom: 67%;" />

带掩码的Masked Self-attention

由于在做生成任务时，生成的句子中单词是一个一个生成的，在计算注意力时不应将未生成的单词计算在内，故去掩码将其先掩盖

未做掩码前，一次性全部并行计算

<img src="NLP学习记录.assets/1665126009053.png" alt="1665126009053" style="zoom:50%;" />

加掩码后逐步计算(先计算已生成的单词)



<img src="NLP学习记录.assets/1665126017001.png" alt="1665126017001" style="zoom:50%;" />

上三角mask与pad mask求并集

<img src="NLP学习记录.assets/1663249568128.png" alt="1663249568128" style="zoom:50%;" />



###### muti-head self-attention

即将输入词向量分成多个通道进行self-attention(从多个维度向结果进行逼近)，然后将每个通道的得分结果重组为一个

工作流程

<img src="NLP学习记录.assets/1665126033726.png" alt="1665126033726" style="zoom:50%;" />

###### position encoding

self-attention的并行提高了计算效率，但增大了计算量（句子中的词与每个词的关系），并且失去了位置信息，故需要将位置信息添入新的词向量中

<img src="NLP学习记录.assets/1665126046995.png" alt="1665126046995" style="zoom:50%;" />



<img src="NLP学习记录.assets/1662695033360.png" alt="1662695033360" style="zoom: 33%;" />

具体公式：

$PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})$

$PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})$

根据
$$
\begin{cases}
sin(α+β)=sinαcosβ+cosαsinβ\\
cos(α+β)=cosαcosβ-sinαsinβ
\end{cases}
$$
将PE根据奇偶性展开得到
$$
\begin{cases}
PE(pos+k,2i)=PE(pos,2i)*PE(k,2i+1)+PE(pos,2i+1)*PE(k,2i)\\
PE(pos+k,2i+1)=PE(pos,2i+1)*PE(k,2i+1)-PE(pos,2i)*PE(k,2i)
\end{cases}
$$
对于pos+k位置的位置向量的某一维度2i或2i+1而言，可以表示为pos位置与k位置的位置向量的2i与2i+1维的线性组合

###### Transformer

seq2seq model with self-attention

![1665126062825](NLP学习记录.assets/1665126062825.png)

是一个seq2seq模型，序列到序列

编码器：把输入变成一个词向量(self-attention)

解码器：获取词向量后生成翻译结果

<img src="NLP学习记录.assets/1665126075823.png" alt="1665126075823" style="zoom:50%;" />

模型中Nx即层数，默认使用了6个编码器(每一块都是一个self-attention)，逐步增强词向量的精准度

编码器具体结构步骤

<img src="NLP学习记录.assets/1665126087879.png" alt="1665126087879" style="zoom:50%;" />

”Thinking“—>词向量x1(通过one-hot、word2vec等得到)+positional encoding—>x1'—>输入到self-attention与其他词向量此处为x2计算注意力—>z1 (该词向量具有位置、句法、语义信息)—>残差连接(避免梯度消失)+归一化—>z1''—>feedforward，Relu(w2(w1x+b1)+b2),此前都只在做线性变换(空间中的平移和放缩)，relu则是非线性变换，如此才可以拟合空间中的任一状态—>r1



<img src="NLP学习记录.assets/1665126095789.png" alt="1665126095789" style="zoom: 67%;" />

decoder的self-attention对已经生成的词进行编码，所以应该加mask

Transformer工作流程，其中decoder(masked)通过已生成的单词得到Q，encoder则用整句话生成K、V,再传给decoder单个块中的第二层Encoder-Decoder Attention做self-attention

<img src="https://media.giphy.com/media/dWA7drKRnF0tVuFWO5/giphy.gif" style="zoom:150%;" />

用Q与源语句K、V做自注意力机制，则可以知道源语句中哪些成分对接下来要生成的词更重要

###### GPT

![1663061246125](NLP学习记录.assets/1663061246125.png)

gpt无法做下游任务改造，因为其训练的是整个模型，即已经带有具体任务。对比于ELMo只用来生成更为精确的词向量，其输出结果可以与不同的下游任务对接

###### bert

参考了ELMO模型的双向编码思想，借鉴了GPT用Transformer作为特征提取器的思路，采用了word2vec所使用的CBOW方法

双向编码导致了需要看下文信息故无法进行生成式任务，而GPT则是单向生成式

对比ELMO使用的LSTM是按序传递，BERT使用的Attention能同时看到上下文信息

bert训练和GPT一样，分两步：

第一阶段：使用容易获取的大规模无标签数据，来训练基础语言模型

第二阶段：根据指定任务的少量带标签数据进行微调训练

由于使用的是Transformer的Encoder所以能看到整个上下文所以无法使用完形填空CBOW的思想

其解决方法是，在输入时将句子mask15%的token，然后让模型预测被masked的token，该方法称为MLM。该方法只在训练阶段使用，测试阶段不使用。

使用Mask后导致模型更多地关注在了被mask掉的token上

由于训练与测试阶段输入数据的不同故将被mask掉的15%又分为80%的[Mask]、10%的不改变、10%的错误替换，这样使得模型对于除mask掉之外的词也会被模型加以注意(否则将会认为除了[Mask]之外的所有数据都是正确的)

###### RNN

$h_0=f(Wx_0+b)$   $y_0=g(Wh_0+b)$

$h_1=f(Wx_1+Wh_0+b)$   $y_1=g(Wh_1+b)$

![1665292265294](NLP学习记录.assets/1665292265294.png)

当前时刻的输出是预测当前时刻的观察，但是输出发生在观察之前。在计算损失时比较Ot和Xt,但Xt用来更新Ht+1

$h_t=f(W_{hh}h_{t-1}+W_{hx}x_{t-1}+b_h)$

去掉$W_{hh}h_{t-1}$将退化成MLP，即增加了前一刻h与当前输出的相关性

$o_t=W_{ho}h_t+b_o$

###### lstm 

输入门作用于当前时刻的输入值，遗忘门作用于上一时刻的记忆值，二者加权求和得到汇总信息

最后通过输出门决定输出值

三个门的计算公式都为$sigmoid(Wx+Wh+b)$

## 论文精读

方法论：

三遍：

1. 标题+摘要+结论+实验部分图表 最终决定是否继续读
2. 重要图表的详细内容+圈出引用文献 
3. 复现作者的思路，并有自己的想法

①标题+作者

②摘要

③结论

④导言

⑤相关工作

⑥模型

⑦实验

⑧评论

### 1.Transformer 

②主流的序列转录模型（由所给序列生成目标序列）大多都基于复杂的循环或卷积神经网络，都有一个编码器和解码器。其中表现最佳的模型也会在编码器和解码器之间使用到注意力机制。基于注意力机制作者提出了一个新的简单的神经网络架构，**Transformer**，该模型仅仅基于注意力机制。

③Transformer 是第一个仅仅使用注意力机制的转录模型，它将之前的在编码解码器之间使用的循环层替换为了multi-head self-attention

在机器翻译这一任务上，Transformer训练地比其他传统的架构都要快。

④RNN对于一个序列的计算是从左往右一步一步做，对于第t个词会计算隐藏状态ht，该ht由前一个词的ht-1和当前词一起决定。该时序性的计算使得并行难以进行。

​	并且Attention机制早已应用于编码器与解码器的结合部，用来使编码器的东西很有效地传给解码器。

​	Transformer不再使用之前的循环神经层，而是仅使用注意力机制去描绘输入和输出之间的全局依赖关系。它支持更强的并行，并且可以在更短时间内完成更为高质量的任务。

⑤  Extended Neural 、GPU ByteNet 、ConvS2S都通过使用卷积神经网络为基本单位进行构建，并行计算所有输入输出位置的隐藏表示，从而减少顺序计算增加并发度。对于这些模型，将来自两个任意输入或输出位置的信号关联起来所需的操作数量随着位置之间的距离而增长，对于ConvS2S来说是线性增长，对于ByteNet来说是对数增长。

​	而在Transformer这些运算的数量被减少到了常量级别，以此为代价的是由于注意力权重位置的平均化导致的辨识度的降低，对于这一缺点，采用Multi-Head Attention机制来解决。

​	Self-attention,或者称为intra-attention，是将一个序列中不同位置关联起来的注意力机制，以计算序列的表示。

​	Transformer是第一个只使用自注意力机制来做encode、decode架构的模型

⑥ 大多数有竞争力的神经网络序列转录模型都有一个encoder-decoder架构。encoder将输入序列的符号表示x(x1,……xn)转换成一个连续的向量表示z(z1,……zn)。对于z decoder将一次解码出一个y最终生成序列y(y1,……yn)，每次生成都是一次auto-regressive自回归，对于yt则需要y1~yt-1作为输入。

 	Transformer也使用了encoder-decoder架构，具体来说该encoder-decoder使用了堆叠起来的self-attention 、point-wise和全连接层

​	编码器结构如下<img src="NLP学习记录.assets/1665126121911.png" alt="1665126121911" style="zoom: 67%;" />

输入先进入嵌入层，将词转换为向量，随后连接的是N层的由Muti-Head Attention以及Feed Forward(前馈神经网络)构成的块，【Add&Norm】中连接到Add的为

**残差连接**

(将浅层输出与深层输出求和 we hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping . To the extreme,  if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers 残差块使得训练很深的网络更加容易)，Norm为LayerNormalization

<img src="NLP学习记录.assets/1665126138890.png" alt="1665126138890" style="zoom:50%;" />

残差连接可以解决**梯度消失**的问题(防止梯度<1相乘后无限接近于0)，残差连接后使得梯度保持在1左右

在使用残差连接之前，更深的神经网络并不能比浅层神经网络具有更好的效果(更深的网络，误差率(训练+测试)反而更高)。

实际上浅层网络构建好后，后加的网络充当一个identity mapping 的话，深层网络的精确度不应该降低，但实际来说SGD并无法实现这一点。

残差连接则是将从前一层传递过来的$H(X)$不直接去学习，而是学习$H(X)-x$, 并在输出时加上那个减去的$x$.

即

《Identity Mappings in Deep Residual Networks》中介绍了各种residual块的设计。

​	LayerNorm与batchNorm比较(蓝色为batchNorm)，layerNorm是对一个样本所有特征进行计算，BatchNorm是对一个mini-batch中的一个特征进行计算

当输入为2D<img src="NLP学习记录.assets/1665126147831.png" alt="1665126147831" style="zoom:50%;" />

二者通过对数据的转置可以达到统一的效果

而RNN、Transformer中输入为3D，如图

<img src="NLP学习记录.assets/1665126156370.png" alt="1665126156370" style="zoom:50%;" />

由于LayerNorm、BatchNorm两种切法不同以及每个序列长度的不固定性，导致了BatchNorm在每次小批量计算时的均值方差的抖动相对较大 ，同时也导致其全局的均值方差不准确（可能新的序列长度过长或过短）；而LayerNorm小批量计算的是每个样本自己的均值和方差，并且也没有必要存储全局均值方差（测试时），故相对稳定。

<img src="NLP学习记录.assets/1665125059900.png" alt="1665125059900" style="zoom:50%;" />

解码器结构如下

<img src="NLP学习记录.assets/1665126164383.png" alt="1665126164383" style="zoom: 67%;" />

解码器的自回归机制(t-1时刻的输出作为t时刻的输入)，以及attention机制中能看到完整的输入，故需要带掩码的注意力机制即Masked Attention，来保证在t时间的输入不会看到t时间之后的内容

​	**Attention机制**就是将query查询内容根据键值对key-value中与key的相似度映射为一个output，其中key-value保持不变，随着query权重分配的变化，将会有不同的output。在计算相似度时，不同的Attention版本有不同的算法。(涉及到的所有数据都是向量)

​	Transformer在计算注意力时使用的是sclaed dot-product attention。该方法中query和key的维度相等，通过计算两个向量的内积来衡量其相似度，内积越大则相似度越高(？)（long相等的前提下）,**Attention(Q,K,V)=softmax(Q K内积/向量长度) V**。除以向量长度是防止两个向量长度比较长时，出现较大值的概率将会增加，该相对差距变大的可能性增加后使得softmax后该值更加靠近于1，剩余的值则更加靠近于0，在该种情况下softmax回归计算时梯度将会很小，不利于尽快收敛。而Transformer中的向量长度都是比较大的故应除以√dk。 计算流程图如下

<img src="NLP学习记录.assets/1665126171051.png" alt="1665126171051" style="zoom:67%;" />

##### Muti-Head Attention

<img src="NLP学习记录.assets/1665126183494.png" alt="1665126183494" style="zoom:67%;" />

相较于单个的注意力函数直接去计算高维的向量，将其投影到低维度并行地去计算更有好处 ，如上图将V、K、Q分别进行投影，投影h次，而每次投影时的W是一直在学习的。
$$
MultiHead(Q,K,V) = Concat(head_1,……,head_h)W^o
$$

$$
head_i= Attention(QW_i^Q,KW_i^K,VW_i^V)
$$

##### Position-wise Feed-Forward Networks

实际上是一个全连接的前馈神经网络，用来作用于每一个词(position)
$$
FFN(x) = max(0, xW1 + b1)W2 + b2
$$
W1将d=512的x扩大到d=2048，线性相加后Relu，然后用W2将维度降回512，最后再线性相加

##### Positional Encoding

用与embedding后数据位数等长的数据来表示该数据原始的位置信息，相加后即携带了该词的位置信息 		

⑦编码器和解码器的embedding 由于使用了统一的字典所以共享权重

⑧**评价**：Attention并不是ALL you need，其中的前馈神经网络、残差连接都缺一不可 。

### 2.Bert

②Abstract

Bidirectional Encoder Representations from Transformers

Bert全称为transformer模型的双向编码器表示

bert使得NLP的语言模型预训练正式出圈，它与最近的语言表示模型不同，bert通过联合所有层中左右的上下文信息，使用无标签的数据来训练深层双向的表示。预训练的bert模型只需要一个额外的输出层就能得到一个不错的结果。

（论文成果在摘要中写明基于什么工作，并且相对于该工作有何提升，再给出具体的实验数据，绝对精度+与当前最优相比提升的精度）

It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MutiNLI accuracy to 86.7% (4.6% absolute improvement) and SQuAD v2.0 Test F1 to 83.1(5.1 point absolute improvment)

已经存在的预训练模型分为基于特征的、基于微调的。

ELMo则属于基于特征的，每个下游任务都要构造一个与其相关的神经网络(RNN架构)，将预训练好的表示作为一个额外的特征同输入一起放入模型，使得模型训练起来比较容易。

GPT则是基于微调的，预训练好的参数在下游只需要微调

以上两个方案在预训练时都使用相同的目标函数，并且都使用单向的语言模型（为预测模型，预测下一个时刻所要输出的语言，故为单向）。

Bert则可用''带掩码的语言模型''(Masked language model, MLM)来减轻语言模型单向的限制，其灵感来自**Cloze task (Taylor，1953)**，具体来说，每次随机从输入中选择一些tokens并将其掩盖，目标函数则去预测这些被盖住的词（相当于进行完形填空），MLM允许去看左右两边的信息，这就使得我们可以训练出双向的深的Transformer

Bert是第一个在句子层面和词元层面取得好成绩的微调模型



③conclusion

最近一些实验表明，大量的、非监督的预训练对于很多语言模型来说是非常好的，这使得一些即使训练样本比较少的任务可以享受深度神经网络。bert的主要成果就是将已有成果拓展到了深的双向的架构上来，使得同样的预训练模型可以处理大量的不一样的NLP任务

⑤相关工作

非监督的基于特征的工作(ELMO)

非监督的基于微调的工作(GPT)

有标号的数据上做迁移学习

⑥BERT

该框架有两个步骤：1）预训练 2）微调

预训练时模型是在没有标号的数据集上训练的。

在微调时bert模型的权重被初始化为预训练时得到的权重，所有权重在微调时都会参与训练，并且使用的是下游任务的有标号的数据。

每一个下游任务都会单独建立一个模型并进行微调。

bert使用的架构是多层双向的Transformer编码器，该架构基于Transformer原始代码。

输入输出的表示上，输入统一为一个序列，从而无差别地表示一个句子或多个句子。这使得一个句子可以是连续文本中的任意跨度，而不是一个真实语义上的句子。

bert使用WordPiece去切词，每个序列的第一个词永远是[CLS(classificiation)]。句子之间用[SEP]特殊标记来分割。并且可以通过学习到的嵌入层来区分token属于A句子还是B句子

 <img src="NLP学习记录.assets/1665126193703.png" alt="1665126193703" style="zoom: 67%;" />

input经过三层embedding后求和，分别是词元本身的向量，所在句子信息向量和整体的position向量，如下图所示。

<img src="NLP学习记录.assets/1665126208275.png" alt="1665126208275" style="zoom:50%;" />

以上为预训练和微调的相同部分。

在预训练时对于每个序列的wordpiece的词元随机选取了15%来进行替换，但由于微调时不存在[MASK]符号，这将导致预训练和微调时数据的不匹配。为了缓和这一问题，将15%被选中的词元中80%的用[MASK]替代，10%的随机替换一个词元，剩余10%不进行操作。以上三种情况都会被标记为用来做预测。

微调时由于句子对放到了一个Transformer块中，所以self-attention可以来回看，比起encoder-decoder架构更优，由此付出的代价是无法再做机器翻译了。

⑦



step1：准备数据集

step2：设计模型即前馈过程中的元素和表达式

step3：构造损失函数和优化器

step4：进行训练epoch(forward、backward、upgrade)