# pytorch深度学习实践

详解：[《PyTorch深度学习实践》完结合集--B站刘二大人学习总结_pytorch 百度网盘_木马苇的博客-CSDN博客[](https://blog.csdn.net/mumawei123/article/details/127969221?ops_request_misc=&request_id=&biz_id=102&utm_term=《PyTorch深度学习实践》&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-127969221.142^v86^control_2,239^v2^insert_chatgpt&spm=1018.2226.3001.4187)

代码中"_"的说明 [Python中各种下划线的操作](https://zhuanlan.zhihu.com/p/105783765?utm_source=com.miui.notes) 

[TOC]



#### 第一个视频

##### 学习目的

学会基本语句和基本块，根据任务需求，组装自己的神经网络系统。

##### 普通算法与机器学习算法的区别：

机器学习算法来自数据集，而非人工设定。

##### 学习系统的发展

![](C:\Users\lxc\Desktop\typora图片\_DEKB2WZ5XQ2H[IIZ0RNL9U.png)

###### 维度诅咒

特征越多，维度越高，需要的数据集越大。

解决方法：映射降维（表示学习）

![](C:\Users\lxc\Desktop\typora图片\1.png)

表示学习：1.特征的挑选也要机器进行学习完成，使用专门的特征器。2.特征学习和模型分开训练，feature是没有标签的无监督学习，mapping是有标签的。

深度学习：1.提取简单的特征，进行基本的处理。2.特征和模型训练同时进行，是端到端的。3.额外的特征提取层。 

##### 神经网络的历史

神经元—感知器—神经网络。

核心：反向传播算法。其本质计算图+链式法则求导。顺着计算图方向是前馈计算。

计算图的优点：只需要确定原子计算，就可以求出所有的导数。

#### 第二个视频 线性模型

过程

1. Data Set：分为训练集（x,y）、开发集(x,y)和验证集(x)。开发集防止过拟合，增强泛化能力,看作预测试。验证集用于应用前测试。
2. Model: y_hat=f(x),先从线性开始，如果不符合，再找更复杂的模型。                    

$$
loss=(y_hat-y)2
$$

3. Training:

   任务：找到W使得平均平方误差（MSE）最低。

   方法：日志，绘图（visdom实时绘图）

   细节：存盘，断点。

4. inferring

   作业答案：[PyTorch学习（一）--线性模型_陈同学爱吃方便面的博客-CSDN博客](https://blog.csdn.net/weixin_44841652/article/details/105017087
   
   ##### 穷举法代码实现
   
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
    
   x_data = [1.0, 2.0, 3.0]
   y_data = [2.0, 4.0, 6.0]
    
    
   def forward(x):
       return x*w
    
    
   def loss(x, y):
       y_pred = forward(x)
       return (y_pred - y)**2
    
    
   # 穷举法
   w_list = []
   mse_list = []
   for w in np.arange(0.0, 4.1, 0.1):
       print("w=", w)
       l_sum = 0
       for x_val, y_val in zip(x_data, y_data):
           y_pred_val = forward(x_val)
           loss_val = loss(x_val, y_val)
           l_sum += loss_val
           print('\t', x_val, y_val, y_pred_val, loss_val)
       print('MSE=', l_sum/3)
       w_list.append(w)
       mse_list.append(l_sum/3)
       
   plt.plot(w_list,mse_list)
   plt.ylabel('Loss')
   plt.xlabel('w')
   plt.show()    
   ```
   
   1.zip把相同索引的元素放在同一个位置。
   
   2.我们使用`plt.plot()`函数来绘制折线图，并使用`plt.title()`、`plt.xlabel()`和`plt.ylabel()`函数来添加标题和坐标轴标签。最后，我们使用`plt.show()`函数来显示图表。
   
   3.NumPy可以与其他Python库（如pandas和matplotlib）一起使用，用于数据分析和可视化。
   
   ##### numpy库的常见用途
   
   - 处理多维数组：NumPy提供了一个称为`ndarray`的多维数组对象，可以用于存储和处理大量数据。这些数组可以是一维、二维或更高维的，可以进行基本的数学运算、逻辑运算、索引和切片等操作。
   
   - 数学函数：NumPy提供了许多数学函数，如三角函数、指数函数、对数函数、线性代数函数等。这些函数可以用于各种科学计算任务，如信号处理、图像处理、物理建模等。
   
   - 随机数生成器：NumPy提供了一个称为`random`的模块，可以用于生成各种类型的随机数。这些随机数可以被用于模拟实验、生成噪声数据、评估算法性能等。
   
     ##### 作业
   
     `from mpl_toolkits.mplot3d import Axes3D`是一个Python模块，用于在matplotlib中创建3D图形。它提供了一个名为`Axes3D`的类，可以用于创建3D图形对象，并在其中添加3D坐标轴、网格线、曲面等元素。
   
     用法：
   
     1. 导入必要的库：
   
     ```
     import matplotlib.pyplot as plt
     from mpl_toolkits.mplot3d import Axes3D
     ```
   
     2.创建3D图形对象：
   
     ```
     fig = plt.figure()
     ax = Axes3D(fig)
     ```
   
     `fig = plt.figure()`语句会创建一个新的`Figure`对象，并返回一个指向该对象的引用。我们可以使用这个引用来添加子图、设置图形属性、保存图形等。
   
     在matplotlib中，所有的图形都是在`Figure`对象中创建的。因此，我们需要先创建一个`Figure`对象，然后在其中添加一个或多个子图（`Axes`对象）。
   
     3.绘制3D图形：
   
     ```
     x = [1, 2, 3, 4, 5]
     y = [2, 4, 6, 8, 10]
     z = [5, 4, 3, 2, 1]
     ax.plot(x, y, z)
     ```
   
     这里，我们使用`ax.plot()`函数绘制一个简单的3D曲线。`x`、`y`和`z`是三个等长的数组，分别表示曲线上的点的x、y、z坐标。
   
     4.添加标签和标题：
   
     ```
     ax.set_xlabel('X Label')
     ax.set_ylabel('Y Label')
     ax.set_zlabel('Z Label')
     ax.set_title('3D Plot')
     ```
   
     这里，我们使用`ax.set_xlabel()`、`ax.set_ylabel()`和`ax.set_zlabel()`函数分别设置x、y和z坐标轴的标签。使用`ax.set_title()`函数设置图形的标题。
   
     5.显示图形：
   
     ```
     plt.show()
     ```
   
   #### 第三个视频 梯度下降算法

穷举法:适合一个w.

分治法：定义几个稀疏的点，找到最小点，再在最小点附近定义几个点，再找最小点。适合光滑的凸函数。

##### 梯度下降算法：

![](C:\Users\lxc\Desktop\typora图片\2.png)

原理：梯度的负方向就是函数减小的方向，由梯度极限定义可证。找到全局最优点。

学习率：下降的幅度，过大不会收敛。

局部最优：因为采取贪心算法，所以只能找到局部最优解。

鞍点：一维，梯度为0的点。多维，马鞍形状。

凸函数：凸函数是指在定义域上的任意两点之间的连线都位于函数图像上方的函数。换句话说，对于定义域上的任意两个点，连接它们的线段都不会穿过函数图像下方。凸函数具有很多重要的性质，例如局部最小值就是全局最小值，因此在优化问题中经常会用到凸函数。常见的凸函数包括线性函数和指数函数等。

代码实现：把表达式转换为代码，打印日志

```
import matplotlib.pyplot as plt

# prepare the training set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# initial guess of weight
w = 1.0#分治法


# define the model linear model y = w*x
def forward(x):
    return x * w


# define the cost function MSE
def cost(xs, ys):#cost是计算所有训练数据的损失
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


# define the gradient function  gd
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


epoch_list = []
cost_list = []
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val  # 0.01 learning rate，更新参数
    print('epoch:', epoch, 'w=', w, 'loss=', cost_val)#梯度总共更新100(epoch)次
    epoch_list.append(epoch)
    cost_list.append(cost_val)
print (w)
print('predict (after training)', 4, forward(4))#最后一个epoch的w
plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()
```

![](C:\Users\lxc\Desktop\typora图片\4.png)

图一：cost曲线太震荡   图二：发散

##### 随机梯度下降：

![](C:\Users\lxc\Desktop\typora图片\5.png)

特点：1.增加了随机性（随机选样产生的随机噪音），使得可能跨越鞍点进行迭代。2.不能并行，因为w的迭代与上一个损失有关。而梯度下降可以并行计算，效率较高。

代码实现：

```
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


# calculate loss function
def loss(x, y):#loss是计算一个训练数据的损失
    y_pred = forward(x)
    return (y_pred - y) ** 2


# define the gradient function  sgd
def gradient(x, y):
    return 2 * x * (x * w - y)


epoch_list = []#用来画图
loss_list = []
print('predict (before training)', 4, forward(4))#训练之前的预测结果
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - 0.01 * grad  # update weight by every grad of sample of training set
        print("\tgrad:", x, y, grad)#逗号输出就是空格，前面缩进
        l = loss(x, y)
    print("progress:", epoch, "w=", w, "loss=", l)#梯度总共更新100(epoch)x3 = 300次。
    epoch_list.append(epoch)
    loss_list.append(l)

print('predict (after training)', 4, forward(4))#训练后的预测结果,w是训练后的一个值，也是损失最小的值。
plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
min_value = min(loss_list)
print(min_value)
```

1、损失函数由cost()更改为loss()。cost是计算所有训练数据的损失，loss是计算一个训练数据的损失。对应于源代码则是少了两个for循环。

2、梯度函数gradient()由计算所有训练数据的梯度更改为计算一个训练数据的梯度。

3、本算法中的随机梯度主要是指，每次拿一个训练数据来训练，然后更新梯度参数。本算法中梯度总共更新100(epoch)x3 = 300次。梯度下降法中梯度总共更新100(epoch)次。

##### 小批量梯度下降算法：

![](C:\Users\lxc\Desktop\typora图片\7.png)

特点：时间度和效率的折中。

#### 第四个视频 反向传播

![](C:\Users\lxc\Desktop\typora图片\8.png)

w数量多，不方便一一表达，将其转化为计算图的形式。

![](C:\Users\lxc\Desktop\typora图片\9.png)

化简后，多出来的w没有意义，为了增加模型复杂度，引入非线性函数。

![](C:\Users\lxc\Desktop\typora图片\10.png)

##### pytorch训练

###### Tensor:可以储存标量和向量。

![](C:\Users\lxc\Desktop\typora图片\11.png)

张量和标量：张量是一个多维数组，它可以包含任意数量的数值，并且每个数值都有一个对应的索引。张量可以是一维的（向量）、二维的（矩阵）或更高维的。

Tensor默认不计算梯度

![](C:\Users\lxc\Desktop\typora图片\12.png)

当标量与Tensor进行计算时，会强制转换为Tensor。涉及Tensor的计算就是构建计算图。

![](C:\Users\lxc\Desktop\typora图片\13.png)

######  代码实现：

```
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])  # w的初值为1.0
w.requires_grad = True  # 需要计算梯度


def forward(x):
    return x * w  # w是一个Tensor


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("predict (before training)", 4, forward(4).item())#Tensor计算后的结果也是Tensor

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)  # l是一个张量，tensor主要是在建立计算图 forward, compute the loss
        l.backward()  # backward,compute grad for Tensor whose requires_grad set to True
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data  # 权重更新时，注意grad也是一个tensor
        w.grad.data.zero_()  # 如果梯度不清零，反向计算时会求和。

    print('progress:', epoch, l.item())  # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）

print("predict (after training)", 4, forward(4).item())
```

1.backward进行反向计算，计算后计算图消失。

2.纯数值的修改更新要使用Tensor里面的data。

3.item将Tensor里面的值取出来，作为标量，防止构建计算图。4.zero表示清零，如果梯度不清零，backward计算导数时，反向计算时会求和。

##### Tensor和tensor的区别

1.[torch](https://so.csdn.net/so/search?q=torch&spm=1001.2101.3001.7020).Tensor()是python类，更明确地说，是默认张量类型torch.FloatTensor()的别名，torch.Tensor([1,2])会调用Tensor类的构造函数__init__，生成单精度浮点类型的张量。

2.torch.[tensor](https://so.csdn.net/so/search?q=tensor&spm=1001.2101.3001.7020)()仅仅是python函数，torch.tensor会从data中的数据部分做拷贝（而不是直接引用），根据原始数据类型生成相应的torch.LongTensor、torch.FloatTensor和torch.DoubleTensor。

##### w.grad.item()和w.grad.data有什么区别

- `w.grad.item()`返回一个Python标量，即张量`w`的第一个元素的梯度值。如果张量`w`不是标量，则会引发一个错误。
- `w.grad.data`返回一个张量，其中包含张量`w`的梯度值。请注意，这个方法返回的是一个张量，而不是一个Python标量。

因此，如果你只需要访问张量`w`的第一个元素的梯度值，那么可以使用`w.grad.item()`。如果你需要访问整个张量`w`的梯度值，那么可以使用`w.grad.data`。

#### 第五个视频 线性模型

##### 总流程

![](C:\Users\lxc\Desktop\typora图片\15.png)

##### 构造数据集

![](C:\Users\lxc\Desktop\typora图片\16.png)

注意数据的维度，向量要用【】。

##### 设计模型

![](C:\Users\lxc\Desktop\typora图片\17.png)

loss最后必须是标量（求和或者求平均），否则不能使用backward。

实例化是指根据一个类创建出一个特定的实例对象的过程。在深度学习中，实例化通常指创建出一个具体的神经网络模型，该模型是基于某个深度学习框架中的特定类所定义的。实例化后，可以使用该模型对数据进行训练和预测。

通用模板：

![](C:\Users\lxc\Desktop\typora图片\18.png)

1.要定义成类，而不是对象。使用类和对象可以使我们更好地组织代码、提高代码复用性、降低代码维护成本等

2.继承nn.Module。

3.init,即构造函数，初始化你的对象时所默认要调用的函数。

4.用module构造出来的对象，它会自动地为你构造计算图，实现backward的过程。

5.如果想自己构造反向计算的函数，可以继承Functions。6.super（类名，self）._init_（）是必须做的，调用父类的init，直接写就行。

7.self.linear是个对象，linear是一个类，继承于module，可以自动计算backward。类名+（）就是构造一个对象，linear包含两个Tensor：权重（w）和偏置（b），可以帮我们计算w*x+b。

8.nn就是神经网络的缩写。

9.实例化：model=LinearModel（）。实例化是指根据一个类创建出一个特定的实例对象的过程。在深度学习中，实例化通常指创建出一个具体的神经网络模型，该模型是基于某个深度学习框架中的特定类所定义的。实例化后，可以使用该模型对数据进行训练和预测。

###### Linear：

![](C:\Users\lxc\Desktop\typora图片\19.png)

1.输入和输出的size必须相等。

2.行表示样本，列表示feature.所以feature表示每一个样本的维度。

3.bias默认值为ture。

![](C:\Users\lxc\Desktop\typora图片\20.png)

*args：将没带参数名的参数变成元组输入。

**kwargs：把带参数名的变量构成词典输入。

##### 构造损失和优化

损失模块：

![](C:\Users\lxc\Desktop\typora图片\21.png)

1.继承module。2.reduce表示求和降维。3.average表求平均。

优化：

![](C:\Users\lxc\Desktop\typora图片\22.png)

1.不继承module。2.params可用model.parameters（）找到。3.lr表示学习率，在pytorch中可以改变。

##### 循环训练

![](C:\Users\lxc\Desktop\typora图片\23.png)

1.loss是个Tensor，在打印是会自动调用str。2.step用于更新。

##### 代码实现：

```
import torch

# prepare dataset
# # x,y是矩阵，3行1列 也就是说总共有3个数据，每个数据只有1个特征
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

# design model using class
"""
our model class should be inherit from nn.Module, which is base class for all neural network modules.
member methods __init__() and forward() have to be implemented
class nn.linear contain two member Tensors: weight and bias
class nn.Linear has implemented the magic method __call__(),which enable the instance of the class can
be called just like a function.Normally the forward() will be called 
"""


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        #self是一个特殊的参数，它指代类的实例对象本身。在类的方法中，self通常作为第一个参数出现，并在方法内部被用来访问实例对象的属性和方法。
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    #覆盖了原有的forward


model = LinearModel()#实例化

# construct loss and optimizer
# criterion = torch.nn.MSELoss(size_average = False)
criterion = torch.nn.MSELoss(reduction='sum')#PyTorch中的一个类，它继承自torch.nn.modules.loss._Loss类，用于计算均方误差损失。
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 方法返回模型中所有可训练参数的迭代器。这些参数包括权重和偏置等变量
'''SGD是随机梯度下降算法，但这里是批量'''
# training cycle forward, backward, update
for epoch in range(100):
    y_pred = model(x_data)  # forward:predict
    loss = criterion(y_pred, y_data)  # forward: loss
    print(epoch, loss.item())

    optimizer.zero_grad()  # the grad computer by .backward() will be accumulated. so before backward, remember set the grad to zero
    loss.backward()  # backward: autograd，自动计算梯度
    optimizer.step()  # update 参数，即更新w和b的值

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
'''这段代码的作用是使用训练好的线性回归模型对一个新的数据样本进行预测，并输出预测结果。'''
x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
```

1.`__call___`: 使得类对象具有类似函数的功能。Module实现了魔法函数__call__()，call()里面有一条语句是要调用forward()。因此新写的类中需要重写forward()覆盖掉父类中的forward()。

2.`self`是一个特殊的参数，它指代类的实例对象本身。在类的方法中，`self`通常作为第一个参数出现，并在方法内部被用来访问实例对象的属性和方法。

#### 第六个视频 逻辑斯蒂回归

回归：在统计学和机器学习中，回归（Regression）是指通过对自变量（Independent variable）和因变量（Dependent variable）之间的关系进行建模，来预测因变量的值。线性回归（Linear Regression）是回归分析中最简单和最常用的一种方法，它假设自变量和因变量之间的关系是线性的，并且通过最小

化预测值与实际值之间的误差来确定模型参数。

分类问题：输出不同种类的概率，选最高的概率分类。服从概率分布，即总和为一。

##### The MNIST Dataset：一堆数字

![](C:\Users\lxc\Desktop\typora图片\25.png)

##### The CIFAR-10 database：彩色图片

![](C:\Users\lxc\Desktop\typora图片\26.png)

##### sigmoid函数

![](C:\Users\lxc\Desktop\typora图片\27.png)

又称为S型生长曲线，是这类饱和函数的统称。其中最典型的是logistic函数（将实数映射到0、1之间，做二分类）：

![](C:\Users\lxc\Desktop\typora图片\28.png)

##### 计算模块

![](C:\Users\lxc\Desktop\typora图片\29.png)

##### 交叉熵

![](C:\Users\lxc\Desktop\typora图片\30.png)

##### 损失函数BCE

![](C:\Users\lxc\Desktop\typora图片\31.png)

![](C:\Users\lxc\Desktop\typora图片\32.png)

- `weight`：一个张量，用于对损失函数的不同样本进行加权。默认值为None。
- `size_average`和`reduce`：这两个参数已经被合并到一个参数`reduction`中，用于指定损失函数计算方式的缩减方式。可选值有"mean"、"sum"和"none"。默认值为"mean"，表示对所有样本的损失值求平均。
- target就是yn，通常指的是样本的真实标签。

##### 代码实现

```
import torch
# import torch.nn.functional as F
 
# prepare dataset
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])
 
#design model using class
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)
 
    def forward(self, x):
        # y_pred = F.sigmoid(self.linear(x))
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
model = LogisticRegressionModel()
 
# construct loss and optimizer
# 默认情况下，loss会基于element平均，如果size_average=False的话，loss会被累加。
criterion = torch.nn.BCELoss(size_average = False) 
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
 
# training cycle forward, backward, update
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
 
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
```

1.逻辑斯蒂回归和线性模型的明显区别是在线性模型的后面，添加了激活函数(非线性变换)

2.预测与标签越接近，BCE损失越小。

3.  torch.sigmoid()、torch.nn.Sigmoid()和torch.nn.functional.sigmoid()三者之间的区别

   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200729151202585.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjYyMTkwMQ==,size_16,color_FFFFFF,t_70)

![image-20230516143508741](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230516143508741.png)

![image-20230516143635029](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230516143635029.png)

可以直接在正向传播中使用。

#### 第七个视频 多维特征的输入

![](C:\Users\lxc\Desktop\typora图片\34.png)

1.多维特征当作矩阵输入，修改linear的参数值。矩阵就是空间变换的函数，可以降维和升维。

2.通过激活函数对线性变化加上非线性变化，来拟合非线性变化。只需要调整线性变化的维度参数。

3.变化的层数决定模型的学习能力，具体数值的选择是超参数搜索问题。层数越多学习能力越强，但是学习能力过强，会把噪声学过来，要提高泛化能力。

##### 第一步 准备数据

![](C:\Users\lxc\Desktop\typora图片\35.png)

1.loadtxt（文件名或文件句柄，分隔符，数据类型,`skiprows`：要跳过的行数，默认为0）只能读入数据，所以要跳过标题行。

- `usecols`：要读取的列索引，默认为所有列。读取的对象。
- `unpack`：如果为True，则
- 返回每列作为单独的数组。默认为False。

2.xy合在一起，用切片输入。第一个‘：’是指读取所有行，第二个：是指从第一列开始，最后一列不要。

3.loadtxt数据集需和源代码放在同一个文件夹内。

##### 第二步 定义模型

![](C:\Users\lxc\Desktop\typora图片\36.png)

1.nn下面的sigmoid是一个模块，也继承于module，看作一个层。没有参数，不需要训练，只需要一个，用来构建计算图。

2.上层的输出是下层的输入，这种模型就用一个变量X。

3.在这个模型中，三层非线性变换可以学习到更加复杂的特征表示，从而提高了模型对输入数据的抽象能力和分类准确率。

##### 降低特征维度

1. 减少计算量：在高维数据中，计算量往往非常大，而降低特征维度可以减少计算量。
2. 去除冗余信息：在实际问题中，数据往往包含大量冗余信息，这些信息对于模型训练并没有太大帮助，反而会增加计算量和噪声。通过降低特征维度可以去除冗余信息，从而提高模型的训练效率和准确率。
3. 避免过拟合：在高维数据中，模型容易出现过拟合的情况，即模型在训练集上表现很好，但在测试集上表现很差。通过降低特征维度可以减少特征数量，从而避免过拟合的发生。
4. 可视化：在一些场景下，高维数据难以可视化，而通过降低特征维度可以将数据投影到二维或三维空间中，从而方便可视化和理解。

##### 最后两步

![](C:\Users\lxc\Desktop\typora图片\37.png)

![](C:\Users\lxc\Desktop\typora图片\38.png)

#### 第八个视频 加载数据库

##### 名词理解：

![](C:\Users\lxc\Desktop\typora图片\39.png)

##### Dataloader示例：

![](C:\Users\lxc\Desktop\typora图片\40.png)

```python
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
 
# prepare dataset
 
 
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0] # shape(多少行，多少列)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
 
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
 
    def __len__(self):
        return self.len
 
 
dataset = DiabetesDataset('diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0) #num_workers 多线程
 
 
# design model using class
 
 
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
 
 
model = Model()
 
# construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
 
# training cycle forward, backward, update
if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0): # train_loader 是先shuffle后mini_batch
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
 
            optimizer.zero_grad()
            loss.backward()
 
            optimizer.step()
```

1、需要mini_batch 就需要import DataSet和DataLoader

2、继承DataSet的类需要重写init，getitem,len魔法函数。分别是为了加载数据集，获取数据索引，获取数据总量。

3、DataLoader对数据集先打乱(shuffle)，然后划分成mini_batch。

4、len函数的返回值 除以 batch_size 的结果就是每一轮epoch中需要迭代的次数。

5、inputs, labels = data中的inputs的shape是[32,8],labels 的shape是[32,1]。也就是说mini_batch在这个地方体现的
6.当一个 Python 文件被直接运行时，它就是作为主程序运行的。如果一个 Python 文件被导入为模块，在其他文件中使用 `import` 命令导入该模块时，它就不是作为主程序运行的。在这种情况下，`if __name__ == '__main__':` 后面的代码块不会被执行。

6.当我们使用 `import` 导入一个 Python 文件时，Python 解释器会执行该文件中的所有代码，并将其中定义的函数、类、变量等对象加载到当前模块的命名空间中，以便在当前模块中使用。

##### 定义数据集

![](C:\Users\lxc\Desktop\typora图片\41.png)

例子：

![](C:\Users\lxc\Desktop\typora图片\43.png)

![](C:\Users\lxc\Desktop\typora图片\44.png)

1.Dataset是一个抽象集，可以创建一个类继承它。

   DataLoader可实例化，参数（数据集，bs，是否打乱，并行计算的程序数）。

2.init有两种选择：All Data，把所有数据读进内存，根据索引提取，适用于有结构的小数据集。把数据打包成文件，然后把文件名初始化为列表，按照索引访问需要的文件，适用于无机构的大数据集。

3.getitem：根据索引提取数据

4.len：数据集的长度

5.shape提取N

6.return返回的是（x，y）的元组，由loader转化为Tensor。

##### Windows中的DataLoader

![](C:\Users\lxc\Desktop\typora图片\42.png)

1.Windows中多线程计算的方式和Linux不同，所以需要加一步封装。

2.`last_drop`是`Dataloader`类中的一个参数，用于指定是否在数据集最后一个batch中丢弃不足batch_size的数据。如果`last_drop`为`True`，则最后一个batch的数据量可能会小于batch_size；如果为`False`，则最后一个batch的数据量会和之前的batch一样，但是可能会有一些数据被重复使用。如果最后一个batch中的数据量小于batch_size，可能会导致模型在最后一轮训练中没有充分地利用GPU的计算能力，从而影响模型的训练效果。此外，如果最后一个batch中的数据量太少，可能会导致模型在训练过程中出现过拟合等问题。因此，在使用batch训练模型时，通常会选择丢弃最后一个batch中不足batch_size的数据，以保证每个batch的数据量都相同。

##### 自带数据库

![](C:\Users\lxc\Desktop\typora图片\45.png)

#### 第九个视频 多分类

##### softmax

![image-20230504103334475](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230504103334475.png)

##### 多分类交叉熵和代码实现

![image-20230504104205527](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230504104205527.png)

![image-20230504105423540](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230504105423540.png)

`torch.nn.CrossEntropyLoss()` 是 PyTorch 中的一个损失函数，通常用于多分类任务中。在训练神经网络时，我们需要定义一个损失函数，用于衡量模型输出与真实标签之间的差异，并通过反向传播算法更新模型的参数。在多分类任务中，通常使用交叉熵损失函数来衡量模型输出与真实标签之间的差异。

具体来说，`torch.nn.CrossEntropyLoss()` 函数将 Softmax 函数和负对数似然损失函数（Negative Log Likelihood Loss）结合起来，计算模型输出与真实标签之间的交叉熵损失。在计算交叉熵损失时，需要将模型输出作为输入，并将真实标签作为目标值。

在使用 `torch.nn.CrossEntropyLoss()` 函数时，通常需要注意以下几点：

- 输入和目标值的形状应该相同，并且输入应该是未经 Softmax 处理的原始输出。
- 目标值应该是一个 LongTensor 类型的张量，其中每个元素都是一个类别标签。
- 如果模型输出已经经过 Softmax 处理，则可以使用 `torch.nn.NLLLoss()` 函数来计算交叉熵损失。

#### MINISET代码实现

![image-20230504110938540](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230504110938540.png)

1.transform是对图像做原始处理的。transforms是将图像由读取进来的PIL（是一个Python图像处理库，可以用来打开、操作和保存多种格式的图像文件）转换为Tensor，并且转换为cwh通道，并将值归一化。

2.使用函数relu和优化器包。

![image-20230504111438550](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230504111438550.png)

这是一个 PyTorch 中的数据预处理管道，它使用了 `transforms.Compose()` 函数将多个数据预处理操作串联起来，以便对数据进行一系列的预处理操作。具体来说，该管道包含两个操作：

- `transforms.ToTensor()`：将数据转换成 PyTorch 中的 Tensor 格式。这个操作会将数据的类型从 PIL.Image.Image 或 numpy.ndarray 转换成 torch.Tensor，并将数据的值缩放到 [0,1] 的范围内。
- `transforms.Normalize((0.1307,), (0.3081,))`：对数据进行标准化处理。这个操作会将数据的均值和方差分别减去和除以给定的参数，从而使数据的均值为 0，方差为 1。这里给定的均值和方差分别是 MNIST 数据集在训练集上的均值和方差。

![image-20230504111715022](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230504111715022.png)

1.三个通道：R、G、V

2.在pytorch里面要把C也就是通道放在前面。1就是通道。

3.softmax的输入不需要再做非线性变换，也就是说softmax之前不再需要激活函数(relu)。softmax两个作用，如果在进行softmax前的input有负数，通过指数变换，得到正数。所有类的概率求和为1。

![image-20230504115922207](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230504115922207.png)

`view()` 是一个函数，用于改变张量的形状。具体来说，这个操作将 `x` 张量的形状从原来的 `(batch_size, num_channels, height, width)` 改变为 `(batch_size, 784)`，其中 `-1` 表示该维度的大小将根据其他维度自动确定。

![image-20230504120031198](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230504120031198.png)

![image-20230504120529153](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230504120529153.png)

1.具体来说，`enumerate` 函数接受一个可迭代对象（如列表、元组、字符串、字典等），并返回一个枚举对象。枚举对象是一个迭代器，每次迭代时都会返回一个包含两个元素的元组，第一个元素是当前元素的索引，第二个元素是当前元素的值

`enumerate` 函数返回的索引默认从 0 开始，可以通过传递第二个参数来指定起始索引。

2.假设 `batch_idx` 是当前数据批次的索引，`running_loss` 是累计的总损失函数值，那么 `if batch_idx % 300 == 299:` 判断当前批次是否为 300 的倍数减 1，如果是，则输出当前平均损失函数值，并将 `running_loss` 重置为 0。这样可以在每处理完 300 个数据批次后输出一次平均损失函数值。

![image-20230504120546205](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230504120546205.png)

![image-20230504120558087](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230504120558087.png)

1.代码中使用 `torch.max` 函数找到输出结果中概率最大的类别，并将其作为模型的预测结果。然后将预测结果与标签进行比较，如果相同则认为预测正确，否则认为预测错误.最后累加预测正确的样本数量和总样本数量，以计算模型在测试集上的准确率。

2. `with torch.no_grad():`表示在测试过程中不需要计算梯度，可以加快计算速度并减少内存消耗

3. 基本思想是:with所求值的对象必须有一个enter()方法，一个exit()方法。

   紧跟with**后面的语句被求值后，返回对象的**__enter__()方法被调用，这个方法的返回值将被赋值给as后面的变量。当with后面的代码块全部被执行完之后，将调用前面返回对象的exit()方法。

4.比较运算

![image-20230504121118827](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230504121118827.png)

序号和标签是否匹配。

#### 10 卷积网络基础篇

##### 总过程

![image-20230504162612451](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230504162612451.png)

##### 卷积的宏观理解

![image-20230504162644405](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230504162644405.png)

##### 卷积的微观理解

参数及组成部分：

卷积层是卷积神经网络中最基本的层之一，它由多个卷积核和偏置项组成。卷积层的参数由以下组成部分构成：

1. 卷积核：卷积核是一组权重，用于对输入张量进行卷积操作。卷积核通常是一个小的矩阵，例如3x3或5x5，并且在整个输入张量上滑动以执行卷积操作。每个卷积核都会产生一个输出通道，因此卷积层中的卷积核数量决定了输出通道数和特征映射数量。
2. 偏置项：偏置项是一组常数，用于调整每个输出通道的值。每个输出通道都有一个对应的偏置项，它与卷积核计算出的结果相加得到最终输出张量中的单个像素值。
3. 激活函数：激活函数是一种非线性函数，用于将输出张量中的每个像素值进行转换。常见的激活函数包括ReLU、Sigmoid和Tanh等。
4. 填充方式：填充是一种在输入张量周围添加额外像素的方法，用于控制输出张量的大小。常见的填充方式包括“valid”和“same”两种。
5. 步幅：步幅是卷积核在输入张量上滑动的距离，用于控制输出张量的大小。步幅越大，输出张量越小。

卷积层的参数是通过反向传播算法进行学习的，以最大化网络对训练数据的拟合能力。

单通道

![image-20230504162711486](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230504162711486.png)

窗口机制加数乘（1*1+6*2+7*3）

多通道

![image-20230504163120504](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230504163120504.png)

1.有几个通道就配置几个核。

![image-20230504163213717](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230504163213717.png)

![image-20230504163354266](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230504163354266.png)

多输出

![image-20230504165726728](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230504165726728.png)

卷积神经网络的输出通道数取决于卷积层中使用的滤波器数量。每个滤波器将产生一个输出通道。因此，如果卷积层使用n个滤波器，则输出通道数为n。在实践中，滤波器的数量通常是通过试验和调整来确定的，以找到一个适合特定任务和计算资源的最佳数量。

计算过程：

![image-20230505141401831](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230505141401831.png)

卷积层最重要的三个值：输入通道、输出通道和卷积核大小。

##### padding（填充）

输出尺寸 = (输入尺寸 - 卷积核尺寸 + 2 x padding) / 步长 + 1

其中，输入尺寸指的是输入数据的宽度或高度，卷积核尺寸指的是卷积核的宽度或高度，步长指的是卷积操作每次移动的步长，padding指的是在输入数据周围添加的额外像素数。如果输出尺寸不是整数，则通常向下取整。

![image-20230505143435442](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230505143435442.png)

计算代码实现：

![image-20230505143458396](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230505143458396.png)

##### stride和maxpooling

![image-20230505150138957](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230505150138957.png)

![image-20230505150224793](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230505150224793.png)

##### 模型

![image-20230505151159481](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230505151159481.png)

##### GPU

###### move model to gpu

![image-20230505151611438](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230505151611438.png)

1.如果有GPU就返回ture，没有就返回false。

2.如果有多张显卡，可以选择，默认第一张显卡cuda0

###### move tensors to gpu

![image-20230505151856951](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230505151856951.png)

1.要转移到与模型的显卡上。

2.测试也要加上这一句。

![image-20230505152040120](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230505152040120.png)

#### 第11个视频 高级的cnn

##### GoogleNet

1.超参数卷积核尺寸难以确定，都使用一下，然后选择最好的。

##### Inception Module

![image-20230507144351547](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230507144351547.png)

1.concatenate:把张量合在一起。

2.1*1张量：

![image-20230507151426279](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230507151426279.png)

作用：减少计算。

###### Inception代码实现：

```python
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
 
# prepare dataset
 
batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # 归一化,均值和方差
 
train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
 
# design model using class
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
 
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)
 
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
 
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)#定义在init里面
 
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
 
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
 
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
 
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)#写在forwaed里面
        branch_pool = self.branch_pool(branch_pool)
 
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1) # b,c,w,h  c对应的是dim=1，把他们堆在一起。
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5) # 88 = 24x3 + 16
 
        self.incep1 = InceptionA(in_channels=10) # 与conv1 中的10对应
        self.incep2 = InceptionA(in_channels=20) # 与conv2 中的20对应
 
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10) 
 
 
    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
 
        return x
 
model = Net()
 
# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
 
# training cycle forward, backward, update
 
 
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
 
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
 
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0
 
 
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100*correct/total))
 
 
if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
```

代码说明：1、先使用类对Inception Moudel进行封装

 2、先是1个卷积层(conv,maxpooling,relu)，然后inceptionA模块(输出的channels是24+16+24+24=88)，接下来又是一个卷积层(conv,mp,relu),然后inceptionA模块，最后一个全连接层(fc)。

3、1408这个数据可以通过x = x.view(in_size, -1)后调用x.shape得到。在fc层包含1408个元素。

##### test

测试回合不是越多越好，有时候可能一半的时候就已经达到最优，继续测试会造成过拟合。

##### 梯度消失


![image-20230507160652867](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230507160652867.png)

1.w得不到更新。

###### 解决方法

![image-20230507162001906](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230507162001906.png)

f（x）尺寸必须和x相同。

##### residual block

###### 示意图

![image-20230507162421153](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230507162421153.png)

###### 代码实现

```
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
 
# prepare dataset
 
batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # 归一化,均值和方差
 
train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
 
# design model using class
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
 
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5) # 88 = 24x3 + 16
 
        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)
 
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(512, 10) # 暂时不知道1408咋能自动出来的
 
 
    def forward(self, x):
        in_size = x.size(0)
 
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
 
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x
 
model = Net()
 
# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
 
# training cycle forward, backward, update
 
 
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
 
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
 
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0
 
 
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100*correct/total))
 
 
if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
```

代码说明：

1、先是1个卷积层(conv,maxpooling,relu)，然后ResidualBlock模块，接下来又是一个卷积层(conv,mp,relu),然后esidualBlock模块模块，最后一个全连接层(fc)。

#### 第12个视频 基础的rnn

##### 权重数量计算公式

![image-20230508114428657](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230508114428657.png)

![image-20230508114449252](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230508114449252.png)

![image-20230508114507286](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230508114507286.png)

##### RNN处理对象

 可以处理带有序列的数据，时间上有先后，逻辑上后面的依赖于前面的。比如自然语言。因为他是顺序处理的。还要使用共享权重减少权重使用量。

##### 示意图

![image-20230508134839263](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230508134839263.png)

1.关系：每个RNN Cell都是同一线性层。

2.代码理解

![image-20230508135213740](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230508135213740.png)

在pytorch中应用

![image-20230509163851627](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230509163851627.png)

![image-20230509163910152](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230509163910152.png)

![image-20230509164146086](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230509164146086.png)

![image-20230509164204022](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230509164204022.png)

例子

数据准备

![image-20230509164258901](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230509164258901.png)

![image-20230509164314439](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230509164314439.png)

独热（One-Hot）向量

设计模型

![image-20230509164655942](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230509164655942.png)

损失和优化

![image-20230509164718246](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230509164718246.png)

训练

![image-20230509164739263](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230509164739263.png)

1.input、label、inputs和labels的维度。

2.loss要用计算图求和，所以不用item。

优化模型：压缩层

![image-20230509165117619](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230509165117619.png)

![image-20230509165129238](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230509165129238.png)

形成一个矩阵

![image-20230509165138244](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230509165138244.png)

通过矩阵乘法提取目标行

![image-20230509165145724](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230509165145724.png)

1.RNN其实是有线性共享的，下图中的输入层x的维度可以和输出hidden层的维度不一致，但最终都会进行线性变换达成一致。

2.输入必须是longtensor。

![image-20230509165340219](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230509165340219.png)

#### 第13个视频 RNN高级篇

精简后的模型

![image-20230510125601289](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230510125601289.png)

循环代码

![image-20230510131237284](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230510131237284.png)

1.RNNclassifier（字符长度，隐层大小，分类数量，RNN层数）

2.把训练和测试的结果记录到列表里，方便绘图。

准备数据：

![image-20230510132411551](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230510132411551.png)

1.77代表128维向量中77为1，其余为0.只需要告诉嵌入层，1的位置就行。

![image-20230510132852134](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230510132852134.png)

保证能够形成一个张量。

代码：

![image-20230510133153000](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230510133153000.png)

1.使用gzip和csv来读取gz文件。

2.rows是由name和language构成。

3.getitem返回的名字是字符串，返回的国家是键值对。

![image-20230510134741582](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230510134741582.png)

1.把列表转化成字典。

![image-20230510135043926](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230510135043926.png)

模型设计

![image-20230510135350142](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230510135350142.png)

1.注意hidden_size、n_layers用在什么地方。

2.hidden=torch.zeros创造一个全0的隐层。

3.bidirectional表示双向（双向direction=2，否则为1）。序列既有关于过去，又有关于未来。前向计算的h又与反向计算的h拼接起来。

![image-20230510140216428](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230510140216428.png)

上面的h是output

![image-20230510141930415](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230510141930415.png)

1.input.t（tanspose）表示矩阵转置，seq*batch是嵌入层需要的矩阵。嵌入层输入维度

![image-20230510142833159](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230510142833159.png)

2.隐层大小

![image-20230510142741037](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230510142741037.png)

3.packsequence便利了gpu的计算，减少了padding后0部分的计算。

![image-20230510143641019](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230510143641019.png)

准备工作：按照序列长度进行逆排序，然后经过嵌入层变化。形成s*b*h。

![image-20230510143913214](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230510143913214.png)

将嵌入层的结果横向整合到data上，并记录长度。在时刻按照长度进行访问数据。

![image-20230510150756179](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230510150756179.png)

1.name2list返回列表本身和列表长度。

2.第一块是把名字转变成ASCII码的列表，第二块创建全0列表，然后复制，达到填充效果。第三块就是进行排序。第四块转化为tensor。

训练

![image-20230510151401719](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230510151401719.png)

测试

![image-20230510152429635](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230510152429635.png)