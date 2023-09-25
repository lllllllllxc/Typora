# 自然语言处理NLP—蓝斯洛特

在 NLP 中，通常需要对文本进行分词、词性标注、命名实体识别、句法分析、情感分析、机器翻译等任务。这些任务都需要对自然语言数据进行处理和分析，而深度学习提供了一种有效的方法来处理这些数据。通过构建深层神经网络，可以将自然语言数据转化为数值型数据，并进行高效的处理和分析。

#### 视频2 one—hot情感分类

##### 数据解读

![image-20230522140635900](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230522140635900.png)

-1代表词典中没有这个词。

##### loadtxt和pd.read_csv的区别

`loadtxt` 是 NumPy 库中的函数，用于从文本文件中加载数据。它可以读取普通的文本文件、CSV 文件等。但是，它不支持读取包含不同数据类型的表格数据，也不能处理缺失值等情况。

使用时，要把csv文件和运行代码放在一个文件夹中，然后读取文件名。

`pd.read_csv` 是 Pandas 库中的函数，专门用于读取 CSV 文件。它可以轻松地读取和处理包含表格数据的 CSV 文件，并支持处理缺失值、不同数据类型等情况。因此，在读取 CSV 文件时，我们通常会使用 `pd.read_csv` 而不是 `loadtxt`。

直接读取文件路径。

##### torch.empty

`torch.empty` 是一个函数，用于创建一个未初始化的张量。参数是张量的尺寸。

##### random.randint

`random.randint(a, b)` 是 Python 中 `random` 模块中的一个函数，用于生成一个在指定范围内的随机整数。

##### `iloc` 是 Pandas 中的一个属性

具体来说，`iloc` 属性可以接受一个整数、整数列表或布尔数组作为参数，用于选择数据。如果参数是整数，则返回相应位置的数据；如果参数是整数列表，则返回相应位置的数据子集；如果参数是布尔数组，则返回相应位置为 True 的数据。

iloc（行，列）

##### classifier.eval()

`classifier.eval()` 是 PyTorch 中用于设置模型为评估模式的函数。在评估模式下，模型不再进行参数更新，而是根据当前的参数对输入数据进行预测，并输出预测结果。

在 PyTorch 中，通过调用 `model.eval()` 将模型设置为评估模式，通过调用 `model.train()` 将模型设置为训练模式。在评估模式下，模型会关闭 Dropout 和 Batch Normalization 等训练时使用的技巧，从而保证模型的稳定性和可靠性。

#### 视频3 CNN姓名分类

##### 数据解读

![image-20230523121431069](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230523121431069.png)

第0列是把姓名用字母的方式表示。

第1列是姓名的分类。

`argmax()`函数

`argmax()`函数将概率分布转换为类别标签。对于每个样本，它返回具有最高概率的类别标签。

`argmax()`函数是一个Numpy函数，用于返回数组中最大值的索引。

#### 视频4 RNN姓名分类

![image-20230528112411690](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230528112411690.png)

分类任务中记忆比较重要，翻译任务中输出也很重要。

##### 视频5 词转向量

![image-20230529151920880](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230529151920880.png)

减少维数，增加了字之间的关联性。

![image-20230529151954007](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230529151954007.png)

##### `torch.mean` 

`torch.mean` 是 PyTorch 中的一个函数，用于计算张量的平均值。它可以沿着指定的维度计算平均值，也可以计算整个张量的平均值。第一个参数是张量，第二个参数是维度。

##### `plt.scatter()` 

`plt.scatter()` 是 matplotlib 库中的一个函数，用于绘制散点图。它可以将多个点在二维平面上绘制出来，其中每个点的位置由横、纵坐标决定，颜色和大小也可以根据需要进行设置。

函数的参数包括：

- `x`：表示点的横坐标，可以是一个数组或列表；
- `y`：表示点的纵坐标，可以是一个数组或列表；
- `s`：表示点的大小，可以是一个标量或与 `x`、`y` 一样长的数组或列表；
- `c`：表示点的颜色，可以是一个标量或与 `x`、`y` 一样长的数组或列表；
- `marker`：表示点的形状，比如圆圈、正方形等；
- `cmap`：表示颜色映射，可以是一个字符串或 Colormap 对象；

#### 视频9 常见NLP计算

##### N to N

翻译和写诗

##### N to 1

分类任务

##### 1 to N

看图说话

##### N to M

![image-20230530103045761](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230530103045761.png)

#### 视频10 RNN和LSTM

![image-20230530104546651](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230530104546651.png)

x与h矩阵拼合在一起，与w矩阵相乘，加上偏置。

#### 视频11 序列到序列的日期翻译

##### `torch.stack` 

`torch.stack` 是一个 PyTorch 中的函数，用于将多个张量按照指定维度进行堆叠。

具体来说，`torch.stack(tensors, dim=0)` 将多个张量按照指定维度 `dim` 进行堆叠，返回一个新的张量。假设输入的张量列表 `tensors` 的形状为 `(N, *)`，则输出的张量的形状为 `(N, *new_shape)`，其中 `*new_shape` 表示除了第 `dim` 维以外的所有维度。

例如，假设有两个形状为 `(3, 4)` 的张量 `a` 和 `b`，则执行 `torch.stack([a, b], dim=0)` 的结果为一个形状为 `(2, 3, 4)` 的新张量，其中：

- 第一维是堆叠维度，即第 `dim` 维，长度为 2；
- 第二维和第三维是原来的维度，即第 0 维和第 1 维，长度分别为 3 和 4。

#### 视频13 注意力翻译日期

##### torch.cat

`torch.cat` 是 PyTorch 中的一个函数，用于沿着指定维度连接张量。它接受一系列张量和要连接的维度，并返回一个新的张量，其中包含连接的数据。

参数：连接矩阵，维度。连接维度上相加。

#### 视频14 ELMO

![image-20230605091049117](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230605091049117.png)

1.数据

读取字典

```
zidian = {}
with open('E:/git库/NLP-Toturials/data/msr_paraphrase/zidian.txt') as fr:
    for line in fr.readlines():
        k, v = line.split(' ')
        zidian[k] = int(v)

zidian['<PAD>'], len(zidian)
```

读取文件，用iloc提取数据。

```
 def __init__(self):
        self.data = pd.read_csv('E:/git库/NLP-Toturials/data/msr_paraphrase/数字化数据.txt', nrows=2000)

    def __getitem__(self, i):
        return self.data.iloc[i]
```

处理数据

```
def to_tensor(data):#数据加载函数
    b = len(data)
    #N句话,每句话30个词
    xs = np.zeros((b * 2,30))

    for i in range(b):
        same, s1, s2 = data[i]

        #添加首尾符号,补0到统一长度
        s1 = [zidian['<SOS>']] + s1.split(',')[:28] + [
            zidian['<EOS>']
        ] + [zidian['<PAD>']] * 28
        xs[i] = s1[:30]

        s2 = [zidian['<SOS>']] + s2.split(',')[:28] + [
            zidian['<EOS>']
        ] + [zidian['<PAD>']] * 28
        xs[b + i] = s2[:30]

    return torch.LongTensor(xs)
```

2.设计模型

正反向rnn网络

```
class ForwardBackward(nn.Module):
    def __init__(self, flip):
        super().__init__()

        self.rnn1 = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)

        self.fc = nn.Linear(in_features=256, out_features=4300)

        self.flip = flip
```

会有三个结果。

定义ELMO网络

```
class ELMo(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(num_embeddings=4300,
                                  embedding_dim=256,
                                  padding_idx=0)

        self.fw = ForwardBackward(flip=False)
        self.bw = ForwardBackward(flip=True)

    def forward(self, x):
        #编码
        #[16,30] -> [16,30,256]
        x = self.embed(x)

        #顺序预测,以当前字预测下一个字,不需要最后一个字
        outs_f = self.fw(x[:, :-1, :])

        #逆序预测,以当前字预测上一个字,不需要第一个字
        outs_b = self.bw(x[:, 1:, :])

        return outs_f, outs_b
```

嵌入层对X编码，将正反向网络作为子层。

3.训练

        model = ELMo()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()
    
    for epoch in range(1):
        for i, x in enumerate(get_dataloader()):
            #x = [b,30]
            opt.zero_grad(
        #模型计算
        outs_f, outs_b = model(x)#正向和反向
    
        #在计算loss的时候,只需要全连接输出
        #[b,29,4300]
        outs_f = outs_f[-1]
        outs_b = outs_b[-1]
    
        #正向预测是以当前字预测下一个字,所以计算loss不需要第一个字
        #[b,30] -> [b,29]
        x_f = x[:, 1:]
        #逆向预测是以当前字预测上一个字,所以计算loss不需要最后一个字
        #[b,30] -> [b,29]
        x_b = x[:, :-1]
    
        #打平,不然计算不了loss
        #[b,29,4300] -> [b*29,4300]
        outs_f = outs_f.reshape(-1, 4300)
        outs_b = outs_b.reshape(-1, 4300)
        #[b,29] -> [b*29]
        x_f = x_f.reshape(-1)
        x_b = x_b.reshape(-1)
    
        #分别计算全向和后向的loss,再求和作为loss
        loss_f = loss_func(outs_f, x_f)
        loss_b = loss_func(outs_b, x_b)
        loss = (loss_f + loss_b) / 2
    
        loss.backward()
        opt.step()

4.计算词向量



#### 视频15 transformer

##### mask

作用：遮住部分数据，防止分散注意力。

- Padding mask: 用于将输入序列中的padding部分进行mask，使得模型不会在padding部分进行计算。在输入序列中，padding部分通常用0进行填充，因此padding mask将输入序列中所有值为0的位置标记为1，其他位置标记为0。
- Sequence mask: 用于将解码器中的未来信息进行mask，使得模型只能依赖于当前时刻之前的信息。在解码器中，每个时刻都需要预测下一个时刻的输出，因此需要将当前时刻之后的信息进行mask。Sequence mask将当前时刻之后的所有位置标记为1，当前时刻及之前的位置标记为0。

在实现过程中，padding mask和sequence mask都是通过创建一个大小为`(batch_size, seq_len)`的二维张量来实现的，其中`batch_size`是批次大小，`seq_len`是序列长度。在二维张量中，每个位置的值为1表示该位置需要进行mask，值为0表示该位置不需要进行mask。然后，这个二维张量被广播为与输入张量相同的形状，并与输入张量进行点乘运算，以实现mask的效果。

##### 位置编码层

在Transformer模型中，位置编码层的实现方式如下：

1. 创建一个大小为`(seq_len, d_model)`的矩阵，其中`seq_len`是输入序列的长度，`d_model`是模型的隐藏层维度。

2. 对于矩阵中的每个位置，计算其对应的位置编码。位置编码由两个正弦函数和余弦函数组成，公式如下：

   `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`

   `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`

   其中`pos`是位置编码的位置，`i`是向量维度的索引。

3. 将计算得到的位置编码矩阵与输入张量相加，得到带有位置编码的输入张量。

通过位置编码层，Transformer模型可以有效地处理变长序列，并且在不同位置上的输入可以得到不同的表示。

##### 注意力计算函数

注意力计算函数是Transformer模型中的核心部分，用于计算每个位置的注意力分数，并将这些分数用于加权求和输入序列的信息。注意力计算函数的实现方式如下：

1. 计算查询向量Q、键向量K和值向量V。这些向量都是从输入序列中的不同位置计算得到的。在Transformer中，Q、K和V都是通过对输入张量进行线性变换得到的。

   Query 张量（Q）：表示需要进行注意力计算的信息，通常是来自前一层的输出或者输入序列的某个元素。在自然语言处理任务中，Query 张量通常表示要生成的单词或者句子的向量表示。

   Key 张量（K）：表示用于计算注意力分数的信息，通常是来自前一层的输出或者输入序列的所有元素。在自然语言处理任务中，Key 张量通常表示输入序列的所有单词或者句子的向量表示。

   Value 张量（V）：表示根据注意力分数加权求和后得到的信息，通常是来自前一层的输出或者输入序列的所有元素。在自然语言处理任务中，Value 张量通常表示输入序列的所有单词或者句子的向量表示。

   我们可以使用 Q 和 K 计算注意力分数张量 S，再将 S 与 V 进行加权求和得到最终的输出。

2. 计算注意力分数。注意力分数是通过将查询向量Q和键向量K进行点积得到的。由于点积得到的结果可能很大，因此可以将点积结果除以一个缩放因子，以避免梯度消失的问题。缩放因子通常为键向量K的维度的平方根。

3. 将注意力分数进行softmax归一化，得到注意力权重。这些权重表示了输入序列中每个位置对查询向量Q的贡献。

   ```
     score = score.masked_fill_(mask, -np.inf)
     score = F.softmax(score, dim=-1)
   ```

4. 将注意力权重与值向量V进行加权求和，得到加权后的值。这些加权后的值表示了输入序列中每个位置对查询向量Q的贡献。最终输出就是这些加权后的值。

   ```
   score = torch.matmul(score, V)
   ```

注意力计算函数是Transformer模型中的重要组成部分，它可以有效地捕捉输入序列中不同位置之间的依赖关系，并在处理序列任务时取得良好的效果。

##### 多头注意力层

多头注意力层是Transformer模型中的一层，用于增强模型对不同位置之间的依赖关系的建模能力。多头注意力层通过将注意力计算函数应用于多组查询、键和值向量，以捕捉输入序列中不同方面的信息。

在多头注意力层中，输入序列首先分别通过三个线性变换，得到多组查询、键和值向量。然后，每组查询、键和值向量都通过注意力计算函数，得到对应的加权值。最后，多组加权值被拼接在一起，并通过另一个线性变换得到最终输出。

具体而言，多头注意力层的实现方式如下：

1. 将输入序列分别通过三个线性变换，得到多组查询、键和值向量。这些向量的维度通常为`d_model`。
2. 对于每组查询、键和值向量，应用注意力计算函数，得到对应的加权值。在每个注意力计算函数中，需要使用不同的权重矩阵进行线性变换。
3. 将多组加权值拼接在一起，并通过另一个线性变换得到最终输出。输出的维度通常为`d_model`。

通过多头注意力层，Transformer模型可以同时考虑不同方面的信息，并在处理序列任务时取得更好的效果。

##### 自注意力

自注意力是指注意力计算函数中的查询向量、键向量和值向量都来自同一个输入序列。自注意力可以帮助模型在处理序列任务时捕捉到输入序列中不同位置之间的依赖关系。

在自注意力中，查询向量、键向量和值向量的计算方式都与传统的注意力计算函数相同。不同之处在于，它们都来自同一个输入序列。

##### fromtimestamp()函数

`datetime.datetime.fromtimestamp(date)`中的`date`是一个Unix时间戳，表示从1970年1月1日零时零分零秒到指定时间的秒数。`fromtimestamp()`函数将这个时间戳转换为一个Python的datetime对象，表示指定时间的日期和时间信息。

需要注意的是，`fromtimestamp()`函数默认将时间戳解释为本地时区的时间。

##### detach()和numpy()

`detach()`方法用于将张量从计算图中分离出来，避免梯度计算对其产生影响。然后，`numpy()`方法将分离的张量转换为NumPy数组。

##### reshape和expand

这段代码将一个形状为`(batch_size, 11)`的掩码张量转换为形状为`(batch_size, 1, 11, 11)`的掩码张量。

首先，`reshape`方法被用来将掩码张量的形状从`(batch_size, 11)`改变为`(batch_size, 1, 1, 11)`，以便后续的操作。

然后，`expand`方法被用来将掩码张量的形状从`(batch_size, 1, 1, 11)`改变为`(batch_size, 1, 11, 11)`。在这个方法中，第一个参数`-1`表示保持该维度的大小不变，第二个参数`1`表示在该维度上的大小为1，第三个参数`11`表示在该维度上的大小为11，最后一个参数`11`表示在最后一个维度上的大小为11。这个操作将掩码张量在第二个维度上进行了复制，使得掩码张量的形状变为`(batch_size, 1, 11, 11)`。

##### unsqueeze(dim)

具体来说，`unsqueeze(dim)`会在张量的`dim`维度上插入一个新的维度，使得张量的维度数加1。

例如，如果一个形状为`(batch_size, seq_len)`的张量`x`，我们可以使用`unsqueeze(1)`将其变为形状为`(batch_size, 1, seq_len)`的张量。这个操作会在第二个维度上增加一个新的维度，使得每个样本都变成了一个形状为`(1, seq_len)`的张量。

##### nn.Embedding(n_vocab, emb_dim

`nn.Embedding(n_vocab, emb_dim)`是一个PyTorch中的嵌入层，用于将一个整数标记序列转换为对应的嵌入向量序列。其中，`n_vocab`表示词汇表大小，`emb_dim`表示嵌入向量的维度。

具体来说，这个嵌入层包含一个形状为`(n_vocab, emb_dim)`的权重矩阵，其中每一行对应一个标记的嵌入向量。给定一个形状为`(n, step)`的整数标记序列`x`，这个嵌入层的前向传播过程会将每个标记转换为对应的嵌入向量，并按照输入序列的顺序输出一个形状为`(n, step, emb_dim)`的嵌入向量序列。

在Transformer模型中，这个嵌入层用于将输入序列中的标记转换为对应的嵌入向量，以便后续处理。

##### 嵌入向量

嵌入向量是一种将离散型数据（如单词、标记等）映射到连续型向量空间中的方法。在自然语言处理中，嵌入向量通常用于表示单词或标记，以便在机器学习模型中进行处理。

##### `torch.matmul` 

`torch.matmul` 是 PyTorch 中的一个函数，用于计算两个张量之间的矩阵乘法。

##### permute

`permute(0, 1, 3, 2)` 是 PyTorch 张量的一种维度变换操作，用于将张量的维度按照指定的顺序进行交换。

##### `BatchNorm1d`和`LayerNorm`

`BatchNorm1d` 对输入数据在每个 mini-batch 中进行归一化处理，使得每个特征的均值为 0，标准差为 1。

`LayerNorm` 是一种层归一化层，用于对神经网络中的每个层进行归一化处理。具体地，`LayerNorm` 对每个样本在特征维度上进行归一化处理，使得每个样本在特征维度上的均值为 0，标准差为 1。

##### `F.dropout` 

`F.dropout` 是 PyTorch 中的一个函数，用于在训练期间对张量进行随机失活(dropout)。Dropout 是一种常用的正则化技术，可防止深度神经网络过拟合。它通过在每个训练批次中随机删除一些神经元的输出来实现这一点。`F.dropout` 函数需要两个参数：输入张量和丢弃概率

#### 视频16 GPT

1.数据

需要区别s1和s2，并且要使用same。

最后整理出x、y和seg。

2.编码层

```
encoder_layer = nn.TransformerEncoderLayer(d_model=256,
                                                   nhead=4,
                                                   dim_feedforward=256,
                                                   dropout=0.2,
                                                   activation='relu')
```

其中，`d_model`表示模型的输入维度，`nhead`表示多头注意力的头数，`dim_feedforward`表示全连接层的隐藏层大小，`dropout`表示dropout概率，`activation`表示激活函数。

3.GPT网络

```
class GPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(num_embeddings=4300, embedding_dim=256)#对x
        self.seg_embed = nn.Embedding(num_embeddings=3, embedding_dim=256)#对seg

        self.position_embed = nn.Parameter(torch.randn(59, 256) / 10)#可学习参数

        self.encoder = encode.Encoder()

        self.fc_x_tail = nn.Linear(in_features=256, out_features=4300)
        self.fc_y = nn.Linear(in_features=59 * 256, out_features=2)

    def forward(self, x_head, seg):
        # [b, 60]
        mask_x = mask.get_key_padding_mask(x_head)

        # 编码,添加位置信息
        x_head = self.embed(x_head) + self.seg_embed(seg) + self.position_embed

        # 编码层计算
        # [b, 60, 256] -> [b, 60, 256]
        x_head = self.encoder(x_head, mask_x)

        x_tail = self.fc_x_tail(x_head)

        y = self.fc_y(x_head.reshape(x_head.shape[0], -1))

        return x_tail, y
```

与transform不同，他的位置矩阵编码是可学习的，不是固定的。

#### 视频17 BERT

##### 1.随机替换函数

1. 先对输入的张量进行克隆，以免影响原来的张量。

2. 定义一个与输入张量`x`相同大小的替换矩阵`replace`，其中所有元素的初始值都为`False`。

3. 遍历输入张量中的每个元素，对于不是特殊符号(`<PAD>`, `<SOS>`, `<EOS>`)的元素，以0.15的概率进行随机替换操作。如果进行了替换操作，就将替换矩阵中对应位置的值设为`True`。

4. 对于替换矩阵中值为True的位置，以一定的概率进行不同的替换操作：

   - 以0.7的概率将其替换为`<MASK>`符号。
   - 以0.15的概率不做任何操作。
   - 以0.15的概率将其替换为随机一个不是特殊符号(`<PAD>`, `<SOS>`, `<EOS>`)的字。

5. 返回替换后的张量`x`和替换矩阵`replace`。

该函数可以用于数据增强，以增加模型对输入数据的鲁棒性。

2.训练

`torch.masked_select`函数

它的作用是从输入张量`pred_x`中选取在`replace_mask.unsqueeze(2)`中值为`True`的位置上的元素，并返回一个一维张量。其中，`replace_mask`是一个二维的布尔型张量，表示哪些位置需要进行替换操作，`unsqueeze(2)`是为了将其扩展为三维张量，以便与`pred_x`进行广播操作。这行代码的作用是将`pred_x`中需要进行替换操作的元素提取出来，以便后面进行随机替换。