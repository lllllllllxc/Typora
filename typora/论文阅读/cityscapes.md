## cityscapes

[城市景观脚本/自述文件.md at master ·mcordts/cityscapesScripts (github.com)](https://github.com/mcordts/cityscapesScripts/blob/master/README.md)

[图像语意分割Cityscapes训练数据集使用方法详解_图像分割cityscape数据集使用介绍_loving____的博客-CSDN博客](https://blog.csdn.net/wang27623056/article/details/106631196?ops_request_misc=%7B%22request%5Fid%22%3A%22169564204216800222823614%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=169564204216800222823614&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-106631196-null-null.142^v94^chatsearchT3_1&utm_term=cityscapesscripts使用&spm=1018.2226.3001.4187)

[cityscapesScripts使用笔记_nefetaria的博客-CSDN博客](https://blog.csdn.net/nefetaria/article/details/105728008?ops_request_misc=%7B%22request%5Fid%22%3A%22169564204216800222823614%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=169564204216800222823614&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-4-105728008-null-null.142^v94^chatsearchT3_1&utm_term=cityscapesscripts使用&spm=1018.2226.3001.4187)

### 文件结构

一般都是拿这5000张精细标注(gt fine)的样本集来进行训练和评估的。当然，还有一个策略就是，先对粗糙标注的样本集进行一个简单的训练，然后再基于精细标注的数据集进行final training。 这里我们只谈gt fine样本集的训练。

1）原始精细标注数据集里面其实每张图片只对应四张标注文件：xxx_gtFine_color.png, xxx_gtFine_instanceIds.png, xxx_gtFine_labelsIds.png以及xxx_gtFine_polygons.json。 xxx_color.png是标注的可视化图片，真正对训练有用的是后面三个文件。xxx_instanceIds.png是用来做实例分割训练用的，而xxx_labelsIds.png是语义分割训练需要的。它们的像素值就是class值。而最后一个文件xxx_polygons.json是用labelme工具标注后所生成的文件，里面主要记录了每个多边形标注框上的点集坐标。

2）至于另外两个xxx_gtFine_instanceTrainIds.png和xxx_gtFine_labelTrainIds.png则是后面使用labels.py （from https://github.com/mcordts/cityscapesScripts/tree/master/cityscapesscripts/helpers）来生成的。因为实际上这5000张精细标注的图片有34类(0~33)，但训练时可能只想关心其中19类(0~18)。所以需要做一个映射来将34类中感兴趣的类别映射到19类中，其它不感兴趣的类别就直接设成255，所以这也是为什么xxx_trainIds.png中有白色像素的原因，因为那些白色像素的类别不是我们感兴趣的，变成255白色了。   当然，这个预变换处理也不一定非要先做，因为有的模型，比如deeplabv3+，训练时，本身就会做这个转换。



路径结构

```
{root}/{type}{video}/{split}/{city}/{city}_{seq:0>6}_{frame:0>6}_{type}{ext}
#最后一部分是照片名
```

- `root`城市景观数据集的根文件夹。我们的许多脚本检查指向此文件夹的环境变量是否存在，并将其用作默认选项。`CITYSCAPES_DATASET`
- `type`数据类型/模式，例如 用于精细的地面真实，或用于左侧 8 位图像。`gtFine``leftImg8bit`
- `split`拆分，即 train/val/test/train_extra/demoVideo。请注意，并非所有拆分都存在所有类型的数据。因此，偶尔发现空文件夹不要感到惊讶。
- `city`记录这部分数据集的城市。
- `seq`使用 6 位数字的序列号。
- `frame`使用 6 位数字的帧号。请注意，在一些城市，尽管记录了很长的序列，但很少，而在某些城市，记录了许多短序列，其中只有第 19 帧被注释。
- `ext`文件的扩展名和可选的后缀，例如 对于地面实况文件`_polygons.json`

可能的值为`split`

- `train`通常用于训练，包含 2975 张带有精细和粗略注释的图像
- `val`应该用于验证超参数，包含 500 张带有精细和粗略注释的图像。也可用于训练。
- `test`用于在我们的评估服务器上进行测试。注释不是公开的，但为了方便起见，我们包括了自我车辆和纠正边界的注释。
- `train_extra`可以选择用于训练，包含带有粗略注释的 19998 图像
- `demoVideo`可用于定性评估的视频序列，**这些视频没有注释**

可能的值为`type`

- `gtFine`精细注释、2975 次训练、500 次验证和 1525 次测试。这种类型的批注用于验证、测试和选择性地用于训练。注记使用包含各个面的文件进行编码。此外，我们还提供图像，其中像素值对标签进行编码。有关详细信息，请参阅 和 中的脚本。`json``png``helpers/labels.py``preparation`
- `gtCoarse`粗略注释，可用于所有训练和验证图像以及另一组 19998 训练图像 （）。这些注释可以与 gtFine 一起使用，也可以在弱监督设置中单独使用。`train_extra`
- `gtBbox3d`车辆的 3D 边界框注释。有关详细信息，请参阅[Cityscapes 3D（Gählert等人，CVPRW '20）。](https://arxiv.org/abs/2006.07864)
- `gtBboxCityPersons`行人边界框注记，适用于所有训练和验证图像。详情请参考[CityPersons（Zhang等人，CVPR '17）。](https://bitbucket.org/shanshanzhang/citypersons)边界框的四个值是 （x， y， w， h），其中 （x， y） 是其左上角，（w， h） 是其宽度和高度。`helpers/labels_cityPersons.py`
- `leftImg8bit`左侧图像为 8 位 LDR 格式。这些是标准的带注释的图像。
- `leftImg8bit_blurred`左侧图像为 8 位 LDR 格式，面部和车牌模糊不清。请计算原始图像的结果，但使用模糊的图像进行可视化。我们感谢[Mapillary](https://www.mapillary.com/)模糊图像。
- `leftImg16bit`左侧图像为 16 位 HDR 格式。这些图像提供每像素 16 位的颜色深度，并包含更多信息，尤其是在场景中非常黑暗或明亮的部分。警告：图像存储为 16 位 png，这是非标准的，并非所有库都支持。
- `rightImg8bit`8 位 LDR 格式的正确立体声视图。
- `rightImg16bit`16 位 HDR 格式的正确立体视图。
- `timestamp`以 ns 为单位的录制时间。每个序列的第一帧始终具有 0 的时间戳。
- `disparity`预先计算的视差深度图。要获得视差值，请计算每个像素 p > 0：d = （ float（p） - 1. ） / 256.，而值 p = 0 是无效的测量值。警告：图像存储为 16 位 png，这是非标准的，并非所有库都支持。

### [Installation](https://github.com/mcordts/cityscapesScripts/blob/master/README.md#installation)

```
python -m pip install cityscapesscripts
```

### [用法](https://github.com/mcordts/cityscapesScripts/blob/master/README.md#usage)

安装将城市景观脚本安装为名为 python 模块并公开以下工具`cityscapesscripts`

- `csDownload`：通过命令行下载城市景观包。
- `csViewer`：查看图像并叠加注释。
- `csLabelTool`：我们用于标记的工具。
- `csEvalPixelLevelSemanticLabeling`：评估验证集上的像素级语义标记结果。此工具还用于评估测试集上的结果。
- `csEvalInstanceLevelSemanticLabeling`：评估验证集上的实例级语义标记结果。此工具还用于评估测试集上的结果。
- `csEvalPanopticSemanticLabeling`：评估验证集上的全景分割结果。此工具还用于评估测试集上的结果。
- `csEvalObjectDetection3d`：评估验证集上的 3D 对象检测。此工具还用于评估测试集上的结果。
- `csCreateTrainIdLabelImgs`：将多边形格式的注释转换为带有标签 ID 的 png 图像，其中像素对可在 中定义的“训练 ID”进行编码。`labels.py`
- `csCreateTrainIdInstanceImgs`：将多边形格式的注释转换为具有实例 ID 的 png 图像，其中像素对由“训练 ID”组成的实例 ID 进行编码。
- `csCreatePanopticImgs`：将标准 png 格式的注释转换为 [COCO 全景分割格式](http://cocodataset.org/#format-data)。
- `csPlot3dDetectionResults`：可视化以 .json 格式存储的 3D 对象检测评估结果。

### [包内容](https://github.com/mcordts/cityscapesScripts/blob/master/README.md#package-content)

该软件包的结构如下

- `helpers`：其他脚本包含的帮助程序文件
- `viewer`：查看图像和注释
- `preparation`：将地面实况注释转换为适合您方法的格式
- `evaluation`：验证您的方法
- `annotation`：用于标注数据集的注记工具
- `download`：城市景观软件包的下载器

请注意，所有文件的顶部都有一个小文档。最重要的文件

- `helpers/labels.py`：定义所有语义类 ID 的中心文件，并提供各种类属性之间的映射。
- `helpers/labels_cityPersons.py`：定义所有CityPersons行人类别的ID的文件，并提供各种类属性之间的映射。
- `setup.py`：运行以启用 cython 插件以加快评估速度。仅针对 Ubuntu 进行了测试。`CYTHONIZE_EVAL= python setup.py build_ext --inplace`

#### 导入数据集

yaml文件中输入数据集名

是的，你可以通过在PyCharm程序中导入YAML文件中的数据集。以下是一般的操作方式：

1. 在你的PyCharm项目中创建一个YAML文件夹，用于存放YAML文件。
2. 将你的YAML文件复制到该文件夹中。
3. 在你的Python代码中导入PyYAML库来解析YAML文件。如果你尚未安装该库，可以使用以下命令在PyCharm的终端中安装：

```
pip install pyyaml
```



1. 在你的Python代码中，使用`open()`函数打开YAML文件，然后使用`yaml.load()`函数将其加载为Python对象。例如：

```
import yaml

# 打开YAML文件并加载数据
with open('YAML文件路径', 'r') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)

# 访问导入的数据集
print(data)
```



请确保将`YAML文件路径`替换为实际的YAML文件路径，注意使用双引号或者单引号来包围文件路径字符串。

这样，YAML文件中的数据集就可以通过`data`变量在PyCharm程序中进行访问和使用了。