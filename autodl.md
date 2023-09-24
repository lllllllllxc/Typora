### autodl

[AutoDL使用教程：1）创建实例 2）配置环境+上传数据 3）PyCharm2021.3专业版下载安装与远程连接完整步骤 4）实时查看tensorboard曲线情况_孟孟单单的博客-CSDN博客](https://blog.csdn.net/LWD19981223/article/details/127085811)

#### 一、创建实例

镜像选择：（1）基础镜像：你的基础镜像环境名叫base，所以每次你需要直接用基础镜像运行代码时，还是需要调用命令`conda activate base`来激活进入到你的基础镜像中。

（2）自己配置：去下载好镜像然后再上传，再配置[虚拟环境](https://so.csdn.net/so/search?q=虚拟环境&spm=1001.2101.3001.7020),可以在代码中用终端安装，但是仍然需要下载好文件（注意yml或txt中的渠道和版本）。

**注意**：![img](https://img-blog.csdnimg.cn/666e81675d1e44d9a638fa514547f15d.png)

#### 二、配置环境+上传数据流程

**当只是需要上传一个`zip压缩包`时，或者其他类型的单个文件时，建议直接进入到`我的网盘`中上传。因为这样可以少了连接到Xftp这个步骤，以及可以少费点钱。**

[AutoDL上传数据详细步骤（自己用的步骤，可能没有其他大佬用的那么高级）_孟孟单单的博客-CSDN博客](https://blog.csdn.net/LWD19981223/article/details/127556122)

**注意事项**：创建实例是在哪个区，就使用哪个区的网盘！避免跨区无法使用一些数据（在这个界面能够看到你的数据，可以下载or删除）

![img](https://img-blog.csdnimg.cn/23fa1eb4fb764881ad3beb3370e517a5.png)

1.开机后就获得了：登陆指令、密码（这个很重要）

那么下图中的`主机`和`端口号`分别是：

- 用户名：root
- 主机HOST：rxxxxn-0.autodl.com （`@`后的所有内容）
- 端口号：12300
- 密码（最后一行）：是步骤（2）直接得到的

xtf上传文件和pycharm连接远端需要。

#### 三、远程连接到本地（PyCharm2021.3专业版）

1. 配置ssh解释器
2. 远程连接：Tools -> Deployment -> Configuration

添加SFTP

![img](https://img-blog.csdnimg.cn/46bc685485e54b1aa2c7bef8245eac6a.png)

输入配置信息

![在这里插入图片描述](https://img-blog.csdnimg.cn/ce5e3cfa39f34d239eeee9c6e0c4d22c.png)

3.找到和更新文件和代码

切换到云目录

![在这里插入图片描述](https://img-blog.csdnimg.cn/87771acbd7254334ac80282f11b6446f.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/b4141f5a8a8441cb83b56af71655180e.png)

连接到远程终端

![在这里插入图片描述](https://img-blog.csdnimg.cn/b70a0623486a483b88e329eb28c78314.png)

查看代码内容，并重新上传更新代码

![在这里插入图片描述](https://img-blog.csdnimg.cn/b892b1f8b0054caba5aa26d673c7b2d8.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/047b076ee8fa4dadba69eb8d7e972508.png)

运行代码

![在这里插入图片描述](https://img-blog.csdnimg.cn/41ad55cbb5c3437e97c1bc39c0e7fa96.png)