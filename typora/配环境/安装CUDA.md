安装CUDA

[成功解决：AssertionError: Torch not compiled with CUDA enabled_安安喜乐的博客-CSDN博客](https://blog.csdn.net/m0_74890428/article/details/130184164)

第一步确定CUDA版本和下列东西的版本

1.windows版本

11.4.3也就是11.4版本第3次更新有win11版本。

2.驱动器版本

更新后再看，决定CUDA版本上限。

打开anaconda的Anaconda Prompt输入命令: nvidia-smi

如果不显示界面，需要更新显卡驱动，找到自己电脑对应显卡的型号，安装对应的驱动就行了。

3.python版本

也在下面网站后面的cp，表示python版本。最好用3.9，因为按照安装Anaconda与PyTorch库（GPU版本）这个攻略，安装库方便。

4.pytorch版本

https://download.pytorch.org/whl/torch_stable.html

能在这个链接可以找到的CUDA版本是11.8，其中没有11.4版本。         

然后torchvision和torchauditor都要一一对应

第二步 下载CUDA

5.如果没有安装CUDA和CUDNN

[(15条消息) CUDA卸载&&重装_@Wufan的博客-CSDN博客](https://blog.csdn.net/weixin_44606139/article/details/127493438?ops_request_misc=%7B%22request%5Fid%22%3A%22168371634116800225519340%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=168371634116800225519340&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-127493438-null-null.142^v86^control_2,239^v2^insert_chatgpt&utm_term=win11卸载cuda&spm=1018.2226.3001.4187)

就按照这个安装和卸载，记住先选版本

第三步 可以先下载whl文件，然后去创建模拟环境和配置一些库。

https://download.pytorch.org/whl/torch_stable.html

或者这样下载，不需要分开下载。

[Previous PyTorch Versions | PyTorch](https://pytorch.org/get-started/previous-versions/)

查看torch

[如何查看电脑安装的pytorch版本？？？（windows和ubuntu系统通用）_怎样在ubuntu系统查看安装的pytorch版本_布丁小芒果的博客-CSDN博客](https://blog.csdn.net/weixin_43382156/article/details/103565151)

[Pytorch 各个GPU版本CUDA和cuDNN对应版本_torchserve:0.6.1-gpu 对应cuda_风信子的猫Redamancy的博客-CSDN博客](https://blog.csdn.net/weixin_45508265/article/details/122006134)

要查看自己的 cuDNN（CUDA Deep Neural Network）版本，可以执行以下步骤：

1. 打开命令行终端（Command Prompt）或者终端（Terminal）。

2. 输入以下命令，并按回车键运行：

   ```
   nvcc --version
   ```

   

   这将显示 CUDA 编译器的版本信息。

3. 输入以下命令，并按回车键运行：

   ```
   cat <path_to_cuda>/include/cudnn.h | grep CUDNN_MAJOR -A 2
   ```

   

   如果你使用的是 Windows 操作系统，将 `/usr/local/cuda/include/cudnn.h` 替换为 `<path_to_cuda>/include/cudnn.h`，其中 `<path_to_cuda>` 是你的 CUDA 安装路径。
   
   
