# Anaconda

首先看项目介绍

其次记得看readme

最后记得看看代码内有没有说明

看csdn

### 配置环境

1.jupyter在网页运行，运行时内核（黑窗口）不能关闭。

2.计算机用户名为英文才可以兼容jupyter。

https://www.bilibili.com/video/BV1eN4y157vj/?spm_id_from=333.788.recommend_more_video.0&vd_source=bee013e22d10b6d2e417a16b33fbc3f5修改用户名



### 熟悉Jupyter环境

1.序号：运行的顺序，如’In[2]，表示第二个运行的语句‘。

2.删除：剪刀是表面删除，实际上仍存在，只是不显示。

​               Kernel（内核） —重启并清空所有输出，内核被清空了。

​               Kernel（内核） —重启并运行所有代码块，上述操作加上重新运行。

### 虚拟环境的基础命令

（1）Prompt 清屏 

`#清屏` 

`cls` 

（2）base 环境下的操作

列出所有的虚拟环境

conda env list

创建名为“环境名”的虚拟环境，并指定 Python 解释器的版本

conda create -n 环境名 python=3.9

删除名为“环境名”的虚拟环境

conda remove -n 环境名 --all

进入名为“环境名”的虚拟环境

`conda activate 环境名`

（3）虚拟环境内的操作 （pip 安装若失败，在【pip intsall 库==版本】后加【 -i https://pypi.tuna.tsinghua.edu.cn/simple 】即可） 

列出当前环境下的所有库 

`conda list`

 安装 NumPy 库，并指定版本 1.12.5 

pip install numpy==1.21.5 -i https://pypi.tuna.tsinghua.edu.cn/simple

 安装 Pandas 库，并指定版本 1.2.4

 `pip install Pandas==1.2.4 -i https://pypi.tuna.tsinghua.edu.cn/simple` 

 安装 Matplotlib 库，并指定版本 3.5.1

pip install Matplotlib==3.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

 查看当前环境下某个库的版本（以 numpy 为例） 

`pip show numpy` 

运行虚拟环境

conda activate

退出虚拟环境 

`conda deactivate`

安装scipy失败

[Python 常安装scipy失败及解决方法_pip装不了scipy_YoPong Yo的博客-CSDN博客](https://blog.csdn.net/FUCCKL/article/details/86696407)

安装整个环境

```
盘
cd 存储路径
pip install -r requirements.txt
```

### 封装安装yml

要使用 `conda env create -f environment.yml` 方式创建虚拟环境，需要创建一个包含依赖包及其版本信息的 YAML 文件。下面是创建 `environment.yml` 文件的一般步骤：

1. 打开文本编辑器（如 Notepad++、Sublime Text 或 Visual Studio Code）。

2. 在文件中添加以下内容来设置新的虚拟环境的名称，例如 `myenv`：

   ```
   name: myenv
   ```

   

3. 添加一个 `dependencies` 字段，用于指定项目所需的依赖包及其版本。你可以使用 `pip` 或 `conda` 注明依赖包来源。

   - 对于 `pip`，示例格式如下：

     ```
     pip:
       - python=3.9
       - numpy>=1.18
       - pandas
       - scikit-learn
     ```

     

   - 对于 `conda`，示例格式如下：

     ```
     dependencies:
       - python=3.9
       - numpy=1.18
       - pandas
       - scikit-learn
     ```

     

4. 如果需要添加其他配置，例如指定使用的 channels（软件源），可以在 YAML 文件中的 `channels` 字段中指定。示例：

   ```
   channels:
     - conda-forge
     - defaults
   ```

   

5. 保存文件为 `environment.yml`。

### 虚拟环境与 Jupyter 内核相连

请在 Prompt 的虚拟环境下操作下列命令

  列出 Jupyter 的内核列表 

jupyter kernelspec list 

安装 ipykernel 

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ipykernel 

 将虚拟环境导入 Jupyter 的 kernel 中 

python -m ipykernel install --user --name=环境名 

删除虚拟环境的 kernel 内核 

jupyter kernelspec remove 环境名

常见问题：                

![image-20230822132449464](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230822132449464.png)

不能带梯子。

找不到conda

[pycharm找不到conda可执行文件怎么办？_聻775的博客-CSDN博客](https://blog.csdn.net/weixin_63350378/article/details/128749544?ops_request_misc=%7B%22request%5Fid%22%3A%22169268648816800182737308%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=169268648816800182737308&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-128749544-null-null.142^v93^control&utm_term=pycharm找不到conda可执行文件&spm=1018.2226.3001.4187)

**有时候终端安装出错，可以在cmd安装成功。**