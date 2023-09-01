#### git是资源管理软件

![image-20230513095847022](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230513095847022.png)

#### git功能

##### 版本控制

![image-20230513095932019](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230513095932019.png)

自动修改文件版本，手动要通过客户端修改。

![image-20230513095940496](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230513095940496.png)

集中式版本控制

![image-20230518133939686](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230518133939686.png)

1.只能在本地修改。

2.文件冲突问题：后面的修改覆盖前面的。

解决方法：（1）加锁解锁，但是效率低。

（2）约束操作的位置，进行文件比对，然后进行合并。

3.中央服务器出问题，容易丢失数据。

分布式版本控制

![image-20230518135108152](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230518135108152.png)

1.在本地也建立了一个资源库。

2.资源安全，本地操作快。但是占用本地空间，上传慢。

##### 提交文件

![image-20230518213746345](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230518213746345.png)

1.把操作文件存储在仓库路径下。

2.与本地资源库不同的情况下，会显示在change。

3.使用commit提交。在history可以看见操作记录。

![image-20230518223526484](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230518223526484.png)

![image-20230518223535427](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230518223535427.png)

修改后也是提交，并且是不同的版本。

![image-20230518224002500](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230518224002500.png)

删除文件也要commit。

##### 分支原理

![image-20230519113944295](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230519113944295.png)

1.增加资源库库副本，减少资源冲突，使得更加稳定。

2.不同的分支就有不同的库。

##### 分支操作

创建、选择和合并分支

![image-20230519115919852](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230519115919852.png)

合并冲突时，删除带符号行和不需要的内容即可

![image-20230519120039970](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230519120039970.png)

#### 标签

在history中🤜，选择tag，使得更加清楚明了。

#### 远程仓库（GitHub网页版）

##### 创建库

![image-20230519124953570](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230519124953570.png)

远程到本地：file—clone

本地到远程：publish repository

删除库：settings—Danger Zone—Delete

##### gitee网页版

远程到本地：file—clone—URL

本地到远程：publish

README：重要的描述信息

1.word文档没有比对功能。

2.便于快速维护和修改。

##### ignore：有些文件或者某些类型文件没必要commit。

##### 文件图标

从上到下：删除、修改、新增

![image-20230519134704317](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230519134704317.png)

##### 比对信息

<img src="C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230519134923478.png" alt="image-20230519134923478" style="zoom:150%;" />

左边表示旧文件的修改，右边表示新文件的修改。

#### 开发工具与GitHub、gitee（两者一样）

从pycharm中添加项目到GitHub、gitee：vcs/git

在pycharm中修改了：右键修改文件—Git—提交—推送

在GitHub中修改：git—pull

多人协作开发：git—克隆

可以在提交时，选择gitee和GitHub。

#### 版本号的作用

1.避免文件合并时的冲突。

2.定位仓库中的文件：前两位代表文件夹+后38位为文件名。

#### 文件操作

1.按照版本号找到文件，在git文件夹界面，右键进入Git bash,输入

```
git cat-file -p 版本号
```

查看文件内容。

#### 版本号的关联

![image-20230520160000397](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230520160000397.png)

1.提交信息之间的parent,关联上一次的提交的文件。

2.tree代表文件状态的版本号。

3.tree后面的版本号联系文件内容的版本号。

4.删除就是文件状态只联系提交时的文件内容。

##### 分支操作

![image-20230520202951043](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230520202951043.png)

1.客户端显示那个分支，HEAD就记录谁。

2.分支会指向最新的提交信息。

#### git基础命令（本地仓库）

![image-20230521115720631](C:\Users\lxc\AppData\Roaming\Typora\typora-user-images\image-20230521115720631.png)

##### git bash的命令

1.查看版本：

```
git -v
```

2.创建本地仓库

```
git init
```

(1)打开路径就是建立库存放的路径。

（2）相比客户端创建，少了一个文件。

（3）有目录，但没有具体的文件夹，因为还没有指向内容（提交信息）。

3.克隆远程仓库

```
git clone http
```

##### 文件基本操作

4.配置

```
git config user.name /user.email
```

```
git config --global
```

或者在客户端：1.option—git       2.rep—rep setting—git config

5.暂存区状态(change)

```
git status
```

6.把untracked file(工作区中的文件)添加到暂存区

```
git add 文件名/*.txt -m#接备注
```

7.从暂存区移动到工作区

```
git rm --cached 文件名
```

8.查看记录(history)

```
git log --oneline
```

##### 误删除

9.远程库存在误删文件（删除操作还没有commit）

恢复

```
git restore
```

10.重置（把版本号之前的操作全重置，丢失记录）

```
git reset --hard 版本号
```

11.还原（恢复到提交前）

```
git revert 前操作版本号
```

##### 分支操作

12.建立分支

```
git branch 分支名
```

13.查看分支

```
git branch -v
```

14.切换分支

```
git checkout 分支名
```

15.创造并切换分支

```
git checkout -b 分支名
```

16.删除分支

```
git branch -d 分支名
```

17.合并分支

```
git merge 被合并的分支名
```

##### 标签操作

18.查看标签

```
git tag
```

19.创造标签

```
git tag 标签名 版本号
```

20.删除标签

```
git tag -d 标签名
```

#### 远程仓库操作

ssh方式

[29 - Git - 命令 - 远程仓库_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1wm4y1z7Dg?p=29&spm_id_from=pageDriver&vd_source=bee013e22d10b6d2e417a16b33fbc3f5)

[TOC]

