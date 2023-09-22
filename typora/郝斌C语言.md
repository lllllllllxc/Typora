## 郝斌C语言

### 课程大纲

![image-20230919153030206](C:\Users\35106\AppData\Roaming\Typora\typora-user-images\image-20230919153030206.png)

### C概述

[7.C概述 - 怎样学C语言_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1sM4y1v7ci?p=7)

教材：C++PrimerPlus

#### 如何使用visual c++编写代码？

创建source c++ file（文件）—输入代码—compile—build—run

```c
#include <stdio.h>
int main(void)
{
	printf("欢迎大家学习C语言!\n");

    return 0;
}
```

#### C语言的起源与发展

![image-20230919111623742](C:\Users\35106\AppData\Roaming\Typora\typora-user-images\image-20230919111623742.png)

#### C语言特点

优点：

代码小，运行速度快。windows、unix和linux的内核都是C语言。

功能强大，有指针。

缺点：

危险性大，非原则性错误不报错。

生产周期长，所以使用面向对象。

可移植性不强。Java可移植性强。

#### 应用领域

![image-20230919121122956](C:\Users\35106\AppData\Roaming\Typora\typora-user-images\image-20230919121122956.png)

#### 参考资料

![image-20230919142822193](C:\Users\35106\AppData\Roaming\Typora\typora-user-images\image-20230919142822193.png)

#### 一元二次方程详解

1. 先写基本框架

```
#include <stdio.h>

int main(void)
{

  return 0;
}
```

2.再写思路框架，定义变量。

```
#include <stdio.h>

int main(void)
{
	//保存三个系数
	int a = 1;
	int b = 2;
	int c = 3;
    float delta;//b*b-4ac
    float x1,x2;//两个解
	delta = b*b-4ac;

	if(delta > 0)
	{
	   两个解
	}else if(delta == 0)
	{
	   唯一解   
	}
	else
	{
	   无解
	}


    return 0;
}
```

3.补充内容，补足引用。

```
#include <stdio.h>
#include <math.h>
int main(void)
{
	//保存三个系数
	int a = 1;
	int b = 2;
	int c = 3;
    double delta;//b*b-4ac
    double x1;
	double x2;//两个解
	delta = b*b-4*a*c;

	if(delta > 0)
	{
	   x1 = (-b+sqrt(delta))/(2*a);
	   x2 = (-b-sqrt(delta))/(2*a);
	   printf("该一元二次方程有两个解,x1=%f,x2=%f\n",x1,x2);

	}else if(delta == 0)
	{
	   x1 = -b/(2*a); 
	   printf("该方程只有唯一解x1=x2=%f\n",x1);
	}
	else
	{
	   printf("无解\n");
	}


    return 0;
}
```

#### VC++ 6.0使用详解

1.保存：CTRL+S

2.代码规范，记得注释和空格。

3.关闭程序，要使用关闭工作空间，否则影响第二个程序运行。

### 编程预备专业知识

#### hello world是如何运行起来的

compile+build 把文件变成了.exe形式，然后虹色感叹号请求操作系统运行。

#### 数据类型

简单类型数据：

整数

整型 —— int、短整型 —— short、长整型 —— long int（4、2、8）

浮点数

float、double float（4、8）

字符

char （1）

复合类型数据

结构体

枚举

#### 变量

数据是内容，变量是容器，数据类型是内容的性质。

或者说变量就是字母，对应内存的空闲单元，是我对内存单元的命名。

#### 变量必须初始化

cpu将程序从外存调入内存运行，内存为其分配一定空间。运行时，空间不允许其他程序使用，运行结束后释放空间，但不清理数据。如果不初始化，就使用遗留的垃圾数据，没有意义。

#### 常量的表示

整数（补码）

十六进制：前面加ox、八进制：前面加0.

小数（IEEE）

传统、科学计数法e

字符（ASCII码和补码）

单个字符用单引号括起来

字符串用双引号括起来
