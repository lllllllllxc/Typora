import torch
from skimage import exposure
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
import cv2
from torch import optim
from torch.optim import optimizer
from math import log, exp


M = np.array([[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]])


# im_channel取值范围：[0,1]
def f(im_channel):
    return np.power(im_channel, 1 / 3) if im_channel > 0.008856 else 7.787 * im_channel + 0.137931



def anti_f(im_channel):
    return np.power(im_channel, 3) if im_channel > 0.206893 else (im_channel - 0.137931) / 7.787
# endregion


# region RGB 转 Lab
# 像素值RGB转XYZ空间，pixel格式:(B,G,R)
# 返回XYZ空间下的值
def __rgb2xyz__(pixel):
    b, g, r = pixel[0], pixel[1], pixel[2]
    rgb = np.array([r, g, b])
    # rgb = rgb / 255.0
    # RGB = np.array([gamma(c) for c in rgb])
    XYZ = np.dot(M, rgb.T)
    XYZ = XYZ / 255.0
    return (XYZ[0] / 0.95047, XYZ[1] / 1.0, XYZ[2] / 1.08883)


def __xyz2lab__(xyz):
    """
    XYZ空间转Lab空间
    :param xyz: 像素xyz空间下的值
    :return: 返回Lab空间下的值
    """
    F_XYZ = [f(x) for x in xyz]
    L = 116 * F_XYZ[1] - 16 if xyz[1] > 0.008856 else 903.3 * xyz[1]
    a = 500 * (F_XYZ[0] - F_XYZ[1])
    b = 200 * (F_XYZ[1] - F_XYZ[2])
    return (L, a, b)


def RGB2Lab(pixel):
    """
    RGB空间转Lab空间
    :param pixel: RGB空间像素值，格式：[G,B,R]
    :return: 返回Lab空间下的值
    """
    xyz = __rgb2xyz__(pixel)
    Lab = __xyz2lab__(xyz)
    return Lab


# endregion

# region Lab 转 RGB
def __lab2xyz__(Lab):
    fY = (Lab[0] + 16.0) / 116.0
    fX = Lab[1] / 500.0 + fY
    fZ = fY - Lab[2] / 200.0

    x = anti_f(fX)
    y = anti_f(fY)
    z = anti_f(fZ)

    x = x * 0.95047
    y = y * 1.0
    z = z * 1.0883

    return (x, y, z)


def __xyz2rgb(xyz):
    xyz = np.array(xyz)
    xyz = xyz * 255
    rgb = np.dot(np.linalg.inv(M), xyz.T)
    # rgb = rgb * 255
    rgb = np.uint8(np.clip(rgb, 0, 255))
    return rgb


def Lab2RGB(Lab):
    xyz = __lab2xyz__(Lab)
    rgb = __xyz2rgb(xyz)
    return rgb
# endregion



    #目标函数是（sourcelave的gama次方-targetlave）平方+（gama-1）平方，所以导数是2（sourcelave的gama次方-targetlave）*
    #gama乘sourcelave的（gama-1）次方+2（gama-1）#没有β
def kaidao(gamma):
    ss=2*(pow(sourcelave,gamma)-targetlave)
    ss=ss*gamma
    x=pow (sourcelave,gamma-1)
    ss=ss*x+2*(gamma-1)
    return ss

def Gradient_Descent_d1(current_x ,learn_rate = 0.001,e = 0.001,count = 50000):
    # current_x initial x value
    # learn_rate 学习率
    # e error
    # count number of iterations
    for i in range(count):
        grad = kaidao(current_x) # 求当前梯度
        if abs(grad) < 0.001 : # 梯度收敛到控制误差内
            break # 跳出循环
        current_x = current_x - grad * learn_rate # 一维梯度的迭代公式
        print("第{}次迭代逼近值为{}".format(i+1,current_x))

    print("最小值为：",current_x)
    print("最小值保存小数点后6位：%.6f"%(current_x))
    return current_x

if __name__ == '__main__':
    source = cv2.imread("example/daytime/aachen_000004_000019_leftImg8bit.png")
    target = cv2.imread("example/nighttime/GOPR0356_frame_000339_rgb_anon.png")

    w1 = source.shape[0]
    h1 = source.shape[1]
    print(w1,h1)
    img_new1 = np.zeros((w1,h1,3))
    lab1 = np.zeros((w1,h1,3))
    lab5=np.zeros((w1,h1,3))
    lab4=np.zeros((w1,h1,3))
    sourcel=np.zeros((w1,h1,1))
    for i in range(w1):
        for j in range(h1):
            Lab1 = RGB2Lab(source[i,j])
            lab1[i, j] = (Lab1[0], Lab1[1], Lab1[2])

#“sourcelave” 和 “targetlave” 都是用来保存亮度值的变量，求亮度值的累加
    sourcelave=0
    targetlave=0
    for i in range(w1):
       for j in range (h1):
           sourcel[i,j]=(lab1[i,j][0])
           sourcelave+=lab1[i,j][0]

    sourcelave=sourcelave/(w1*h1)
    w2 = target.shape[0]
    h2 = target.shape[1]
    print(w2, h2)
    img_new2 = np.zeros((w2, h2, 3))
    lab2 = np.zeros((w2, h2, 3))
    tarlab=np.zeros((w2,h2,3))
    targetl=np.zeros((w2,h2,1))
    for i in range(w2):
        for j in range(h2):
            Lab2 = RGB2Lab(target[i, j])
            lab2[i, j] = (Lab2[0], Lab2[1], Lab2[2])

    for i in range(w2):
        for j in range(h2):
            targetl[i,j]=(lab2[i,j][0])



    for i in range(w2):
        for j in range(h2):
            targetlave+=lab2[i,j][0]

    targetlave=targetlave/(w2*h2)
    #以上对target重复source的操作

    gama=log(targetlave,sourcelave)#初始值
    print("当前gama",gama)
    aves=0
    avel=0
    sums=0
    suml=0
    gamaa=Gradient_Descent_d1(gama)
    print("求导后gama",gamaa)
    print("sourceave",sourcelave)
    print("targetave",targetlave)
    multi=True if source.shape[-1]>1 else False
    matched=exposure.match_histograms(source,target)

#通过 matched 数组获取图像的宽度和高度，分别保存在变量 w3 和 h3 中。
    w3=matched.shape[0]
    h3=matched.shape[1]

    img_new3 = np.zeros((w3, h3, 3))
    for i in range(w3):
        for j in range(h3):
            rgb = RGB2Lab(matched[i, j])
            img_new1[i, j] = (rgb[0], rgb[1], rgb[2])

    for i in range(w3):
        for j in range(h3):
            rgb = Lab2RGB(img_new1[i, j])
            img_new3[i, j] = (rgb[2], rgb[1], rgb[0])



    for i in range(w3):
        for j in range(h3):
            img_new1[i,j][0]=sourcel[i,j][0]

    first = np.zeros((w3, h3, 3))

    for i in range(w3):
        for j in range(h3):
            rgb = Lab2RGB(img_new1[i, j])
            first[i, j] = (rgb[2], rgb[1], rgb[0])


    for i in range(w3):
        for j in range(h3):
            sourcel[i,j][0]=pow(sourcel[i,j][0],gamaa)

    print("X")
    for i in range(w3):
        for j in range(h3):
            img_new1[i,j][0]=sourcel[i,j][0]

    last=np.zeros((w3,h3,3))
    for i in range(w3):
        for j in range(h3):
            rgb = Lab2RGB(img_new1[i, j])
            last[i, j] = (rgb[2], rgb[1], rgb[0])


    cv2.imwrite(r"matched.png",last)

