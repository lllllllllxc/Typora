import cv2
import numpy as np
from math import log

from PIL import Image
from matplotlib import pyplot as plt
from skimage import color

#打印直方图
def his_save(a,b,c,name):
    plt.figure()
    plt.plot(a)
    plt.plot(b)
    plt.plot(c)
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    # 保存直方图
    plt.savefig(name)

def com_his(image):
    input_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    input_a, input_b = input_lab[:, :, 1], input_lab[:, :, 2]
    input_c = input_lab[:, :, 0]
    input_hist_a = cv2.calcHist([input_a], [0], None, [256], [0, 256])
    input_hist_b = cv2.calcHist([input_b], [0], None, [256], [0, 256])
    input_hist_l = cv2.calcHist([input_c], [0], None, [256], [0, 256])
    his_save(input_hist_a,input_hist_b,input_hist_l,'histogram3.png')
    return 0

def rgb_to_lab(image):
    # 将输入图像转换为64位浮点数
    image_float = image.astype(np.float64) / 255.0
    # 使用SciPy的color库执行RGB到LAB转换
    lab_image = color.rgb2lab(image_float)
    return lab_image


def lab_to_rgb(lab_image):
    lab_image = cv2.convertScaleAbs(lab_image)  # 将图像深度转换为 CV_8U 类型
    rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    return rgb_image


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


def gamma_correction(l_channel, gamma):
    gamma_corrected_l_channel = np.power(l_channel, gamma)
    return gamma_corrected_l_channel

def histogram_matching(input_image, target_image):
    # 转换输入图像和目标图像到Lab颜色空间
    input_lab = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB)

    # 提取输入图像和目标图像的a和b通道
    input_a, input_b = input_lab[:, :, 1], input_lab[:, :, 2]
    input_c = input_lab[:, :, 0]
    target_a, target_b = target_lab[:, :, 1], target_lab[:, :, 2]
    target_c = target_lab[:, :, 0]

    # 计算输入图像和目标图像的直方图
    input_hist_a = cv2.calcHist([input_a], [0], None, [256], [0, 256])
    input_hist_b = cv2.calcHist([input_b], [0], None, [256], [0, 256])
    target_hist_a = cv2.calcHist([target_a], [0], None, [256], [0, 256])
    target_hist_b = cv2.calcHist([target_b], [0], None, [256], [0, 256])
    target_hist_l = cv2.calcHist([target_c], [0], None, [256], [0, 256])
    input_hist_l = cv2.calcHist([input_c], [0], None, [256], [0, 256])
    # 计算输入图像和目标图像的累积分布函数（CDF）
    input_cdf_a = input_hist_a.cumsum()
    input_cdf_b = input_hist_b.cumsum()
    input_cdf_c = input_hist_l.cumsum()
    target_cdf_a = target_hist_a.cumsum()
    target_cdf_b = target_hist_b.cumsum()
    target_cdf_c = target_hist_l.cumsum()


    # 归一化累积分布函数（将其缩放到 [0, 255] 范围）
    input_cdf_normalized_a = (input_cdf_a - input_cdf_a.min()) * 255 / (input_cdf_a.max() - input_cdf_a.min())
    input_cdf_normalized_b = (input_cdf_b - input_cdf_b.min()) * 255 / (input_cdf_b.max() - input_cdf_b.min())
    input_cdf_normalized_c = (input_cdf_c - input_cdf_c.min()) * 255 / (input_cdf_c.max() - input_cdf_c.min())
    target_cdf_normalized_a = (target_cdf_a - target_cdf_a.min()) * 255 / (target_cdf_a.max() - target_cdf_a.min())
    target_cdf_normalized_b = (target_cdf_b - target_cdf_b.min()) * 255 / (target_cdf_b.max() - target_cdf_b.min())
    target_cdf_normalized_c = (target_cdf_c - target_cdf_c.min()) * 255 / (target_cdf_c.max() - target_cdf_c.min())

    # # 创建空白输出图像，与输入图像相同大小
    output_image = np.zeros_like(input_image)

    # 根据CDF之间的映射函数将输入图像的a和b通道映射到目标图像的a和b通道
    lookup_table_a = np.interp(input_cdf_normalized_a, target_cdf_normalized_a, np.arange(256))
    lookup_table_b = np.interp(input_cdf_normalized_b, target_cdf_normalized_b, np.arange(256))
    lookup_table_c = np.interp(input_cdf_normalized_c, target_cdf_normalized_c, np.arange(256))
    output_a = cv2.LUT(input_a, lookup_table_a)
    output_b = cv2.LUT(input_b, lookup_table_b)
    output_c = cv2.LUT(input_c, lookup_table_c)

    gamma_corrected_l_ms = gamma_correction(input_c, optimized_gamma)

    # 将映射后的a和b通道与伽马校正的L通道组合成Lab图像
    output_lab = cv2.merge([np.uint8(gamma_corrected_l_ms), np.uint8(output_a), np.uint8(output_b)])

    his_save(input_hist_a,input_hist_b,input_hist_l,'histogram1.png')
    his_save(target_hist_a, target_hist_b, target_hist_l, 'histogram2.png')
    # his_save(lookup_table_a, lookup_table_b, gamma_corrected_l_ms, 'histogram3.png')

    # 将输出图像转换回RGB颜色空间
    output_image = lab_to_rgb(output_lab)

    return output_image


source_image = cv2.imread('example/daytime/bochum_000000_021325_leftImg8bit.png')  # 替换成实际的文件路径和名称
target_image = cv2.imread('example/nighttime/GOPR0356_frame_000339_rgb_anon.png')  # 替换成实际的文件路径和名称


w1 = source_image.shape[0]
h1 = source_image.shape[1]

lab1 = rgb_to_lab(source_image)
sourcel = lab1[:, :, 0]
# sourceGrayAve = np.sum(sourceGray_image) / (w1 * h1)
source_a = lab1[:, :, 1]
sourcelave = np.sum(sourcel) / (w1 * h1)


w2 = target_image.shape[0]
h2 = target_image.shape[1]
lab2 = rgb_to_lab(target_image)
targetl = lab2[:, :, 0]
targetlave = np.sum(targetl) / (w1 * h1)

# 设置初始猜测值gamma0
gama=log(targetlave,sourcelave)
print(targetlave)
print(sourcelave)

beta = 1
learning_rate = 0.001

optimized_gamma = Gradient_Descent_d1(gama)
print(optimized_gamma)
matched=histogram_matching(source_image, target_image)
image_pil = Image.fromarray(np.uint8(matched))
com_his(matched)
image_pil.save('result_final.png')













