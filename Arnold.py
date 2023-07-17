from importlib.resources import path
from os import times
from re import X
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pylab
from PIL import Image
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False

"""
function: Arnold 加密算法
params: 
    image: 输入的原始图像
    a,b: Arnold 置乱参数
    shuffle_times: 置乱次数
return:
    arnold_encode: Arnold 置乱后的图像
"""


def arnold_encode(image, a, b, shuffle_times):

    # 创建新图像
    arnold_image = np.zeros(shape=image.shape)

    # 计算N
    h, w = image.shape[0], image.shape[1]
    N = h   # 或N=w

    # 遍历像素坐标变换
    for ori_x in range(h):
        for ori_y in range(w):
            # 按照公式坐标变换
            new_x = (1*ori_x + b*ori_y) % N
            new_y = (a*ori_x + (a*b+1)*ori_y) % N
            arnold_image[new_x, new_y, :] = image[ori_x, ori_y, :]

    if shuffle_times == 0:
        return image
    elif shuffle_times == 1:
        return arnold_image
    else:
        return arnold_encode(arnold_image, a, b, (shuffle_times - 1))


def arnold_decode(image, a, b, shuffle_times):

    # 创建新图像
    decode_image = np.zeros(shape=image.shape)

    # 计算N
    h, w = image.shape[0], image.shape[1]
    N = h  # 或N=w

    # 遍历像素坐标变换
    for time in range(shuffle_times):
        for ori_x in range(h):
            for ori_y in range(w):
                # 按照公式坐标变换
                new_x = ((a*b+1)*ori_x + (-b) * ori_y) % N
                new_y = ((-a)*ori_x + ori_y) % N
                decode_image[new_x, new_y, :] = image[ori_x, ori_y, :]
    if shuffle_times == 0:
        return image
    if shuffle_times == 1:
        return decode_image
    else:
        return arnold_decode(decode_image,  a, b, (shuffle_times - 1))


"""
function: Arnold 加密算法后保存加密图像
params: 
    image: 输入的原始图像
    a,b: Arnold 置乱参数
    times_list: 置乱次数列表
"""


def save_image(image, a, b, times_list):
    for t in times_list:
        name = 'arnold_image' + '_' + str(a) + '_' + str(b) + '_' + str(t)
        mpimg.imsave('./save_img/' + name + '.png',
                     arnold_encode(image, a, b, t))  # 保存路径

a = 1
b = 1
shuffle_times = 20
'''
剪切攻击
'''
bg = mpimg.imread('Arnold_test_img\Japan_Mount1920x1080.jpg')
plt.subplot(3, 3, 1), plt.title('载体图像')
plt.imshow(bg), plt.axis('off')

wm = mpimg.imread('Arnold_test_img/fei_logo_135x135.png')
plt.subplot(3, 3, 2), plt.title('水印图像')
plt.imshow(wm), plt.axis('off')

en_wm = arnold_encode(wm, a, b, shuffle_times)
mpimg.imsave('Arnold_test_img/arnoldwm.png',en_wm)  # 保存路径
plt.subplot(3, 3, 3), plt.title('Arnold变换后')
plt.imshow(en_wm), plt.axis('off')

bg_wm = mpimg.imread('save_img\未遭受攻击.png')
plt.subplot(3, 3, 4), plt.title('含水印图像')
plt.imshow(bg_wm), plt.axis('off')

wm_ext = mpimg.imread('save_img\正常提取水印.png')
plt.subplot(3, 3, 5), plt.title('提取水印')
plt.imshow(wm_ext), plt.axis('off')

wm_ext_de = arnold_decode(wm_ext, a, b, shuffle_times)
mpimg.imsave('Arnold_test_img/正常提取水印.png',wm_ext_de)  # 保存路径
plt.subplot(3, 3, 6), plt.title('逆Arnold变换后')
plt.imshow(wm_ext_de), plt.axis('off')

cut_bg_wm = mpimg.imread('save_img\剪切.png')
plt.subplot(3, 3, 7), plt.title('剪切攻击')
plt.imshow(cut_bg_wm), plt.axis('off')

cut_wm_ext = mpimg.imread('save_img\剪切攻击后提取水印.png')
plt.subplot(3, 3, 8), plt.title('剪切攻击后提取水印')
plt.imshow(cut_wm_ext), plt.axis('off')

cut_wm_ext_de = arnold_decode(cut_wm_ext, a, b, shuffle_times)
mpimg.imsave('Arnold_test_img/剪切攻击后提取水印.png',cut_wm_ext_de)  # 保存路径
plt.subplot(3, 3, 9), plt.title('逆Arnold变换后')
plt.imshow(cut_wm_ext_de), plt.axis('off')

plt.show()
pylab.show()

