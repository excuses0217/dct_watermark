from email.mime import base
import math
import cv2
import numpy as np
import skimage.util as skiu
from skimage import transform, metrics
import matplotlib.image as mpimg
import math


if __name__ == '__main__':
    root = ".."
    base = cv2.imread(
        r"Robustness_test/normal.png".format(root), cv2.IMREAD_GRAYSCALE)
    imgs = list()
    for i in [50, 30, 15, 8, 2]:
        imgs.append(cv2.imread('Robustness_test/' +
                    str(i) + '.png', cv2.IMREAD_GRAYSCALE))
    info = ["50", "30", "15", "8", "2"]
    i = 0
    for img in imgs:
        PSNR = metrics.peak_signal_noise_ratio(base, img, data_range=255)
        SSIM = metrics.structural_similarity(base, img, full=True, win_size=7)
        print("CUTATTACK: {}% | PSNR:{} | SSIM:{}".format(info[i], PSNR, SSIM[0]))
        i += 1
