# Author: Xuechao Zhang
# Date: Oct 9th, 2020
# Description: 在同一个窗口下循环显示多张图片

import numpy as np
import os
import cv2
import sys
import random

if __name__ == "__main__":
    print("Python.version: "+sys.version)
    print("OpenCV.version: " + cv2.__version__)
    
    # while(True):
    #     path = "test_img"
    #     random_filename = random.choice([x for x in os.listdir(path)])
    #     print(path + "/" + random_filename)
    #     img_raw = cv2.imread(path + "/" + random_filename)

    #     cv2.imshow('frame', img_raw)
    #     cv2.waitKey(1000)

    # 因为是生成随机数做测试，设置固定随机数种子，可以保证每次结果一致
    np.random.seed(0)
    RGB = np.random.randint(0, 255, (5, 5, 3))
    alpha = np.random.randint(0, 255, (5, 5))
    RGBA = np.dstack((RGB, alpha))
    print('RGB = \n {}'.format(RGB))
    print('alpha = \n {}'.format(alpha))
    print('RGBA = \n {}'.format(RGBA))
    print('RGBA[:, :, 3] = \n {}'.format(RGBA[:, :, 3]))

    H_T0 = [0.731, -0.036, 0.001]
    H_T1 = [0.778, 0.195, 0.004]
    H_T2 = [0.513, 0.244, -0.001]
    claw_list = np.array([np.array(H_T0)[0:2], np.array(H_T1)[0:2], np.array(H_T2)[0:2]], np.float32)
    depth_align = np.array(H_T0)[3]
