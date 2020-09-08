# Author: Xuechao Zhang
# Date: Aug 25th, 2020
# Description: AprilTag 识别复现

import os
import sys
import cv2
import random
import datetime
import numpy as np
from matplotlib import pyplot as plt

def detect(frame, debug = False):
    """
    随意拼凑一下 Aug 31st, 2020
    """
    gray = np.array(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))    

    blur = cv2.GaussianBlur(gray, (3, 3), 0.8)

    canny = cv2.Canny(blur, 50, 350, apertureSize=3)  # adaptive thresholding or canny

    if debug:
        cv2.imshow('frame', frame)
        cv2.imshow('gray', gray)
        cv2.imshow('blur', blur)
        cv2.imshow('canny', canny)

    _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # findContours函数
    # 参数：二值图；检索模式；轮廓的近似表示方法
    # contours是个list，其中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示
    # hierarchy表示contours之间的关系
    # https://blog.csdn.net/hjxu2016/article/details/77833336


    """
    compute convex hulls and find maximum inscribed quadrilaterals
    计算凸包 以及 最大内切四边形 还没看懂
    """
    quads = []  # array of quad including four peak points
    hulls = []
    for i in range(len(contours)):
        if (hierarchy[0, i, 3] < 0 and contours[i].shape[0] >= 4):  # 第i个轮廓不存在父轮廓？ 且其中的点多于4个
            area = cv2.contourArea(contours[i]) # 计算面积
            if area > 400:
                hull = cv2.convexHull(contours[i]) # 计算凸包
                if (area / cv2.contourArea(hull) > 0.8): # 要求不凹太多
                    hulls.append(hull)
                    quad = cv2.approxPolyDP(hull, 8, True)  # 多边形拟合 平滑处理 (# maximum_area_inscribed)
                    if (len(quad) == 4):
                        areaqued = cv2.contourArea(quad)
                        areahull = cv2.contourArea(hull)
                        if areaqued / areahull > 0.8 and areahull >= areaqued: # 要求拟合后的多边形面积比拟合前小一点点？
                            quads.append(quad)

    framecopy = np.copy(frame)
    cv2.drawContours(frame, quads, -1, (0, 255, 0), 2)
    cv2.drawContours(framecopy, hulls, -1, (0, 255, 0), 2)
    plt.figure().set_size_inches(19.2, 10.8)
    plt.subplot(211)
    plt.imshow(frame)
    plt.subplot(212)
    plt.imshow(framecopy)
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_test():
    """
    随机选一张测试图片
    """
    path = "test_img"
    random_filename = random.choice([x for x in os.listdir(path)])
    print(path + "/" + random_filename)
    img_raw = cv2.imread(path + "/" + random_filename)
    detect(img_raw)

if __name__ == "__main__":
    print("")
    print("Python.version: "+sys.version)
    print("OpenCV.version: "+cv2.__version__)

    img_test()

    # # 计时开始
    # starttime = datetime.datetime.now()
    # print('**********Start!**********')

    # #计时结束
    # print('**********Stop!**********')
    # endtime = datetime.datetime.now()
    # print('Elapsed time:', end='')
    # print(endtime - starttime)
    # print('')


