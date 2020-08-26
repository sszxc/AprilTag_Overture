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

if __name__ == "__main__":
    print("")
    print("Python.version: "+sys.version)
    print("OpenCV.version: "+cv2.__version__)

    print(random.randint(0, 9))

    path = "test_img"
    random_filename = random.choice([x for x in os.listdir(path)])
    print(path + "/" + random_filename)
    img_raw = cv2.imread(path + "/" + random_filename)
    cv2.imshow('image', img_raw)

    # # 计时开始
    # starttime = datetime.datetime.now()
    # print('**********Start!**********')

    # #计时结束
    # print('**********Stop!**********')
    # endtime = datetime.datetime.now()
    # print('Elapsed time:', end='')
    # print(endtime - starttime)
    # print('')

    cv2.waitKey(0)
    cv2.destroyAllWindows()
