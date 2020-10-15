# Author: Xuechao Zhang
# Date: Oct 9th, 2020
# Description: 在同一个窗口下循环显示多张图片

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
