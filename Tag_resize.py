# Author: Xuechao Zhang
# Date: Aug 25th, 2020
# Description: AprilTag 随机取一张图，最近邻插值放大

import os
import sys
import cv2
import random

if __name__ == "__main__":
    print("")
    print("Python.version: "+sys.version)
    print("OpenCV.version: "+cv2.__version__)

    print(random.randint(0, 9))

    path = "tag36h11"
    random_filename = random.choice([x for x in os.listdir(path)\
        if (os.path.splitext(x)[0][:3] == "tag" and \
            os.path.splitext(x)[-1][1:] == "png")])  # 选择开头为tag的png图片
    print(path + "/" + random_filename)
    img_raw = cv2.imread(path + "/" + random_filename)
    sp = img_raw.shape # 看看大小
    print("img_size:" + str(sp))
    multiple = 40 # 放大
    img_resize = cv2.resize(img_raw, (sp[1] * multiple, sp[0] * multiple), interpolation=cv2.INTER_NEAREST)
        
    cv2.imshow('image', img_resize)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
