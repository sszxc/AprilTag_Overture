# Author: Xuechao Zhang
# Date: Oct 16th, 2020
# Description: 相机坐标系与机器人坐标系的仿射变换

import numpy as np
import math
import cv2

# 拖动示教 机器人末端触碰对应Tag时记录示教器给出的坐标
H_T0 = [0.597009, 0.266322, 0.001992]
H_R0 = [-176.961823, -5.470199, 169.642593]

H_T1 = [0.652023, -0.046341, 0.001979]
H_R1 = [-176.952682, -5.476172, 141.537018]

H_T2 = [0.454048, -0.066969, 0.009171]
H_R2 = [-172.948593, -9.487500, 141.045471]

def eulerAnglesToRotationMatrix(theta):
    """
    # Author: Xuechao Zhang
    # Date: Sept 8th, 2020
    # Description: 欧拉角到旋转矩阵的转换
    #       https://blog.csdn.net/ouyangandy/article/details/105965898
    """
    # demo:
    # eulerAngles = [13.8972, -175.3867, 29.0579]
    # print(eulerAnglesToRotationMatrix(eulerAngles))

    theta = [x / 180.0 * 3.14159265 for x in theta]  # 角度转弧度
    R_x = np.array([[1,                  0,                   0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0,  math.sin(theta[0]), math.cos(theta[0])]])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0,                  1,                  0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]),  0],
                    [0,                  0,                   1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def data_conv(R, T):
    """
    欧拉角和位置向量 拼成4*4矩阵
    """
    # demo
    # H_0 = data_conv(H_R0, H_T0)
    # H_1 = data_conv(H_R1, H_T1)
    # H_2 = data_conv(H_R2, H_T2)
    _34 = np.c_[eulerAnglesToRotationMatrix(R), np.array(T).T] # 加一列
    _44 = np.r_[_34, np.array([[0, 0, 0, 1]])]  # 加一行
    return _44


def cal_affine(tag_list=np.array([[398, 273], [197, 289], [244, 128]], np.float32)):
    """
    仿射变换
    """
    src = tag_list
    # dst = np.array([np.array(H_T0)[0:2], np.array(H_T1)[0:2], np.array(H_T2)[0:2]], np.float32)
    dst = np.float32([[100, 100], [100, 300], [300, 300]])
    
    Affine_result = cv2.getAffineTransform(src, dst)
    return Affine_result

if __name__ == "__main__":
    Affine_result = cal_affine()
    print("fhuaihfw")


