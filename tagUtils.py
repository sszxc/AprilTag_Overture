# Author:rain
# E-Mail: raindown95@outlook.com
# Link: https://github.com/BlackJocker1995/Apriltag_python
# Description: AprilTag的一些功能函数
#               包括像素坐标系到相机坐标系的转换

import numpy as np
import cv2
import math
from scipy.optimize import fsolve

# 相机内参 Oct.15th
# 1103.26946498287	0	0
# 0	1102.68487162187	0
# 347.415754825587	269.670874761139	1
# realSense内参 Oct.21st
# 639.898749592195	0	0
# 0	638.829623034055	0
# 327.955458515705	264.288936727808	1
Kmat = np.array([[639.898749592195, 0, 327.955458515705],
                 [0, 638.829623034055, 264.288936727808],
                 [0, 0, 1]])

ss = 0.5
src = np.array([[-ss, -ss, 0],
                    [ss, -ss, 0],
                    [ss, ss, 0],
                    [-ss, ss, 0]])
#500 is scale
# 原作者内参
# Kmat = np.array([[700, 0, 0],
#                      [0, 700, 0],
#                      [0, 0, 1]])* 1.0
disCoeffs= np.zeros([4, 1]) * 1.0
edges = np.array([[0, 1],
                              [1, 2],
                              [2, 3],
                              [3, 0]])
def project(H,point):
    x = point[0]
    y = point[1]
    z = H[2,0]*x +H[2,1]*y+H[2,2]

    point[0] = (H[0,0]*x+H[0,1]*y+H[0,2])/z*1.0
    point[1] = (H[1, 0]*x+H[1, 1] *y + H[1, 2]) / z*1.0
    return point

def project_array(H):
    ipoints = np.array([[-1,-1],
                        [1,-1],
                        [1,1],
                        [-1,1]])
    for point in ipoints:
        point = project(H,point)

    return ipoints

def sovle_coord(R1,R2,R3,edge = 1060):
    x = -(R2*R2 - R1*R1 - edge**2) / (2.0*edge)
    y = -(R3*R3 - R1*R1 - edge**2) / (2.0*edge)
    z =  (np.sqrt(R1*R1 - x * x - y * y))-edge
    return x,y,z


def verify_z(x,y,R4,edge = 1060):
        x = edge - x
        y = edge - y
        rand2 = x**2+y**2
        h = np.sqrt(R4**2 - rand2)
        return edge - h


def get_Kmat(H):
    campoint = project_array(H)*1.0
    opoints = np.array([[-1.0, -1.0, 0.0],
                        [1.0, -1.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [-1.0, 1.0, 0.0]])
    opoints = opoints*0.5
    rate, rvec, tvec = cv2.solvePnP(opoints, campoint, Kmat, disCoeffs)
    return rvec,tvec

def get_pose_point(H):
    """
    将空间坐标转换成相机坐标
    Trans the point to camera point
    :param H: homography
    :return:point
    """
    rvec, tvec =  get_Kmat(H)
    point, jac = cv2.projectPoints(src, rvec, tvec, Kmat, disCoeffs)
    return np.int32(np.reshape(point,[4,2]))

def get_pose_point_noroate(H):
    """
    将空间坐标转换成相机坐标但是不旋转
    Trans the point to camera point but no rotating
    :param H: homography
    :return:point
    """
    rvec, tvec = get_Kmat(H)
    point, jac = cv2.projectPoints(src, np.zeros(rvec.shape), tvec, Kmat, disCoeffs)
    return np.int32(np.reshape(point,[4,2]))

def average_dis(point,k):
    return np.abs( k/np.linalg.norm(point[0] - point[1]))
def average_pixel(point):
    return np.abs( np.linalg.norm(point[0] - point[1]))
def get_distance(H,t): # 没怎么看懂 四个点只取走了两个 还用的二范数 t很像比例系数
    points = get_pose_point_noroate(H)
    return average_dis(points,t)
def get_min_distance(array_detections,t):
    min = 65535;
    for detection in array_detections:
        #print(detection.id)
        dis = get_distance(detection.homography,t)
        if dis < min:
            min = dis
    return min;

def get_pixel(H):
    points = get_pose_point_noroate(H)
    return average_pixel(points)

def pixel2camera(center_x, center_y, dis):
    """
    像素坐标系转换到相机坐标系
    """
    coord = np.matmul(np.linalg.inv(Kmat) * dis, np.array([[center_x, center_y, 1]]).T)
    return coord

def set_coordinate(img, detections, dis_offset = 38000):
    """
    利用0,1,2三个标签 建立坐标系 绘图
    dis_offset为测距补偿量
    """
    tag_id = []
    center_list = []
    for detection in detections:
        tag_id.append(detection.id)
        point = get_pose_point(detection.homography)  # 用拟合出的变换矩阵再求一次角点
        # dis = round(get_distance(detection.homography, 122274), 2)
        # dis = round(get_distance(detection.homography, 55000), 2)  # 边长55mm标签
        dis = round(get_distance(detection.homography, dis_offset), 2)  # 边长64mm标签
        center_x = int(sum(point[:, 0]) / 4)
        center_y = int(sum(point[:, 1]) / 4)
        center_list.append(np.array([center_x, center_y]))

        pixel2camera(center_x, center_y, dis)
        # 显示id和相机坐标
        cv2.putText(img, str(detection.id), (center_x, center_y - 20), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 4)
        cv2.putText(img, str(np.round(pixel2camera(center_x, center_y, dis),2).ravel()), (center_x, center_y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 1)
        
    line = []
    if 0 in tag_id and 1 in tag_id:
        line.append(np.array([center_list[tag_id.index(0)],  center_list[tag_id.index(1)]], np.int32).reshape((-1, 1, 2)))
    if 1 in tag_id and 2 in tag_id:
        line.append(np.array([center_list[tag_id.index(1)],  center_list[tag_id.index(2)]], np.int32).reshape((-1, 1, 2)))
    if line:
        cv2.polylines(img, line, True, (0, 255, 255), 2)
    center_list_sort = [] # 如果三码齐全 按顺序排好 输出
    if len(line) == 2:
        center_list_sort.append(center_list[tag_id.index(0)])
        center_list_sort.append(center_list[tag_id.index(1)])
        center_list_sort.append(center_list[tag_id.index(2)])
    return img, center_list_sort

if __name__ == "__main__":
    print(str(pixel2camera(100, 200, 60).ravel()))
