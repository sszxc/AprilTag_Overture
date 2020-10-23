# Author: Xuechao Zhang
# Date: Oct 22nd, 2020
# Description: STF俱乐部 智能机器人系统综合设计 集成
#               RealSense调用、坐标平面映射

import CoordAlign as coa
import tagUtils as tud
from AprilTag import Apriltag
import time
import pyrealsense2 as rs
import numpy as np
import cv2

def init_RealSense():
    """
    RealSense初始化
    获取校正对象
    """
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    return pipeline, align

def eject_RealSense(pipeline):
    pipeline.stop()

def get_RS_frame(pipeline, align, aligned = True):
    """
    color_image 颜色图
    depth_image 深度图
    depth_colormap 深度图上色
    """
    while (True):
        try:
            frames = pipeline.wait_for_frames()
            if aligned:
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
            else:
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

            if  depth_frame and color_frame:
                break
        except Exception as e:  # 中途断连
            print(e)
            return
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    return color_image, depth_image, depth_colormap

def AprilTag_detection(frame):
    """
    检测AprilTag
    如果三码齐全则输出坐标序列tag_list
    frame为可视化效果
    """
    ap = Apriltag()
    ap.create_detector(debug=False)

    quads, detections = ap.detect(frame)
    cv2.drawContours(frame, quads, -1, (0, 255, 0), 2)
    # print(detections)

    cv2.drawContours(frame, quads, -1, (0, 255, 0), 2)
    frame, tag_list = tud.set_coordinate(frame, detections, 38000)

    return tag_list, frame

def set_claw_list():
    """
    记录拖动示教得到的claw_list
    需要和上位机操作一下这部分代码
    """
    # 拖动示教 机器人末端触碰对应Tag时记录示教器给出的坐标
    H_T0 = [0.731, -0.036, 0.001]
    H_T1 = [0.778, 0.195, 0.004]
    H_T2 = [0.513, 0.244, -0.001]
    claw_list = np.array([np.array(H_T0)[0:2], np.array(H_T1)[0:2], np.array(H_T2)[0:2]], np.float32)
    return claw_list


def C2R_solve(Affine, src):
    """
    输入像素坐标系
    输出机器人坐标系
    """
    return np.dot(Affine, src)

if __name__ == "__main__":
    pipeline, align = init_RealSense()
    
    # 标定阶段
    claw_list = set_claw_list()
    while (True):
        color_image, _, _ = get_RS_frame(pipeline, align, aligned=True)  # 获取图像
        tag_list, _ = AprilTag_detection(color_image) # Tag检测
        if tag_list:
            break
    Affine_result = coa.cal_affine(np.array(tag_list, np.float32), claw_list)  # claw_list暂时没加
    # 结束标定阶段

    _, depth_image, _ = get_RS_frame(pipeline, align, aligned=True)  # 传递深度图
    # target = Dex-net(depth_image)
    target = np.array([[475, 160, 1]]).T
    des = C2R_solve(Affine_result, target)  # 解算 加上深度坐标就可以输出

    # 逃出
    eject_RealSense(pipeline)
    print("finish")