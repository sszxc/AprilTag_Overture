# Author: Xuechao Zhang
# Date: Oct 9th, 2020
# Description: 调用 RealSense 连续测试

import CoordAlign as coa
import tagUtils as tud
from AprilTag import Apriltag
import time
import pyrealsense2 as rs
import numpy as np
import cv2

if __name__ == "__main__":

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    ap = Apriltag()
    ap.create_detector(debug=False)
    fps_real = fps = 24

    while (True):  # 创建无限循环，用于播放每一帧图像
        starttime = time.time()

        # Wait for a coherent pair of frames: depth and color
        try:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
        except Exception as e: # 中途断连
            print(e)
            break

        frame = np.asanyarray(color_frame.get_data())
        # depth_image = np.asanyarray(depth_frame.get_data())
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        quads, detections = ap.detect(frame)  # 检测过程
        cv2.drawContours(frame, quads, -1, (0, 255, 0), 2)
        # print(detections)

        cv2.drawContours(frame, quads, -1, (0, 255, 0), 2)
        frame, tag_list = tud.set_coordinate(frame, detections)
        if tag_list:  # 三码齐全
            Affine_result = coa.cal_affine(np.array(tag_list, np.float32)) # 仿射变换
            rows, cols = frame.shape[:2]
            # frame = cv2.warpAffine(frame, Affine_result, (rows, cols))

        cv2.putText(frame, "FPS:" + str(fps_real), (0, 25), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)
        cv2.imshow('frame', frame)  # 显示帧

        #等待1毫秒，判断此期间有无按键按下，以及按键的值是否是Esc键
        if cv2.waitKey(1000//int(fps)) & 0xFF == 27:
            break
        
        endtime = time.time()
        fps_real = round(1 / (endtime - starttime), 2) # 保留两位

    pipeline.stop()  # Stop streaming
    cv2.destroyAllWindows()  # 释放视频播放窗口
