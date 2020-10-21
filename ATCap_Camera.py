# Author: Xuechao Zhang
# Date: Oct 9th, 2020
# Description: 调用摄像头 连续测试

import cv2
import time
from AprilTag import Apriltag
import tagUtils as tud
import numpy as np
import CoordAlign as coa

if __name__ == "__main__":
    ap = Apriltag()
    ap.create_detector(debug=False)
    cap = cv2.VideoCapture(1)
    fps_real = fps = 24
    
    while (True):  # 创建无限循环，用于播放每一帧图像        
        starttime = time.time()
        ret, frame = cap.read()  # 读取图像的每一帧
        if ret == True:
            quads, detections = ap.detect(frame)  # 检测过程
            cv2.drawContours(frame, quads, -1, (0, 255, 0), 2)
            # print(detections)
            
            cv2.drawContours(frame, quads, -1, (0, 255, 0), 2)
            frame, tag_list = tud.set_coordinate(frame, detections)
            if tag_list:
                Affine_result = coa.cal_affine(np.array(tag_list, np.float32))
                rows, cols = frame.shape[:2]
                frame = cv2.warpAffine(frame, Affine_result, (rows, cols))

            cv2.putText(frame, "FPS:" + str(fps_real), (0, 25), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)
            cv2.imshow('frame', frame)  # 显示帧

            #等待1毫秒，判断此期间有无按键按下，以及按键的值是否是Esc键
            if cv2.waitKey(1000//int(fps)) & 0xFF == 27:
                break
        else:
            print ("Cap failed!")
            break

        endtime = time.time()
        fps_real = round(1 / (endtime - starttime),2) # 保留两位

    cap.release()  # 释放ideoCapture对象
    cv2.destroyAllWindows()  # 释放视频播放窗口
