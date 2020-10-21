import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()

cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(cfg)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("深度比例系数为：", depth_scale)
# 深度比例系数为： 0.0010000000474974513
# 测试了数个摄像头，发现深度比例系数都相同，甚至D435i的也一样。

align = rs.align(rs.stream.color)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Show images
        cv2.namedWindow('RealSense1', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense1', depth_colormap)

        cv2.namedWindow('RealSense2', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense2', color_image)

        
        
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

        # width = aligned_depth_frame.get_width()
        # height = aligned_depth_frame.get_height()
        # # print(width, '', height)
        # # 640  480
        # # print(int(width / 2))
        # # 320
        # dist_to_center = aligned_depth_frame.get_distance(int(width / 2), int(height / 2))
        # # 不知为什么没有用到深度比例系数
        # print(dist_to_center)
        # # 0.0
        # # 0.37800002098083496
        # # 0.3840000033378601
        # # 0.1940000057220459
        # # ...
finally:
    pipeline.stop()
