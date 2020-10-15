# Author:rain
# E-Mail: raindown95@outlook.com
# Link: https://github.com/BlackJocker1995/Apriltag_python

import numpy as np
import cv2
class TagDetection(object):

    def __init__(self):
        self.good = False
        self.obsCode = -1
        self.matchCode = -1
        self.id = -1
        self.hammingDistance = -1
        self.rotation = -1
        self.points = None
        self.homography = None

    def addHomography(self):
        """
        find homography
        """
        self._recomputeHomography()

    def addPoint(self,points):
        """
        add quad`points to detection
        :param points: quad`s points
        """
        self.points = points

    def _recomputeHomography(self):
        """
        从单位矩形到像素坐标下目标矩形的变换矩阵
        find Homography
        :return: Homography
        """
        src = np.array([
          [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1],
        ]).reshape(-1,1,2)
        dst = np.array(self.points)
        retval,mark = cv2.findHomography(np.array(src),np.array(dst))
        self.homography = retval

