# 单位矩阵 旋转矩阵 缩放矩阵

import numpy as np
import math


class Matrix:
    def __init__(self):
        self.size = 3
        self.m = np.eye(self.size)

    def identity(self):
        self.m = np.eye(self.size)

    # 缩放
    def scale(self, x, y):
        m = np.eye(self.size)
        m[0][0] = x
        m[1][1] = y
        self.m = np.dot(self.m, m)

    # 旋转
    def rotation(self, angle):
        m = np.eye(self.size)
        angle = angle * math.pi / 180
        sin = math.sin(angle)
        cos = math.cos(angle)
        m[0][0] = cos
        m[0][1] = -sin
        m[1][0] = sin
        m[1][1] = cos
        self.m = np.dot(self.m, m)

    # 平移translation
    def translation(self, x, y):
        m = np.eye(self.size)
        m[0][2] = x
        m[1][2] = y
        self.m = np.dot(self.m, m)

    # dot
    def dot(self, arr):
        return np.inner(arr, self.m)


if __name__ == "__main__":
    m = Matrix()
    # m.translation(2, 2)
    m.rotation(30)
    w = 1
    a = np.array([[1, 2, w], [3, 4, w], [5, 6, w]])
    # a = np.array([[1, 2, w]])
    # a = np.array([[1], [2], [3]])

    # print(np.inner(a, m.m))
    print(m.dot(a))
