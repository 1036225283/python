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
    def dot(self, arr, height, width):
        center(arr, height, width)
        a = np.inner(arr, self.m)
        cartesian(a, height, width)
        return a


# 笛卡尔坐标系->中心坐标系
def center(arr, height, width):
    for p in arr:
        p[0] = p[0] - width / 2
        p[1] = p[1] - height / 2


# 中心坐标系->笛卡尔坐标系
def cartesian(arr, height, width):
    for p in arr:
        p[0] = p[0] + width / 2
        p[1] = p[1] + height / 2


if __name__ == "__main__":
    m = Matrix()
    m.translation(3, 3)
    # m.rotation(30)
    # m.scale(2, 2)
    w = 1
    a = np.array([[1, 2, w], [3, 4, w], [5, 6, w]])
    # a = np.array([[1, 2, w]])
    # a = np.array([[1], [2], [3]])

    # print(np.inner(a, m.m))
    print(m.dot(a, 0, 0))
