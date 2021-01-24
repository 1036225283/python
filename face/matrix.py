# 单位矩阵 旋转矩阵 缩放矩阵

import numpy as np
import math
import warnings


class Matrix:
    def __init__(self, height=1, width=1):
        self.size = 3
        self.height = height
        self.width = width
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
        if x > 1:
            x = 1
            warnings.warn("translation.x must <= 1.")

        if x < -1:
            x = -1
            warnings.warn("translation.x must >= -1.")
        if y > 1:
            y = 1
            warnings.warn("translation.x must <= 1.")

        if y < -1:
            y = -1
            warnings.warn("translation.y must >= -1.")

        m = np.eye(self.size)
        m[0][2] = x
        m[1][2] = y
        self.m = np.dot(self.m, m)

    # 平移translation
    def translation_point(self, x, y):
        if x > 1:
            x = 1
            warnings.warn("translation.x must <= 1.")

        if x < -1:
            x = -1
            warnings.warn("translation.x must >= -1.")
        if y > 1:
            y = 1
            warnings.warn("translation.x must <= 1.")

        if y < -1:
            y = -1
            warnings.warn("translation.y must >= -1.")

        m = np.eye(self.size)
        m[0][2] = x / 2 * self.width
        m[1][2] = y / 2 * self.height
        self.m = np.dot(self.m, m)

    # dot
    def dot(self, arr):
        self.center(arr)
        a = np.inner(arr, self.m)
        self.cartesian(a)
        return a

    # dot point 68
    def dot_point_68(self, points):
        newpoints = np.arange(68 * 3, dtype=float).reshape(68, 3)

        for i, p in enumerate(points):
            newpoints[i][0] = p[0]
            newpoints[i][1] = p[1]
            newpoints[i][2] = 1

        pp = self.dot(newpoints)
        newpoints = points.copy()
        for i, p in enumerate(pp):
            points[i][0] = p[0] / self.width
            points[i][1] = p[1] / self.height

            newpoints[i][0] = p[0]
            newpoints[i][1] = p[1]

        return (points, newpoints)

    def chagne_height_width(self):
        v = np.array([[self.height, self.width, 1]])
        v = self.dot(v)
        self.height = v[0][0]
        self.width = v[0][1]

    # affine_grid
    def to_theta(self):
        return np.delete(self.m, -1, 0)

    # 笛卡尔坐标系->中心坐标系
    def center(self, arr):
        for p in arr:
            p[0] = p[0] - self.width / 2
            p[1] = p[1] - self.height / 2

    # 中心坐标系->笛卡尔坐标系
    def cartesian(self, arr):
        for p in arr:
            p[0] = p[0] + self.width / 2
            p[1] = p[1] + self.height / 2


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
    print(m.dot(a))
    print(m.to_theta())
    print(np.delete(m.m, -1, 0))
