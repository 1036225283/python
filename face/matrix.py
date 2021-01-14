# 单位矩阵 旋转矩阵 缩放矩阵

import numpy as np


class Matrix:
    def __init__(self, size):
        self.size = size
        self.m = np.eye(size)

    def identity(self):
        self.m = np.eye(self.size)

    def scale(self, x, y):
        m = np.eye(self.size)
        m[0][0] = x
        m[1][1] = y
        print(m)
        self.m = np.dot(self.m, m)


# 缩放


if __name__ == "__main__":
    m = Matrix(3)
    m.scale(2, 2)
    w = 0
    a = np.array([[1, 2, w], [3, 4, w], [5, 6, w]])
    # a = np.array([[1, 2, w]])
    # a = np.array([[1], [2], [3]])

    print(np.inner( a,m.m))
