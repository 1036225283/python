import numpy as np
import torch

# read file
def readText(filePath):
    try:
        f = open(filePath, "r")
        str = f.read()
        return str
    finally:
        if f:
            f.close()


str = readText("/home/xws/Downloads/300w_cropped/01_Indoor/indoor_300.pts")


def textToPoint(text):
    lines = text.split("\n")
    print(len(lines))
    na = np.zeros((68, 2))
    for i, val in enumerate(lines):
        # print("序号：%s   值：%s" % (i + 1, val))
        if i < 3:
            continue
        if i > 70:
            continue
        # print("append：%s   值：%s" % (i + 1, val))
        vals = val.split(" ")
        na[i - 3][0] = float(vals[0])
        na[i - 3][1] = float(vals[1])
    return na


def pointToTensor(points):
    # na = np.zeros(68 * 2)
    # for i, p in enumerate(points):
    #     print("ww", i, p)
    #     na[i * 2] = p[0]
    #     na[i * 2 + 1] = p[1]

    # print("end ")
    return torch.from_numpy(points.reshape(68*2))


def tensorToPoint(tensor):
    na = tensor.numpy().reshape(68, 2)
    return na


a = textToPoint(str)
# for val in a:
#     print(val[0] + val[1])

na = pointToTensor(a)
print(" na = ", na)

# na = tensorToPoint(na)
# print(" na = ", na)