from torchvision import transforms as tfs
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import torch
import os
import time
import Config

# torch.set_default_tensor_type(torch.DoubleTensor)


# read file
def readText(filePath):
    try:
        f = open(filePath, "r")
        str = f.read()
        return str
    finally:
        if f:
            f.close()


def textToPoint(path):
    text = readText(path)
    lines = text.split("\n")
    # print(len(lines))
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
    return torch.from_numpy(points.reshape(68 * 2)).type(torch.DoubleTensor)


def tensorToPoint(tensor):
    na = tensor.numpy().reshape(68, 2)
    return na


# -*- coding:utf-8 -*-

pic_strong = tfs.Compose([tfs.ColorJitter(0.5, 0.3, 0.3, 0.1), tfs.ToTensor()])
unloader = tfs.ToPILImage()


def imageToTensor(path):
    img = Image.open(path)
    width = img.size[0]
    height = img.size[1]
    img = img.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    imgTensor = pic_strong(img)
    return imgTensor.type(torch.DoubleTensor), width, height


def tensorToImage(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def getFiles():
    rootdir = "/home/xws/Downloads/300w_cropped/01_Indoor"
    a = []
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            if list[i].endswith(".png"):
                a.append((path, path.replace(".png", ".pts")))
                # print(path, list[i].replace(".png", ".pts"))
        # # 你想对文件的操作
        # print(path, list[i])

    a.sort(key=lambda a: a[0])
    return a


def loadOneIBUG(path):
    imgInfo = imageToTensor(path[0])
    imgTensor = imgInfo[0]

    width = imgInfo[1]
    height = imgInfo[2]
    points = textToPoint(path[1])
    for p in points:
        p[0] = p[0] / width
        p[1] = p[1] / height
    pointTensor = pointToTensor(points)
    return (imgTensor, pointTensor)


def loadIBUG(paths):
    datas = []
    for path in paths:
        if len(datas) >= Config.DATA_SIZE:
            continue

        print("path = ", path)
        data = loadOneIBUG(path)
        if data[0].size()[0] != 3:
            continue
        # print(data[0].size()[0])
        datas.append(data)
    return datas


def get_all_files_and_bboxes(is_train=True):
    if is_train:
        file = open("/home/xws/Downloads/300w_cropped/01_Indoor")
    else:
        file = open("/home/xws/Downloads/300w_cropped/01_Indoor")
    datas = []
    for line in file:
        if line.find(".jpg") >= 0:
            # bboxes = getOne(file)
            datas.append({"img": line, "bboxes": bboxes})
        else:
            continue
    file.close()
    return datas


def getTime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def show(plt, X, L):
    plt.cla()
    img = tensorToImage(X)
    plt.imshow(img)
    points = tensorToPoint(L.cpu().detach())
    for p in points:
        plt.plot(p[0] * Config.IMAGE_SIZE, p[1] * Config.IMAGE_SIZE, "r.")
    plt.savefig("/home/xws/Downloads/python/python/face/img/test0.png")


if __name__ == "__main__":
    getTime()


# a = getFiles()

# loadIBUG(a)

# t = imageToTensor("/home/xws/Downloads/300w_cropped/01_Indoor/indoor_300.png")
# print(t)
# str = readText("/home/xws/Downloads/300w_cropped/01_Indoor/indoor_300.pts")

# a = textToPoint("/home/xws/Downloads/300w_cropped/01_Indoor/indoor_300.pts")
# for val in a:
#     print(val[0] + val[1])

# na = pointToTensor(a)
# print(" na = ", na)

# na = tensorToPoint(na)
# print(" na = ", na)
