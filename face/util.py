from torchvision import transforms as tfs
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import torch
import os
import time
import Config
import matrix

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


def getFiles(rootdir="/home/xws/Downloads/300w_cropped/01_Indoor"):
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
    return (imgTensor, pointTensor, path[0])


# 对图像进行旋转
def rorateData(path, img, points, height, width, angle):
    points = points.copy()
    new_img = tfs.functional.rotate(img, -angle)
    m = matrix.Matrix(height, width)
    m.rotation(angle)

    points = m.dot_point_68(points)

    pointTensor = pointToTensor(points[0])

    new_img = new_img.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    imgTensor = pic_strong(new_img)
    return (imgTensor.type(torch.DoubleTensor), pointTensor, path[0])


# 对图像进行平移
def translationData(path, img, points, height, width, x, y):
    imgTensor = pic_strong(img)
    m = matrix.Matrix(height, width)
    m_point = matrix.Matrix(height, width)
    m.translation(x, y)
    m_point.translation_point(x, y)
    theta = torch.from_numpy(m.to_theta())
    img_torch = imgTensor
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size(), False)
    output = F.grid_sample(img_torch.unsqueeze(0), grid)

    newpoints = m_point.dot_point_68(points)
    pointTensor = pointToTensor(points)

    new_img = tensorToImage(output)
    new_img = new_img.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    imgTensor = pic_strong(new_img)
    return (imgTensor.type(torch.DoubleTensor), pointTensor, path[0], points)


# 对图像进行旋转
def rorate_op(path, img, points, height, width, angle):
    new_points = points.copy()
    new_img = tfs.functional.rotate(img, -angle)
    m = matrix.Matrix(height, width)
    m.rotation(angle)

    new_points = m.dot_point_68(new_points)
    pointTensor = pointToTensor(new_points[0])

    out_img = new_img.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    imgTensor = pic_strong(out_img)
    return (
        imgTensor.type(torch.DoubleTensor),
        pointTensor,
        path[0],
        new_img,
        new_points[1],
    )


# 对图像进行平移
def translation_op(path, img, points, height, width, x, y):
    imgTensor = pic_strong(img)
    m = matrix.Matrix(height, width)
    m_point = matrix.Matrix(height, width)
    m.translation(-x, -y)
    m_point.translation_point(x, y)
    theta = torch.from_numpy(m.to_theta())
    img_torch = imgTensor.type(torch.DoubleTensor)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size(), False)
    output = F.grid_sample(img_torch.unsqueeze(0), grid)

    newpoints = m_point.dot_point_68(points)
    pointTensor = pointToTensor(newpoints[0])

    new_img = tensorToImage(output)
    out_img = new_img.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    imgTensor = pic_strong(out_img)
    return (
        imgTensor.type(torch.DoubleTensor),
        pointTensor,
        path[0],
        new_img,
        newpoints[1],
    )


# 对图像进行缩放
def scale_op(path, img, points, height, width, x, y):
    imgTensor = pic_strong(img)
    m = matrix.Matrix(height, width)
    m_point = matrix.Matrix(height, width)
    m.scale(1 / x, 1 / y)
    m_point.scale(x, y)
    theta = torch.from_numpy(m.to_theta())
    img_torch = imgTensor.type(torch.DoubleTensor)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size(), False)
    output = F.grid_sample(img_torch.unsqueeze(0), grid)

    newpoints = m_point.dot_point_68(points)
    pointTensor = pointToTensor(newpoints[0])

    new_img = tensorToImage(output)
    out_img = new_img.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    imgTensor = pic_strong(out_img)
    return (
        imgTensor.type(torch.DoubleTensor),
        pointTensor,
        path[0],
        new_img,
        newpoints[1],
    )


# 进行数据增强
def loadTheIBUG(path, angle):
    data = []
    img = Image.open(path[0])
    width = img.size[0]
    height = img.size[1]
    points = textToPoint(path[1])

    item = translation_op(path, img, points, height, width, 0.03, 0.03)
    # showImgTensorAndPointTensor((item[0], item[1]))
    # showImgAndPoint((item[3], item[4], height, width))
    data.append((item[0], item[1], path[0]))
    item = rorate_op(path, item[3], item[4], height, width, angle)
    # showImgTensorAndPointTensor((item[0], item[1]))
    # showImgAndPoint((item[3], item[4], height, width))
    data.append((item[0], item[1], path[0]))

    item = scale_op(path, item[3], item[4], height, width, 1.1, 1.1)
    # showImgTensorAndPointTensor((item[0], item[1]))
    # showImgAndPoint((item[3], item[4], height, width))
    data.append((item[0], item[1], path[0]))

    return data


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

        data = loadTheIBUG(path, 30)
        datas.extend(data)

        # data = loadTheIBUG(path, 60)
        # datas.append(data)

        data = loadTheIBUG(path, 90)
        datas.extend(data)

        # data = loadTheIBUG(path, 120)
        # datas.append(data)

        # data = loadTheIBUG(path, 150)
        # datas.append(data)

        data = loadTheIBUG(path, 180)
        datas.extend(data)
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


def showImgTensorAndPointTensor(data):
    imgTensor = data[0]
    points = data[1]
    plt.cla()
    img = tensorToImage(imgTensor)
    plt.imshow(img)
    points = tensorToPoint(points)
    points = points.reshape(68, 2)
    for p in points:
        plt.plot(p[0] * Config.IMAGE_SIZE, p[1] * Config.IMAGE_SIZE, "r.")
    plt.show()


def showImgAndPoint(data):
    img = data[0]
    points = data[1]
    plt.cla()
    plt.imshow(img)
    points = points.reshape(68, 2)
    for p in points:
        plt.plot(p[0], p[1], "r.")
        pass
    plt.show()


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
