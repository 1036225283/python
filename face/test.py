import matplotlib.pyplot as plt
from torchvision import transforms as tfs
import torch.nn.functional as F


from PIL import Image, ImageDraw
import numpy as np
import torch

import util
import matrix
import Config


def test0():
    plt.ion()

    # #读取图像到数组中
    # im = array(Image.open('/home/xws/Downloads/boll.jpeg'))
    f = plt.figure()
    # #绘制图像
    t = util.imageToTensor("/home/xws/Downloads/300w_cropped/01_Indoor/indoor_001.png")
    img = util.tensorToImage(t[0])
    # img = plt.imread("/home/xws/Downloads/300w_cropped/01_Indoor/indoor_300.png")
    plt.imshow(img)
    width = t[1]
    height = t[2]

    # text = util.readText("/home/xws/Downloads/300w_cropped/01_Indoor/indoor_300.pts")
    points = util.textToPoint(
        "/home/xws/Downloads/300w_cropped/01_Indoor/indoor_001.pts"
    )
    # print("points = ", points)
    points = util.pointToTensor(points)
    # print("pointToTensor = ", points)
    points = util.tensorToPoint(points)
    # print("tensorToPoint = ", points)

    # # 使用红色星状物标记绘制点
    for p in points:
        plt.plot(p[0], p[1], "r+")

    # #绘制前两个点的线
    # plt.plot(x[:2], y[:2])

    # #添加标题，显示绘制的图像
    plt.title('Plotting:"pic1.png"')
    plt.ioff()
    plt.savefig("/home/xws/Downloads/python/python/face/test.png")
    # plt.show()

    print("this is end")


transform = tfs.Compose([tfs.RandomRotation(degrees=30)])

transform1 = tfs.Compose(
    [
        tfs.RandomRotation(
            30,
            resample=Image.BICUBIC,
            expand=False,
        )
    ]
)

tensor_t = tfs.Compose(
    [
        tfs.RandomRotation(
            30,
            expand=False,
        )
    ]
)

pic_strong = tfs.Compose([tfs.ColorJitter(0.5, 0.3, 0.3, 0.1), tfs.ToTensor()])

# 测试图片旋转
def test1():

    path = "/home/xws/Downloads/300w_cropped/01_Indoor/indoor_300.png"
    img = Image.open(path)
    new_img = transform(img)
    # imgTensor = pic_strong(img)
    # img = util.tensorToImage(imgTensor)
    plt.imshow(new_img)
    plt.show()


# 测试tensor的旋转
def test2():
    img = torch.tensor([1, 2, 3, 4])
    img = img.view(1, 2, 2)
    print(img)
    img = tensor_t(img)
    print(img)


# 测试旋转图片和标记数据
def test3():
    m = matrix.Matrix()

    # #读取图像到数组中
    # im = array(Image.open('/home/xws/Downloads/boll.jpeg'))
    t = util.imageToTensor("/home/xws/Downloads/300w_cropped/01_Indoor/indoor_001.png")
    img = util.tensorToImage(t[0])
    # img = plt.imread("/home/xws/Downloads/300w_cropped/01_Indoor/indoor_300.png")

    path = "/home/xws/Downloads/300w_cropped/01_Indoor/indoor_001.png"
    img = Image.open(path)
    new_img = transform(img)
    new_img = tfs.functional.rotate(img, -30)
    plt.imshow(new_img)

    width = t[1]
    height = t[2]

    # text = util.readText("/home/xws/Downloads/300w_cropped/01_Indoor/indoor_300.pts")
    points = util.textToPoint(
        "/home/xws/Downloads/300w_cropped/01_Indoor/indoor_001.pts"
    )

    # 将point转换成百分比
    # for p in points:
    #     p[0] = p[0] / width
    #     p[1] = p[1] / height

    # 将point按照照片实际比例进行放大
    # for p in points:
    #     p[0] = p[0] * Config.IMAGE_SIZE
    #     p[1] = p[1] * Config.IMAGE_SIZE

    # 将point扩充一下
    newpoints = np.arange(68 * 3, dtype=float).reshape(68, 3)
    for i, p in enumerate(points):
        print("ww", i, p)
        newpoints[i][0] = p[0]
        newpoints[i][1] = p[1]
        newpoints[i][2] = 1

    print(newpoints)

    m.rotation(30)
    pp = m.dot(newpoints, height, width)
    for i, p in enumerate(pp):
        print("ww", i, p)
        points[i][0] = p[0]
        points[i][1] = p[1]

    # points = util.pointToTensor(points)
    # points = util.tensorToPoint(points)

    # # 使用红色星状物标记绘制点
    i = 1
    for p in points:
        plt.plot(p[0], p[1], "r_")
        # plt.text(p[0], p[1], i)
        i = i + 1

    # plt.text(points[8][0], points[8][1], 0)
    # plt.text(points[17][0], points[17][1], 0)
    # plt.text(points[26][0], points[26][1], 0)
    # plt.text(points[27][0], points[27][1], 0)
    # plt.text(points[33][0], points[33][1], 0)
    # plt.text(points[48][0], points[48][1], 0)
    # plt.text(points[54][0], points[54][1], 0)

    # plt.plot(points[8][0], points[8][1], "g|")
    plt.plot(points[17][0], points[17][1], "g|")
    plt.plot(points[26][0], points[26][1], "g|")
    # plt.plot(points[27][0], points[27][1], "g|")
    plt.plot(points[33][0], points[33][1], "g|")
    plt.plot(points[48][0], points[48][1], "g|")
    plt.plot(points[54][0], points[54][1], "g|")

    # #添加标题，显示绘制的图像
    plt.show()

    print("this is end")


# test1()
# test2()
test3()
# load ibug a img and show the point