import matplotlib.pyplot as plt
from torchvision import transforms as tfs


from PIL import Image, ImageDraw
import numpy as np
import torch

import util


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


transform = tfs.Compose(
    [
        tfs.RandomAffine(
            degrees=30, translate=(0, 0), scale=(0.9, 1), shear=(2, 3), fillcolor=66
        )
    ]
)

transform = tfs.Compose(
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


# test1()
test2()
# load ibug a img and show the point