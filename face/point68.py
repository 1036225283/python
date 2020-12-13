# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F  # 加载nn中的功能函数
import torch.optim as optim  # 加载优化器有关包
import torch.utils.data as Data
from torchvision import datasets, transforms  # 加载计算机视觉有关包
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import util
import dataset
import os
import math
import model as models
import Config

torch.set_default_tensor_type(torch.DoubleTensor)


MODEL_SAVE_PATH = "/home/xws/Downloads/python/python/face/model/point68.pt"


# 加载小批次数据，即将MNIST数据集中的data分成每组batch_size的小块，shuffle指定是否随机读取
train_loader = Data.DataLoader(
    dataset=dataset.IBUGDataSet(Config.BATCH_SIZE),
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
)


# load test
# testTensor = util.imageToTensor("/home/xws/Downloads/test.jpeg")
testTensor = util.imageToTensor(
    "/home/xws/Downloads/300w_cropped/01_Indoor/indoor_001.png"
)
imgTensor = testTensor[0]

img = util.tensorToImage(testTensor[0])
plt.imshow(img)


def show(plt, X, L):
    plt.cla()
    img = util.tensorToImage(X)
    plt.imshow(img)
    points = util.tensorToPoint(L.cpu().detach())
    for p in points:
        plt.plot(p[0] * Config.IMAGE_SIZE, p[1] * Config.IMAGE_SIZE, "r.")
    plt.savefig("/home/xws/Downloads/python/python/face/img/test0.png")


def show2(plt, X, O, N):
    plt.cla()
    img = util.tensorToImage(X)
    plt.imshow(img)
    points = util.tensorToPoint(O.cpu().detach())
    for p in points:
        plt.plot(p[0] * Config.IMAGE_SIZE, p[1] * Config.IMAGE_SIZE, "r_")
    points = util.tensorToPoint(N.cpu().detach())
    for p in points:
        plt.plot(p[0] * Config.IMAGE_SIZE, p[1] * Config.IMAGE_SIZE, "g|")
    plt.savefig("/home/xws/Downloads/python/python/face/img/testt0.png")


model = models.Point68()  # 实例化全连接层

if os.path.isfile(MODEL_SAVE_PATH):
    print("loading ...")
    state = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(state["net"])
    start_epoch = state["epoch"]
    print("loading over")


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print("device = ", device)
# device = "cpu"

if device != "cpu":
    model.to(device)


# model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.0001 * math.e)
ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

# optimizer = optim.SGD(model.parameters(), lr=1 * math.e, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.000005 * math.e)
loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
epoch = 0
for epoch in range(Config.EPOCH):
    train_loss = 0  # 定义训练损失
    model.train()  # 将网络转化为训练模式
    # print("startTime = ", util.getTime())
    for i, (X, label) in enumerate(train_loader):  # 使用枚举函数遍历train_loader
        if device != "cpu":
            # show(plt, X, label)
            X = X.cuda()  # 包装tensor用于自动求梯度
            label = label.cuda()

        for x in range(1):
            optimizer.zero_grad()  # 优化器梯度归零
            out = model(X)  # 正向传播

            lossvalue = torch.nn.functional.smooth_l1_loss(label, out)
            # lossvalue = loss_fn(out, label)
            print("lossvalue = ", lossvalue)

            lossvalue.backward()  # 反向转播，刷新梯度值
            optimizer.step()  # 优化器运行一步，注意optimizer搜集的是model的参数

            # 计算损失
            train_loss += float(lossvalue)
            if epoch % 20 == 0:
                show2(plt, X, label, out)
                state = {
                    "net": model.state_dict(),
                    "epoch": epoch,
                }
                torch.save(state, MODEL_SAVE_PATH)

    # print("endTime = ", util.getTime())

    # print("epoch:" + " " + str(epoch))
    # if epoch % 10 == 0:
    #     print("epoch:" + " " + str(epoch))

    #     model.eval()  # 模型转化为评估模式

    #     if device == "cpu":
    #         X = imgTensor  # 包装tensor用于自动求梯度
    #     else:
    #         X = imgTensor.cuda()  # 包装tensor用于自动求梯度

    #     X = X.view(1, 3, 224, 224)
    #     testout = model(X)