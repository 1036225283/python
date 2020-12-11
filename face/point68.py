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

torch.set_default_tensor_type(torch.DoubleTensor)


BATCH_SIZE = 32
MODEL_SAVE_PATH = "/home/xws/Downloads/python/python/face/model/point68.pt"
EPOCH = 500


# 加载小批次数据，即将MNIST数据集中的data分成每组batch_size的小块，shuffle指定是否随机读取
train_loader = Data.DataLoader(
    dataset=dataset.IBUGDataSet(BATCH_SIZE), batch_size=BATCH_SIZE, shuffle=True
)


# load test
testTensor = util.imageToTensor("/home/xws/Downloads/test.jpeg")
# testTensor = util.imageToTensor(
#     "/home/xws/Downloads/300w_cropped/01_Indoor/indoor_001.png"
# )
imgTensor = testTensor[0]

img = util.tensorToImage(testTensor[0])
plt.imshow(img)


# 定义网络模型亦即Net
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.full1 = nn.Linear(7 * 7 * 64, 68 * 2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, bias=False, padding=1)
        self.conv11 = nn.Conv2d(64, 64, kernel_size=3, bias=False, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, bias=False, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, bias=False, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, bias=False, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, bias=False, padding=1)
        self.down_sample1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_sample2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_sample3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_sample4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_sample5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(3)

        self.relu = nn.ReLU()
        self.se = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # nn.Softmax

    def forward(self, X):
        X = self.bn1(X)
        X = self.conv1(X)  # 224*224
        X = self.conv11(X)  # 224*224
        X = self.relu(X)
        X = self.down_sample1(X)  # 112*112

        X = self.conv2(X)  # 110*110
        X = self.relu(X)
        X = self.down_sample2(X)  # 55*55

        X = self.conv3(X)  # 54*54
        X = self.relu(X)
        X = self.down_sample3(X)  # 27*27

        X = self.conv4(X)  # 25*25
        X = self.relu(X)
        X = self.down_sample4(X)  # 27*27

        X = self.conv5(X)  # 25*25
        X = self.relu(X)
        X = self.down_sample5(X)  # 27*27

        X = X.view(-1, 7 * 7 * 64)
        # print("view size = ",X.size())
        X = self.full1(X)
        X = self.tanh(X)

        return X


model = Model()  # 实例化全连接层

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

optimizer = optim.SGD(model.parameters(), lr=0.2 * math.e)
# optimizer = optim.SGD(model.parameters(), lr=1 * math.e, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.000005 * math.e)

epoch = 0
for epoch in range(EPOCH):
    train_loss = 0  # 定义训练损失
    model.train()  # 将网络转化为训练模式
    # print("startTime = ", util.getTime())
    for i, (X, label) in enumerate(train_loader):  # 使用枚举函数遍历train_loader
        if device != "cpu":
            X = X.cuda()  # 包装tensor用于自动求梯度
            label = label.cuda()

        for x in range(1):
            optimizer.zero_grad()  # 优化器梯度归零
            out = model(X)  # 正向传播
            lossvalue = torch.nn.functional.smooth_l1_loss(out, label)

            lossvalue.backward()  # 反向转播，刷新梯度值
            optimizer.step()  # 优化器运行一步，注意optimizer搜集的是model的参数

            # 计算损失
            train_loss += float(lossvalue)

    # print("endTime = ", util.getTime())

    # print("epoch:" + " " + str(epoch))
    if epoch % 2 == 0:
        print("epoch:" + " " + str(epoch))

        model.eval()  # 模型转化为评估模式

        if device == "cpu":
            X = imgTensor  # 包装tensor用于自动求梯度
        else:
            X = imgTensor.cuda()  # 包装tensor用于自动求梯度

        X = X.view(1, 3, 224, 224)
        testout = model(X)
        points = util.tensorToPoint(testout.cpu().detach())
        for p in points:
            plt.plot(p[0] * 224, p[1] * 224, "r+")

        plt.savefig(
            "/home/xws/Downloads/python/python/face/img/test" + str(epoch) + ".png"
        )
        plt.cla()
        plt.imshow(img)

        state = {
            "net": model.state_dict(),
            "epoch": epoch,
        }
        torch.save(state, MODEL_SAVE_PATH)
