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


BATCH_SIZE = 64

# 加载小批次数据，即将MNIST数据集中的data分成每组batch_size的小块，shuffle指定是否随机读取
train_loader = Data.DataLoader(
    dataset=dataset.IBUGDataSet(BATCH_SIZE), batch_size=BATCH_SIZE, shuffle=True
)


# load test
testTensor = util.imageToTensor("/home/xws/Downloads/test.jpeg")
imgTensor = testTensor[0]
img = util.tensorToImage(testTensor[0])
plt.imshow(img)


# 定义网络模型亦即Net
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.full1 = nn.Linear(1600, 68 * 2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, bias=False)
        self.convlast = nn.Conv2d(64, 12, kernel_size=1, bias=False, stride=1)
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)
        # nn.Softmax

    def forward(self, X):
        # print("tensor size = ",X.size())
        X = self.conv1(X)  # 224*224
        # print("conv1 size = ",X.size())

        X = self.down_sample(X)  # 112*112
        # print("down_sample size = ",X.size())

        X = self.conv2(X)  # 110*110
        X = self.down_sample(X)  # 55*55

        X = self.conv2(X)  # 54*54
        X = self.down_sample(X)  # 27*27

        X = self.conv2(X)  # 25*25
        X = self.down_sample(X)  # 27*27

        X = self.conv2(X)  # 25*25
        X = self.down_sample(X)  # 27*27

        # X = self.convlast(X)  # 25*25
        X = X.view(-1, 1600)
        # print("view size = ",X.size())
        X = self.full1(X)

        return X


model = Model()  # 实例化全连接层
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print("device = ", device)
# device = "cpu"

if device != "cpu":
    model.to(device)


# model.to(device)
# model = model.cuda()

loss = nn.CrossEntropyLoss()  # 损失函数选择，交叉熵函数
optimizer = optim.SGD(model.parameters(), lr=0.1)
num_epochs = 10

for echo in range(num_epochs):
    train_loss = 0  # 定义训练损失
    model.train()  # 将网络转化为训练模式
    print("startTime = ", util.getTime())
    for i, (X, label) in enumerate(train_loader):  # 使用枚举函数遍历train_loader
        # X = X.view(-1,784)       #X:[64,1,28,28] -> [64,784]将X向量展平
        print("x.size = ", X.size())
        if device == "cpu":
            X = Variable(X)  # 包装tensor用于自动求梯度
            label = Variable(label)
        else:
            X = Variable(X).cuda()  # 包装tensor用于自动求梯度
            label = Variable(label).cuda()
        out = model(X)  # 正向传播
        lossvalue = loss(out, label)  # 求损失值
        optimizer.zero_grad()  # 优化器梯度归零
        lossvalue.backward()  # 反向转播，刷新梯度值
        optimizer.step()  # 优化器运行一步，注意optimizer搜集的是model的参数

        # 计算损失
        train_loss += float(lossvalue)

    print("endTime = ", util.getTime())

    print("echo:" + " " + str(echo))
    print("lose:" + " " + str(train_loss / len(train_loader)))
    eval_loss = 0
    eval_acc = 0
    model.eval()  # 模型转化为评估模式

    if device == "cpu":
        X = imgTensor  # 包装tensor用于自动求梯度
    else:
        X = imgTensor.cuda()  # 包装tensor用于自动求梯度

    testout = model(X)
    points = util.tensorToPoint(testout)
    for p in points:
        plt.plot(p[0], p[1], "r+")
    plt.savefig("/home/xws/Downloads/python/python/face/test.png")
