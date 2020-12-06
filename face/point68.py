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

# 加载torchvision包内内置的MNIST数据集 这里涉及到transform:将图片转化成torchtensor
test_dataset = datasets.MNIST(
    root="~/data/", train=False, transform=transforms.ToTensor()
)

# 加载小批次数据，即将MNIST数据集中的data分成每组batch_size的小块，shuffle指定是否随机读取
train_loader = Data.DataLoader(
    dataset=dataset.IBUGDataSet, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = Data.DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

# load test
testTensor = util.imageToTensor("/home/xws/Downloads/test.jpeg")
imgTensor = testTensor[0]
img = util.tensorToImage(testTensor[0])

# 定义网络模型亦即Net 这里定义一个简单的全连接层784->10
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.full1 = nn.Linear(3872, 10)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, bias=False)
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)
        # nn.Softmax

    def forward(self, X):
        # print("tensor size = ",X.size())
        X = self.conv1(X)  # 26*26
        # print("conv1 size = ",X.size())

        X = self.down_sample(X)  # 13*13
        # print("down_sample size = ",X.size())

        X = self.conv2(X)  # 11*11
        # print("conv2 size = ",X.size())

        X = X.view(-1, 3872)
        # print("view size = ",X.size())
        X = self.full1(X)
        # print("full1 size = ",X.size())

        return F.relu(X)


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

# 以下四个列表是为了可视化（暂未实现）
losses = []
acces = []
eval_losses = []
eval_acces = []

for echo in range(num_epochs):
    train_loss = 0  # 定义训练损失
    train_acc = 0  # 定义训练准确度
    model.train()  # 将网络转化为训练模式
    print("startTime = ", util.getTime())
    for i, (X, label) in enumerate(train_loader):  # 使用枚举函数遍历train_loader
        # X = X.view(-1,784)       #X:[64,1,28,28] -> [64,784]将X向量展平
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
        # 计算精确度
        _, pred = out.max(1)
        num_correct = (pred == label).sum()
        acc = int(num_correct) / X.shape[0]
        train_acc += acc

    print("endTime = ", util.getTime())

    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    print("echo:" + " " + str(echo))
    print("lose:" + " " + str(train_loss / len(train_loader)))
    print("accuracy:" + " " + str(train_acc / len(train_loader)))
    eval_loss = 0
    eval_acc = 0
    model.eval()  # 模型转化为评估模式

    if device == "cpu":
        X = Variable(X)  # 包装tensor用于自动求梯度
        label = Variable(label)
    else:
        X = Variable(X).cuda()  # 包装tensor用于自动求梯度
        label = Variable(label).cuda()

    testout = model(X)
    testloss = loss(testout, label)
