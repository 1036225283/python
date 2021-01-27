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
import model as models
import Config

torch.set_default_tensor_type(torch.DoubleTensor)


def show(model, data, path):
    imgTensor = data[0]
    X = imgTensor.view(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    testout = model(X)
    points = util.tensorToPoint(testout.cpu().detach())
    for p in points:
        plt.plot(p[0] * Config.IMAGE_SIZE, p[1] * Config.IMAGE_SIZE, "r+")
    img = util.tensorToImage(imgTensor)
    plt.imshow(img)
    plt.savefig(path[0].replace("300w_cropped/01_Indoor", "test"))
    # plt.show()
    plt.cla()


# load all image
paths = util.getFiles()


# 加载所有图片,并进行测试,将测试结果保存起来
model = Config.model()  # 实例化全连接层
model.eval()  # 模型转化为评估模式
if os.path.exists(Config.MODEL_SAVE_PATH):
    print("loading ...")
    state = torch.load(Config.MODEL_SAVE_PATH)
    model.load_state_dict(state["net"])
    start_epoch = state["epoch"]
    print("loading over")

for path in paths:
    print("path = ", path[0].replace("300w_cropped/01_Indoor", "test"))
    data = util.loadOneIBUG(path)
    if data[0].size()[0] != 3:
        continue
    show(model, data, path)
