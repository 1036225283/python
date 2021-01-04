import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class CnnBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1):
        super(CnnBlock, self).__init__()
        self.conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(planes)
        self.mish = Mish()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.mish(out)
        out = F.dropout(out, 0.1)
        return out


class CnnAlign(nn.Module):
    def __init__(self):
        super(CnnAlign, self).__init__()
        self.cnn1 = CnnBlock(3, 16)
        self.cnn2 = CnnBlock(16, 32)
        self.cnn3 = CnnBlock(32, 64)
        self.cnn4 = CnnBlock(64, 128)
        self.cnn5 = CnnBlock(128, 64)
        self.max_pool = nn.MaxPool2d(2)
        self.drop_3 = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 2 * 194)

    def forward(self, x):
        x1 = self.cnn1(x)  # 224
        x2 = self.max_pool(x1)  # 112
        x3 = self.cnn2(x2)  # 112
        x4 = self.max_pool(x3)  # 56
        x5 = self.cnn3(x4)  # 56
        x6 = self.max_pool(x5)  # 28
        x7 = self.cnn4(x6)  # 28
        x8 = self.max_pool(x7)  # 14
        x9 = self.cnn5(x8)  # 14
        x10 = self.max_pool(x9)  # 7
        x11 = self.fc1(x10.view(-1, 7 * 7 * 64))
        x12 = self.drop_3(x11)
        x13 = self.fc2(x12)
        return x13


class CnnAlignHard(nn.Module):
    def __init__(self):
        super(CnnAlignHard, self).__init__()
        self.cnn1 = CnnBlock(3, 16)
        self.cnn2 = CnnBlock(16, 32)
        self.cnn3 = CnnBlock(32, 64)
        self.cnn4 = CnnBlock(64, 128)
        self.cnn5 = CnnBlock(128, 256)
        self.cnn6 = CnnBlock(256, 512, padding=0)
        self.cnn7 = nn.Conv2d(512, 388, kernel_size=1, padding=0)
        self.max_pool = nn.MaxPool2d(2)
        self.mean_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x1 = self.cnn1(x)  # 224
        x2 = self.max_pool(x1)  # 112
        x3 = self.cnn2(x2)  # 112
        x4 = self.max_pool(x3)  # 56
        x5 = self.cnn3(x4)  # 56
        x6 = self.max_pool(x5)  # 28
        x7 = self.cnn4(x6)  # 28
        x8 = self.max_pool(x7)  # 14
        x9 = self.cnn5(x8)  # 14
        x10 = self.max_pool(x9)  # 7
        x11 = self.cnn6(x10)  # 5
        x12 = self.mean_pool(x11)  # 1
        x13 = self.cnn7(x12)  # 5
        return torch.flatten(x13, 1)


def test_model():
    net = CnnAlignHard()
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(y.size())


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.mish = Mish()
        self.stride = stride

        self.conv4 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mish(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.mish(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # down sample
        residual = self.conv4(residual)
        residual = self.bn4(residual)

        out += residual
        out = self.mish(out)

        return out


# 3层
class Bottleneck3(nn.Module):
    def __init__(self, input, output, stride=1, use_res_connect=True, expand_ratio=2):
        super(Bottleneck3, self).__init__()
        the_stride = stride
        self.conv1 = nn.Conv2d(
            input, input * expand_ratio, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(input * expand_ratio)
        self.conv2 = nn.Conv2d(
            input * expand_ratio,
            input * expand_ratio,
            kernel_size=3,
            stride=the_stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(input * expand_ratio)
        self.conv3 = nn.Conv2d(input * expand_ratio, output, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output)
        self.mish = Mish()
        self.use_res_connect = use_res_connect

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mish(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.mish(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_res_connect:
            return x + out
        else:
            return out


class BaseBlock(nn.Module):
    def __init__(self, input, output, kernel=3, stride=1, padding=0):
        super(BaseBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            input,
            output,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(output)
        self.mish = Mish()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.mish(x)
        return x


class Point68_y(nn.Module):
    path = "/home/xws/Downloads/python/python/face/model/point68_y.pt"

    def __init__(self):
        super(Point68_y, self).__init__()
        self.b1 = BaseBlock(3, 68, 5, 1, 2)
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bx1 = Bottleneck(68, 68, 1)
        self.bx1_1 = Bottleneck(68, 68, 1)
        self.bx2 = Bottleneck(68, 128, 1)
        self.bx2_1 = Bottleneck(128, 128, 1)
        self.bx3 = Bottleneck(128, 128, 1)
        self.bx4 = Bottleneck(128, 128, 1)
        self.bx5 = Bottleneck(128, 64, 1)
        self.full1 = nn.Linear(7 * 7 * 64, 68 * 2)
        self.relu = nn.ReLU()
        self.se = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.conv_last = nn.Conv2d(64, 68 * 2, kernel_size=7, bias=False, padding=0)

    def forward(self, x):
        # vgg to get feature
        x = self.b1(x)  # 416
        x = self.down_sample(x)  # 208
        x = self.bx1(x)  # 208
        x = self.bx1_1(x)  # 208
        x = self.down_sample(x)  # 104
        x = self.bx2(x)  # 104
        x = self.bx2_1(x)  # 104
        x1 = self.down_sample(x)  # 52
        x1 = self.bx3(x1)  # 52
        x2 = self.down_sample(x1)  # 26
        x2 = self.bx4(x2)  # 26
        x3 = self.bx5(x2)

        # X = x3.view(-1, 7 * 7 * 64)
        # X = self.full1(X)
        # X = self.se(X)
        X = self.conv_last(x3)
        X = self.tanh(X)

        return X


class Point68_y1(nn.Module):
    path = "/home/xws/Downloads/python/python/face/model/point68_y1.pt"

    def __init__(self):
        super(Point68_y1, self).__init__()
        self.b1 = BaseBlock(3, 68, 5, 1, 2)
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bx1 = Bottleneck(68, 68, 1)
        self.bx1_1 = Bottleneck(68, 68, 1)
        self.bx2 = Bottleneck(68, 128, 1)
        self.bx2_1 = Bottleneck(128, 128, 1)
        self.bx3 = Bottleneck(128, 128, 1)
        self.bx4 = Bottleneck(128, 128, 1)
        self.bx5 = Bottleneck(128, 64, 1)
        self.full1 = nn.Linear(7 * 7 * 64, 68 * 2)
        self.relu = nn.ReLU()
        self.se = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.conv_last = nn.Conv2d(128, 68 * 2, kernel_size=7, bias=False, padding=0)

    def forward(self, x):
        # vgg to get feature
        x = self.b1(x)  # 416
        x = self.down_sample(x)  # 208
        x = self.bx1(x)  # 208
        x = self.bx1_1(x)  # 208
        x = self.down_sample(x)  # 104
        x = self.bx2(x)  # 104
        x = self.bx2_1(x)  # 104
        x1 = self.down_sample(x)  # 52
        x1 = self.bx3(x1)  # 52
        x2 = self.down_sample(x1)  # 26

        X = self.conv_last(x2)
        X = self.tanh(X)

        return X


class Point68PLDF(nn.Module):
    path = "/home/xws/Downloads/python/python/face/model/point68_pldf.pt"

    def __init__(self):
        super(Point68PLDF, self).__init__()
        self.conv_first = BaseBlock(3, 64, 3, 2, 1)
        self.conv_second = BaseBlock(64, 64, 3, 2, 1)
        self.avg_pool_14 = nn.AvgPool2d(14)
        self.avg_pool_7 = nn.AvgPool2d(7)
        self.conv_56_1 = Bottleneck3(64, 64, 1, True)
        self.conv_56_2 = Bottleneck3(64, 64, 2, False)
        self.conv_28_1 = Bottleneck3(64, 64, 1, True)
        self.conv_28_2 = Bottleneck3(64, 64, 1, True)
        self.conv_28_3 = Bottleneck3(64, 128, 2, False)
        self.conv_14_1 = Bottleneck3(128, 128, 1, True, 4)
        self.conv_14_2 = Bottleneck3(128, 128, 1, True, 4)
        self.conv_14_3 = Bottleneck3(128, 128, 1, True, 4)
        self.conv_14_4 = Bottleneck3(128, 128, 1, True, 4)
        self.conv_14_5 = Bottleneck3(128, 16, 1, False, 4)
        self.conv_o2 = BaseBlock(16, 32, 3, 2, 1)
        self.full1 = nn.Linear(184, 68 * 2)
        self.relu = nn.ReLU()
        self.se = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.conv_last = nn.Conv2d(32, 68 * 2, kernel_size=7, bias=False, padding=0)

    def forward(self, x):
        # vgg to get feature
        x = self.conv_first(x)  # 112
        x = self.conv_second(x)  # 56
        x = self.conv_56_1(x)
        x = self.conv_56_2(x)
        x = self.conv_28_1(x)
        x = self.conv_28_2(x)
        x = self.conv_28_3(x)
        x = self.conv_14_1(x)
        x = self.conv_14_2(x)
        x = self.conv_14_3(x)
        x = self.conv_14_4(x)
        x = self.conv_14_5(x)
        o1 = self.avg_pool_14(x)
        o1 = o1.view(o1.size(0), -1)
        x = self.conv_o2(x)
        o2 = self.avg_pool_7(x)  # 7
        o2 = o2.view(o2.size(0), -1)

        # X = x3.view(-1, 7 * 7 * 64)
        # X = self.full1(X)
        # X = self.se(X)
        o3 = self.conv_last(x)
        o3 = o3.view(o3.size(0), -1)

        o = torch.cat([o1, o2, o3], 1)
        o = self.full1(o)
        o = self.se(o)

        return o


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


class Point68_max(nn.Module):
    path = "/home/xws/Downloads/python/python/face/model/point68_max.pt"

    def __init__(self):
        super(Point68_max, self).__init__()
        self.conv_first = BaseBlock(3, 68, 3, 1, 1)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_112_1 = Bottleneck3(68, 68, 1, True)
        self.conv_112_2 = Bottleneck3(68, 68, 1, True)
        self.conv_112_3 = Bottleneck3(68, 68, 1, True)
        self.conv_112_4 = Bottleneck3(68, 68, 1, False)
        self.conv_56_1 = Bottleneck3(68, 68, 1, True)
        self.conv_56_2 = Bottleneck3(68, 68, 1, True)
        self.conv_56_3 = Bottleneck3(68, 68, 1, True)
        self.conv_28 = Bottleneck3(68, 68, 1, True)
        self.conv_14 = Bottleneck3(68, 64, 1, False)
        self.bx3 = Bottleneck3(68, 68, 1, True)
        self.bx4 = Bottleneck3(68, 68, 1, True)
        self.bx5 = Bottleneck3(68, 64, 1, False)
        self.full1 = nn.Linear(7 * 7 * 64, 68 * 2)
        self.relu = nn.ReLU()
        self.se = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.conv_last = nn.Conv2d(64, 68 * 2, kernel_size=7, bias=False, padding=0)

    def forward(self, x):
        # vgg to get feature
        x = self.conv_first(x)  # 224
        x = self.max_pool_1(x)  # 112
        x = self.conv_112_1(x)  # 112
        x = self.conv_112_2(x)  # 112
        x = self.conv_112_3(x)  # 112
        x = self.conv_112_4(x)  # 112
        x = self.max_pool_2(x)  # 56
        x = self.conv_56_1(x)  # 56
        x = self.conv_56_2(x)  # 56
        x = self.conv_56_3(x)  # 56
        x = self.max_pool_3(x)  # 28
        x = self.conv_28(x)  # 28
        x = self.avg_pool_1(x)  # 14
        x = self.conv_14(x)  # 14
        x = self.avg_pool_2(x)  # 7

        # X = x3.view(-1, 7 * 7 * 64)
        # X = self.full1(X)
        # X = self.se(X)
        X = self.conv_last(x)
        X = self.se(X)

        return X


# 残差无多尺寸
class Point68_residual(nn.Module):
    path = "/home/xws/Downloads/python/python/face/model/point68_residual.pt"

    def __init__(self):
        super(Point68_residual, self).__init__()
        self.conv_first = BaseBlock(3, 64, 3, 2, 1)
        self.conv_second = BaseBlock(64, 64, 3, 2, 1)
        self.conv_56_1 = BaseBlock(64, 64, padding=1)
        self.conv_56_2 = BaseBlock(64, 64, padding=1)
        self.conv_56_3 = BaseBlock(64, 64, padding=1)
        self.conv_56_4 = BaseBlock(64, 64, padding=1)
        self.conv_56_5 = BaseBlock(64, 64, padding=1)
        self.conv_56_6 = BaseBlock(64, 64, padding=1)
        self.conv_28_1 = BaseBlock(64, 64, padding=1)

        self.max_pool_56 = nn.MaxPool2d(2, 2)
        self.max_pool_28 = nn.MaxPool2d(2, 2)
        self.full1 = nn.Linear(184, 68 * 2)
        self.relu = nn.ReLU()
        self.se = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.conv_last = nn.Conv2d(64, 68 * 2, kernel_size=14, bias=False, padding=0)

    def forward(self, x):
        # vgg to get feature
        x = self.conv_first(x)  # 112
        x = self.conv_second(x)  # 56
        x1 = self.conv_56_1(x)
        x2 = self.conv_56_2(x1)
        x3 = self.conv_56_3(x2)
        x4 = self.conv_56_4(x3)
        # x5 = self.conv_56_5(x4 + x3)
        # x6 = self.conv_56_6(x5 + x4)
        x = self.max_pool_56(x4)
        x = self.conv_28_1(x)
        x = self.max_pool_28(x)
        o = self.conv_last(x)
        o = self.relu(o)

        return o


# 基础层
# 输出层1
# 输出层2
if __name__ == "__main__":
    test_model()