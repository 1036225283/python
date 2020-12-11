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


class BaseBlock(nn.Module):
    def __init__(
        self, inplanes, planes, kernel_size=3, stride=1, padding=0, active=True
    ):
        super(BaseBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.active = active
        if active:
            self.mish = Mish()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        if self.active:
            x = self.mish(x)
        return x


def get_m_index(data):
    indexs = []
    for i in range(len(data)):
        if data[i] == "M":
            indexs.append(i)
    return indexs


class MPred(nn.Module):
    def __init__(self):
        super(MPred, self).__init__()

    def forward(self, x1, x2, x3, img_size=416, anchor=[16, 48, 144]):
        out1_cxy = torch.sigmoid(x1[:, 0:3, :])
        out1_wh = anchor[0] * torch.exp(x1[:, 3:5, :]) / img_size
        out1 = torch.cat([out1_cxy, out1_wh], dim=1)

        out2_cxy = torch.sigmoid(x2[:, 0:3, :])
        out2_wh = anchor[1] * torch.exp(x2[:, 3:5, :]) / img_size
        out2 = torch.cat([out2_cxy, out2_wh], dim=1)

        out3_cxy = torch.sigmoid(x3[:, 0:3, :])
        out3_wh = anchor[2] * torch.exp(x3[:, 3:5, :]) / img_size
        out3 = torch.cat([out3_cxy, out3_wh], dim=1)
        return out1, out2, out3


class MSSD(nn.Module):
    def __init__(self):
        super(MSSD, self).__init__()
        self.b1 = BaseBlock(3, 64, 5, 1, 2)
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bx1 = Bottleneck(64, 128, 1)
        self.bx1_1 = Bottleneck(128, 256, 1)
        self.bx2 = Bottleneck(256, 128, 1)
        self.bx2_1 = Bottleneck(128, 128, 1)
        self.bx3 = Bottleneck(128, 128, 1)
        self.bx4 = Bottleneck(128, 128, 1)
        self.bx5 = Bottleneck(128, 128, 1)

        self.o1 = BaseBlock(128, 5, 3, 1, 1, active=False)
        self.o2 = BaseBlock(128, 5, 3, 1, 1, active=False)
        self.o3 = BaseBlock(128, 5, 3, 1, 1, active=False)

        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear")

        self.pred = MPred()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        x3 = self.down_sample(x2)  # 13
        x3 = self.bx5(x3)

        u2 = self.up_sample(x3)
        u1 = self.up_sample(x2)

        o1 = self.o1(x3)  # 13
        o2 = self.o2(x2 + u2)  # 26
        o3 = self.o3(x1 + u1)  # 52

        o1 = o1.view(o1.shape[0], o1.shape[1], o1.shape[2] * o1.shape[3])
        o2 = o2.view(o2.shape[0], o2.shape[1], o2.shape[2] * o2.shape[3])
        o3 = o3.view(o3.shape[0], o3.shape[1], o3.shape[2] * o3.shape[3])
        out = torch.cat(self.pred(o3, o2, o1), dim=2)
        out = out.permute(0, 2, 1)
        return out


class Point68(nn.Module):
    def __init__(self):
        super(Point68, self).__init__()
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
        x3 = self.down_sample(x2)  # 13
        x3 = self.bx5(x3)

        X = x3.view(-1, 7 * 7 * 64)
        # print("view size = ",X.size())
        X = self.full1(X)
        X = self.se(X)

        return X


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


if __name__ == "__main__":
    test_model()