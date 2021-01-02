import torch
import torch.optim as optim  # 加载优化器有关包
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random

seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.set_printoptions(precision=8)


print("torch version = ", torch.__version__)

use_cuda = torch.cuda.is_available()
print("use_cuda = ", use_cuda)

print("versino = ", torch.version.cuda)


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.full = nn.Linear(1, 1, False)

    def forward(self, x):
        x = self.full(x)
        return x


model = Test()
a = torch.rand(1)
print(a)
a = torch.square(a)
print(a)
a = torch.sqrt(a)
print(a)


# 单个输入测试
def test1():
    x = torch.tensor([2.0])
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    print("b ", list(model.full.named_parameters())[0])
    y = model(x)
    print("y = ", y)
    y.backward()
    optimizer.step()
    print("b ", list(model.full.named_parameters())[0])

# 批量输入测试
def test2():
    # a = [2, 2, 2, 2, 2]
    a = [2, 2.1, 2.2, 2.3, 2.4]
    x = torch.tensor(a, dtype=torch.float32)
    x = x.view(5, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    print("b ", list(model.full.named_parameters())[0])

    y = model(x)
    print("y = ", y)
    m = torch.sum(y / 5)
    print("m = ", m)
    m.backward()
    optimizer.step()
    print("b ", list(model.full.named_parameters())[0])





# print("b ", list(model.full.named_parameters())[0])

# y = model(x)

# print("y = ", y)
# m = torch.sum(y / 5)
# print("m = ", m)




# optimizer.step()
# print("b ", list(model.full.named_parameters())[0])

# test1()
print("*******************************************************************************")
test2()