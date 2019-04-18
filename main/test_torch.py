from __future__ import print_function
import torch

# 构造一个5*3的矩阵
x = torch.empty(5, 3)
print(x)

# 构造一个随机初始化的5*3的矩阵
x = torch.rand(5, 3)
print(x)

# 构造一个初始化为0的5*3的矩阵 元素类型为long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 直接用数据构造一个张量
x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
print(x)

# 用指定的数据类型覆盖x
x = torch.randn_like(x, dtype=torch.float)  # override dtype!
print(x)  # result has the same size

# 打印出张量的大小
print(x.size())

y = torch.rand(5, 3)
print(x + y)
