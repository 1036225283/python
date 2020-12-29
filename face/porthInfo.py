import torch

print("torch version = ", torch.__version__)

use_cuda = torch.cuda.is_available()
print("use_cuda = ", use_cuda)

print("versino = ", torch.version.cuda)


a = torch.rand(1)
print(a)
a = torch.square(a)
print(a)
a = torch.sqrt(a)
print(a)

# 测试梯度和反向传播

x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([2.0], requires_grad=True)

y = x ** 2
z = w * y + 1.0

z.backward()

print(x.grad)
print(w.grad)

