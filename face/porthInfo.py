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