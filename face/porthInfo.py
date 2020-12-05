import torch

print("torch version = ", torch.__version__)

use_cuda = torch.cuda.is_available()
print("use_cuda = ", use_cuda)

print("versino = ", torch.version.cuda)
