import torch

print("torch version = ", torch.__version__)

use_cuda = torch.cuda.is_available()
print("use_cuda = ", use_cuda)

print("versino = ", torch.version.cuda)

for epoch in range(100):

    if epoch % 20 == 0:
        print("test:" + " " + str(epoch), epoch % 20)

    # print(epoch, epoch % 20)
