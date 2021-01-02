from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from torch.utils.data import Dataset
import torch
import random
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import util


class IBUGDataSet(Dataset):
    def __init__(self, batchSize, is_train=True, is_random=True):
        self.batchSize = batchSize
        paths = util.getFiles()
        paths_noface = util.getFiles("/home/xws/Downloads/300w_cropped/noface")
        paths = paths + paths_noface

        self.datas = util.loadIBUG(paths)
        print("datas.length = ", len(self.datas))

    def __len__(self):
        if len(self.datas) < self.batchSize:
            return self.batchSize
        else:
            return len(self.datas)

    def __getitem__(self, item):
        # print("getitem ", item, " ", item % (len(self.datas) - 1))
        if item < len(self.datas):
            return self.datas[item]
        else:
            return self.datas[item % (len(self.datas) - 1)]
