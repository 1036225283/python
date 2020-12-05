from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from torch.utils.data import Dataset
import torch
import random
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import util


class IBUGDetectSet(Dataset):
    def __init__(self, img_size, is_train=True, is_random=True):
        paths = util.getFiles()
        self.datas = util.loadIBUG(paths)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        return self.datas[item]
