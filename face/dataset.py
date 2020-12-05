from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from torch.utils.data import Dataset
import torch
import random
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt



def getOne(file):
    len = int(file.readline())
    bboxes = []
    for i in range(len):
        tmp = file.readline()
        datas = tmp.split(" ")
        bbox = [float(datas[0]), float(datas[1]), float(datas[2]), float(datas[3])]
        bboxes.append(bbox)
    return bboxes


class IBUGDetectSet(Dataset):
    def __init__(self, img_size, is_train=True, is_random=True):
        self.is_train = is_train
        self.is_random = is_random
        if is_train:
            self.PIC_PATH = TRAIN_IMG_PATH
        else:
            self.PIC_PATH = VAL_IMG_PATH
        self.img_size = img_size
        self.datas = get_all_files_and_bboxes(is_train)
        self.pic_strong = tfs.Compose(
            [tfs.ColorJitter(0.5, 0.3, 0.3, 0.1), tfs.ToTensor()]
        )

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        # img
        img_path = self.PIC_PATH + self.datas[item]["img"]
        img_path = img_path.replace("\n", "")
        img_origin = Image.open(img_path)
        img, scaled_bboxes = pic_resize2square(
            img_origin, self.img_size, self.datas[item]["bboxes"], self.is_random
        )
        img_tensor = self.pic_strong(img)

        # label
        feature_map = [0, 0, 0]
        feature_map[0] = self.img_size / (2 ** 3)
        feature_map[1] = self.img_size / (2 ** 4)
        feature_map[2] = self.img_size / (2 ** 5)
        label_tensor = bbox2tensor(scaled_bboxes, self.img_size, feature_map)
        if self.is_train:
            return img_tensor, label_tensor
        else:
            return img_tensor, label_tensor, (img_path, img_origin.size)
