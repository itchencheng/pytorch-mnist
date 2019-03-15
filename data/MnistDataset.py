#coding:utf-8

import os
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset

class MnistDataset(Dataset):

    def __init__(self, dir_path, is_labeled, is_vali, transform_func):
        super(MnistDataset, self).__init__()
        self.dir_path = dir_path
        label_file = os.path.join(dir_path, 'info.txt')
        self.img_list = []
        self.is_labeled = is_labeled
        self.is_vali = is_vali
        self.transform_func = transform_func

        fi = open(label_file)
        lines = fi.readlines()
        for line in lines:
            line = line.strip()
            if (self.is_labeled):
                if (',' in line):
                    line = line.split(',')
                    tag = int(line[0])
                    img_name = os.path.join(dir_path, line[1])
                    self.img_list.append((tag, img_name))
            else:
                img_name = os.path.join(dir_path, line)
                self.img_list.append(img_name)
        
        fi.close()

        if (is_labeled):
            ratio = 0.9
            cut = int(ratio * len(self.img_list))
            if (self.is_vali):
                self.img_list = self.img_list[cut:]
            else:
                self.img_list = self.img_list[:cut]


    def __getitem__(self, idx):
        if (self.is_labeled):
            (tag, img_name) = self.img_list[idx]
        else:
            img_name = self.img_list[idx]

        img = np.fromfile(img_name)
        img.dtype = np.float32
        img.shape = (28, 28)
        img = Image.fromarray(img).convert('L')
        img = self.transform_func(img)

        if (self.is_labeled):
            return (img, tag)
        else:
            return img


    def __len__(self):
        return len(self.img_list)

