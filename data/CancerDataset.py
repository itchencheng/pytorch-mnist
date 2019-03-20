#coding:utf-8

import os
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
import pandas as pd
import random


class CancerDataset(Dataset):

    def __init__(self, dir_path, is_labeled, is_vali, train_ratio, transform_func):
        super(CancerDataset, self).__init__()

        if (is_labeled):
            label_file = os.path.join(dir_path, 'train_labels.csv')
            dir_path = os.path.join(dir_path, 'train')
        else:
            label_file = os.path.join(dir_path, 'test_images.csv')
            dir_path = os.path.join(dir_path, 'test')

        self.is_labeled = is_labeled
        self.is_vali = is_vali
        self.transform_func = transform_func
        self.img_list = []

        fi = open(label_file)
        lines = fi.readlines()
        lines = lines[1:] # splt title
        for line in lines:
            line = line.strip()
            if (',' in line):
                line = line.split(',')
                tag = int(line[1])
                img_name = os.path.join(dir_path, line[0]+'.tif')
                self.img_list.append((img_name, tag))
        
        fi.close()

        if (is_labeled):
            random.seed(0)
            random.shuffle(self.img_list)
            ratio = train_ratio
            cut = int(ratio * len(self.img_list))
            if (self.is_vali):
                self.img_list = self.img_list[cut:]
            else:
                self.img_list = self.img_list[:cut]


    def __getitem__(self, idx):
        if (self.is_labeled):
            (img_name, tag) = self.img_list[idx]
        else:
            img_name = self.img_list[idx][0]

        img = Image.open(img_name)
        img = self.transform_func(img)

        if (self.is_labeled):
            return (img, tag)
        else:
            return (img, img_name)


    def __len__(self):
        return len(self.img_list)

