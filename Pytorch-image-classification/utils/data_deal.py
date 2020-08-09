#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Authors:hujun06
Date:2020-07-16
"""

import torch, cv2, os, random
from torch.utils import data
from PIL import Image
import matplotlib.pyplot as plt

class CustomDataset(data.Dataset):
    """
    CustomDataset
    """

    def __init__(self, data_dir, label, transform=None, size):
        """
        :param data_dir:
        :param label:
        :param transform:
        :param size:
        """
        self.transform = transform
        self.size = size
        self.imgList = []
        self.labelList = []
        for xx in label:
            x = xx.split(' ')
            self.imgList.append(data_dir + '/' + x[0])
            self.labelList.append(int(x[1]))

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        imgPath = self.imgList[index]
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.open(imgPath)
        if self.size is not None:
            img = cv2.resize(img, self.size)
        if self.transform is not None:
            img = self.transform(img)
        label = torch.IntTensor([self.labelList[index]])
        return img, label

    def __len__(self):
        """
        :return:
        """
        return len(self.imgList)

def get_labels(path):
    """r
    :param path:
    :return:
    """
    trainPath = os.path.join(path)
    with open(trainPath, 'r') as f:
        labels_train = f.readlines()
    return labels_train

