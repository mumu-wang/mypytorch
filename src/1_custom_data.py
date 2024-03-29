# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/25 17:43
@Author  : Lin Wang
@File    : 1_read_data.py

"""

from torch.utils.data import Dataset
import cv2
import os
import torch

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
        pass

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.path, img_name)
        img = cv2.imread(str(img_item_path))
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = 'dataset/train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)
print(ants_dataset[0])

print(torch.__version__)
tensor = torch.Tensor([1.]).to(device='mps')
print(tensor)