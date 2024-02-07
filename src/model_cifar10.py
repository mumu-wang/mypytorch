# -*- coding: utf-8 -*-
"""
@Time    : 2024/2/4 14:35
@Author  : Lin Wang
@File    : model_cifar10.py

"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
import torch.optim as op

device = 'mps'


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(OrderedDict([
            ('conv1', Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2, device=device)),
            ('pool1', MaxPool2d(kernel_size=2)),
            ('conv2', Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, device=device)),
            ('pool2', MaxPool2d(kernel_size=2)),
            ('conv3', Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, device=device)),
            ('pool3', MaxPool2d(kernel_size=2)),
            ('flatten', Flatten()),
            ('full_connect1', Linear(1024, 128, device=device)),
            ('full_connect2', Linear(128, 10, device=device))
        ]))

    def forward(self, x):
        return self.model(x)
