# -*- coding: utf-8 -*-
"""
@Time    : 2024/2/1 09:50
@Author  : Lin Wang
@File    : 10_exercise.py

"""
from collections import OrderedDict

import torch
from torch.nn import Linear, Module, Sequential, Conv2d, MaxPool2d, Flatten
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2


class Model(Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(OrderedDict([
            ('conv1', Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)),
            ('pool1', MaxPool2d(kernel_size=2)),
            ('conv2', Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)),
            ('pool2', MaxPool2d(kernel_size=2)),
            ('conv3', Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)),
            ('pool3', MaxPool2d(kernel_size=2)),
            ('flatten', Flatten()),
            ('full_connect1', Linear(1024, 128)),
            ('full_connect2', Linear(128, 10))
        ]))

    def forward(self, x):
        return self.model(x)


input = torch.ones(64, 3, 32, 32)
output = Model()(input)
print(output.shape)

writer = SummaryWriter('logs')
writer.add_graph(Model(), input_to_model=input)
writer.close()