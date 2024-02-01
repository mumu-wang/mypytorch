# -*- coding: utf-8 -*-
"""
@Time    : 2024/2/1 09:40
@Author  : Lin Wang
@File    : 9_linear.py

"""

import torch
from torch.nn import Linear, Module
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2


class Model(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(10, 5)
        self.linear2 = Linear(5, 3)

    def forward(self, x):
        x = self.linear1(x)
        return self.linear2(x)


input = torch.ones(2, 10)
output = Model()(input)
print(f' input is {input} \n output is {output} \n shape is {output.shape}')
