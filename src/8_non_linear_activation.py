# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/31 16:09
@Author  : Lin Wang
@File    : 8_non_linear_activation.py

"""

import torch
from torch.nn import Module
from torch.nn import Sigmoid
from torch.nn import ReLU
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from PIL import Image

device = 'mps'


class Model(Module):
    def __init__(self):
        super().__init__()
        self.non_linear = Sigmoid()
        # self.non_linear = ReLU()

    def forward(self, x):
        return self.non_linear(x)

image_path = './resources/image/view.jpg'
image_original = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(Image.open(image_path)) \
    .to(device=device)
image_non_liner = Model()(image_original)

writer = SummaryWriter('logs')
writer.add_image('8_non_linear_activation_original', image_original, global_step=1)
writer.add_image('8_non_linear_activation_non_linear', image_non_liner, global_step=1)
writer.close()
