# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/31 15:37
@Author  : Lin Wang
@File    : 7_pooling.py

"""
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module
from torch.nn import MaxPool2d
from torchvision.transforms import v2
from PIL import Image

device = 'mps'


class Model(Module):
    def __init__(self):
        super().__init__()
        self.max_pooling = MaxPool2d(kernel_size=10, ceil_mode=True)

    def forward(self, x):
        return self.max_pooling(x)


image_path = './resources/image/view.jpg'
image_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(Image.open(image_path)) \
    .to(device=device)
image_pooling = Model()(image_tensor)

writer = SummaryWriter('logs')
writer.add_image('7_pooling_original', img_tensor=image_tensor, global_step=1)
writer.add_image('7_pooling_max_pool', img_tensor=image_pooling, global_step=1)
writer.close()
