# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/29 13:37
@Author  : Lin Wang
@File    : 4_torchvision.py

"""
import ssl
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
import torch

ssl._create_default_https_context = ssl._create_unverified_context

# download CIFAR10
train_set = torchvision.datasets.CIFAR10(root='cifar10', train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root='cifar10', train=False, download=True)

image_2_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
writer = SummaryWriter('logs')
for index in range(10):
    image, _ = test_set[index]
    tensor_img = image_2_tensor(image)
    writer.add_image('4_torchvision', image_2_tensor(image), global_step=index)
writer.close()
