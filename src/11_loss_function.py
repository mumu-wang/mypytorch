# -*- coding: utf-8 -*-
"""
@Time    : 2024/2/1 16:24
@Author  : Lin Wang
@File    : 11_loss_function.py

"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

device = 'cpu'


# x = torch.tensor([0.2, 0.3, 0.2], requires_grad=True)
# x = torch.reshape(x, (1, 3))
# target = torch.tensor([1], dtype=torch.long)
# target = torch.reshape(target, (1,))
# loss_cross = nn.CrossEntropyLoss()
# cross_result = loss_cross(x, target)
# print(cross_result)
# cross_result.backward()
# print(1)


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


image_2_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
dataset = torchvision.datasets.CIFAR10(root='cifar10', train=False, transform=image_2_tensor, download=True)
data_loader = DataLoader(dataset=dataset, batch_size=64)
loss_cross = nn.CrossEntropyLoss()

image, target = next(iter(data_loader))
image, target = image.to(device=device, dtype=torch.float32), target.to(device=device, dtype=torch.long)
my_model = Model()
image_result = my_model(image)
result = loss_cross(image_result, target)
result.backward()

print(result)
