# -*- coding: utf-8 -*-
"""
@Time    : 2024/2/1 17:51
@Author  : Lin Wang
@File    : 12_optim.py

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


image_2_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
dataset = torchvision.datasets.CIFAR10(root='cifar10', train=False, transform=image_2_tensor, download=True)
data_loader = DataLoader(dataset=dataset, batch_size=64)

my_model = Model()
loss_cross = nn.CrossEntropyLoss()
optim = op.SGD(my_model.parameters(), lr=0.01)

for epoch in range(1, 20):
    epoch_loss = 0
    for image, target in data_loader:
        image, target = image.to(device=device, dtype=torch.float32), target.to(device=device, dtype=torch.long)
        image_result = my_model(image)
        result_loss = loss_cross(image_result, target)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        epoch_loss += result_loss
    print(f'epoch {epoch}: loss is {epoch_loss}')

