# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/29 15:07
@Author  : Lin Wang
@File    : 5_dataloader.py

"""
import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
test_data = torchvision.datasets.CIFAR10('cifar10', train=False, transform=to_tensor, download=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
writer = SummaryWriter('logs')

for idx, data in enumerate(test_loader):
    imgs, labels = data
    writer.add_images(tag='5_dataloader', img_tensor=imgs, global_step=idx)

writer.close()
