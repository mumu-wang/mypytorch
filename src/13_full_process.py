# -*- coding: utf-8 -*-
"""
@Time    : 2024/2/4 14:33
@Author  : Lin Wang
@File    : full_process.py

"""

import cv2
import torch
import torch.nn as nn
from torch.optim import SGD
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10
from model_cifar10 import *
import time

device = torch.device('mps')

transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, True)])

dataset_train = CIFAR10('cifar10', train=True, transform=transform)
dataset_test = CIFAR10('cifar10', train=False, transform=transform)

train_loader = DataLoader(dataset_train, 64)
test_loader = DataLoader(dataset_test, 64)

epoch = 10

# model
model = Model()
model = model.to(device=device)

# loss
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device=device)

# opti
optim = SGD(model.parameters(), lr=0.001)

total_train_loss = 0
total_test_loss = 0

for idx in range(101):
    total_train_loss = 0
    total_test_loss = 0
    start = time.time()
    model.train()
    for image, target in train_loader:
        image, target = image.to(device=device, dtype=torch.float32), target.to(device=device, dtype=torch.long)
        output = model(image)
        loss = loss_fn(output, target)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_loss += loss.item()
    print(f'epoch is {idx}, total train loss is {round(total_train_loss, 2)}')

    model.eval()
    for image_test, target_test in test_loader:
        image_test, target_test = image_test.to(device=device, dtype=torch.float32), target_test.to(device=device,
                                                                                                    dtype=torch.long)
        with torch.no_grad():
            output_test = model(image_test)
            loss = loss_fn(output_test, target_test)
            total_test_loss += loss.item()
    print(f'epoch is {idx}, total test loss is {round(total_test_loss, 2)}')
    end = time.time()
    print(f'time cost is {end - start}')

    if (idx + 1) % 20 == 0:
        torch.save(model, f'./model/cifar10_batch_{idx + 1}.pt')
