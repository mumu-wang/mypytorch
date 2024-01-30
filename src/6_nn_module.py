# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/29 17:31
@Author  : Lin Wang
@File    : 6_nn_module.py

"""
import torch
from torch import nn
import torch.nn.functional as F

device = 'cpu'


# model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, device=device)
        self.conv2 = nn.Conv2d(20, 20, 5, device=device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


input = torch.randn(1, 1, 50, 100, device=device)
model = Model()
output = model(input)
print(output)

# conv
input_tensor = torch.tensor([[1, 2, 0, 3, 1],
                             [0, 1, 2, 3, 1],
                             [1, 2, 1, 0, 0],
                             [5, 2, 3, 1, 1],
                             [2, 1, 0, 1, 1]])
input_tensor = torch.reshape(input_tensor, [1, 1, 5, 5])
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])
kernel = torch.reshape(kernel, [1, 1, 3, 3])
conv_result = F.conv2d(input_tensor, kernel)
print(conv_result)
