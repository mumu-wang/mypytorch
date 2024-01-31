# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/29 17:31
@Author  : Lin Wang
@File    : 6_nn_module.py

"""
import torch
from torch import nn
from torchvision.transforms import v2
import torch.nn.functional as F
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import time

device = 'mps'

# manual conv
input_tensor = torch.tensor([[1, 2, 0, 3, 1],
                             [0, 1, 2, 3, 1],
                             [1, 2, 1, 0, 0],
                             [5, 2, 3, 1, 1],
                             [2, 1, 0, 1, 1]])
input_tensor = torch.reshape(input_tensor, [1, 1, 5, 5]).to(device=device, dtype=torch.float32)
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])
kernel = torch.reshape(kernel, [1, 1, 3, 3]).to(device=device, dtype=torch.float32)
conv_result = F.conv2d(input_tensor, kernel)
print(conv_result)


# # model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 5, padding=2, device=device)

    def forward(self, x):
        return self.conv(x)


start_time = time.time()

image_path = './resources/image/view.jpg'
image_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(Image.open(image_path))\
    .to(torch.device(device=device))
image_conv = Model()(image_tensor)

end_time = time.time()
print(f'process running time is {round(end_time - start_time, 2)}')

writer = SummaryWriter('logs')
writer.add_image('6_nn_module_original', img_tensor=image_tensor, global_step=1)
writer.add_image('6_nn_module_conv', img_tensor=image_conv, global_step=1)
writer.close()
