# -*- coding: utf-8 -*-
"""
@Time    : 2024/2/7 14:20
@Author  : Lin Wang
@File    : 14_load_model.py

"""

import torch
from torchvision.transforms import v2
from PIL import Image
from torchvision.datasets import CIFAR10

device = torch.device('mps')
image_path = './resources/image/airplane.png'

model = torch.load('./model/cifar10_batch_100.pt', map_location=device)

image = Image.open(image_path)
image = image.convert('RGB')
img_2_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((32, 32))])
image_tensor = img_2_tensor(image)
image_tensor = torch.reshape(image_tensor, (1, 3, 32, 32)).to(device=device)

dataset = CIFAR10('cifar10', train=False, download=True)
idx_to_class = {value: key for key, value in dataset.class_to_idx.items()}

model.eval()
with torch.no_grad():
    output = model(image_tensor)
    print(output)
    # print(output.argmax(1).item())
    print(idx_to_class)
    print(idx_to_class[output.argmax(1).item()])
    # print(f'{output.argmax(1).item()}, {output.flatten().numpy()[output.argmax(1).item()]}')
