# -*- coding: utf-8 -*-
"""
@Time    : 2024/2/7 09:18
@Author  : Lin Wang
@File    : 1_dataset_dataloader.py

"""
import cv2
import torch
from torchvision.transforms import v2
from torchvision.datasets import *
from torch.utils.data import DataLoader
import numpy as np


def dataset_study():
    img_2_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    dataset = CIFAR10('cifar10', train=False, download=True, transform=img_2_tensor)
    print(dataset)
    print(dataset.class_to_idx)
    return dataset


def dataloader_study():
    dataset = dataset_study()
    dataloader = DataLoader(dataset=dataset, batch_size=64)
    img, target = next(iter(dataloader))
    numpy_images = img.permute(0, 2, 3, 1).numpy()
    merged_image = np.concatenate(np.split(numpy_images, 8, axis=0), axis=1)
    merged_image = np.concatenate(merged_image, axis=1)
    cv2.imshow('pic', merged_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


print(dataloader_study())
