# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/28 09:05
@Author  : Lin Wang
@File    : 3_transform.py

"""
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import os

current_path = os.path.dirname(os.path.abspath(__file__))
img = './dataset/train/bees/29494643_e3410f0d37.jpg'
img_path = str(os.path.join(current_path, img))


# tensorboard show
def tensorboard_show(tag, image, step):
    writer = SummaryWriter(log_dir='logs')
    writer.add_image(tag=tag, img_tensor=image, global_step=step)
    writer.close()


# 1. ToTensor
def toTensor(img_path):
    image = Image.open(img_path)
    tensor_trans = transforms.ToTensor()
    tensor_img = tensor_trans(image)
    return tensor_img


# 2. Normalize
def normalize(img_path):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    trans_norm = transforms.Normalize(mean, std)
    img_norm = trans_norm(toTensor(img_path))
    return img_norm


# 3. Resize
def resize(img_path):
    image_compose = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])
    image_tensor = image_compose(Image.open(img_path))
    return image_tensor


tensorboard_show('3_transform_img', toTensor(img_path), 1)
tensorboard_show('3_transform_img_normalize', normalize(img_path), 1)
tensorboard_show('3_transform_img_resize', resize(img_path), 1)
