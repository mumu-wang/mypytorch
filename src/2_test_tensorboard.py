# -*- coding: utf-8 -*-
"""
@Time    : 2024/1/27 20:30
@Author  : Lin Wang
@File    : 2_test_tensorboard.py

"""

# python3 -m tensorboard.main --logdir logs

from torch.utils.tensorboard import SummaryWriter
import cv2
import os

current_path = os.path.dirname(os.path.abspath(__file__))
img_path = './dataset/train/bees/29494643_e3410f0d37.jpg'

writer = SummaryWriter('logs')
image = cv2.imread(str(os.path.join(current_path, img_path)))
writer.add_image('2_test_tensorboard', image, 1, dataformats='HWC')
for i in range(100):
    writer.add_scalar('y=x', i, i)
writer.close()
