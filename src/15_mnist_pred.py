# -*- coding: utf-8 -*-
"""
@Time    : 2024/2/22 09:40
@Author  : Lin Wang
@File    : 15_mnist_pred.py

"""
from collections import OrderedDict
import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
import time

device = torch.device('mps')
num_epochs = 40

img_2_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, True)])
mnist_train = MNIST(root='mnist', train=True, transform=img_2_tensor, download=True)
mnist_test = MNIST(root='mnist', train=False, transform=img_2_tensor, download=True)

mnist_train_loader = DataLoader(dataset=mnist_train, batch_size=64)
mnist_test_loader = DataLoader(dataset=mnist_test, batch_size=12)


# 1. visualization dataset
def show_pic(dataset_loader):
    image, label = next(iter(dataset_loader))
    img = v2.ToPILImage()(image[0])
    img.show()
    print(label)


# show_pic(mnist_train_loader)


# 2. define model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('flatten', nn.Flatten()),
            ('linear1', nn.Linear(in_features=3136, out_features=128)),
            ('linear2', nn.Linear(in_features=128, out_features=10))
        ]))

    def forward(self, x):
        y = self.model(x)
        return y


def training_model():
    writer = SummaryWriter(log_dir='logs')
    model = MyModel().to(device=device)

    loss_fun = nn.CrossEntropyLoss().to(device=device)
    optim = SGD(params=model.parameters(), lr=0.001)

    print('start training')
    model.train()
    for epoch in range(1, num_epochs + 1):
        start = time.time()
        total_train_loss = 0
        for images, labels in mnist_train_loader:
            images, labels = images.to(device=device), labels.to(device=device)
            train_out = model(images)
            loss = loss_fun(train_out, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_train_loss += loss.item()
        end = time.time()
        print(f'epoch: {epoch}, total loss is {round(total_train_loss, 2)}, cost {int(end - start)} s')
        writer.add_scalar('mnist_train', total_train_loss, epoch)
    torch.save(model, f'./model/mnist_batch_{num_epochs}.pt')
    writer.close()


# training_model()


device = torch.device('cpu')


def test_model():
    model = torch.load(f'./model/mnist_batch_{num_epochs}.pt', map_location=device)
    model.eval()

    writer = SummaryWriter(log_dir='logs')
    with torch.no_grad():
        images, labels = next(iter(mnist_test_loader))
        images, labels = images.to(device=device), labels.to(device=device)
        output = model(images)
        result = output.argmax(1)
        print(f'predict result is {result.cpu().numpy()}')
        writer.add_images(tag='mnist_test', img_tensor=images)


test_model()
