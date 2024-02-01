[TOC]

#### 0. 安装pytorch

https://pytorch.org/get-started/locally/

```
pip3 install torch torchvision torchaudio tensorboard opencv-python packaging six
```

学习视频：https://www.bilibili.com/video/BV1hE411t7RN/



#### 1. pytorch加载数据

Dataset: 提供一种方式获取**数据**和**标签**

- 抽象类，继承from **torch.utils.data** import Dataset 可以定义自己的数据集
- torchvision.datasets.xxx 提供常用的数据集

Dataloader: 按指定方式加载数据集

- from **torch.utils.data** import DataLoader



#### 2. tensorboard使用

##### 2.1. opencv

pip install opencv-python

import cv2

用来读取和显示图片，读取图片cv2.imread()返回值为numpy.array，shape 为 HWC， 在tensorboard中需要显示指定dataformat

##### 2.2 tensorboard可视化

可以添加和显示**图片**和**标量**数据

add_image 添加单张图片, add_images 添加多张图片

```
from torch.utils.tensorboard import SummaryWriter
import cv2

img_path = '/Users/linwang/Documents/source_code/py_test/mypytorch/dataset/train/bees/29494643_e3410f0d37.jpg'
writer = SummaryWriter('logs') # 在当前目录logs里面生成待展示文件

image = cv2.imread(img_path) # image type is tensor or numpy.array
writer.add_image('bees', image, 1, dataformats='HWC') # dataformats 显示指定图片格式为 height weight channel
for i in range(100):
    writer.add_scalar('y=x', i, i)
writer.close()
```

python3 -m tensorboard.main --logdir logs --port=6006 / tensorboard --logdir logs --port=6006

在 http://localhost:6006/#timeseries 中查看



#### 3. transforms的使用

transforms做为一个工具箱，里面提供的工具可以对**图片**进行各种转换

| <img src="./assets/image-20240128093245775.png" alt="image-20240128093245775" style="zoom: 25%;" /> |
| ------------------------------------------------------------ |
| <img src="./assets/image-20240128091238815.png" alt="image-20240128091238815" style="zoom:50%;" /> |

##### 3.1 transforms怎样使用

```
from torchvision import transforms
from PIL import Image
image = Image.open(img_path) # 读取图片，PIL image, numpy.array 
tensor_trans = transforms.ToTensor() # 创建具体工具
tensor_img = tensor_trans(image) # 使用工具
```

##### 3.2 tensor数据类型 

tensor类型里面包含 backward, data, auto_grad等反向神经网络常用的方法

##### 3.3 常见的transforms

输入，输出，作用

**ToTensor**:*Convert a PIL Image or ndarray to tensor and scale the values accordingly*

**Normalize**: *Normalize a tensor image with mean and standard deviation.*

**Resize**: *Resize the input image to the given size*

**Compose**: *Composes several transforms together. This transform does not support torchscript*

##### 3.4 V1, V2版本 TODO 区别

Torchvision supports common computer vision transformations in the `torchvision.transforms` and `torchvision.transforms.v2` modules. Transforms can be used to transform or augment data for training or inference of different tasks (image classification, detection, segmentation, video classification).



#### 4. torchvison中数据集使用

https://pytorch.org/ 中Docs里面介绍常用的模块，PyTorch, torchaudio, torchtext, torchvision

案例，在tensorboard中显示cifar10的10张图片

```
import ssl
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
import torch

ssl._create_default_https_context = ssl._create_unverified_context # 避免下载错误

# download CIFAR10
train_set = torchvision.datasets.CIFAR10(root='cifar10', train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root='cifar10', train=False, download=True)

image_2_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]) #ToTensor
writer = SummaryWriter('logs')
for index in range(10):
    image, _ = test_set[index]
    tensor_img = image_2_tensor(image)
    writer.add_image('CIFAR10', image_2_tensor(image), global_step=index)
writer.close()
```



#### 5. DataLoader使用

| <img src="./assets/image-20240129150053354.png" alt="image-20240129150053354" style="zoom:33%;" /> |
| ------------------------------------------------------------ |

torch.utils.data.DataLoader https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

输出： images, labels



#### 6. nn.Module和conv使用
1.怎样计算卷积：矩阵对应位置相乘最后相加, https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

| <img src="./assets/image-20240131094857047.png" alt="image-20240131094857047" style="zoom:50%;" /> |
| ------------------------------------------------------------ |
| Example: Blue maps are inputs, and cyan maps are outputs. padding=0, strides=1. |
| <img src="./assets/no_padding_no_strides.gif" alt="no_padding_no_strides" style="zoom:50%;" /> |

2.常用参数

- **in_channels** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Number of channels in the input image
- **out_channels** ([*int*](https://docs.python.org/3/library/functions.html#int)) – Number of channels produced by the convolution
- **kernel_size** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)) – Size of the convolving kernel
- **stride** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple)*,* *optional*) – Stride of the convolution. Default: 1
- **padding** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple) *or* [*str*](https://docs.python.org/3/library/stdtypes.html#str)*,* *optional*) – Padding added to all four sides of the input. Default: 0
- **bias** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – If `True`, adds a learnable bias to the output. Default: `True`

3.**torch.reshape**(*input*, *shape*) 可以修改input tensor的形态

4.自定义模型需要继承 **torch.nn.Module**, 自己实现 init, forward 方法

5.使用方法

```
conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2, device=device)
conv(image_tensor)
```



#### 7. 池化使用

1.池化层定义

| <img src="./assets/image-20240131153805741.png" alt="image-20240131153805741" style="zoom:50%;" /> |
| ------------------------------------------------------------ |

2.常用参数

- **kernel_size** ([*Union*](https://docs.python.org/3/library/typing.html#typing.Union)*[*[*int*](https://docs.python.org/3/library/functions.html#int)*,* [*Tuple*](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[*int*](https://docs.python.org/3/library/functions.html#int)*,* [*int*](https://docs.python.org/3/library/functions.html#int)*]]*) – the size of the window to take a max over
- **stride** ([*Union*](https://docs.python.org/3/library/typing.html#typing.Union)*[*[*int*](https://docs.python.org/3/library/functions.html#int)*,* [*Tuple*](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[*int*](https://docs.python.org/3/library/functions.html#int)*,* [*int*](https://docs.python.org/3/library/functions.html#int)*]]*) – the stride of the window. Default value is `kernel_size`
- **padding** ([*Union*](https://docs.python.org/3/library/typing.html#typing.Union)*[*[*int*](https://docs.python.org/3/library/functions.html#int)*,* [*Tuple*](https://docs.python.org/3/library/typing.html#typing.Tuple)*[*[*int*](https://docs.python.org/3/library/functions.html#int)*,* [*int*](https://docs.python.org/3/library/functions.html#int)*]]*) – Implicit negative infinity padding to be added on both sides

- **ceil_mode** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – when True, will use ceil instead of floor to compute the **output shape**. 当设置为True时，遇到边界时，池化层结果会保留所有结果，如果为False时，会舍弃不完整图形计算的结果

3.使用方法

```
max_pooling = nn.MaxPool2d(kernel_size=10, ceil_mode=True)
max_pooling(image_tensor)
```



#### 8. 非线性变换

1.常用变换

[`nn.ReLU`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU)

[`nn.Sigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid)

2.使用方法

```
non_linear = Sigmoid()
non_linear(image_tensor)
```



#### 9. 线性操作

1.作用于全连接网络

2.使用方法

```
m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
```

3.torch.flatten()可以将tensor降维，类似torch.reshape()子集



#### 10. 神经网络搭建实战 # TODO

输入：CIFAR10 数据

输出： 分类结果

model structure

| <img src="./assets/image-20240201105029162.png" alt="image-20240201105029162" style="zoom:50%;" /> |
| ------------------------------------------------------------ |

```
self.model = Sequential(OrderedDict([
            ('conv1', Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)),
            ('pool1', MaxPool2d(kernel_size=2)),
            ('conv2', Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)),
            ('pool2', MaxPool2d(kernel_size=2)),
            ('conv3', Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)),
            ('pool3', MaxPool2d(kernel_size=2)),
            ('flatten', Flatten()),
            ('full_connect1', Linear(1024, 128)),
            ('full_connect2', Linear(128, 10))
        ]))
```

使用nn.Sequential() 构建网络模型

nn.Flatten()可以将tensor变量降维。Flattens a contiguous range of dims into a tensor.

tensordboard 通过 add_graph() 可以可视化网络模型，对学习很有帮助



#### 11. 损失函数与反向传播

1.常见损失函数

[`nn.L1Loss`](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss)  Creates a criterion that measures the mean absolute error (MAE) 

[`nn.MSELoss`](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)  Creates a criterion that measures the mean squared error

[`nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)  

2.作用

计算预测值和目标的差异，调用backward方法，可以计算模型中每个权重的梯度

3.使用方法

```
# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward() #执行此方法会计算模型权重的梯度
```



#### 12. 优化器

1.在torch.optim包中

2.使用方法

```
for input, target in dataset:
    optimizer.zero_grad() #每一轮将梯度值清零
    output = model(input) # 模型预测
    loss = loss_fn(output, target) #计算损失
    loss.backward() #反向传播，计算梯度
    optimizer.step() #优化器，根据梯度更新权重
```


#### 13. 使用和修改现有模型
1. torchvision.models.vgg16
2. TODO 修改vgg模型 model.add_model, model.classifier.add_model, model.classifier[x]





