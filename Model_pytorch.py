# -*- coding: utf-8 -*-

import torch
import numpy as np


# 预先计算好的图像文件和其最相近其他类别图像文件
# 图像文件存有np数组形为 [10, ] ，每个数组成员形如 [该类图片数, 784]
# 图片邻居文件存有np数组形为 [10, ], 每个数组成员形如 [该类图片数, 9, 784]
# 784 = 28*28 为 MNIST 图片形状
IMG_PATH = '/Users/haojieyuan/Desktop/Data/RobustNN/images.npy'
IMG_Neighbor_PATH = '/Users/haojieyuan/Desktop/Data/RobustNN/nearest_images.npy'

images = np.load(IMG_PATH)
neighbors = np.load(IMG_Neighbor_PATH)

# 设置训练参数
torch.device("cpu")
dtype = torch.float
batch_size = 1


# 构建 MNIST 分类网络

# 网络输入
# TODO: 输入随机化
IMG_input = torch.randn(batch_size, 784)
Neighbor_input = torch.randn(batch_size, 9, 784)
label_input = torch.rand(batch_size, 1)

IMG_input = IMG_input.view(batch_size, 28, 28)

# 网络推理部分
Conv = torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, 5),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(32, 64, 5),
    torch.nn.ReLU,
    torch.nn.MaxPool2d(2)
)

Linear = torch.nn.Sequential(
    torch.nn.Linear(7*7*64, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 10)
)


Conv_out = Conv(IMG_input)
Conv_out = Conv_out.view(-1, 7*7*64)
Linear_out = Linear(Conv_out)
output = torch.nn.functional.log_softmax(Linear_out, dim=1)


# TODO：初步训练网络正确率达到 98%
