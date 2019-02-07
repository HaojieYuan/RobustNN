# -*- coding: utf-8 -*-

from utils import *


# 预先计算好的图像文件和其最相近其他类别图像文件
# 图像文件存有np数组形为 [10, ] ，每个数组成员形如 [该类图片数, 784]
# 图片邻居文件存有np数组形为 [10, ], 每个数组成员形如 [该类图片数, 9, 784]
# 784 = 28*28 为 MNIST 图片形状
IMG_PATH = '/Users/haojieyuan/Desktop/Data/RobustNN/images.npy'
IMG_Neighbor_PATH = '/Users/haojieyuan/Desktop/Data/RobustNN/nearest_images.npy'


# 读取数据
images = np.load(IMG_PATH)
neighbors = np.load(IMG_Neighbor_PATH)

# 对数据进行变形预处理
# 处理结束后，各变量尺寸如下
# image_labels [图片总数, 10] 1-hot型变量
# neighbor_labels [图片总数, 9, 10] 1-hot型变量
# images [图片总数, 784]
# neighbors [图片总数, 9, 784]
image_labels, neighbor_labels = get_labels(images)
images = np_flatten(images)
neighbors = np_flatten(neighbors)

# 对数据进行随机排列
shuffle_index = np.random.permutation(len(images))
image_labels = image_labels[shuffle_index]
neighbor_labels = neighbor_labels[shuffle_index]
images = images[shuffle_index]
neighbors = neighbors[shuffle_index]

# 切分 训练集/验证集/测试集
# 45000:5000:5000
train_set = dict()
dev_set = dict()
test_set = dict()

train_set['images'] = images[:45000]
train_set['labels'] = image_labels[:45000]
train_set['neighbors'] = neighbors[:45000]
train_set['neighbor_labels'] = neighbor_labels[:45000]

dev_set['images'] = images[45000:50000]
dev_set['labels'] = image_labels[45000:50000]

test_set['images'] = images[50000:]
test_set['labels'] = image_labels[50000:]

save_obj(train_set, 'train_set')
save_obj(dev_set, 'dev_set')
save_obj(test_set, 'test_set')
