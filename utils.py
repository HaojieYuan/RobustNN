# -*- coding: utf-8 -*-

import numpy as np
import pickle


# Utils
def np_flatten(in_array):
    """
    用于将不规则的np数组展开成一维
    """
    for i in range(len(in_array)):
        if i == 0:
            out_array = in_array[i]
        else:
            out_array = np.concatenate((out_array, in_array[i]))

    return out_array


def to_onehot(y, num_classes=None):
    """
    参考keras的to_catagoricial
    输入：任意形状的数组，最后一维为代表one-hot向量1的位置的数字
    输出：one-hot型的数组
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def get_labels(images):
    """
    生成label数组方便后期训练使用
    """
    image_labels = []
    neighbor_labels = []
    for i in range(len(images)):
        labels = np.zeros([images[i].shape[0], 10], dtype=np.int) + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        image_labels.append(labels[:, i])
        neighbor_labels.append(np.delete(labels, i, axis=1))

    image_labels = np_flatten(image_labels)
    neighbor_labels = np_flatten(neighbor_labels)

    return to_onehot(image_labels), to_onehot(neighbor_labels)


# 由于test set在最终测试之前不能动
# 所以考虑将数据随机排序后存成文件，而不是每次都随机排序
# 下面定义了存取变量的函数
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def next_batch(train_set, num):
    batch_index = np.random.randint(len(train_set['images']), size=num)
    return [train_set['images'][batch_index], train_set['labels'][batch_index],
            train_set['neighbors'][batch_index], train_set['neighbor_labels'][batch_index]]