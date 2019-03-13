import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from pyflann import *
from itertools import chain
from time import time
import threading

# 近邻算法类
flann = FLANN()

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

# 整理MNIST图像类
images = []
for i in range(10):
    images.append(mnist.train.images[mnist.train.labels==i])


class Neighbor(threading.Thread):
    def __init__(self, images, i):
        threading.Thread.__init__(self)
        self.images = images
        self.i = i
        self.class_i = []
        self.count = 0
        self.total = len(self.images[self.i])

    def run(self):
        print("thread", self.i, "start time:", time())
        for image in self.images[self.i]:
            nearests = []

            for j in chain(range(0, self.i), range(self.i + 1, 10)):
                nearest, dist = flann.nn(images[j], np.array([image]), 1, algorithm="kmeans", branching=32,
                                         iterations=7, checks=16)
                nearests.append(np.array(images[j][nearest[0]]))

            self.class_i.append(np.array(nearests))
            if self.count % 100 == 0:
                print("thread", self.i, "progress:", self.count, "/", self.total)
            self.count += 1
        print("thread", self.i, "end time:", time())

    def get_result(self):
        threading.Thread.join(self)
        return np.array(self.class_i)


# 开始计时
start = time()

nearest_images = []
Neighbors = []

for i in range(10):
    Neighbors.append(Neighbor(images, i))
    Neighbors[i].start()

for i in range(10):
    nearest_images.append(Neighbors[i].get_result())

nearest_images = np.array(nearest_images)

stop = time()
print("Total run time:" + str(stop - start))

