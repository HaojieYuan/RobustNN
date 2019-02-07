# -*- coding: utf-8 -*-

import tensorflow as tf
from utils import load_obj, next_batch


# 读取数据
train_set = load_obj('train_set')
dev_set = load_obj('dev_set')

# 训练参数
batch_size = 32
learning_rate = 1e-4
drop_out_rate = 0.5
train_epochs = 10000

epsilon = 0.1


# 网络推理过程
# 权值初始化
def weight_variable(shape):
    # 用正态分布来初始化权值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # 本例中用ReLU激活函数，所以用一个很小的正偏置较好
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积层
def conv2d(x, W):
    # 默认 strides[0]=strides[3]=1, strides[1]为x方向步长，strides[2]为y方向步长
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# pooling 层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Tensorflow 向量夹角余弦值计算
def tensor_cos(x, y):
    x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))
    y_norm = tf.sqrt(tf.reduce_sum(tf.square(y), axis=1))
    xy = tf.reduce_sum(tf.multiply(x, y), axis=1)
    xy_cos = tf.divide(xy, tf.multiply(x_norm, y_norm))

    return xy_cos


# 输入部分
X_input = tf.placeholder(tf.float32, [None, 784])
X_neighbor = tf.placeholder(tf.float32, [None, 9, 784])
y_input = tf.placeholder(tf.float32, [None, 10])
y_neighbor = tf.placeholder(tf.float32, [None, 9, 10])

# 把X转为卷积所需要的形式
X = tf.reshape(X_input, [-1, 28, 28, 1])
# 第一层卷积：5×5×1卷积核32个 [5，5，1，32],h_conv1.shape=[-1, 28, 28, 32]
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)

# 第一个pooling 层[-1, 28, 28, 32]->[-1, 14, 14, 32]
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积：5×5×32卷积核64个 [5，5，32，64],h_conv2.shape=[-1, 14, 14, 64]
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# 第二个pooling 层,[-1, 14, 14, 64]->[-1, 7, 7, 64]
h_pool2 = max_pool_2x2(h_conv2)

# flatten层，[-1, 7, 7, 64]->[-1, 7*7*64],即每个样本得到一个7*7*64维的样本
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

# fc1
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout: 输出的维度和h_fc1一样，只是随机部分值被值为零
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_out = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 训练
sess = tf.InteractiveSession()

# 计算基本loss
cross_entropy = -tf.reduce_sum(y_input*tf.log(y_out))
total_loss = cross_entropy

# 计算每个错误类别的梯度，使其相似于 错误类别最接近于该类别的样本与输入样本的差
# 通常产生对抗样本 adv_X = X - gradients
# 我们希望 adv_X = X_new
# 即 gradients = X - X_new
# 采用两个向量的余弦值作为loss
# 训练目标是使cos值最大，所以将1-loss加入total_loss
for i in range(9):
    adv_loss = -tf.reduce_sum(y_neighbor[:, i, :]*tf.log(y_out))
    adv_gradients = tf.gradients(adv_loss, X_input)
    difference = X_input - X_neighbor[:, i, :]
    cos_loss = tensor_cos(adv_gradients, difference)
    total_loss += epsilon*(1 - cos_loss)


train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_input, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())

for i in range(train_epochs):
    batch = next_batch(train_set, batch_size)
    if i % 1000 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            X_input: batch[0], y_input: batch[1], keep_prob: drop_out_rate})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={X_input: batch[0], y_input: batch[1], keep_prob: drop_out_rate})

print("test accuracy %g" % accuracy.eval(feed_dict={
    X_input: dev_set['images'], y_input: dev_set['labels'], keep_prob: 1.0}))
