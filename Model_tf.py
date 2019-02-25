# -*- coding: utf-8 -*-

import tensorflow as tf
from utils import load_obj, next_batch
from tensorflow.python import debug as tf_debug


DEBUG = False

# 读取数据
train_set = load_obj('train_set')
dev_set = load_obj('dev_set')

# 训练参数
batch_size = 32
learning_rate = 1e-4
drop_out_rate = 0.5
train_epochs = 1000000

epsilon = 0.01


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
    '''
    x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))
    y_norm = tf.sqrt(tf.reduce_sum(tf.square(y), axis=1))
    xy = tf.reduce_sum(tf.multiply(x, y), axis=1)
    xy_cos = tf.divide(xy, tf.multiply(x_norm, y_norm)+1e-8)

    return xy_cos
    '''
    x_norm = tf.sqrt(tf.reduce_sum(x * x, 1))
    y_norm = tf.sqrt(tf.reduce_sum(y * y, 1))
    xy = tf.reduce_sum(x * y, 1)
    xy_cos = tf.div(xy, x_norm * y_norm + 1e-8)
    
    return xy_cos


def udf_cross_entropy(x, y, name=None):
    safe_y = tf.where(tf.equal(x, 0.), tf.ones_like(y), y)
    return -tf.reduce_sum(x * tf.log(safe_y), name=name)


# 输入部分
X_input = tf.placeholder(tf.float32, [None, 784], name='X_input')
X_neighbor = tf.placeholder(tf.float32, [None, 9, 784], name='X_neighbor')
y_input = tf.placeholder(tf.float32, [None, 10], name='y_input')
y_neighbor = tf.placeholder(tf.float32, [None, 9, 10], name='y_neighbor')

# 把X转为卷积所需要的形式
X = tf.reshape(X_input, [-1, 28, 28, 1], name='X_reshape')
# 第一层卷积：5×5×1卷积核32个 [5，5，1，32],h_conv1.shape=[-1, 28, 28, 32]
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name='W_conv1')
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv1')
h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1, name='h_conv1')

# 第一个pooling 层[-1, 28, 28, 32]->[-1, 14, 14, 32]
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积：5×5×32卷积核64个 [5，5，32，64],h_conv2.shape=[-1, 14, 14, 64]
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name='W_conv2')
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv2')
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name='h_conv2')

# 第二个pooling 层,[-1, 14, 14, 64]->[-1, 7, 7, 64]
h_pool2 = max_pool_2x2(h_conv2)

# flatten层，[-1, 7, 7, 64]->[-1, 7*7*64],即每个样本得到一个7*7*64维的样本
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name='h_pool2_flat')

# fc1
W_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1), name='W_fc1')
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name='b_fc1')
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='h_fc1')

# dropout: 输出的维度和h_fc1一样，只是随机部分值被值为零
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

# 输出层
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), name='W_fc2')
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]), name='b_fc2')
y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_out = tf.nn.softmax(y_logits, name='y_out')


# 训练
sess = tf.InteractiveSession()
if DEBUG:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)


# 计算基本loss
#cross_entropy = -tf.reduce_sum(y_input*tf.log(y_out+1e-8), name='ce_loss')
#cross_entropy = udf_cross_entropy(y_input, y_out, name='ce_loss')
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y_input), name='ce_loss')
total_loss = cross_entropy

# 计算每个错误类别的梯度，使其相似于 错误类别最接近于该类别的样本与输入样本的差
# 通常产生对抗样本 adv_X = X - gradients
# 我们希望 adv_X = X_new
# 即 gradients = X - X_new
# 采用两个向量的余弦值作为loss
# 训练目标是使cos值最大，所以将1-loss加入total_loss
for i in range(9):
    #adv_loss = -tf.reduce_sum(y_neighbor[:, i, :]*tf.log(y_out+1e-8))
    #adv_loss = udf_cross_entropy(y_neighbor[:, i, :], y_out)
    adv_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y_neighbor[:, i, :]))
    adv_gradients = tf.gradients(adv_loss, X_input)[0]
    difference = X_input - X_neighbor[:, i, :]
    cos_loss = tensor_cos(adv_gradients, difference)
    total_loss += epsilon*tf.reduce_sum(tf.abs(1-cos_loss))


train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_input, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

for i in range(train_epochs):
    batch = next_batch(train_set, batch_size)

    if i % 1000 == 0:
        train_accuracy, loss_normal, loss_sum = sess.run([accuracy, cross_entropy, total_loss],
                                                         feed_dict={X_input: batch[0], y_input: batch[1],
                                                                    X_neighbor: batch[2], y_neighbor: batch[3],
                                                                    keep_prob: drop_out_rate})
        print("step %d, training accuracy %g, cross entropy %g, regulizer %g" % (i, train_accuracy,
                                                                                 loss_normal, loss_sum - loss_normal))
        saver.save(sess, '/home/SENSETIME/yuanhaojie/RobustNN_ckpt/bs32ep100', global_step=i)
    train_step.run(feed_dict={X_input: batch[0], y_input: batch[1], X_neighbor: batch[2], y_neighbor: batch[3],
                              keep_prob: drop_out_rate})
    if i % 10000 == 0:
        print("test accuracy %g" % accuracy.eval(feed_dict={
            X_input: dev_set['images'], y_input: dev_set['labels'], keep_prob: 1.0}))
