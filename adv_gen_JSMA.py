import tensorflow as tf
from utils import load_obj
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

model_path = "/home/SENSETIME/yuanhaojie/RobustNN_ckpt/bs32ep100-170000"
meta_graph = '/home/SENSETIME/yuanhaojie/RobustNN_ckpt/bs32ep100-170000.meta'
dev_set = load_obj('dev_set')
target = 0
epsilon = 0.01
#increase = True

sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph(meta_graph)
saver.restore(sess, model_path)
graph = tf.get_default_graph()

X_input = graph.get_tensor_by_name('X_input:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')
y_out = graph.get_tensor_by_name('y_out:0')
y_input = graph.get_tensor_by_name('y_input:0')
max_prob = tf.reduce_max(y_out)

loss = graph.get_tensor_by_name('ce_loss:0')

x = dev_set['images'][np.random.randint(5000)].reshape([1, 784])
target_label = np.zeros([1, 10])
target_label[0][target] = 1

class_changed = tf.equal(tf.argmax(y_out, 1), tf.argmax(target_label, 1))

target_gradients = tf.gradients(y_out*target_label, X_input)[0]
other_gradients = tf.reduce_sum(tf.gradients(y_out*(1-target_label), X_input), axis=0)
# scores_mask = ((target_gradients > 0) & (other_gradients < 0))
# saliency_map = target_gradients*tf.abs(other_gradients)*scores_mask


def get_pairs(map1, map2):
    max_ = 0
    p_ = 0
    q_ = 0
    increase = True
    for p in range(len(map1)):
        for q in range(p, len(map1)):
            alpha = map1[p] + map1[q]
            beta = map2[p] + map2[q]
            if alpha > 0 and beta < 0 and -alpha*beta > max_:
                p_, q_ = p, q
                max_ = -alpha*beta
                increase = True
            if alpha < 0 and beta > 0 and -alpha*beta > max_:
                p_, q_ = p, q
                max_ = -alpha*beta
                increase = False

    return p_, q_, increase


plt.imsave('/home/SENSETIME/yuanhaojie/results/origin_JSMA.png', x.reshape([-1, 28]), cmap='gray')
terminte_flag = False
i = 0
mask = np.ones([784])
prob = 0
while (not terminte_flag) or (prob < 0.8):
    target_gradients_e = target_gradients.eval(feed_dict={X_input: x, keep_prob: 1.0})[0]
    other_gradients_e = other_gradients.eval(feed_dict={X_input: x, keep_prob: 1.0})[0]
    #best = tf.arg_max(saliency_map)
    p, q, increase = get_pairs(target_gradients_e*mask, other_gradients_e*mask)
    if p == 0 and q == 0:
        print('Failed')
        break
    plt.imsave('/home/SENSETIME/yuanhaojie/results/JSMA.png', x.reshape([-1, 28]), cmap='gray')
    if increase:
        x[0][p] = 1
        x[0][q] = 1
    else:
        x[0][p] = 0
        x[0][q] = 0
    mask[p] = 0
    mask[q] = 0
    i += 1
    terminte_flag, prob = sess.run([class_changed, max_prob], feed_dict={X_input: x, y_input: target_label, keep_prob: 1.0})

print('iterations: %d' % i)
plt.imshow(x.reshape([-1, 28]), cmap='gray')
plt.show()
plt.imsave('/home/SENSETIME/yuanhaojie/results/JSMA.png', x.reshape([-1, 28]), cmap='gray')
