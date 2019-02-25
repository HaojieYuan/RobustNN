import tensorflow as tf
from utils import load_obj
import numpy as np
import matplotlib.pyplot as plt

model_path = ''
dev_set = load_obj('dev_set')
target = 0
epsilon = 0.1

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, model_path)
graph = tf.get_default_graph()

X_input = graph.get_tensor_by_name('X_input:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')
y_out = graph.get_tensor_by_name('y_out:0')
y_input = graph.get_tensor_by_name('y_input:0')

loss = graph.get_tensor_by_name('ce_loss:0')
gradients = tf.gradients(loss, X_input)[0]

x = dev_set['images'][np.random.randint(5000)]
target_label = np.zeros([1, 10])
target_label[0][target] = 1


grad = gradients.eval(feed_dict={X_input: x, y_input: target_label, keep_prob: 1.0})
class_changed = correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(target_label, 1))
x -= epsilon*np.sign(grad)


print(class_changed)
plt.imshow(x.reshape([-1, 28]), cmap='gray')
plt.show()
