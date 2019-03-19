from cleverhans.attacks import LBFGS
import tensorflow as tf
import tensorflow.keras as keras
from utils import cifar10_load
import numpy as np
from cleverhans.utils_keras import KerasModelWrapper

model_file = 'ResNet32_acc893.h5'
(x_train, y_train), (x_dev, y_dev), (x_test, y_test) = cifar10_load()
model = keras.models.load_model(model_file)

lbfgs_params = {'batch_size':batch_size,
                'clip_min':0.,
                'clip_max':1.}

model_wrap = KerasModelWrapper(model)

x = tf.placeholder(tf.float32, shape=(None,32,32,3))
sess = keras.backend.get_session()
attack_wrap = LBFGS(model_wrap, sess=sess)
adv_x = attack_wrap.generate(x, **lbfgs_params)

adv_images = np.array(sess.run(adv_x, feed_dict={x: x_train[:16]}))
