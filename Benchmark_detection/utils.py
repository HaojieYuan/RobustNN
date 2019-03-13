import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data
from keras.utils.np_utils import to_categorical
import tensorflow as tf

def cifar10_load():
    (x_train_n, y_train_n), (x_test, y_test) = load_data()
    x_train = np.copy(x_train_n[:45000])
    y_train = np.copy(y_train_n[:45000])
    x_dev = np.copy(x_train_n[45000:])
    y_dev = np.copy(y_train_n[45000:])
    y_train = to_categorical(y_train)
    y_dev = to_categorical(y_dev)
    y_test = to_categorical(y_test)
    
    return (x_train/255.0, y_train), (x_dev/255.0, y_dev), (x_test/255.0, y_test)

def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    #print([str(i.name) for i in not_initialized_vars]) # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
