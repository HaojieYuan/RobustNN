import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data
from keras.utils.np_utils import to_categorical

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