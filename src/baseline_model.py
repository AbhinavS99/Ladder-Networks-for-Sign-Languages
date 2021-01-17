from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf
from keras.callbacks import TensorBoard

def get_baseline_model():
    model = Sequential()
    model.add(Reshape((28,28,1), input_shape=(784,)))
    model.add(Conv2D(32, kernel_size=3, padding='valid', activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(64, kernel_size=3, padding='valid', activation='relu'))
    model.add(Conv2D(64, kernel_size=3, padding='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(25, activation='softmax'))
    return model