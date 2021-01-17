from src.data_utils import Data_Extracter
from src.baseline_model import get_baseline_model

from keras.callbacks import TensorBoard
from time import time
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import keras
from sklearn.metrics import accuracy_score
import random
import tensorflow as tf

data_extracter = Data_Extracter()
x_train, y_train = data_extracter.extract_data('./dataset/sign_mnist_train.csv')
x_test, y_test = data_extracter.extract_data('./dataset/sign_mnist_test.csv')

x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

sampled = range(x_train.shape[0])
random.seed(0)
random.shuffle(sampled)
sampled = sampled[:1000]

x_train = x_train[sampled]
y_train = y_train[sampled]

model = get_baseline_model()

model.compile(tf.keras.optimizers.Adam(lr=0.001) , 'categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
for _ in range(100):
    model.fit(x_train, y_train, batch_size=128, epochs=20)
    model.save('model.h5')
    y_test_pr = model.predict(x_test, batch_size=100)
    print("test accuracy" , accuracy_score(y_test.argmax(-1) , y_test_pr.argmax(-1)))