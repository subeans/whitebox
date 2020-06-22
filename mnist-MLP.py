from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from packaging import version
import math
import time
import pickle

import os
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs,default=10,type=int)
parser.add_argument('--prof_start_batch', default=500, type=int)
parser.add_argument('--prof_end_batch', default=520, type=int)

args = parser.parse_args()

device_name = tf.test.gpu_device_name()
if not device_name:
	raise SystemError('GPU Device Not Found')
print('Found GPU at :{}'.format(device_name))

batch_size = args.batch_size
num_classes = 10

num_data = 60000
batch_data = math.ceil(num_data / args.batch_size)
epochs = math.ceil(args.prof_end_batch / batch_data)

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# 간단한 방법은 MNIST가 0-255 사이 값만을 가진다는 것을 알기 때문에 
# 단순히 255를 나눠도 Feature Scaling이 가능
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

#whitebox predict할 때 layer 2개 뿐인 MLP 를 구성하였으므로 똑같이 구성 

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

class BatchTimeCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.all_times = []

    def on_train_end(self, logs=None):
        time_file = open('time_list', 'ab')
        pickle.dump(self.all_times, time_file)
        time_file.close()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_times = []

    def on_epoch_end(self, epoch, logs=None):
        self.all_times.append(self.epoch_times)

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_time_start = time.time()

    def on_train_batch_end(self, batch, logs=None):
        self.epoch_times.append(time.time() - self.batch_time_start)

batch_time_callback = BatchTimeCallback()
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks = [batch_time_callback])
