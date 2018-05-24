#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Fri May 11 21:00:23 2018

@author: snigdha
"""
"""
Question 2: 
Design a convnet that sorts numbers. Operators are ReLU, Conv, and Pooling. 
E.g. input: 5, 3, 6, 2; output: 2, 3, 5, 6

Below is the Keras implementation of the convnet to sort numbers.

This scripts defines the actual convent model and trains it using n subsets of 
4 numbers ranging from 0-50.

Some reference for Keras implementation for 1D convnet has been taken from
https://datascience.stackexchange.com/questions/29345

Methodology: 1D convnet will be used for this problem.

The brief description of the convnet is as follows:
    
    conv1D(no of filters = 128, kernel = 3)
    |
    v
    conv1D(no of filters = 64, kernel = 3)
    |
    v
    Max pool(pool size 2)
    |
    v
    conv1D(no of filters = 32, kernel = 3)
    |
    v
    Max pool(pool size 2)
    |
    v
    Flatten
    |
    v
    Dense layer(4)
    
Kernel of size 3 was chosen to capture the order of numbers
when they are sorted.(Intuition).

The network predicts a set of 4 numbers which are replaced with 
numbers they are closest to in the original list.
    
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, Reshape
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import load_model

import numpy as np

"""
n Random permutations of size 4 in the range 0-50 are taken to 
train the network. Similar technique is followed to create
the test set.
"""
n = 500000
x_train = np.zeros((n,4))
for i in range(n):
    x_train[i,:] = np.random.permutation(50)[0:4]

x_train = x_train.reshape(n, 4, 1)
y_train = np.sort(x_train, axis=1).reshape(n, 4,)

n = 1000
x_test = np.zeros((n,4))
for i in range(n):
    x_test[i,:] = np.random.permutation(50)[0:4]

x_test = x_test.reshape(n, 4, 1)
y_test = np.sort(x_test, axis=1).reshape(n, 4,)

input_shape = (4,1)

"""
Convolutional neural network model to learn sorting
"""
model = Sequential()
model.add(Conv1D(128, kernel_size=(3),
                 activation='relu',
                 input_shape=input_shape,
                 padding='same'))

model.add(Conv1D(64, (3), activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Reshape((64,2)))

model.add(Conv1D(32, (3), activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=(2)))

model.add(Flatten())
model.add(Dense(4))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

"""
Training the model (using batches for faster training)
"""
epochs = 30
batch_size = 128
# Fitting the model weights.
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

"""
Plotting the accuracy and loss curves for training
and validation(or test)
"""
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

"""
Checking one test case to see how the model works
and training parameters(like epochs, dataset size) need to be changed.

Numbers predicted are replaced with the "numbers closest to"
the ones in original test list

"""
test_list = [5,3,6,2]
pred = model.predict(np.asarray(test_list).reshape(1,4,1))
print(test_list)
print(pred)

print([np.asarray(test_list).reshape(4,)[np.abs(np.asarray(test_list).reshape(4,) - i).argmin()] for i in list(pred[0])])

"""
Saving this Keras model

model.save('best_model.h5')
"""