#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 22:35:13 2018

@author: snigdha
"""

"""
This is the script for testing the 
best obtained Keras model.

Loading the best obtained Keras model 
and using it to sort numbers
"""
from keras.models import load_model
import numpy as np

model = load_model('best_model.h5')

"""
Given test case in the question
"""
test_list = [5,3,6,2]
pred = model.predict(np.asarray(test_list).reshape(1,4,1))
print("Input list",test_list)
#print(pred)

print("Sorted list:",[np.asarray(test_list).reshape(4,)[np.abs(np.asarray(test_list).
                  reshape(4,) - i).argmin()] for i in list(pred[0])])

"""
Wider range of numbers
"""
test_list = [42,37,15,2]
pred = model.predict(np.asarray(test_list).reshape(1,4,1))
print("Input list",test_list)
#print(pred)
print("Sorted list:",[np.asarray(test_list).reshape(4,)[np.abs(np.asarray(test_list).
                  reshape(4,) - i).argmin()] for i in list(pred[0])])

"""
Numbers the network hasn't seen
"""
test_list = [88, 70, 93, 63]
pred = model.predict(np.asarray(test_list).reshape(1,4,1))
print("Input list",test_list)
#print(pred)

print("Sorted list:",[np.asarray(test_list).reshape(4,)[np.abs(np.asarray(test_list).
                  reshape(4,) - i).argmin()] for i in list(pred[0])])

"""
Closer numbers the network hasn't seen
"""
test_list = [88, 87, 85, 81]
pred = model.predict(np.asarray(test_list).reshape(1,4,1))
print("Input list",test_list)
#print(pred)

print("Sorted list:",[np.asarray(test_list).reshape(4,)[np.abs(np.asarray(test_list).
                  reshape(4,) - i).argmin()] for i in list(pred[0])])

