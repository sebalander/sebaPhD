# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 18:02:38 2017

@author: ulises
"""

""" Data Augmentation, using Keras, test"""


from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import  mnist
from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras.layers.convolutional import Convolution2D
from keras.utils.np_utils import to_categorical
from keras.layers.pooling import  GlobalAveragePooling2D
import numpy as np
import matplotlib 

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((60000,28,28,1))
X_test  = X_test.reshape( (10000,28,28,1))
y_train = y_train.reshape((1,60000))
y_test  = y_test.reshape( (1,10000))
n_classes = 10

y_tr = to_categorical(y_train)
y_ts = to_categorical(y_test)

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=False,
    zoom_range=2)

aug = datagen.flow(X_test,y_ts,batch_size=10000)
[x,y] = aug.next()
    
for i in range(50):
    plt.figure()
    plt.imshow(x[i,:,:,0])
