# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:13:10 2016

@author: nicolasfaedo
"""

# test in cifar cnn with cam


from keras.datasets import  mnist
from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.utils.np_utils import to_categorical
from keras.layers.pooling import  GlobalAveragePooling2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((60000,28,28,1))
X_test  = X_test.reshape( (10000,28,28,1))
y_train = y_train.reshape((1,60000))
y_test  = y_test.reshape( (1,10000))

# this returns a tensor
inputs = Input(shape=(None,None,1))

# a layer instance is callable on a tensor, and returns a tensor
x = Convolution2D(10,5,5, border_mode='same')(inputs)
x = Activation('relu')(x)
x = Convolution2D(12,3,3, border_mode='same')(x)
x = Activation('relu')(x)
x = Convolution2D(10,3,3, border_mode='same')(x)

y = GlobalAveragePooling2D(dim_ordering='default')(x)

predictions = Activation('softmax')(y)

model=Model(input=inputs, output=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model2 = Model(input=inputs, output=x)
model2.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
y_tr = to_categorical(y_train.reshape(60000,1))
y_ts = to_categorical(y_test.reshape(10000,1))

model.fit(X_train, y_tr)  # starts training
#
model.save('ModeloCompleto.h5')
model2.save('ModeloSinsoft.h5')



#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------
#
m1 = load_model('ModeloCompleto.h5')
m2 = load_model('ModeloSinsoft.h5')
##Hago una prueba

ind=173
a=X_test[ind].reshape(28,28)
A=a.reshape(1,28,28,1)
plt.imshow(a)

b=m1.predict(A)
B=(b==b.max())*y_ts[ind]

Q=m2.predict(A)

