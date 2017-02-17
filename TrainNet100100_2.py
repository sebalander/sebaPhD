# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 15:13:10 2016

@author: nicolasfaedo
"""

# test in cifar cnn with cam

from keras.datasets import  mnist
from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras.layers.convolutional import Convolution2D
from keras.utils.np_utils import to_categorical
from keras.layers.pooling import  GlobalAveragePooling2D
import numpy as np



#import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((60000,28,28,1))
X_test  = X_test.reshape( (10000,28,28,1))
y_train = y_train.reshape((1,60000))
y_test  = y_test.reshape( (1,10000))
n_classes = 10

#Recorto el set de train a la mitad
X_train = X_train[:len(X_train),:,:,:]
y_train = y_train[:,:len(y_train.transpose())]


y_tr = to_categorical(y_train)
y_ts = to_categorical(y_test)



[n_x,n_y] = [100,100]
[o_x,o_y] = X_train.shape[1:3]
n_ims = len(X_train)

im=np.zeros([n_ims,n_x,n_y,1])
for l in range(n_ims):
    r_x = np.random.randint(n_x-o_x)
    r_y = np.random.randint(n_y-o_y)
    for i in range(o_x):
        for j in range(o_y):
            im[l,r_x+i,r_y+j,:]= X_train[l,i,j,:]
            
X_train = im + np.random.randint(0,high=20,size=[len(im),n_x,n_y,1])

# this returns a tensor
inputs = Input(shape=(None,None,1))

# a layer instance is callable on a tensor, and returns a tensor
x = Convolution2D(10,5,5, border_mode='same')(inputs)
x = Activation('relu')(x)
x = Convolution2D(12,3,3, border_mode='same')(x)
x = Activation('relu')(x)
x = Convolution2D(10,3,3, border_mode='same')(x)

y = GlobalAveragePooling2D(dim_ordering='default')(x)

predictions = Dense(10,activation='softmax',bias=False)(y)

model=Model(input=inputs, output=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model2 = Model(input=inputs, output=x)
model2.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
               
model.fit(X_train, y_tr,batch_size=15,nb_epoch=10)  # starts training
#
#model.save('ModeloCompleto_2.h5')
#model2.save('ModeloSinsoft_2.h5')
#

