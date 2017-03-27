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
import cv2
import numpy as np


print("cargo los datos \n")

#import matplotlib.pyplot as plt
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_train = X_train.reshape((60000,28,28,1))
#X_test  = X_test.reshape( (10000,28,28,1))
#y_train = y_train.reshape((1,60000))
#y_test  = y_test.reshape( (1,10000))
#n_classes = 10


##Recorto el set de train a la mitad
#X_train = X_train[:len(X_train),:,:,:]
#y_train = y_train[:,:len(y_train.transpose())]

dir = '/home/ulises/Code/visionUNQ extra/CNN extra/resultados preliminares/'
vars=(np.load(dir+'augDataset1.npy')).item()

X_train = vars['x_train']
y_test = vars['y_test']
#pos_test = vars['post_ts']

X_tr = np.zeros([0,100,100,1])
Y_tr = np.zeros(0)
for i in range(len(X_train)):
    X_tr = np.concatenate((X_tr,X_train[i]),axis=0)
#    Y_tr = np.concatenate((Y_tr,y_test[i]))


y_tr = to_categorical(Y_tr)


#X_train = np.load("X_train_varTam.npy")[0]

# %%
"""si no tengo los datos guardados los creo con lo siguiente"""
n_ims = len(X_train)
low=15
high=98
print("cambio tamaño de imagenes \n")
X=[]
for i in range(n_ims):
    tamNumN = np.random.randint(low,high=high)
    X.append(cv2.resize(X_train[i,:,:,0],(tamNumN,tamNumN)))


[n_x,n_y] = [100,100]


im=np.zeros([n_ims,n_x,n_y,1])
for l in range(n_ims):
    [o_x,o_y] = X[l].shape
    r_x = np.random.randint(n_x-o_x)
    r_y = np.random.randint(n_y-o_y)
    for i in range(o_x):
        for j in range(o_y):
            im[l,r_x+i,r_y+j,:]= X[l][i,j]
    
print("agrego ruido gausiano y salt and pepper \n")        
X_train = im + np.random.randint(0,high=200,size=[len(im),n_x,n_y,1]) 
sp =(np.random.rand(len(im),100,100,1)>.97)
X_train[sp] = 255



# %%
print("creo el modelo\n")
# this returns a tensor
inputs = Input(shape=(None,None,1))

# a layer instance is callable on a tensor, and returns a tensor
x = Convolution2D(15,7,7, border_mode='same')(inputs)
x = Activation('relu')(x)
x = Convolution2D(12,7,7, border_mode='same')(x)
x = Activation('relu')(x)
x = Convolution2D(10,7,7, border_mode='same')(x)
x = Activation('relu')(x)
x = Convolution2D(10,7,7, border_mode='same')(x)
x = Activation('relu')(x)
x = Convolution2D(10,7,7, border_mode='same')(x)
x = Activation('relu')(x)
x = Convolution2D(10,7,7, border_mode='same')(x)


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
              
print("comienzo el entrenamiento\n")           
model.fit(X_train, y_tr,batch_size=15,nb_epoch=5)  # starts training
#
print("guardo los datos")
model.save('ModeloCompleto_6de7x7_varTam2.h5')
model2.save('ModeloSinsoft_6de7x7_varTam2.h5')

#
#data= [X_train]

#np.save('X_train_varTam.npy',data)

