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


#Recorto el set de train a la mitad
X_train = X_train[:len(X_train)/2,:,:,:]
y_train = y_train[:,:len(y_train)/2]

[n_x,n_y] = [100,100]
[o_x,o_y] = X_train.shape[1:3]
n_ims = len(X_train)

im=np.zeros([n_ims,n_x,n_y,1])
for l in range(n_ims):
    r_x = np.random.randint(n_x-o_x)
    r_y = np.random.randint(n_y-o_y)
    for i in range(o_x):
        for j in range(o_y):
            im[l,r_x+i,r_y+i,:]= X_train[l,i,j,:]
            
X_train = im + 20*np.random.randint(0,high=255,size=[len(im),n_x,n_y,1])

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
              
y_tr = to_categorical(y_train.reshape(len(y_train),1))
y_ts = to_categorical(y_test.reshape(len(y_test),1))

model.fit(X_train, y_tr)  # starts training
#
model.save('ModeloCompletos.h5')
model2.save('ModeloSinsoft.h5')



#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

#Cargo los modelos
m1 = load_model('ModeloCompleto.h5')
m2 = load_model('ModeloSinsoft.h5')
##Hago una prueba

ind=173
a=X_test[ind].reshape(28,28)
A=a.reshape(1,28,28,1)
plt.imshow(a)

b=model.predict(A)
B=(b==b.max())*y_ts[ind]

Q=model2.predict(A)

acum=np.zeros(shape(Q)[1:3])
for cont in range(shape(b)[1]):
    acum = acum + Q[0,:,:,cont]*b[0,cont]
    
plt.imshow(acum)




#una segunda prueba
#vamos a generar una imagen que tenga un numero en alguna parte

n_x  = 100
n_y  = 100
n_im = 53
[o_x,o_y] = shape(X_test)[1:3]

#im=zeros([1,n_x,n_y,1]) #defino tamaño de la nueva imagen
im=np.random.randint(0,high=255,size=[1,n_x,n_y,1])

r_x=np.random.randint(n_x-o_x)
r_y=np.random.randint(n_y-o_y)
for i in range(o_x):
   for j in range(o_y):
       im[0,r_x+i,r_y+j,0] = X_test[n_im,i,j,0]
       
       
clasif      = model.predict(im)
FeatureMaps = model2.predict(im)

acum=np.zeros(shape(FeatureMaps)[1:3])
for cont in range(shape(clasif)[1]):
    acum = acum + FeatureMaps[0,:,:,cont]*clasif[0,cont]
    
plt.imshow(acum)
    