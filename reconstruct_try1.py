# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:54:08 2017

@author: ulises
"""

from keras.datasets import  mnist
from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras.layers.convolutional import Convolution2D
from keras.utils.np_utils import to_categorical
from keras.layers.pooling import  GlobalAveragePooling2D
import cv2
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

image = cv2.imread('/home/ulises/Downloads/joxg8DZD.png')
img = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )

#l = (np.array([[1,0,0],[0,1,0],[0,0,1]]).reshape(3,3,1,1),(3,))#np.random.rand(1))
l = (np.random.rand(3,3,1,1),np.random.rand(1))


# this returns a tensor
inputs = Input(shape=(None,None,1))

# a layer instance is callable on a tensor, and returns a tensor
x = Convolution2D(1,3,3, border_mode='valid')(inputs)
x = Activation('relu')(x)



model=Model(input=inputs, output=x)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



model.set_weights(l)


input = img.reshape(1,50,50,1)
input2 = np.lib.pad(input,((0,0),(1,1),(1,1),(0,0)),'constant',constant_values=0)

out = model.predict(input2)[0,:,:,0]
o2 = out - model.get_weights()[-1]
#hasta aca pasada



w = model.get_weights()[0][:,:,0,0]


w_s = w.shape
o_s = out.shape
i_s = input[0,:,:,0].shape
#in_pad = np.lib.pad(input[0,:,:,0] ,((1,w_s[0]-1),(1,w_s[1]-1)),'constant',constant_values=0)
#w_pad  = np.lib.pad(w              ,((1,i_s[0]-1),(1,i_s[1]-1)),'constant',constant_values=0)
#op1    = np.lib.pad(out            ,((1,w_s[0]-1),(1,w_s[1]-1)),'constant',constant_values=0)



#fourier de la entrada padeada, el w padeado y la salida padeada

f_in  = (np.fft.fft2(input[0,:,:,0],s=[i_s[0]+w_s[0]-1,i_s[1]+w_s[1]-1]))
f_op1 = (np.fft.fft2(out,s=[i_s[0]+w_s[0]-1,i_s[1]+w_s[1]-1]))
f_w   = (np.fft.fft2(w,s=[i_s[0]+w_s[0]-1,i_s[1]+w_s[1]-1]))


# %% calculo la convolución por fourier y comparo con la de keras


ir = f_in*f_w

r  = np.real((np.fft.ifft2(ir)))
f0 = np.concatenate((out,r[1:-1,1:-1]),axis=1)

plt.figure()
plt.title('salida de la red y conv fourier')
plt.imshow(f0,interpolation='none')



# %% hago la inversa

f_i   = f_op1/f_w
inv = np.real((np.fft.ifft2(f_i)))
#inv = np.real((np.fft.ifft2(f_i)))

i = np.roll(np.roll(inv,2,axis=0),2,axis=1)
inv2 = i[1:-1,1:-1]

f1 = np.concatenate((input[0,:,:,0],inv2),axis=1)


out2 = model.predict(i.reshape(1,52,52,1))[0,:,:,0]
f2 = np.concatenate((out,out2),axis=1)


plt.figure()
plt.title('entrada y reconstrucción')
plt.imshow(f1,interpolation='none')



plt.figure()
plt.title('salida y salida de reconstruct')
plt.imshow(f2,interpolation='none')
#

# %%
"""las cosas dan algo distinto, propongo filtrar la imagen  porque esas lineas 
   me molestan e intuyo que en el fondo se parecen bastante la original y la 
   reconstruida"""

#creo un filtro tipo pasabajos básico
q=np.ones((3,3))/9
#q=np.array([[0.33],[0.33],[0.33]])
si=inv2.shape
sq=q.shape
#transformo a fourier la inversa de la imagen y el filtro
tr1 = np.fft.fft2(inv2,s=(si[0]+sq[0]-1,si[1]+sq[1]-1))
tr2 = np.fft.fft2(q,s=(si[0]+sq[0]-1,si[1]+sq[1]-1))
# multiplico
prod = tr1*tr2
#saco la inversaiprod = np.real(np.fft.ifft2(prod))
iprod = np.abs(np.fft.ifft2(prod))
#concateno y muestro
f3 = np.concatenate((input[0,:,:,0],iprod[1:-1,:]),axis=1)

plt.figure()
plt.title('imagen original, y reconstruida filtrada')
plt.imshow(f3,interpolation='none')


#model.fit(X_train, y_tr,batch_size=15,nb_epoch=5)  # starts training
##
#model.save('ModeloCompleto_6de7x7_varTam.h5')
#model2.save('ModeloSinsoft_6de7x7_varTam.h5')
#
##
#data= [X_train,y_tr]
#
#np.save('X_y_train_varTam',data)