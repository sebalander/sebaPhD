# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:15:52 2017

Generar DataSet aumentado, guardando los bounding Box que devuelve opencv


@author: ulises
"""

from keras.datasets import  mnist
from keras.models import Model, load_model
#from keras.layers.convolutional import Convolution2D
from keras.utils.np_utils import probas_to_classes
#from keras.layers.pooling import  GlobalAveragePooling2D
import numpy as np
from numpy import uint8
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import cv2

#cargo los test de mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((60000,28,28,1))
X_test  = X_test.reshape( (10000,28,28,1))
y_train = y_train.reshape((1,60000))
y_test  = y_test.reshape( (1,10000))
n_classes = 10


tam_num= np.arange(8,98,18,dtype=uint8)
#tam_num = [54]

X_tr = [] # clasificacion correcta (1 o 0)
pos_tr = []
X_ts = []
pos_ts = []

[n_x,n_y] = [100,100] #new image size



for k in tam_num:
    s=time.time()
    #creo
    X= []
    for i in range(len(X_test)):
        tam = np.random.randint(k,high=k+18,dtype=uint8)
        X.append(cv2.resize(X_test[i,:,:,0],(tam,tam)))
        
    n_test= len(X)
    im_test=np.zeros([n_test,n_x,n_y,1],dtype=uint8) #new image matrix, n° of images, size
    #add number to image

    x = np.zeros(len(im_test),dtype=uint8)
    y = np.zeros(len(im_test),dtype=uint8)
    lx = np.zeros(len(im_test),dtype=uint8)
    ly = np.zeros(len(im_test),dtype=uint8)
    for l in range(n_test):
        [o_x,o_y] = X[l].shape #old size 
        r_x = np.random.randint(0,high=n_x-o_x,dtype=uint8)
        r_y = np.random.randint(0,high=n_y-o_y,dtype=uint8)
        for i in range(o_x):
            for j in range(o_y):
                im_test[l,r_x+i,r_y+j,0] = X[l][i,j]
                x[l],y[l],lx[l],ly[l] = cv2.boundingRect(np.uint8(im_test[l,:,:,0]))
    X_ts.append(im_test)
    pos_ts.append([x,y,lx,ly])
    
    X= []
    for i in range(len(X_train)):
        tam = np.random.randint(k,high=k+18,dtype=uint8)
        X.append(cv2.resize(X_train[i,:,:,0],(tam,tam)))
        
    n_train = len(X)
    im_train=np.zeros([n_train,n_x,n_y,1],dtype=uint8) #new image matrix, n° of images, size
    #add number to image
    x = np.zeros(len(im_train),dtype=uint8)
    y = np.zeros(len(im_train),dtype=uint8)
    lx = np.zeros(len(im_train),dtype=uint8)
    ly = np.zeros(len(im_train),dtype=uint8)
    for l in range(n_train):
        [o_x,o_y] = X[l].shape #old size     
        r_x = np.random.randint(0,high=n_x-o_x,dtype=uint8)
        r_y = np.random.randint(0,high=n_y-o_y,dtype=uint8)
        for i in range(o_x):
            for j in range(o_y):
                im_train[l,r_x+i,r_y+j,0] = X[l][i,j]
                x[l],y[l],lx[l],ly[l] = (cv2.boundingRect(np.uint8(im_train[l,:,:,0])))
    X_tr.append(im_train)
    pos_tr.append([x,y,lx,ly])
    
    t = time.time()-s
    print("Para tamaño %d El tiempo es: %f" %(k,t))
    
    
    


data = {'x_train':X_tr,'y_train':y_train,'pos_tr':pos_tr,'x_test':X_ts,'y_test':y_test,'post_ts':pos_ts}
np.save('/home/ulises/Code/visionUNQ extra/CNN extra/resultados preliminares/augDataset1.npy',data)

