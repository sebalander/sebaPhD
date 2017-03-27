# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 19:47:59 2017

@author: ulises
"""


from keras.datasets import  mnist
from keras.models import Model, load_model
#from keras.layers.convolutional import Convolution2D
from keras.utils.np_utils import probas_to_classes
#from keras.layers.pooling import  GlobalAveragePooling2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

#import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((60000,28,28,1))
X_test  = X_test.reshape( (10000,28,28,1))
y_train = y_train.reshape((1,60000))
y_test  = y_test.reshape( (1,10000))
n_classes = 10

#cut the train set to half
#X_train = X_train[:len(X_train)/2,:,:,:]
#y_train = y_train[:,:len(y_train.transpose())/2]


#load Models
m1 = load_model('ModeloCompleto_6de7x7_varTam2_1p.h5')
m2 = load_model('ModeloSinsoft_6de7x7_varTam2_1p.h5')


#------------------------------------------------------------------------------
# La imagen es de tamaño constante, se hace un barrido cambiando el tamanio del
# numero
#------------------------------------------------------------------------------


import cv2

tam_num= np.arange(8,98,2)
acc = [] # np.zeros(len(K))


[n_x,n_y] = [100,100] #new image size

for k in tam_num:
    ## create an image with a handmade number somewhere
    s=time.time()
    X=np.zeros([len(X_test),k,k,1])
    for i in range(len(X_test)):
        X[i,:,:,0]=cv2.resize(X_test[i,:,:,0],(k,k))

    [n_test,o_x,o_y] = X.shape[0:3] #old size 
    
    im_test=np.zeros([n_test,n_x,n_y,1]) #new image matrix, n° of images, size
    #add number to image
    for l in range(n_test):
        r_x=np.random.randint(n_x-o_x)
        r_y=np.random.randint(n_y-o_y)
        for i in range(o_x):
            for j in range(o_y):
                im_test[l,r_x+i,r_y+j,0] = X[l,i,j,0]
    #add noise intensity 20 (~8% noise)
    im_test= im_test + np.random.randint(0,high=20,size=[1,n_x,n_y,1])
    
    #get classification
    
    classif      = m1.predict(im_test) 
    e=time.time()
    #probas_to_classes vuelve de la predicción a las clases como números de 0 a 9
    
    # transform probability to classes
    predictedClasses = probas_to_classes(classif)
    #correct prediction vector
    Tclas = predictedClasses == y_test
    
    #create confussion Matrix
#    
#    for l in range(n_test):    
#        M_conf[k,y_test[0,l],predictedClasses[l]] += 1    
#    
#    #normalize Conf Matrix
#    for l in range(n_classes):
#        M_conf[k,:,l] /= M_conf[k,:,l].sum()
#        
#    #calculate Accuracy
    acc.append(Tclas.sum()/n_test*100)
    t=e-s
    
    print("Para tamaño %d la precisión de prediccion es: %.1f%% " %(k,acc[-1]))
    print("El tiempo es: %f" %t )
    
##
##
#plt.plot(tam_num[0:(len(acc))],acc,'+-')
#plt.plot([15,15],[10,100])
#plt.plot([98,98],[10,100])
#plt.plot([8,98],[90,90])
#plt.savefig('/home/ulises/Code/visionUNQ extra/CNN extra/resultados preliminares/Acc_vs_tamNum_6_7x7_varTam2_1p.png')
#####
#Vars=[tam_num,acc]
#np.save('/home/ulises/Code/visionUNQ extra/CNN extra/resultados preliminares/Acc_vs_tamNum_6_7x7_varTam2_1p',Vars)
##


''' asd'''
######################################
#%%

## Corrijo los graficos
# Las imágenes tienen 28x28 pero dentro los números son de hasta 20x20, por lo 
# que se propone utilizar este tamaño máximo dentro de la imágen de 28x28
# para eso basta con afectar la coordenada X por 20/28
#coef = 20/28
#[t0,ac0]=np.load('/home/ulises/Code/visionUNQ extra/CNN extra/resultados preliminares/Acc_vs_tamNum_7_5x5_tam10.npy',)
#[t1,ac1]=np.load('/home/ulises/Code/visionUNQ extra/CNN extra/resultados preliminares/Acc_vs_tamNum_7_5x5.npy',)
#[t2,ac2]=np.load('/home/ulises/Code/visionUNQ extra/CNN extra/resultados preliminares/Acc_vs_tamNum_7_5x5_tam44x44.npy',)
#plt.figure()
#plt.title('Clasificación en función de tamaño corregida ')
#plt.plot(coef*t0[0:len(ac0)],ac0,'+-')
#plt.plot(coef*t1[0:len(ac1)],ac1,'+-')
#plt.plot(coef*t2[0:len(ac2)],ac2,'+-')
#
#plt.plot(coef*np.array([10,10]),[10,100])
#plt.plot(coef*np.array([28,28]),[10,100])
#plt.plot(coef*np.array([44,44]),[10,100])
#
#plt.plot([0,70],[90,90])
#plt.grid()
#
#plt.savefig('/home/ulises/Code/visionUNQ extra/CNN extra/resultados preliminares/figs/Acc_tamNum_7_5x5_3deTamFijo.png')
#plt.close()

