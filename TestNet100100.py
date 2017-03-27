# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:59:33 2017

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
import cv2

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
m1 = load_model('ModeloCompleto_6de7x7_varTam_2pool.h5')
m2 = load_model('ModeloSinsoft_6de7x7_varTam_2pool.h5')

## create an image with a handmade number somewhere
 
[n_x,n_y] = [100,100] #new image size
[n_test,o_x,o_y] = X_test.shape[0:3] #old size 

im_test=np.zeros([n_test,n_x,n_y,1]) #new image matrix, n° of images, size

#add number to image
for l in range(n_test):
    r_x=np.random.randint(n_x-o_x)
    r_y=np.random.randint(n_y-o_y)
    for i in range(o_x):
        for j in range(o_y):
            im_test[l,r_x+i,r_y+j,0] = X_test[l,i,j,0]
       
#add noise intensity 20 (~8% noise)
im_test= im_test + np.random.randint(0,high=20,size=[1,n_x,n_y,1])


#%%
#get classification
classif      = m1.predict(im_test) 
#probas_to_classes vuelve de la predicción a las clases como números de 0 a 9

# transform probability to classes
predictedClasses = probas_to_classes(classif)
#correct prediction vector
Tclas = predictedClasses == y_test

#create confussion Matrix
M_conf = np.zeros([n_classes,n_classes])
for l in range(n_test):    
    M_conf[y_test[0,l],predictedClasses[l]] += 1    

#normalize Conf Matrix
for l in range(n_classes):
    M_conf[:,l] /= M_conf[:,l].sum()
    
#calculate Accuracy
acc = Tclas.sum()/n_test*100


print("La precisión de prediccion es: %.1f%% " %acc)
print("La matriz de confunsión es:")
print(M_conf.round(2))


#%%------------------------------------------------------------------
im= im_test[12,:,:,0].reshape([1,n_x,n_y,1])
FeatureMaps = m2.predict(im)
cl = m1.predict(im)

weights=m1.get_weights()[-1]


acum=np.zeros(FeatureMaps.shape[1:])
for clase in range(cl.shape[1]):
    for feta in range(FeatureMaps.shape[-1]):
        acum[:,:,clase] += (FeatureMaps[0,:,:,feta]*weights[feta,clase])
    

#plt.imshow(acum[:,:,0])



dosNums= np.zeros([1,n_x,n_y,1])
for i in range(o_x):
    for j in range(o_y):
        dosNums[0,22+i,60+j,0]=X_test[7,i,j,0]
        dosNums[0,50+i,60+j,0]=X_test[7,i,j,0]
   
plt.imshow(dosNums[0,:,:,0])

clasDosNums = m1.predict(dosNums)
FeatDosNums = m2.predict(dosNums)[0].transpose(2,0,1)
acum=np.zeros(FeatureMaps.shape[1:3])

weights=m1.get_weights()[-1]
ultimasFetas = np.zeros(FeatDosNums.shape)

for clase in range(clasDosNums.shape[1]):
    for feta in range(FeatDosNums.shape[0]):
        ultimasFetas[clase,:,:] += FeatDosNums[feta,:,:]*weights[feta,clase]

#ultimasFetas = (FeatDosNums[0]*weights).transpose(2,0,1)



vmin = np.min(ultimasFetas)
vmax = np.max(ultimasFetas)

negro = np.zeros([100,100])


# %%
for feta in ultimasFetas:
    plt.figure()
    minimo = np.min(feta)
    maximo = np.max(feta)
    normalizada = (feta-minimo) / (maximo - minimo)
    plt.imshow(feta,vmin=vmin,vmax=vmax,cmap='inferno')
    #acum +=  FeatDosNums[0,:,:,cont]*weights[cont]
    
#
#cmparada = ultimasFetas[0]<ultimasFetas[1]
#plt.imshow(cmparada,cmap='gray')

# %%
acum
plt.figure()
plt.imshow(acum)
##--------------------#--------------#-------------------------------
#

# %% Graficar Primera Capa de Kernels
q=m1.get_weights()[0]
q=q.transpose(2,3,0,1)[0]

for feta in q:
    plt.figure()
    plt.imshow(feta,
               cmap='gray',
               interpolation='none')
               
# %%               