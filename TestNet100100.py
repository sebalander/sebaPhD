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


#import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((60000,28,28,1))
X_test  = X_test.reshape( (10000,28,28,1))
y_train = y_train.reshape((1,60000))
y_test  = y_test.reshape( (1,10000))
n_classes = 10

#cut the train set to half
X_train = X_train[:len(X_train)/2,:,:,:]
y_train = y_train[:,:len(y_train.transpose())/2]


#load Models
m1 = load_model('ModeloCompletos.h5')
m2 = load_model('ModeloSinsoft.h5')

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


#------------------------------------------------------------------
im= im_test[12,:,:,0].reshape([1,n_x,n_y,1])
FeatureMaps = m2.predict(im)
cl = m1.predict(im)
acum=np.zeros(FeatureMaps.shape[1:3])
for cont in range(FeatureMaps.shape[3]):
    acum = acum + FeatureMaps[0,:,:,cont]*cl[0,cont]
    
plt.imshow(acum)
c=probas_to_classes(cl)    
plt.imshow(FeatureMaps[0,:,:,9]*cl[0,9])


dosNums= np.zeros([1,n_x,n_y,1])
for i in range(o_x):
    for j in range(o_y):
        dosNums[0,3+i,3+j,0]=X_test[13,i,j,0]
        dosNums[0,50+i,60+j,0]=X_test[39,i,j,0]
   
plt.imshow(dosNums[0,:,:,0])

clasDosNums = m1.predict(dosNums)
FeatDosNums = m2.predict(dosNums)
#--------------------#--------------#-------------------------------

for i in range(10):
    plt.figure()
    plt.imshow(FeatDosNums[0,:,:,i])