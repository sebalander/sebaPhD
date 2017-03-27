# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:56:29 2017

@author: ulises
"""

from keras.datasets import  mnist
import matplotlib.pyplot as plt
import numpy as np
import cv2


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape((60000,28,28))
X_test  = X_test.reshape( (10000,28,28))
y_train = y_train.reshape((1,60000))
y_test  = y_test.reshape( (1,10000))


si=[]
for x in X_train:
  si.append(cv2.boundingRect(x)[2:])


s =np.asarray(si)
plt.figure()
plt.subplot(2,1,1)
plt.title('ancho')
plt.hist(s[:,0],bins=15)
plt.subplot(2,1,2)
plt.title('alto')
plt.hist(s[:,1],bins=15)


plt.savefig('/home/ulises/Code/visionUNQ extra/CNN extra/resultados preliminares/HistogramaTamanos.png')
