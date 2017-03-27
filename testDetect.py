# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:42:28 2017

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
import cv2

#cargo los test de mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_test  = X_test.reshape( (10000,28,28,1))
y_test  = y_test.reshape( (1,10000))
n_classes = 10

#cargo modelos
m1 = load_model('../sebaPhD/ModeloCompleto_6de7x7_varTam.h5')
m2 = load_model('../sebaPhD/ModeloSinsoft_6de7x7_varTam.h5')



# tamaños de numeros
#tam_num= np.arange(8,98,2)
tam_num= np.arange(8,98,2)
#tam_num = [54]

Tc = [] # clasificacion correcta (1 o 0)
pos = [] # posición sacada de la imagen (x y) de donde empieza el cuadrado
leng = [] # largo del rectangulo
posPred = []# posición sacada del cam (pred_x,pred_y)
lenPred = []# largo del rectangulo sacado del cam
winner = []
[n_x,n_y] = [100,100] #new image size
weights=m1.get_weights()[-1]


for k in tam_num:
    s=time.time()
    #creo
    X=np.zeros([len(X_test),k,k,1])
    for i in range(len(X_test)):
        X[i,:,:,0]=cv2.resize(X_test[i,:,:,0],(k,k))
        
    [n_test,o_x,o_y] = X.shape[0:3] #old size 
    im_test=np.zeros([n_test,n_x,n_y,1]) #new image matrix, n° of images, size
    #add number to image
    r_x = np.random.randint(0,high=n_x-o_x,size=n_test)
    r_y = np.random.randint(0,high=n_y-o_y,size=n_test)
    x = np.zeros(len(im_test))
    y = np.zeros(len(im_test))
    lx = np.zeros(len(im_test))
    ly = np.zeros(len(im_test))
    for l in range(n_test):
        for i in range(o_x):
            for j in range(o_y):
                im_test[l,r_x[l]+i,r_y[l]+j,0] = X[l,i,j,0]
                x[l],y[l],lx[l],ly[l] = cv2.boundingRect(np.uint8(im_test[l,:,:,0]))
    #add noise intensity 20 (~8% noise)
    im_test= im_test + np.random.randint(0,high=20,size=[1,n_x,n_y,1])
    # de acá al print tengo que agregarle un indent

    classif=np.zeros((len(im_test),10))
    e=time.time()
    
    # Predicciones 
    pr = m1.predict(im_test)
    fm = m2.predict(im_test)
    
    idx_win = np.zeros(len(pr))
    cam = np.zeros(fm.shape[0:-1])
    pred_x = np.zeros(len(im_test))
    pred_y = np.zeros(len(im_test))
    pred_lx = np.zeros(len(im_test))
    pred_ly = np.zeros(len(im_test))
    
    for l in range(len(pr)):
        idx_win[l] = np.argsort(pr[l,:])[-1]
        cam[l,:,:] = (fm[l,:,:,:]*weights[:,idx_win[l]]).sum(axis=2)
        camN = np.uint8(255* (cam[l,:,:]-cam[l,:,:].min())/(cam[l,:,:].max()-cam[l,:,:].min()))     #llevo a 0 255
        thc = cv2.adaptiveThreshold(camN,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,15,10)
        pred_x[l],pred_y[l],pred_lx[l],pred_ly[l]=cv2.boundingRect(255-thc)
    
    # transform probability to classes
    predictedClasses = probas_to_classes(pr)
    #correct prediction vector
    Tclas = predictedClasses == y_test
    #    #calculate Accuracy
    
    
    Tc.append(Tclas.sum())
    pos.append([x,y])
    leng.append([lx,ly])
    winner.append(idx_win)
    posPred.append([pred_x,pred_y])
    lenPred.append([pred_lx,pred_ly])
    t=e-s
    
    print("Para tamaño %d la precisión de prediccion es: %.1f%% " %(k,Tc[-1]/100))
    print("El tiempo es: %f" %t )
    
##
#
#
#IoU = np.zeros(len(x))
#for i in range(len(x)):
#    rx = set(range(np.int(x[i]),np.int(x[i]+lx[i])))
#    ry = set(range(np.int(y[i]),np.int(y[i]+ly[i])))
#    px = set(range(np.int(pred_x[i]),np.int(pred_x[i]+pred_lx[i])))
#    py = set(range(np.int(pred_y[i]),np.int(pred_y[i]+pred_ly[i])))
#    i_x = len(rx.intersection(px))
#    i_y = len(ry.intersection(py))
#    i_area = i_x*i_y
#    upi_area = len(rx)*len(ry)+len(px)*len(py)
#    IoU[i] = i_area/(upi_area-i_area)



#plt.plot(tam_num[0:(len(acc))],acc,'+-')
#plt.plot([15,15],[10,100])
#plt.plot([98,98],[10,100])
#plt.plot([8,98],[90,90])
#plt.savefig('/home/ulises/Code/visionUNQ extra/CNN extra/resultados preliminares/Acc_vs_tamNum_6_7x7_varTam2_1p.png')
#####
Vars={'Tc':Tc,'pos':pos,'leng':leng,'posPred':posPred,'lenPred':lenPred}
np.save('/home/ulises/Code/visionUNQ extra/CNN extra/resultados preliminares/testdetct.npy',Vars)
