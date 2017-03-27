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

#import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_train = X_train.reshape((60000,28,28,1))
X_test  = X_test.reshape( (10000,28,28,1))
#y_train = y_train.reshape((1,60000))
y_test  = y_test.reshape( (1,10000))
n_classes = 10


m1 = load_model('../sebaPhD/ModeloCompleto_6de7x7_varTam.h5')
m2 = load_model('../sebaPhD/ModeloSinsoft_6de7x7_varTam.h5')


#funcion para obtener piramide
def pyramid(image, scale=1.5, minSize=(20, 20),maxSize=(250, 250)):
    # yield the original image
    oi = image
    ims=[]
    ims.append(image)
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = np.uint8(255 * (cv2.resize(np.uint8(image), (w,w))>20))
        #image = 255 * (image>20)
         # if the resized image does not meet the supplied minimum
         # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
         # yield the next image in the pyramid
        ims.append(image)
    image = oi
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] * scale)
        image = cv2.resize(image, (w,w))
        #image = 255* (image>20)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] > maxSize[1] or image.shape[1] > maxSize[0]:
            break
        # yield the next image in the pyramid
        ims.append(image)
    return ims



# tamaños de numeros
#tam_num= np.arange(8,104,2)
tam_num = [54]
acc = [] # np.zeros(len(K))
pos = []
posPred = []
lenPred = []
[n_x,n_y] = [100,100] #new image size
weights=m1.get_weights()[-1]


for k in tam_num:
    ## create an image with a handmade number somewhere
    s=time.time()
    X=np.zeros([len(X_test),k,k,1])
    for i in range(len(X_test)):
        X[i,:,:,0]=cv2.resize(X_test[i,:,:,0],(k,k))
        
    [n_test,o_x,o_y] = X.shape[0:3] #old size 
    im_test=np.zeros([n_test,n_x,n_y,1]) #new image matrix, n° of images, size
    #add number to image
    r_x = np.random.randint(0,high=n_x-o_x,size=n_test)
    r_y = np.random.randint(0,high=n_y-o_y,size=n_test)
    for l in range(n_test):
        for i in range(o_x):
            for j in range(o_y):
                im_test[l,r_x[l]+i,r_y[l]+j,0] = X[l,i,j,0]
    
    #add noise intensity 20 (~8% noise)
    im_test= im_test + np.random.randint(0,high=20,size=[1,n_x,n_y,1])
    # de acá al print tengo que agregarle un indent
x = np.zeros(len(im_test))
y = np.zeros(len(im_test))
lx = np.zeros(len(im_test))
ly = np.zeros(len(im_test))
classif=np.zeros((len(im_test),10))
e=time.time()

for n_im in range(len(im_test)):
    pyram = pyramid(im_test[n_im,:,:,0])
    pico = 0
    cams=[]
    fm=[]
    acumCams = np.zeros([100,100])
    for im in pyram:
        s = im.shape
        pr = m1.predict(im.reshape(1,s[0],s[1],1))[0]
        fm = m2.predict(im.reshape(1,s[0],s[1],1))[0]
        #ordeno los indices para encontrar ganador en la pred
        ipr_s = np.argsort(pr)
        #calculo el class activation map ganador
        cam = cv2.resize((fm*weights[:,ipr_s[-1]]).sum(axis=2),(100,100))
        #busco el pico en el CAM
        aux = pr[ipr_s[-1]]
        #aux = np.max(cam)- np.median(cam)
        #guardo los cams
        cams.append(cam)
        #los sumo para tener el cam "final"    
        acumCams += cam
        #comparo el pico con el pico anterior, si gana en intensidad guardo ganadores
        if aux>pico:
            pico = aux
            fetas = fm
            prWin = pr
            camWin = cam
            twin = s
    pr=prWin
    cam=camWin
    camN = np.uint8(255* (cam-cam.min())/(cam.max()-cam.min()))     #llevo a 0 255
    thc = cv2.adaptiveThreshold(camN,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,15,10)
    x[n_im],y[n_im],lx[n_im],ly[n_im]=cv2.boundingRect(255-thc)
    classif[n_im,:] = pr
# transform probability to classes
predictedClasses = probas_to_classes(classif)
#correct prediction vector
Tclas = predictedClasses == y_test
#    #calculate Accuracy
acc.append(Tclas.sum()/n_test*100)
pos.append([r_x,r_y])
posPred.append([x,y])
lenPred.apend([lx,ly])
t=e-s

print("Para tamaño %d la precisión de prediccion es: %.1f%% " %(k,acc[-1]))
print("El tiempo es: %f" %t )
    
##
#



#
#
#plt.plot(tam_num[0:(len(acc))],acc,'+-')
#plt.plot([15,15],[10,100])
#plt.plot([98,98],[10,100])
#plt.plot([8,98],[90,90])
#plt.savefig('/home/ulises/Code/visionUNQ extra/CNN extra/resultados preliminares/Acc_vs_tamNum_6_7x7_varTam2_1p.png')
#####
#Vars=[tam_num,acc]
#np.save('/home/ulises/Code/visionUNQ extra/CNN extra/resultados preliminares/Acc_vs_tamNum_6_7x7_varTam2_1p',Vars)
