# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:42:28 2017

@author: ulises
"""
# %% Bloque 1 imports y variables del set de test


from keras.models import load_model
from keras.utils.np_utils import probas_to_classes
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from keras.utils.io_utils import print_function

dir = '/home/ulises/Code/visionUNQ extra/CNN extra/resultados preliminares/'
vars=(np.load(dir+'augDataset1.npy')).item()

X_test = vars['x_test']
y_test = vars['y_test']
pos_test = vars['post_ts']
#cargo modelos
m1 = load_model('../sebaPhD/ModeloCompleto_6de7x7_varTam2.h5')
m2 = load_model('../sebaPhD/ModeloSinsoft_6de7x7_varTam2.h5')


# %% Bloque 2 predicción y detección


Tc = [] # clasificacion correcta (1 o 0)
posPred = []# posición sacada del cam (pred_x,pred_y)
winner = []
[n_x,n_y] = [100,100] #new image size
weights=m1.get_weights()[-1]


for k in range(len(X_test)):
    s=time.time()
    X=X_test[k]
    [n_test,o_x,o_y] = X.shape[0:3] #old size from image
    im_test=np.zeros([n_test,n_x,n_y,1]) #new image matrix, n° of images, size
    
    #add noise intensity 20 (~8% noise)
    im_test= X + np.random.randint(0,high=20,size=[1,n_x,n_y,1])
    sp =(np.random.rand(len(im_test),100,100,1)>.97)
    im_test[sp] = 255
    # Predicciones 
    pr = m1.predict(im_test) #class prediction
    fm = m2.predict(im_test) # last layer featureMaps
    
    idx_win = np.zeros(len(pr)) #winner's index
    cam = np.zeros(fm.shape[0:-1]) #class activation maps
    pred_x = np.zeros(len(im_test)) # predicted x start of image
    pred_y = np.zeros(len(im_test)) # predicted y start of image
    pred_lx = np.zeros(len(im_test)) # predicted x length of image
    pred_ly = np.zeros(len(im_test)) # predicted y length of image
    
    for l in range(len(pr)):
        idx_win[l] = np.argsort(pr[l,:])[-1] #Sort index 
        #Calculate Class Activation Map
        cam[l,:,:] = (fm[l,:,:,:]*weights[:,idx_win[l]]).sum(axis=2)
        # Uint8 Normalization of CAM
        camN = np.uint8(255* (cam[l,:,:]-cam[l,:,:].min())/(cam[l,:,:].max()-cam[l,:,:].min()))
        # aplication of Adaptive Threshold
        thc = cv2.adaptiveThreshold(camN,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,15,10)
        #get predicted x, y lx, ly
        pred_x[l],pred_y[l],pred_lx[l],pred_ly[l]=cv2.boundingRect(255-thc)
    
    # transform probability to classes
    predictedClasses = probas_to_classes(pr)
    #correct predictions vector
    Tclas = predictedClasses == y_test
    #    #calculate Accuracy
    
    #append to list  True class, winner, Predicted Pos
    Tc.append(Tclas)
    winner.append(idx_win)
    posPred.append([pred_x,pred_y,pred_lx,pred_ly])
    t=time.time()-s
    
    #print time of iteration over set to not get bored
    print("El tiempo es: %f" %t )
##

#
# %% Bloque 3 cálculo de la intersección sobre la union, predicción vs opencv
#
IoU = [] #intersection over union quotient as score for detection
for k in range(len(Tc)):
    #get real x, y, lx, ly
    xt = pos_test[k][0]
    yt = pos_test[k][1]
    lxt= pos_test[k][2]
    lyt= pos_test[k][3]
    #get predicted x, y, lx, ly
    xp = posPred[k][0]
    yp = posPred[k][1]
    lxp= posPred[k][2]
    lyp= posPred[k][3]
    #IoU auxiliar vector for itearion 
    iou = np.zeros(len(Tc[k].transpose()))
    for i in range(len(xt)):
        #define real rectangle
        rx = set(range(np.int(xt[i]),np.int(xt[i]+lxt[i])))
        ry = set(range(np.int(yt[i]),np.int(yt[i]+lyt[i])))
        #define predicted rectangle
        px = set(range(np.int(xp[i]),np.int(xp[i]+lxp[i])))
        py = set(range(np.int(yp[i]),np.int(yp[i]+lyp[i])))
        i_x = len(rx.intersection(px))#find intersection between real x and predicted x
        i_y = len(ry.intersection(py))#find intersection between real y and predicted y
        i_area = i_x*i_y #calculate intersection area
        #get union + intersection areas as sum of areas
        upi_area = len(rx)*len(ry)+len(px)*len(py)
        #get IoU
        iou[i] = i_area/(upi_area-i_area)
    #append IoU to list
    IoU.append(iou)

## Save variables
Vars={'Tc':Tc,'posPred':posPred,'IoU':IoU}
np.save('/home/ulises/Code/visionUNQ extra/CNN extra/resultados preliminares/testdetct.npy',Vars)


# %% Bloque 4 parámetros de interés 

#calculo parámetros de interés, es necesario correr el primer bloque y este

#Load variables calculated in 2 and 3
Vars =np.load('/home/ulises/Code/visionUNQ extra/CNN extra/resultados preliminares/testdetct.npy').item()
Tc = Vars['Tc']
posPred = Vars['posPred']
IoU = Vars['IoU']

#Define Lists for sections of dataset:
#Acc accuracy, 
#Dtc Score of IoU for Trueclass
#DNtc Score of IoU for Mistclass (not True class)
#D Score of IoU 
Acc  = []
Dtc  = []
DNtc = []
D    = []

# define vector for total True class, IoU and position
Tc_tot = np.zeros(0)
IoU_tot = np.zeros(0)
size_tot = np.zeros(0)
a_test = np.zeros(0)
a_pred = np.zeros(0)

for i in range(len(Tc)):
    #concatenate to get total vectors
    Tc_tot  = np.concatenate((Tc_tot,Tc[i].reshape(10000,)))
    size_tot = np.concatenate((size_tot,pos_test[i][3]))
    IoU_tot = np.concatenate((IoU_tot,IoU[i]))
    a_test = np.concatenate((a_test,np.float64(pos_test[i][2])*np.float64(pos_test[i][3])))
    a_pred = np.concatenate((a_pred,posPred[i][2]*posPred[i][3]))
    #Calculate and append acc, Dtc, DNtc and D for partial vector
    Acc.append(np.sum(Tc[i])/len(Tc[i].transpose()))
    Dtc.append(np.sum(Tc[i]*IoU[i])/np.sum(Tc[i]))
    DNtc.append(np.sum(~Tc[i]*IoU[i])/np.sum(~Tc[i]))
    D.append(np.sum(IoU[i])/len(Tc[i].transpose()))

#Calculate and append acc, Dtc, DNtc and D for total vectors
Acc_tot  = np.sum(Tc_tot)/len(Tc_tot)
Dtc_tot  = np.sum( Tc_tot*IoU_tot)/np.sum( Tc_tot)
DNtc_tot = np.sum((1-Tc_tot)*IoU_tot)/np.sum((1-Tc_tot))
D_tot    = np.sum(IoU_tot)/len(Tc_tot)

#get size order
orden = np.argsort(size_tot)


#
##get ordered size and IoU not really necesary
#IoUOrd = IoU_tot[orden]
#posOrd = size_tot[orden]
#

cocienteAreas =a_pred/a_test

m      = np.uint(size_tot.min()) #min value of sizes
M      = np.uint(size_tot.max()) #max value of sizes
mn     = np.zeros(M-m) #means vector
mnclas = np.zeros(M-m)
index  = np.zeros(M-m) #index
c_A    = np.zeros(M-m)
for idx in range(m,M):
    tamanio = np.argwhere(size_tot==idx)
    mn[idx-m] = np.mean(IoU_tot[tamanio])
    mnclas[idx-m] = np.mean(Tc_tot[tamanio])
    c_A [idx-m] = np.mean(cocienteAreas[tamanio])
    index[idx-m] = idx
    


aux =IoU_tot*Tc_tot
mnT = np.zeros(M-m) #means vector
indexT = np.zeros(M-m) #index
for idx in range(m,M):
    mnT[idx-m] = np.mean(aux[np.argwhere(size_tot==idx)])
    indexT[idx-m] = idx
# %% Plots
figDir = '/home/ulises/Code/visionUNQ extra/CNN extra/resultados preliminares/figs/varTam2/'

#plot( IoU vs number size)
plt.figure()
plt.title('IoU vs tamaño de numero')
plt.plot(size_tot[orden],IoU_tot[orden],'*')
plt.xlabel('tamaño numero')
plt.ylabel('Score IoU')
plt.savefig(figDir +'Iou.png')

#plot(clasificaioon en función de tamaño y score IoU)
plt.figure()
plt.subplot(211)
plt.plot(index,mnclas,label="media de la clasificación")
plt.legend(loc='best')
plt.grid()
plt.xlabel('tamaño numero')
plt.ylabel('Score clasificacion')

plt.subplot(212)
plt.plot(index,mn,label="media del IoU")
plt.plot(indexT,mnT,label="media del IoU para Trueclass")
plt.legend(loc='best')
plt.grid()
plt.xlabel('tamaño numero')
plt.ylabel('Score IoU media por tamaño')

plt.savefig(figDir +'class_IoUmean.png')

#plot( mean of IoUs)
plt.figure()
plt.title('promedio de los IoU para cada tamaño')
plt.plot(index,mn,label="media del IoU para todos")
plt.plot(indexT,mnT,label="media del IoU para Trueclass")
plt.legend(loc='best')
plt.grid()
plt.xlabel('tamaño numero')
plt.ylabel('Score IoU media por tamaño')
plt.savefig(figDir +'IoUmean.png')


plt.figure()
plt.title('cociente de promedios casos acertados sobre casos totales')
plt.plot(index,mnT/mn)
plt.grid()
plt.xlabel('tamaño numero')
plt.ylabel('Cociente Score IoU')
plt.savefig(figDir +'IoUtrue_sobre_tot.png')

plt.figure()
plt.title('Cociente de áreas')
plt.semilogy(index,c_A,label="AreaPredicha/AreaReal")
plt.grid(True,which="both",ls=":")
plt.xlabel('tamaño numero')
plt.ylabel('Relacion area predicha-real (log)')
plt.legend(loc='best')
plt.savefig(figDir +'CocienteA_pred_A_real.png')

#plot_model(m1, to_file=figDir+'model.png')

