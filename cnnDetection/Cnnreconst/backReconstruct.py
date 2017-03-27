# -*- coding: utf-8 -*
'''
tratar de graficar los pesos/kernels de las redes convolutivas

feb 2017
@author: sebalander
'''
# %%
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import  mnist
from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation
from keras.layers.convolutional import Convolution2D
from keras.utils.np_utils import to_categorical
from keras.layers.pooling import  GlobalAveragePooling2D
import numpy as np

# %%
model = load_model('../ModeloCompleto_7_5x5_pooling.h5')
#kernels = np.load('kernels.npy')
kernels = model.get_weights()
# imprimo los tamaños
tamanios = [k.shape for k in kernels]


# %%

k=[]
b=[]
w=[]
for i in range(len(tamanios)):
    if len(tamanios[i])!=2:
        if len(tamanios[i])>2:     
            k.append(kernels[i])
        else:
            b.append(kernels[i])
    else:
        w.append(kernels[i])
        


# %% para ir usandolo a mano

#    
#if relu:
#    s[-2][s[-2]<0]=0
#        
        



#


#
def bakDraw(c,w,k,b,relu=False,transpose=False):
    
    n_el = len(k)+len(w)
    s=[None] * n_el
    
    
    
    if transpose:
        s[-1] = w[0][:,c]
    else:
        s[-1] = w[0][c]
    s[-2] = np.tensordot(k[-1],s[-1],axes=((3),(0)))
    
    #s[-2]=s[-2]-b[-1][c]
    if relu:
        s[-2][s[-2]<0]=0
    
 

    '''
    los tamanios  de los objetos son
    (3, 3, 10, 12) y (3, 3, 12) referidos a k2 y s2 respectivamente
    donde n1=n2=m1=m2 = 3
    y quiero obtener espacialmente algo de (n1+n2-1)x(m1+m2-1) = 5x5
    y de profundidad va a tener 10
    '''
    
    
    for i in range(n_el-2,0,-1):
        ss = s[i].shape
        ks = k[i-1].shape
        newSize = (ks[0]+ss[0]-1, ks[1]+ss[1]-1, ks[2])
        s[i-1] = np.zeros(newSize)
        
        for l in range(ss[0]):
            for j in range(ss[1]):
                tira = s[i][l,j]
                comp = tira*k[i-1]
                s[i-1][l:l+ks[0],j:j+ks[1]] += comp.sum(3)
            #s[i-1] = s[i-1]-b[i-1][c]    
            if relu:
                    s[i-1][s[i-1]<0]=0
            
            
                     
    return s[0][:,:,0]
#
##
### %%
##


plt.figure()
plt.suptitle("sin 'invertir' relu y con trasposision de W");
for c in range(9):
    plt.subplot('33'+str(c+1))
    im = bakDraw(c,w,k,b,relu=True,transpose=True)
    plt.imshow(im,cmap='gray',interpolation='none')
    plt.title('numero '+str(c))
    plt.axis('off')


# %%
#
#s1 = np.tensordot(k2,s2,axes=((0,1,3),(0,1,2)))
#s0 = np.tensordot(k1,s1,axes=((3),(0)))
#
#im = s0[:,:,0]
#
#plt.imshow(im,cmap='gray',interpolation='none')
#
## %% testear cuentas y cositas auxiliares
#tam = (4,4,2,3)
#
#k = np.arange(np.prod(tam)).reshape(tam,order='F')
#printK(k)
#
#tira = np.arange(tam[-1])
#
#sp = k*tira
#printK(sp)
#
## %%
#def printK(k):
#    ks = k.shape
#    for i in range(ks[2]):
#        for j in range(ks[3]):
#            print(k[:,:,i,j])
#
#def printS(s):
#    ss = s.shape
#    for i in range(ss[2]):
#        print(s[:,:,i])
#
