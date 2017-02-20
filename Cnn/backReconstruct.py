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
from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras.layers.convolutional import Convolution2D
from keras.utils.np_utils import to_categorical
from keras.layers.pooling import  GlobalAveragePooling2D
import numpy as np

# %%
model = load_model('ModeloCompleto_7_5x5.h5')
#kernels = np.load('kernels.npy')
kernels = model.get_weights()
# imprimo los tamaños
tamanios = [k.shape for k in kernels]


# %%




k1 = kernels[0]
b1 = kernels[1]

k2 = kernels[2]
b2 = kernels[3]

k3 = kernels[4]
b3 = kernels[5]

k4 = kernels[6]
b4 = kernels[7]

k5 = kernels[8]
b5 = kernels[9]

k6 = kernels[10]
b6 = kernels[11]


w = kernels[12]


# %% para ir usandolo a mano


#def bakDraw(c,w,k3,k2,k1,relu=False,transpose=False):
def bakDraw(c,w,k6,k5,k4,k3,k2,k1,relu=False,transpose=False):    
    
    if transpose:
        s6 = w[:,c]
    else:
        s6 = w[c]
    s5 = np.tensordot(k6,s6,axes=((3),(0)))
    
    if relu:
        s5[s5<0]=0
#    
#    
#    if transpose:
#        s3 = w[:,c]
#    else:
#        s3 = w[c]
#    s2 = np.tensordot(k3,s3,axes=((3),(0))) # todavia no es necesario hacer un for
#    
#    if relu:
#        s2[s2<0]=0 # pongo a cerlo lo negativo
#    
    
    #
    '''
    los tamanios  de los objetos son
    (3, 3, 10, 12) y (3, 3, 12) referidos a k2 y s2 respectivamente
    donde n1=n2=m1=m2 = 3
    y quiero obtener espacialmente algo de (n1+n2-1)x(m1+m2-1) = 5x5
    y de profundidad va a tener 10
    '''
    ss = s5.shape
    ks = k5.shape # son los sizes
    newSize = (ks[0]+ss[0]-1, ks[1]+ss[1]-1, ks[2])
    #    s1 = np.zeros(newSize)
    s4 = np.zeros(newSize)
    #printK(k2)
    
    for i in range(ss[0]):
        for j in range(ss[1]):
            tira = s5[i,j]
            comp = tira*k5
            s4[i:i+ss[0],j:j+ss[1]] += comp.sum(3)
    # recorrer s2, cada tira de 12, el 
    #    for i in range(ss[0]):
    #        for j in range(ss[1]):
    #            tira = s2[i,j]
    #            comp = tira*k2
    #            s1[i:i+ss[0],j:j+ss[1]] += comp.sum(3)
    #    
    #printS(s1)
    
    if relu:
        s4[s4<0]=0
    #    if relu:
    #        s1[s1<0]=0 # pongo a cerlo lo negativo
    
    #
    
    ss = s4.shape
    ks = k4.shape
    newSize = (ks[0]+ss[0]-1, ks[1]+ss[1]-1, ks[2])
    s3 = np.zeros(newSize)
    
    #    ss = s1.shape
    #    ks = k1.shape # son los sizes
    #    newSize = (ks[0]+ss[0]-1, ks[1]+ss[1]-1, ks[2])
    #    s0 = np.zeros(newSize)
    
    #printK(k1)
    
    for i in range(ss[0]):
        for j in range(ss[1]):
            tira = s4[i,j]
            comp = tira*k4
            s3[i:i+ss[0],j:j+ss[1]] += comp.sum(3)
    # recorrer s2, cada tira de 12, el 
    #    for i in range(ss[0]):
    #        for j in range(ss[1]):
    #            tira = s1[i,j]
    #            comp = tira*k1
    #            s0[i:i+ss[0],j:j+ss[1]] += comp.sum(3)
    
    if relu:
        s3[s3<0]=0 # pongo a cerlo lo negativo
    
    
    ss = s3.shape
    ks = k3.shape
    newSize = (ks[0]+ss[0]-1, ks[1]+ss[1]-1, ks[2])
    s2 = np.zeros(newSize)
    
    for i in range(ss[0]):
        for j in range(ss[1]):
            tira = s3[i,j]
            comp = tira*k3
            s2[i:i+ss[0],j:j+ss[1]] += comp.sum(3)
    
    if relu:
        s2[s2<0]=0 # pongo a cerlo lo negativo   
    
    
    
    ss = s2.shape
    ks = k2.shape
    newSize = (ks[0]+ss[0]-1, ks[1]+ss[1]-1, ks[2])
    s1 = np.zeros(newSize)
    
    for i in range(ss[0]):
        for j in range(ss[1]):
            tira = s2[i,j]
            comp = tira*k2
            s1[i:i+ss[0],j:j+ss[1]] += comp.sum(3)
    
    if relu:
        s1[s1<0]=0 # pongo a cerlo lo negativo   
        
    
        
    ss = s1.shape
    ks = k1.shape
    newSize = (ks[0]+ss[0]-1, ks[1]+ss[1]-1, ks[2])
    s0 = np.zeros(newSize)
    
    for i in range(ss[0]):
        for j in range(ss[1]):
            tira = s1[i,j]
            comp = tira*k1
            s0[i:i+ss[0],j:j+ss[1]] += comp.sum(3)
    
    if relu:
        s0[s0<0]=0 # pongo a cerlo lo negativo   
    
         
    return s0[:,:,0]


# %%

img=bakDraw(0,w,k6,k5,k4,k3,k2,k1,relu=False,transpose=True)

plt.figure()
plt.suptitle("sin 'invertir' relu y con trasposision de W");
for c in range(9):
    plt.subplot('33'+str(c+1))
    im = bakDraw(c,w,k3,k2,k1,relu=False,transpose=True)
    plt.imshow(im,cmap='gray',interpolation='none')
    plt.title('numero '+str(c))
    plt.axis('off')


# %%

s1 = np.tensordot(k2,s2,axes=((0,1,3),(0,1,2)))
s0 = np.tensordot(k1,s1,axes=((3),(0)))

im = s0[:,:,0]

plt.imshow(im,cmap='gray',interpolation='none')

# %% testear cuentas y cositas auxiliares
tam = (4,4,2,3)

k = np.arange(np.prod(tam)).reshape(tam,order='F')
printK(k)

tira = np.arange(tam[-1])

sp = k*tira
printK(sp)

# %%
def printK(k):
    ks = k.shape
    for i in range(ks[2]):
        for j in range(ks[3]):
            print(k[:,:,i,j])

def printS(s):
    ss = s.shape
    for i in range(ss[2]):
        print(s[:,:,i])








