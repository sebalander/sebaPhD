#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 20:36:20 2017

testear

@author: sebalander
"""

# %% imports
import numpy as np
import scipy.linalg as ln
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy as dc

# %%

def plotPointsRefFrame(xm, ym, xc, yc, zc, xi, yi):
    plt.subplot(121)
    plt.scatter(xm,ym)
    plt.subplot(122)
    plt.scatter(xi,yi)
    
    fig = plt.figure()
    s = np.mean(ln.norm([xc,yc,zc], axis=0)) / 4
    ax = fig.gca(projection='3d')
    ax.scatter(xc, yc, zc)  # puntos rototrasladados
    ax.plot([0, s], [0, 0], [0, 0], '-r')
    ax.plot([0, 0], [0, s], [0, 0], '-b')
    ax.plot([0, 0], [0, 0], [0, s], '-k')

0

# %%

rRod = np.array([np.pi*0.1, 0, 0], dtype=float)
R = cv2.Rodrigues(rRod)[0]
# no tiene sentido que la coord T[2] sea negativa porque eso significa que los
# puntos estan detras de la camara
T = np.array([3, -7, 10], dtype=float)

# este es el vector que tiene que dar:
m = np.array([R[0,0], R[1,0], R[2,0],
              R[0,1], R[1,1], R[2,1],
              T[0], T[1], T[2]])

gr = np.linspace(-10,10,13)
xm, ym = np.meshgrid(gr,gr)
xm.shape = -1
ym.shape = -1

gr = np.linspace(-15,-10,4)
xAux, yAux = np.meshgrid(gr,gr)
xAux.shape = -1
yAux.shape = -1

xm = np.concatenate((xm, xAux-1))
ym = np.concatenate((ym, yAux+5))


# aplico rototraslacion
xc, yc, zc = np.dot(R[:,:2],[xm, ym]) + T.reshape(-1,1)
## es lo mismo que hacer esta cuenta
#x2 = m0[0] * x1 + m0[3] * y1 + m0[6]
#y2 = m0[1] * x1 + m0[4] * y1 + m0[7]
#z2 = m0[2] * x1 + m0[5] * y1 + m0[8]

# proyecto a coords homogeneas
xi = xc / zc
yi = yc / zc



# sumo ruido
xm += np.random.randn(xm.shape[0]) * 0.3
ym += np.random.randn(ym.shape[0]) * 0.3
xi += np.random.randn(xi.shape[0]) * 0.1
yi += np.random.randn(yi.shape[0]) * 0.1

plotPointsRefFrame(xm, ym, xc, yc, zc, xi, yi)


# %% armo la matriz de datos
zer = np.zeros_like(xm)
ons = np.ones_like(xm)

A1 = np.array([xm, zer, -xi*xm, ym, zer, -xi*ym, ons,  zer, -xi])
A2 = np.array([zer, xm, -yi*xm, zer, ym, -yi*ym,  zer, ons, -yi])

# tal que A*m = 0
A = np.concatenate((A1,A2), axis=1).T

## la hago cuadrada
#Asq = np.dot(A.T,A)
## chequeo que con el vector deseado da cero
#np.dot(Asq,m0) # valores chicos ~1e-11
#
## saco los autovalores ya utovectores
#l, v = np.linalg.eig(Asq)

u, s, v = ln.svd(A)

plt.figure()
plt.plot(s)
plt.semilogy()


np.linalg.matrix_rank(A)
# ta claro que el ultimo valor singular es cero
m1 = v[-1]

m1n = m1 / np.sqrt(ln.norm(m1[:3])*ln.norm(m1[3:6])) * np.sign(m1[-1])
m1n

plt.figure()
plt.plot(m,'k+')
plt.plot(m1n,'rx')

# %% pruebo a ver como da

# ahora lo grafico con el anterior
R1 = np.array([m1n[:3], m1n[3:6], np.cross(m1n[:3], m1n[3:6])]).T  # en marco ref camara
# donde quiero que este la camara
T1 = m1n[6:]

# aplico rototraslacion
xc1, yc1, zc1 = np.dot(R1[:,:2],[xm, ym]) + T1.reshape(-1,1)
# proyecto a coords homogeneas
xi1 = xc1 / zc1
yi1 = yc1 / zc1

plotPointsRefFrame(xm, ym, xc1, yc1, zc1, xi1, yi1)

plt.figure()
plt.scatter(xi, yi)
plt.scatter(xi1, yi1)











