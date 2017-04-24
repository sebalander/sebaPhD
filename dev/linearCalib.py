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

# pruebo a ver como da

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
plt.plot(m,'k+')
plt.plot(m1n,'rx')


# %% pruebo a ver como dan

m0norm = np.linalg.norm(m0)
p = np.empty_like(m0)

for i in range(len(v)):
    m = v[i]
    # que tna paralelo es a m0
    p = np.dot(m0,m)
    ang = np.arccos(p / np.linalg.norm(m) / m0norm)
    r1 = np.linalg.norm(m[:3])
    r2 = np.linalg.norm(m[3:6])
    
    print(ang, r1/r2, m[6:])
0

# %% caso unidimensional
# pongo la rotacion deseada

th = np.pi * 0.1
ct = np.cos(th)
st = np.sin(th)
Rc = np.array([[ct,-st],[st,ct]])  # en marco ref camara
# donde quiero que este la camara
Tm = np.array([-3, -8])
Tc = -np.dot(R.T,Tm)

agregoUno = False

gr = np.linspace(-10,10,30)
x1 = gr
y1 = np.zeros_like(x1)
if agregoUno:
    # pongo un solo punto lejos no alineado
    x1[0] = 0.0
    y1[0] = -10.0


m0 = np.array([ct, st, Tc[0], Tc[1]])

x2 = ct * x1 - st * y1 + Tc[0]
y2 = st * x1 + ct * y1 + Tc[1]

x = x2 / y2

plt.figure(1)
plt.scatter(x1,x)

# ploteo para ver como estan orientados y eso
plt.figure(2)
plt.scatter(x1,y1)
plt.plot([Tm[0], Tm[0] + ct], [Tm[1], Tm[1] -st], 'r') # eje x
plt.plot([Tm[0], Tm[0] + st], [Tm[1], Tm[1] + ct], 'k') # eje y

# %% armo la matriz y la resuelvo
ons = np.ones_like(x1)
A = np.array([x1-x*y1, -y1-x*x1, ons, -x]).T

Asq = np.dot(A.T,A)

# saco los autovalores ya utovectores
l, v = np.linalg.eig(Asq)

# %% pruebo a ver como dan

#m0norm = np.linalg.norm(m0)
#p = np.empty_like(m0)
#
#for i in range(len(v)):
#    m = v[i]
#    # que tna paralelo es a m0
#    p = np.dot(m0,m)
#    ang = np.arccos(p / np.linalg.norm(m) / m0norm)
#    r = np.linalg.norm(m[:2])
#    
#    print(p, ang, r, m[2:])
#0
m1 = v[-1]
r = np.linalg.norm(m[:2])
m1 /= r

m0
m1

# ahora lo grafico con ela nterior
Rc1 = np.array([[m1[0],-m1[1]],[m1[1], m1[0]]])  # en marco ref camara
# donde quiero que este la camara
Tm1 = - np.dot(Rc1.T, m1[2:])
plt.figure(2)
plt.plot([Tm1[0], Tm1[0] + m1[0]], [Tm1[1], Tm1[1] - m1[1]], 'r') # eje x
plt.plot([Tm1[0], Tm1[0] + m1[1]], [Tm1[1], Tm1[1] + m1[0]], 'k') # eje y













