#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 01:29:33 2017

testear si la proyeccion de elipses a traves de conoide es equivalente a una
aproximacion lineal

@author: sebalander
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from calibration import calibrator as cl

# %%
N = 100
# mismo centro para todos los puntos
xp, yp = np.array([[1,1]]*N).T
xm = np.empty_like(xp)
ym = np.empty_like(xp)
rV, tV = np.random.rand(2,3)

# genero muchas covarianzas diferentes
#Cp = np.random.rand(N,2,2)*10 - 5  # np.array([[1, 0.3],[0.3, 1]])/10

c = np.random.rand(2,2)*10 - 5
c = c.dot(c.T)
scale = np.linspace(0.1,10,num=N)

Cp = np.empty((N,2,2), dtype=float)
Cm = np.empty_like(Cp)

# mapeo las coordenadas y las covarianzas
for i in range(N):
    Cp[i] = c * scale[i]
    xm[i], ym[i], Cm[i] = cl.xypToZplane(xp[i], yp[i], rV, tV, Cp=Cp[i])

# suma de autovalores (area)
lp = Cp[:,[0,1],[0,1]].sum(axis=1)
lm = Cm[:,[0,1],[0,1]].sum(axis=1)

plt.plot(lp, lm, '+')

'''
esto me demuestra que el tamaño de la elipse es lineal si se deja la rototraslacion fija, o sea que deberia ser lo mismo que hacer la 
pero el centro de la elipse seguramente se mueve... eso no esta bien

'''