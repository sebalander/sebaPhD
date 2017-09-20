#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 19:47:01 2017

este scrip suelto es para testear el fiteo de la hiperparabola con una parabola
perfectamente conocida aun poniendole ruido

@author: sebalander
"""
# %% imports
import numpy as np
from dev import multipolyfit as mpf
import matplotlib.pyplot as plt

# %% useful funcs
def coefs2mats(coefs, n=8):
    const = coefs[0]
    jac = coefs[1:n+1]
    hes = np.zeros((n,n))
    hes[np.triu_indices(n)] = hes.T[np.triu_indices(n)] = coefs[n+1:]
    
    hes[np.diag_indices(n)] *= 2
    return const, jac, hes


# invento ejemplo para testear el fiteo de hiperparabola

# vert = np.random.rand(8) # considero centrado en el origen, no pierdo nada
jac = np.random.rand(8) * 2 -1
hes = np.random.rand(8,8) * 2 - 1
hes = (hes + hes.T) / 2 # lo hago simetrico

deltaX = np.random.rand(10000, 8)
#y = deltaX.dot(jac)
#y += (deltaX.reshape((-1,8,1,1))
#      * hes.reshape((1,8,8,1))
#      * deltaX.T.reshape((1,1,8,-1))).sum(axis=(1,2)).diagonal() / 2

y = np.array([jac.dot(x) + x.dot(hes).dot(x) / 2 for x in deltaX])

# %% hago el fiteo
ruidoY = np.random.randn(y.shape[0]) * 0.1

coefs, exps = mpf.multipolyfit(deltaX, y + ruidoY, 2, powers_out=True)

constRan, jacRan, hesRan = coefs2mats(coefs)


testRan = [constRan + jacRan.dot(x) + x.dot(hesRan).dot(x) / 2 for x in deltaX]

plt.figure()
plt.plot(y, testRan, 'x')
emin, emax = [np.min(y), np.max(y)]
plt.plot([emin, emax],[emin, emax])


plt.figure()
plt.plot(hes[np.tril_indices_from(hes)], hesRan[np.tril_indices_from(hesRan)], '+')

plt.figure()
plt.plot(y, y+ruidoY, 'x')

# %%

hUli = np.array([[2,-10],[-10,2]])
xx = np.random.rand(1000, 2) * 2 - 1

yy = [x.dot(hUli).dot(x) / 2 for x in xx]

coefs, exps = mpf.multipolyfit(xx, yy, 2, powers_out=True)

constRan, jacRan, hesRan = coefs2mats(coefs,n=2)
