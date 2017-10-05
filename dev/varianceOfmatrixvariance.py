#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 19:01:13 2017

@author: sebalander
"""
# %%
import numpy as np
# import scipy as sc

# %%
N = int(5e2)  # number of data
M = int(1e5)  # number of realisations

mu = np.array([7, 10])  # mean of data
c = np.array([[5, -3],[-3, 7]])

# generate data
x = np.random.multivariate_normal(mu, c, (N, M))

# estimo los mu y c de cada relizacion
muest = np.mean(x, axis=0)
dif = x - muest.reshape((1,-1,2))
cest = np.sum(dif.reshape((N,M,2,1)) * dif.reshape((N,M,1,2)), axis=0) / (N - 1)


# saco la media y varianza entre todas las realizaciones
muExp = np.mean(muest, axis=0)
difmu = muest - muExp.reshape((1,2))
muVar = np.sum(difmu.reshape((M,2,1)) * difmu.reshape((M,1,2)), axis=0) / (M - 1)

cExp = np.mean(cest, axis=0)
difc = cest - cExp.reshape((1,2,2))
cVar = np.sum(difc.reshape((M,2,2,1,1)) *
               difc.reshape((M,1,1,2,2)).transpose((0,1,2,4,3)),
               axis=0) / (M - 1)

cVar2 = np.zeros((M,2,2,2,2))
for i in range(M):
    cVar2[i] = difc[i].reshape((2,2,1,1)) * difc[i].reshape((1,1,2,2))

cVar2 = np.sum(cVar2 / (M - 1), axis=0)

cVar2 = np.zeros_like(cVar)
for i in [0,1]:
    for j in [0,1]:
        for k in [0,1]:
            for l in [0,1]:
                cVar2[i,j,k,l] = np.sum(difc[:,i,j] * difc[:,k,l])

# saco las cuentas analiticas
expMu = mu
VarMu = c / N
expC = c
# no es necesario trasponer porque c es simetrica
VarC = (c.reshape((2,2,1,1)) *
        c.reshape((1,1,2,2)).transpose((0,1,3,2))) * (2 * N - 1) / (N - 1)**2

print('numerico: \n', cVar, '\n\n\n\n analitico \n', VarC)

reldif = np.abs((cVar - VarC) / VarC)
reldif > 1e-1
