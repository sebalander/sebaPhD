#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 19:01:13 2017

@author: sebalander
"""
# %%
import numpy as np
import scipy as sc

# %%
N = int(5e1)  # number of data
M = int(1e5)  # number of realisations

mu = np.array([7, 10])  # mean of data
c = np.array([[3, -2],[-2, 4]])

# generate data
x = np.random.multivariate_normal(mu, c, (N, M))

# estimo los mu y c de cada relizacion
muest = np.mean(x, axis=0)
dif = x - muest.reshape((1,-1,2))
cest = np.sum(dif.reshape((N,M,2,1)) * dif.reshape((N,M,1,2)), axis=0) / (N - 1)

# saco la media y varianza entre todas las realizaciones
muExp = np.mean(muest, axis=0)
difmu = muest - muExp.reshape((1,2))
muVar = np.mean(difmu.reshape((M,2,1)) * difmu.reshape((M,1,2)), axis=0)

cExp = np.mean(cest, axis=0)
difc = cest - cExp.reshape((1,2,2))
cVar = np.mean(difc.reshape((M,2,2,1,1)) *
               difc.reshape((M,1,1,2,2)).transpose((0,1,2,4,3)), axis=0)

# saco las cuentas analiticas
expMu = mu
VarMu = c / N
expC = c
VarC = (c.reshape((2,2,1,1)) *
        c.reshape((1,1,2,2)).transpose((0,1,3,2))
    )* (2 * N - 1) / (N - 1)**2

print('numerico: \n', cVar, '\n\n\n\n analitico \n', VarC)


