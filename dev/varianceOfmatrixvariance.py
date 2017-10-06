#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 19:01:13 2017

@author: sebalander
"""
# %%
import numpy as np

# %%
N = int(1e3)  # number of data
M = int(1e5)  # number of realisations

mu = np.array([7, 10])  # mean of data
c = np.array([[5, 3], [3, 7]])

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
difc = (cest - cExp).reshape((M,2,2,1,1))
cVarAux = difc * difc.transpose((0,4,3,2,1))
cVar = np.sum(cVarAux  / (M - 1), axis=0)


# saco las cuentas analiticas
expMu = mu
VarMu = c / N
expC = c
# no es necesario trasponer porque c es simetrica
cShaped = c.reshape((2,2,1,1))
VarCnn = cShaped * cShaped.transpose((3,2,1,0))
nn = (2 * N - 1) / (N - 1)**2

VarC = VarCnn * nn
cVarnn = cVar / nn

print('media de media numerico: \n', muExp,
      '\n\n media de media analitico \n', expMu)

print('\n\n varianza de media numerico: \n', muVar,
      '\n\n varianza de media analitico \n', VarMu)

print('\n\n media de varianza numerico: \n', cExp,
      '\n\n media de varianza analitico \n', expC)

print('\n\n varianza de varianza numerico (sin normalizar):\n', cVarnn,
      '\n\n varianza de varianza analitico (sin normalizar)\n', VarCnn)

#reldif = np.abs((cVar - VarC) / VarC)
#reldif > 1e-1

# %% pruebo wishart
#  probabilidad de la covarianza estimada
import scipy.stats as sts
# N degrees of freedom
# inv(c) is precision matrix
wpdf = sts.wishart(N, c)
wpdf.pdf(cest[3])






