#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 15:10:42 2018

@author: sebalander
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
#x = diabetes.data[:, np.newaxis, 2].reshape(-1)

x = np.random.randn(100)
t = np.linspace(0, 1, len(x))
# leo aplano para tener como referencia
#x -= np.mean(x)

# %%

pi2 = 2 * np.pi
pi2sq = np.sqrt(pi2)

def prob1vs0(t, x, adaptPrior=True):
    m0, c0 = np.polyfit(t, x, 0, cov=True)
    m1, c1 = np.polyfit(t, x, 1, cov=True)

    dif0 = x - m0[0]
    var0 = np.mean((dif0)**2)
    dif1 = x - m1[0] * t - m1[1]
    var1 = np.mean((dif1)**2)

    # defino los priors
    if adaptPrior:
        pConst0 = 1 / (np.max(dif0) - np.min(dif0)) # prior de la constante
        deltaDif1 = np.max(dif1) - np.min(dif1)
        pConst1 = 1 / deltaDif1
        penDelta = deltaDif1 / (t[-1] - t[0])
        pPendi1 = 1 / penDelta / 2  # prior de la pendiente

        pWgH0 = pConst0
        pWgH1 = pConst1 * pPendi1
    else:
        pWgH0 = 1.0
        pWgH1 = 1.0

    pDagWH0 = sc.stats.multivariate_normal.pdf(dif0, cov=var0)
    pDagWH1 = sc.stats.multivariate_normal.pdf(dif1, cov=var1)

    deltaW0 = pi2sq * np.sqrt(c0)[0, 0]
    deltaW1 = pi2 * np.sqrt(np.linalg.det(c1))

    prob1_0 = np.prod(pDagWH1 / pDagWH0)
    prob1_0 *= pWgH1 * deltaW1 / pWgH0 / deltaW0

    return prob1_0

def probPenSig(a, s, adaptPrior=True):
    x1 = x * s + a * t
    return prob1vs0(t, x1, adaptPrior=True)

prob1vs0(t, x)
print(probPenSig(0.1, 1), probPenSig(0.1, 1, False))

# %%

noPriorList = list()
wiPriorList = list()
nDatos = 100
t = np.linspace(0, 1, nDatos)

for i in range(1000):
    x = np.random.randn(nDatos)
    noPriorList.append(prob1vs0(t, x, False))
    wiPriorList.append(prob1vs0(t, x))

noPrior = np.log(noPriorList)
wiPrior = np.log(wiPriorList)

plt.scatter(wiPrior, noPrior)
plt.xlabel('log no prior')
plt.ylabel('log con prior')

accNoPrior = np.mean(noPrior < 1)
accWiPrior = np.mean(wiPrior < 1)

print('claramente anda mejor la prediccion con prior', nDatos)
print('acc No prior', accNoPrior, 'acc with prior', accWiPrior)


# %%

probs = list()
aList = np.linspace(0, 10)
sList = np.linspace(1e-2, 2)

A, S = np.meshgrid(aList, sList)

P = [probPenSig(a, s, False)for a, s in zip(A.reshape(-1), S.reshape(-1))]
P = np.log(np.reshape(P, A.shape))


#import matplotlib
#import numpy as np
#import matplotlib.cm as cm
#import matplotlib.mlab as mlab
#import matplotlib.pyplot as plt
#
#plt.figure()
#CS = plt.contour(A, S, P)
#plt.clabel(CS, inline=1, fontsize=10)


from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(A, S, P, cmap='viridis')
ax.contour(A, S, P, zdir='z', cmap='magma')
ax.set_xlabel('pendiente')
ax.set_ylabel('sigma')





