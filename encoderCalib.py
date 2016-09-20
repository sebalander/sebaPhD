# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 18:16:05 2016

@author: sebalander
"""
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters

# %%
# cargamos tvecs y encoders
tiltFile = "./resources/encoders/Experimento 2 Pan 20 fotos/tvecs2.txt"
tiltEncsFile = "./resources/encoders/Experimento 2 Pan 20 fotos/resultados2.csv"
panFile = "./resources/encoders//Experimento 2 Pan 20 fotos/tvecs2.txt"
panEncsFile = "./resources/encoders/Experimento 2 Pan 20 fotos/resultados2.csv"

tltAll = np.loadtxt(tiltEncsFile, delimiter=',')
panAll = np.loadtxt(panEncsFile, delimiter=',')
#tlt = tltAll[:,2].T
#tlt = np.concatenate((tlt, panAll[:,2].T))
#pan = tltAll[:,1].T
#pan = np.concatenate((pan, panAll[:,1].T))

encoders = tltAll[:,1:3]
encoders = np.concatenate((encoders, panAll[:,1:3]),axis=0)

tVecs = tltAll[:,6:]
tVecs =  np.concatenate((tVecs, panAll[:,6:]),axis=0)

# %% ROTACIONES
#
# esta no se necesita
#def rotBtoI(a,b,tB):
#    ca = np.cos(a)
#    sa = np.sin(a)
#    cb = np.cos(b)
#    sb = np.sin(b)
#    
#    tI = np.array([ca*tB[0] - sa*cb*tB[1] + sa*sb*tB[2],
#                   sa*tB[0] + ca*cb*tB[1] - ca*sb*tB[2],
#                   sb*tB[1] + cb*tB[2]])
#    
#    return tI

def rotItoB(a,b,tI):
    ca = np.cos(a)
    sa = np.sin(a)
    cb = np.cos(b)
    sb = np.sin(b)
    
    if len(tI.shape) == 1:
        tB = np.array([ca*tI[0] + sa*tI[1],
                       -sa*cb*tI[0] + ca*cb*tI[1] + sb*tI[2],
                       sa*sb*tI[0] - ca*sb*tI[1] + cb*tI[2]])
    else:
        tB = np.array([ca*tI[:,0] + sa*tI[:,1],
               -sa*cb*tI[:,0] + ca*cb*tI[:,1] + sb*tI[:,2],
               sa*sb*tI[:,0] - ca*sb*tI[:,1] + cb*tI[:,2]]).T
    
    return tB

# %% RESIDUAL FUNCTION

def residual(params, tVecs, encoders):
    
    tVecsB = rotItoB(params["ka"].value*encoders[:,0]+params["la"].value,
                     params["kb"].value*encoders[:,1]+params["lb"].value,
                     tVecs)
    
    tMean = np.mean(tVecsB,axis=0)
    
    return tVecs - tMean

# %%
params = Parameters()
params.add("ka",value=np.pi, vary=True)
params.add("la",value=0, vary=True)
params.add("kb",value=np.pi, vary=True)
params.add("lb",value=0, vary=True)

# da muy mallllll
out = minimize(residual, params, args=(tVecs, encoders))

