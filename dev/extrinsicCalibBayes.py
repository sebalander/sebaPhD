#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 01:12:17 2017

@author: sebalander
"""


# %%
#import glob
import cv2
import numpy as np
#import scipy.linalg as ln
import matplotlib.pyplot as plt
#from importlib import reload
from copy import deepcopy as dc
#import pyproj as pj
from calibration import calibrator as cl
import time
import corner

import scipy.linalg as ln
import scipy.stats as sts
from dev import bayesLib as bl
from calibration import lla2eceflib as llef
from numpy.random import rand as rn

#from multiprocess import Process, Queue
# https://github.com/uqfoundation/multiprocess/tree/master/py3.6/examples

# %% LOAD DATA
# input
plotCorners = False
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
modelos = ['poly', 'rational', 'fisheye', 'stereographic']
model = modelos[3]

imagesFolder = "./resources/intrinsicCalib/" + camera + "/"
cornersFile =      imagesFolder + camera + "Corners.npy"
patternFile =      imagesFolder + camera + "ChessPattern.npy"
imgShapeFile =     imagesFolder + camera + "Shape.npy"

# model data files
#distCoeffsFile =   imagesFolder + camera + model + "DistCoeffs.npy"
#linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
#tVecsFile =        imagesFolder + camera + model + "Tvecs.npy"
#rVecsFile =        imagesFolder + camera + model + "Rvecs.npy"

# model data files
intrinsicParamsFile = imagesFolder + camera + model + "intrMetrResults.npy"
#intrinsicParamsFile = imagesFolder + camera + model + "intrinsicParamsML"
#intrinsicParamsFile = intrinsicParamsFile + "0.5.npy"


# output
extrinsicResultsFile = imagesFolder + camera + model + "extrMetrResults.npy"
# calibration points
calibptsFile = "/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/2016-11-13 medicion/calibrExtr/puntosCalibracion.txt"

extrinsicFolder = "./resources/intrinsicCalib/" + camera + "/"


# load model specific data
#distCoeffs = np.load(distCoeffsFile)
#cameraMatrix = np.load(linearCoeffsFile)

# load intrinsic calib uncertainty
resultsML = np.load(intrinsicParamsFile).all()  # load all objects
#Nmuestras = resultsML["Nsamples"]
Xint = resultsML['paramsMU']
covarDist = resultsML['paramsVAR']
#muCovar = resultsML['paramsMUvar']
#covarCovar = resultsML['paramsVARvar']
Ns = resultsML['Ns']
cameraMatrix, distCoeffs = bl.flat2int(Xint, Ns, model)

if model==modelos[3]:
    Cf = np.zeros((4,4))
    Cf[2:,2:] = covarDist[:2,:2]
    
    Ck = covarDist[2,2]
else:
    Cf = covarDist[:4,:4]
    Ck = covarDist[4:,4:]


# Calibration points
dataPts = np.loadtxt(calibptsFile)
xIm = dataPts[:,:2]  # PUNTOS DE CALIB EN PIXELES
n = 1
m = dataPts.shape[0]

# EGM96 sobre WGS84: https://geographiclib.sourceforge.io/cgi-bin/GeoidEval?input=-34.629344%2C+-58.370350&option=Submit
geoideval = 16.1066
elev = 7 # altura msnm en parque lezama segun google earth
h = elev + geoideval # altura del terreno sobre WGS84

# according to google earth, a good a priori position is
# -34.629344, -58.370350
hcam = 15.7 # camera heigh measured
LLAcam = np.array([-34.629344, -58.370350, h + hcam]) # a priori position of camera

# lat lon alt of the data points
hvec = np.zeros_like(dataPts[:,3]) + h
LLA = np.concatenate((dataPts[:,2:], hvec.reshape((-1,1))), axis=1)
LLA0 = np.array([LLAcam[0], LLAcam[1], h]) # frame of reference origin

calibPts = llef.lla2ltp(LLA, LLA0)  # PUNTOS DE CALIB EN METROS
camPrior = np.array([np.pi, 0, 0, 0, 0, hcam]) # # PRIOR CAMERA COORDS EN METROS
# pongo 3m de sigma en x,y y 20cm en z
W = np.array([0, 0, 0, 1 / 3, 1 / 3, 1 / 0.2])**2


plt.figure()
plt.plot(calibPts[:,0], calibPts[:,1], '+')
plt.plot(camPrior[0], camPrior[1], 'x')

# %% determino la pose a prior
# saco una pose inicial con opencv


# saco condiciones iniciales para los parametros uso opencv y heuristica

#reload(cl)
if model is modelos[3]:
    # saco rVecs, tVecs de la galera
    tvecIni = camPrior
    rvecIni = np.array([np.pi, 0, 0]) # camara apuntando para abajo
    
else:
    ret, rvecIni, tvecIni, inliers = cv2.solvePnPRansac(calibPts, xIm, cameraMatrix, distCoeffs)

xpr, ypr, c = cl.inverse(xIm, rvecIni, tvecIni, cameraMatrix, distCoeffs, model=model)

plt.plot(xpr, ypr, '*')

# %%

std = 2.0
# output file
extrinsicParamsOutFile = extrinsicFolder + camera + model + "intrinsicParamsAP"
extrinsicParamsOutFile = extrinsicParamsOutFile + str(std) + ".npy"
Ci = np.repeat([ std**2 * np.eye(2)],n*m, axis=0).reshape(n,m,2,2)

params = dict()
params["n"] = n
params["m"] = m
params["imagePoints"] = xIm.reshape((1,1,-1,2)) 
params["model"] = model
params["chessboardModel"] = calibPts[:,:2].reshape((1,-1,2))
params["Cccd"] = Ci
params["Cf"] = Cf
params["Ck"] = Ck
params["Crt"] = False
#params = [n, m, xIm.reshape((1,1,-1,2)),model, calibPts[:,:2].reshape((1,-1,2)), Ci]

Xext0 = bl.ext2flat(rvecIni, tvecIni)

# %% ahora arranco con Metropolis, primera etapa
stdAng = np.pi / 180 * 0.3 # 0.3 grados de std
covar0 = np.diag([stdAng,stdAng,stdAng, 0.1, 0.1, 0.1])**2
sampleador = sts.multivariate_normal(Xext0, covar0)

mH = bl.metHasExt(Xint, Ns, params, sampleador, camPrior, W)
E0 = mH.etotalExt(Xext0)

dims = 6 # cantidad de parametros de pose
Nini = 3**dims

paraMuest = list() # np.zeros((Nmuestras, Xint.shape[0]))
errorMuestras = list() # np.zeros(Nmuestras)

paraMuest.append(Xext0)
errorMuestras.append(E0)

# primero saco muestras solo cambiando el mean
# estas se descartan son el burn in period
# corro eta parte hata que ve en el grafico que ya no se desplaza demasiado
for i in range(1, Nini):
    paM, erM = mH.nuevo(paraMuest[-1], errorMuestras[-1])
    paraMuest.append(paM)
    errorMuestras.append(erM)
    
    sampleador.mean = paraMuest[-1]


# grafico para ver que haya pasado el burn in period
plt.figure()
plt.plot(paraMuest - paraMuest[-1])




# %%
import corner 

paraMuest = list() # np.zeros((Nmuestras, Xint.shape[0]))
errorMuestras = list() # np.zeros(Nmuestras)
covMuestras = list()
covMuestras.append(covar0)

paraMuest.append(paM)
errorMuestras.append(erM)

Ntot = np.int(6.5**dims)

# segunda tanda de muestras solo cambiando el mean
# estas ya se usan en la tirada definitva
for i in range(1, Ntot):
    paM, erM = mH.nuevo(paraMuest[i-1], errorMuestras[i-1])
    paraMuest.append(paM)
    errorMuestras.append(erM)
    
    sampleador.mean = paraMuest[-1]
    covar = np.cov(np.array(paraMuest).T)
    if np.linalg.matrix_rank(covar) == dims:
        # leave this loop if covariance matrix has rank 6
        covMuestras.append(covar)
        break

# muestras cambiando el mean y covarianza, arranco desde donde se dejó
for i in range(i, Ntot):
    paM, erM = mH.nuevo(paraMuest[-1], errorMuestras[-1])
    paraMuest.append(paM)
    errorMuestras.append(erM)
    
    sampleador.mean = paraMuest[-1]
    sampleador.cov = np.cov(np.array(paraMuest).T)
    
    covMuestras.append(sampleador.cov)



plt.figure()
covarianzasFlattened = np.array(covMuestras).reshape((-1, dims**2))
plt.plot(covarianzasFlattened - covarianzasFlattened[-1])

paraMuest = np.array(paraMuest)
corner.corner(paraMuest,50)





results = dict()
results['Nsamples'] = paraMuest.shape[0]
results['paramsMU'] = np.mean(paraMuest,0)
results['paramsVAR'] = np.cov(paraMuest.T)
results['paramsMUvar'] = results['paramsVAR'] / results['Nsamples']
results['paramsVARvar'] = bl.varVarN(results['paramsVAR'], results['Nsamples'])
results['Ns'] = Ns





# %% SAVE CALIBRATION
np.save(extrinsicResultsFile, results)

# tambien las muestras de la corrida solo por las dudas
np.save("/home/sebalander/Documents/datosMHextrinsicstereo.npy", paraMuest)





# %%

mu2Covar = covar2 / Nmuestras
extr = np.max(np.abs(mu2Covar))
plt.matshow(mu2Covar, cmap='RdBu', vmin=-extr, vmax=extr)

print('error relativo del mu')
eRelMu = np.abs(mu2Covar / (mu2.reshape((-1,1)) * mu2.reshape((1,-1))))
plt.matshow(np.log(eRelMu), cmap='inferno')

wm, vm = ln.eigh(mu2Covar)
plt.matshow(vm, cmap='RdBu', vmin=-1, vmax=1)

wMu, vMu = ln.eigh(eRelMu)
np.prod(wMu)
plt.matshow(vMu, cmap='RdBu')

print(np.sqrt(np.diag(mu2Covar)) / mu2)

covar2Covar = bl.varVarN(covar2, Nmuestras)
print('error relativo de la covarianza')
eRelCovar = covar2Covar / (covar2.reshape((-1,1)) * covar2.reshape((1,-1)))
wCo, vCo = ln.eigh(eRelCovar)
np.prod(wCo)
plt.matshow(vCo, cmap='RdBu')

print(eRelCovar)
print(np.sqrt(np.mean(eRelCovar**2)))

resultsAP = dict()

resultsAP['Nsamples'] = Nmuestras
resultsAP['paramsMU'] = mu2
resultsAP['paramsVAR'] = covar2
resultsAP['paramsMUvar'] = mu2Covar
resultsAP['paramsVARvar'] = covar2Covar
resultsAP['Ns'] = Ns

# %%
save = False
if save:
    np.save(extrinsicParamsOutFile, resultsAP)

load = False
if load:
    resultsAP = np.load(extrinsicParamsOutFile).all()  # load all objects
    
    Nmuestras = resultsAP["Nsamples"]
    mu2 = resultsAP['paramsMU']
    covar2 = resultsAP['paramsVAR']
    mu2Covar = resultsAP['paramsMUvar']
    covar2Covar = resultsAP['paramsVARvar']
    Ns = resultsAP['Ns']

# %%

xm, ym, Cm = cl.inverse(xIm, mu2[:3], mu2[3:], cameraMatrix,
                        distCoeffs, model, params['Cccd'], params['Cf'], params['Ck'], Crt=covar2)

fig = plt.figure()
ax = fig.gca()

ax.plot(calibPts[:,0], calibPts[:,1], '+')
ax.plot(camPrior[0], camPrior[1], 'x')
ax.plot(mu2[3], mu2[4], '.r')
cl.plotEllipse(ax, mu2Covar[3:5,3:5], mu2[3], mu2[4], 'r')
cl.plotPointsUncert(ax, Cm, xm, ym, 'b')

# %% calculate mahalanobis distance


mahErr = bl.errorCuadraticoImagen(mu2, Xint, Ns, params, 0, mahDist=True)

# hago el cumulativo
count = np.arange(m + 1) / m
indSort = np.argsort(mahErr)
mahCum = np.concatenate(([0], mahErr[indSort]))

#mahRange = np.linspace(0, mahCum[-1], 100)
chiError = sts.chi2(2)
chi2cdf = chiError.cdf(mahCum)
chi2Invese = chiError.ppf(count)

# %%
plt.figure()
plt.step(mahCum, count, where='post', label='data')
plt.step(chi2Invese, count, where='post')
plt.step(mahCum, chi2cdf, where='post')
plt.xlabel('Mah Dist squared')
plt.legend()
plt.xscale('log')


# %% difference of mah dist with inverse of distribution
mahTeo = np.empty_like(mahErr)
mahTeo[indSort] = chi2Invese[1:]  # distancia a la que "tendria que haber dado"

# ratio to calculate teoretical position
ratio = np.sqrt(mahTeo / mahErr)

Xproj = np.array([xm, ym]).T 
Xerror = calibPts[:,:2]- Xproj # vector error
calibTeo = Xproj + Xerror * ratio.reshape((-1,1)) # "theoretical position"
# rearrange to plot
xDif = np.array([calibTeo[:,0], calibPts[:,0]]).T
yDif = np.array([calibTeo[:,1], calibPts[:,1]]).T

cameradist = ln.norm(calibPts - mu2[3:6], axis=1)

fig = plt.figure()
ax = fig.gca()

ax.plot(calibPts[:,0], calibPts[:,1], '+')
ax.plot(camPrior[0], camPrior[1], 'x')
ax.plot(mu2[3], mu2[4], '.r')
cl.plotEllipse(ax, mu2Covar[3:5,3:5], mu2[3], mu2[4], 'r')
cl.plotPointsUncert(ax, Cm, xm, ym, 'b')
ax.plot(xDif.T, yDif.T, '-r')
