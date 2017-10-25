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
modelos = ['poly', 'rational', 'fisheye']
model = modelos[2]

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
intrinsicParamsOutFile = imagesFolder + camera + model + "intrinsicParamsML"
intrinsicParamsOutFile = intrinsicParamsOutFile + "0.5.npy"

# calibration points
calibptsFile = "/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/2016-11-13 medicion/calibrExtr/puntosCalibracion.txt"

extrinsicFolder = "./resources/intrinsicCalib/" + camera + "/"


# load model specific data
#distCoeffs = np.load(distCoeffsFile)
#cameraMatrix = np.load(linearCoeffsFile)

# load intrinsic calib uncertainty
resultsML = np.load(intrinsicParamsOutFile).all()  # load all objects
#Nmuestras = resultsML["Nsamples"]
Xint = resultsML['paramsMU']
covarDist = resultsML['paramsVAR']
#muCovar = resultsML['paramsMUvar']
#covarCovar = resultsML['paramsVARvar']
Ns = resultsML['Ns']
cameraMatrix, distCoeffs = bl.flat2int(Xint, Ns)

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
camPrior = np.array([0,0,hcam]) # # PRIOR CAMERA COORDS EN METROS
# pongo 3m de sigma en x,y y 20cm en z
W = 1 / np.array([3, 3, 0.2])**2
def ePrior(Xext):
    '''
    el error de prior solo va para la posicion y no para los angulos porque no
    hay info facil de los angulos a priori
    '''
    return W.dot((Xext[3:] - camPrior)**2)

plt.figure()
plt.plot(calibPts[:,0], calibPts[:,1], '+')
plt.plot(camPrior[0], camPrior[1], 'x')

# %% determino la pose a prior
# saco una pose inicial con opencv
ret, rvecIni, tvecIni, inliers = cv2.solvePnPRansac(calibPts, xIm, cameraMatrix, distCoeffs)

xpr, ypr, c = cl.inverse(xIm, rvecIni, tvecIni, cameraMatrix, distCoeffs, model=model)

plt.plot(xpr, ypr, '*')

# %%

# error total
def etotal(Xext, Ns, Xint, params):
    '''
    calcula el error total como la suma de los errore de cada punto en cada
    imagen
    '''
    ep = ePrior(Xext)
    
    return ep + bl.errorCuadraticoImagen(Xext, Xint, Ns, params, 0).sum()

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
params["Cf"] = covarDist[:4,:4]
params["Ck"] = covarDist[4:,4:]
params["Crt"] = False
#params = [n, m, xIm.reshape((1,1,-1,2)),model, calibPts[:,:2].reshape((1,-1,2)), Ci]

Xext0 = bl.ext2flat(rvecIni, tvecIni)

# %% metropolis hastings

def nuevo(old, oldE):
    global generados
    global aceptados
    global avance
    global retroceso
    global sampleador
    
    # genero nuevo
    new = sampleador.rvs() # rn(8) * intervalo + cotas[:,0]
    newE = etotal(new, Ns, Xint, params)
    generados += 1
    print(generados, aceptados, avance, retroceso)
    
    # cambio de error
    deltaE = newE - oldE
    
    if deltaE < 0:
        aceptados +=1
        avance += 1
        print("avance directo")
        return new, newE # tiene menor error, listo
    else:
        # nueva opoertunidad, sampleo contra rand
        pb = np.exp(- deltaE / 2)
         
        if pb > rn():
            aceptados += 1
            retroceso += 1
            print("retroceso, pb=", pb)
            return new, newE # aceptado a la segunda oportunidad
        else:
            # vuelvo recursivamente al paso2 hasta aceptar
            print('rechazado, pb=', pb)
            new, newE = nuevo(old, oldE)
    
    return new, newE

class pdfSampler:
    '''
    para samplear de la uniforme entre las cotas puestas a mano
    '''
    def __init__(self, a, ab):
        self.a = a
        self.ab = ab
    
    def rvs(self, n=False):
        if not n:
            return np.random.rand(6) * self.ab + self.a
        else:
            return np.random.rand(n,6) * self.ab + self.a


# %% ahora arranco con Metropolis, primera etapa
mu0 = Xext0
covar0 = np.diag(mu0*1e-2)**2
sampleador = sts.multivariate_normal(mu0, covar0)

Nmuestras = int(1e3)

generados = 0
aceptados = 0
avance = 0
retroceso = 0

paraMuest = np.zeros((Nmuestras,6))
errorMuestras = np.zeros(Nmuestras)
probMuestras = np.zeros(Nmuestras)

# primera
start = sampleador.rvs() # dc(Xint)
startE = etotal(start, Ns, Xint, params)
paraMuest[0], errorMuestras[0] = nuevo(start, startE)

# primera parte saco 10 puntos asi como esta
for i in range(1, 20):
    paraMuest[i], errorMuestras[i] = nuevo(paraMuest[i-1], errorMuestras[i-1])
    sampleador.mean = paraMuest[i] # corro el centro


# ahora actualizo con media y covarianza moviles
for i in range(20, 200):
    paraMuest[i], errorMuestras[i] = nuevo(paraMuest[i-1], errorMuestras[i-1])
    
    sampleador.mean = sampleador.mean *0.7 + 0.3 * paraMuest[i]
    sampleador.cov = sampleador.cov * 0.7 + 0.3 * np.cov(paraMuest[i-10:i].T)


probMuestras[:i] = np.exp(- errorMuestras[:i] / 2)


# ahora actualizo pesando por la probabilidad
for i in range(200, Nmuestras):
    paraMuest[i], errorMuestras[i] = nuevo(paraMuest[i-1], errorMuestras[i-1])
    probMuestras[i] = np.exp(- errorMuestras[i] / 2)
    
    sampleador.mean = np.average(paraMuest[:i], 0, weights=probMuestras[:i])
    sampleador.cov = np.cov(paraMuest[:i].T, ddof=0, aweights=probMuestras[:i])


# saco la media pesada y la covarinza pesadas
mu1 = np.average(paraMuest, 0, weights=probMuestras)
covar1 = np.cov(paraMuest.T, ddof=0, aweights=probMuestras)

ln.eigvals(covar1)


# %% ultima etapa de metropolis, no se actualiza online la pds sampling
sampleador = sts.multivariate_normal(mu2, covar2)

Nmuestras = int(1e5)

generados = 0
aceptados = 0
avance = 0
retroceso = 0

paraMuest2 = np.zeros((Nmuestras,6))
errorMuestras2 = np.zeros(Nmuestras)

# primera
start = sampleador.rvs() # sampleador() # rn(8) * intervalo + cotas[:,0]
startE = etotal(start, Ns, Xint, params)
paraMuest2[0], errorMuestras2[0] = (start, startE)


tiempoIni = time.time()
for i in range(1, Nmuestras):
    paraMuest2[i], errorMuestras2[i] = nuevo(paraMuest2[i-1], errorMuestras2[i-1])
    
    tiempoNow = time.time()
    Dt = tiempoNow - tiempoIni
    frac = i / Nmuestras
    DtEstimeted = (tiempoNow - tiempoIni) / frac
    stringTimeEst = time.asctime(time.localtime(tiempoIni + DtEstimeted))
    print('Transcurrido: %.2fmin. Avance %.4f. Tfin: %s'
          %(Dt/60, frac, stringTimeEst) )

# new estimated covariance
mu2 = np.mean(paraMuest2, axis=0)
covar2 = np.cov(paraMuest2.T)

corner.corner(paraMuest2, 50)

# %%

mu2Covar = covar2 / Nmuestras
extr = np.max(np.abs(mu2Covar))
plt.matshow(mu2Covar, cmap='RdBu', vmin=-extr, vmax=extr)

print('error relativo del mu')
eRelMu = mu2Covar / (mu2.reshape((-1,1)) * mu2.reshape((1,-1)))

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
save = True
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

# %%
fig = plt.figure()
ax = fig.gca()

ax.plot(calibPts[:,0], calibPts[:,1], '+')
ax.plot(camPrior[0], camPrior[1], 'x')
ax.plot(mu2[3], mu2[4], '.r')
cl.plotEllipse(ax, mu2Covar[3:5,3:5], mu2[3], mu2[4], 'r')
cl.plotPointsUncert(ax, Cm, xm, ym, 'b')
