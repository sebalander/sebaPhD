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
distCoeffsFile =   imagesFolder + camera + model + "DistCoeffs.npy"
linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
tVecsFile =        imagesFolder + camera + model + "Tvecs.npy"
rVecsFile =        imagesFolder + camera + model + "Rvecs.npy"

# model data files
intrinsicParamsOutFile = imagesFolder + camera + model + "intrinsicParamsML"
intrinsicParamsOutFile = intrinsicParamsOutFile + "0.5.npy"

# calibration points
calibptsFile = "/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/2016-11-13 medicion/calibrExtr/puntosCalibracion.txt"

# load model specific data
distCoeffs = np.load(distCoeffsFile)
cameraMatrix = np.load(linearCoeffsFile)

# load intrinsic calib uncertainty
resultsML = np.load(intrinsicParamsOutFile).all()  # load all objects
#Nmuestras = resultsML["Nsamples"]
Xint = resultsML['paramsMU']
covarDist = resultsML['paramsVAR']
#muCovar = resultsML['paramsMUvar']
#covarCovar = resultsML['paramsVARvar']
Ns = resultsML['Ns']
cameraMatrixOut, distCoeffsOut = bl.flat2int(Xint, Ns)

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
ret, rvecIni, tvecIni, inliers = cv2.solvePnPRansac(calibPts, xIm, cameraMatrixOut, distCoeffsOut)

xpr, ypr, c = cl.inverse(xIm, rvecIni, tvecIni, cameraMatrixOut, distCoeffsOut, model=model)

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
extrinsicParamsOutFile = imagesFolder + camera + model + "extrinsicParamsAP"
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

#E0 = etotal(Xext0, Ns, Xint, params)
#difE = 6 * np.log(10)
#Emax = E0 + difE

## %%
#cotas = np.array([[2.4944859, 2.82924],  # [0]
#                  [1.11983, 1.4141],    # [1]
#                  [-0.5572, -0.2562328], # [2]
#                  [-0.11, 0.596],    # [3]
#                  [1.4521, 2.7999], # [4]
#                  [10.0466, 16.0076]])  # [5] 
#
#errores = np.zeros((6,2))
#
#for i in [4]: # range(6)
#    Xext2 = dc(Xext0)
#    Xext2[i] = cotas[i,0]
#    errores[i,0] = etotal(Xext2, Ns, Xint, params)
#
#    Xext2[i] = cotas[i,1]
#    errores[i,1] = etotal(Xext2, Ns, Xint, params)
#    print((errores[i] - E0) / difE)
#
##print((errores - E0) / difE)
#
#print(cotas[:,0] < Xext0)
#
#print(Xext0 < cotas[:,1])

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

## %% sampleo simplemente para sacar una pdf desde donde proponer
#
#sampleador = pdfSampler(cotas[:,0],  cotas[:,1] - cotas[:,0])
#
## saco 1000 muestras
#Xsamples = sampleador.rvs(100)
#esamples = np.array([etotal(x, Ns, Xint, params) for x in Xsamples])
#plt.hist(esamples)
##meErr = (np.max(esamples) + np.min(esamples)) / 2
## pongo una constante que me baja la escala porque me dan nros muy grandes
## eto me deforma la gaussiana per es un comienzo
#probsamples = np.exp(- esamples * 1e-2 / 2)
#
#for i in range(6):
#    plt.figure()
#    plt.hist(Xsamples[:,i],100, weights=probsamples, normed=True)
#
## %%
## media pesada por la probabilidad
#xaverg = np.average(Xsamples, axis=0, weights=probsamples)
#eaverg = etotal(xaverg, Ns, Xint, params)
#paverg = np.exp(- eaverg * 1e-2 / 2 )
## covarianza pesada por la probabilidad
#xcovar = np.cov(Xsamples.T, ddof=0, aweights=probsamples)
#
#ln.eigvals(xcovar)
#
## %% segunda ronda de proponer samples para sacar una pdf
## repetir este bloque varias veces me ayuda a ir corriendo la pdf
## sampleo de una pdf, no uniforme como antes
#
#sampleador = sts.multivariate_normal(xaverg, xcovar)
#
## saco 1000 muestras
#Xsamples2 = sampleador.rvs(1000)
#esamples2 = np.array([etotal(x, Ns, Xint, params) for x in Xsamples2])
#plt.figure(6)
#plt.hist(esamples2, 30)
##meErr = (np.max(esamples) + np.min(esamples)) / 2
## pongo una constante que me baja la escala porque me dan nros muy grandes
## eto me deforma la gaussiana per es un comienzo
#probsamples2 = np.exp(- (esamples2 - eaverg2) / 2)
#
#for i in range(7,13):
#    plt.figure(i)
#    plt.hist(Xsamples2[:,i-7],100, weights=probsamples2, normed=True)
#
## media pesada por la probabilidad
#xaverg2 = np.average(Xsamples2, axis=0, weights=probsamples2)
#xcovar2 = np.cov(Xsamples2.T, ddof=0, aweights=probsamples2)
#eaverg2 = etotal(xaverg2, Ns, Xint, params)
#
#ln.eigvals(xcovar2)

'''
aca los resultados para no tener que correr todo de nuevo:
    
xaverg = np.array([  2.63083507,   1.17600421,  -0.25493875,   0.58739594,
         2.77222218,  14.43985677])

xcovar = np.array([[  2.92403190e-08,   1.29185352e-08,   4.04071807e-08,
          7.73672211e-07,   9.08206166e-08,   7.74763196e-07],
       [  1.29185352e-08,   2.23554054e-08,   2.57425254e-08,
          1.19069512e-07,  -5.50744593e-08,   6.22587094e-08],
       [  4.04071807e-08,   2.57425254e-08,   8.09630927e-08,
          1.48667380e-06,   1.88325788e-07,   5.77312247e-07],
       [  7.73672211e-07,   1.19069512e-07,   1.48667380e-06,
          3.62999187e-05,   6.32766340e-06,   1.52757301e-05],
       [  9.08206166e-08,  -5.50744593e-08,   1.88325788e-07,
          6.32766340e-06,   1.37343873e-06,   2.15822184e-06],
       [  7.74763196e-07,   6.22587094e-08,   5.77312247e-07,
          1.52757301e-05,   2.15822184e-06,   3.17442496e-05]])
'''


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


# saco la media pesada y la covarinza si peso porque asi es metropolis
mu1 = np.average(paraMuest, 0, weights=probMuestras)
covar1 = np.cov(paraMuest.T, ddof=0, aweights=probMuestras)

ln.eigvals(covar1)

# %%

#plt.plot(paraMuest-np.mean(paraMuest, axis=0))

for i in range(6):
    plt.figure()
    plt.hist(paraMuest[200:,i],30)



# %% ultima etapa de metropolis, no se actualiza online la pds sampling
sampleador = sts.multivariate_normal(mu1, covar1)

Nmuestras = int(2e5)

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

cameraMatrixOut, distCoeffsOut = bl.flat2int(mu2, Ns)

mu2Covar = covar2 / Nmuestras
covar2Covar = bl.varVarN(covar2, Nmuestras)

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
    np.save(extrinsicParamsOutFile, resultsML)

load = False
if load:
    resultsML = np.load(extrinsicParamsOutFile).all()  # load all objects
    
    Nmuestras = resultsML["Nsamples"]
    mu3 = resultsML['paramsMU']
    covar3 = resultsML['paramsVAR']
    mu3Covar = resultsML['paramsMUvar']
    covar3Covar = resultsML['paramsVARvar']
    Ns = resultsML['Ns']
    
    cameraMatrixOut, distCoeffsOut = bl.flat2int(mu3, Ns)


# %%
#
#
## el error relativo aproximadamente
#np.sqrt(np.diag(covar3)) / mu3
#
#
#
#print(np.concatenate([[xaverg], [mu2], [mu3]], axis=0).T)
#
#
## %% ahora le pongo un prior en los parametros extrinsecos
#
## pongo 3m de sigma en x,y y 20cm en z
#camPrior
#W = 1 / np.array([3,3,0.2])**2
#
#def ePrior(Xext):
#    return W.dot((Xext[3:] - camPrior)**2)
#
## error total
#def etotal(Xext, Ns, Xint, params):
#    '''
#    calcula el error total como la suma de los errore de cada punto en cada
#    imagen
#    '''
#    ep = ePrior(Xext)
#    
#    return ep + bl.errorCuadraticoImagen(Xext, Xint, Ns, params, 0).sum()
#
#ePrior(mu3)
#etotal(mu3, Ns, Xint, params)
#
## saco una gaussiana que es el producto de lo que se obtiene por ML y la priori
## solo en la posicion
#covar4 = np.zeros_like(covar3)
#covar4[:3,:3] = covar3[:3,:3] # para angulos dejo como esta
#S1 = ln.inv(covar3[3:,3:])
#S2 = np.diag(W)
#covar4[3:,3:] = ln.inv(S1 + S2)
#
#mu4 = dc(mu3)
#mu4[3:] = covar4[3:,3:].dot(S1.dot(mu3[3:]) + S2.dot(camPrior)) # cambio la media pesandola
## parece que no va a afectar casi nada porque el likelihood pesa muchisimo mas,
## la gaussiana es mucho mas finita.
#
## %%
#sampleador = sts.multivariate_normal(mu5, covar5)
#
#Nmuestras = int(1e6)
#
#generados = 0
#aceptados = 0
#avance = 0
#retroceso = 0
#
#paraMuest5 = np.zeros((Nmuestras,6))
#errorMuestras5 = np.zeros(Nmuestras)
#
## primera
#start = sampleador.rvs() # sampleador() # rn(8) * intervalo + cotas[:,0]
#startE = etotal(start, Ns, Xint, params)
#paraMuest5[0], errorMuestras5[0] = (start, startE)
#
#
#
#tiempoIni = time.time()
#for i in range(1, Nmuestras):
#    paraMuest5[i], errorMuestras5[i] = nuevo(paraMuest5[i-1], errorMuestras5[i-1])
#    tiempoNow = time.time()
#    Dt = tiempoNow - tiempoIni
#    frac = i / Nmuestras
#    DtEstimeted = (tiempoNow - tiempoIni) / frac
#    stringTimeEst = time.asctime(time.localtime(tiempoIni + DtEstimeted))
#    print('Transcurrido: %.2fmin. Avance %.4f. Tfin: %s'
#          %(Dt/60, frac, stringTimeEst) )
#
##
##for i in range(6):
##    plt.figure()
##    plt.hist(paraMuest5[:,i],30)
#
#corner.corner(paraMuest5, 50)
#
## %%
#mu5 = np.mean(paraMuest5, axis=0)
#covar5 = np.cov(paraMuest5.T)
#cameraMatrixOut, distCoeffsOut = bl.flat2int(mu5, Ns)
#
#mu5Covar = covar5 / Nmuestras
#covar5Covar = bl.varVarN(covar5, Nmuestras)
#
#resultsAP = dict()
#
#resultsAP['Nsamples'] = Nmuestras
#resultsAP['paramsMU'] = mu5
#resultsAP['paramsVAR'] = covar5
#resultsAP['paramsMUvar'] = mu5Covar
#resultsAP['paramsVARvar'] = covar5Covar
#resultsAP['Ns'] = Ns
#
## %%
#extrinsicParamsOutFile = imagesFolder + camera + model + "extrinsicParamsAP"
#extrinsicParamsOutFile = extrinsicParamsOutFile + str(std) + ".npy"
#
#save = False
#if save:
#    np.save(extrinsicParamsOutFile, resultsAP)
#
#load = False
#if load:
#    resultsAP = np.load(extrinsicParamsOutFile).all()  # load all objects
#    
#    Nmuestras = resultsAP["Nsamples"]
#    mu5 = resultsAP['paramsMU']
#    covar5 = resultsAP['paramsVAR']
#    mu5Covar = resultsAP['paramsMUvar']
#    covar5Covar = resultsAP['paramsVARvar']
#    Ns = resultsAP['Ns']
#    

# %%

xm, ym, Cm = cl.inverse(xIm, mu2[:3], mu2[3:], cameraMatrix,
                        distCoeffs, model, params['Cccd'], params['Cf'], params['Ck'], Crt=covar2)

fig = plt.figure()
ax = fig.gca()

ax.plot(calibPts[:,0], calibPts[:,1], '+')
ax.plot(camPrior[0], camPrior[1], 'x')
cl.plotPointsUncert(ax, Cm, xm, ym, 'b')
