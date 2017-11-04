#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 20:35:03 2017

test functions that propagate uncertanty

@author: sebalander
"""
# %%
#import time
#import timeit

import numpy as np
import numdifftools as ndf
from calibration import calibrator as cl
import matplotlib.pyplot as plt

from dev import bayesLib as bl
from importlib import reload


# %% LOAD DATA
# input
plotCorners = False
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
modelos = ['poly', 'rational', 'fisheye']
model = modelos[2]

imagesFolder = "/home/sebalander/Desktop/Code/sebaPhD/resources/intrinsicCalib/" + camera + "/"
cornersFile =      imagesFolder + camera + "Corners.npy"
patternFile =      imagesFolder + camera + "ChessPattern.npy"
imgShapeFile =     imagesFolder + camera + "Shape.npy"

# model data files
distCoeffsFile =   imagesFolder + camera + model + "DistCoeffs.npy"
linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
tVecsFile =        imagesFolder + camera + model + "Tvecs.npy"
rVecsFile =        imagesFolder + camera + model + "Rvecs.npy"

# %% load data
imagePoints = np.load(cornersFile)
n = len(imagePoints)  # cantidad de imagenes
#indexes = np.arange(n)
#
#np.random.shuffle(indexes)
#indexes = indexes
#
#imagePoints = imagePoints[indexes]
#n = len(imagePoints)  # cantidad de imagenes

chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
#images = glob.glob(imagesFolder+'*.png')

# Parametros de entrada/salida de la calibracion
objpoints = np.array([chessboardModel]*n)
m = chessboardModel.shape[1]  # cantidad de puntos

# load model specific data
distCoeffs = np.load(distCoeffsFile)
cameraMatrix = np.load(linearCoeffsFile)
rVecs = np.load(rVecsFile)#[indexes]
tVecs = np.load(tVecsFile)#[indexes]

#
#
## %% simplest test
#j = 0
#
#plt.figure()
#plt.scatter(chessboardModel[0,:,0], chessboardModel[0,:,1],
#            marker='+', c='k', s=100)
#
#for j in range(0,n):
#    xm, ym, Cm = cl.inverse(imagePoints[j,0], rVecs[j], tVecs[j],                                        
#                            cameraMatrix, distCoeffs, model)
#    
#    plt.scatter(xm, ym, marker='x', c='b')
#
#
## %%
#ep = 0.01  # realtive standard deviation in parameters
#
## matriz de incerteza de deteccion en pixeles
#Cccd = np.repeat([np.eye(2,2)*0.1**2],imagePoints[j,0].shape[0], axis=0)
#Cf = np.diag(cameraMatrix[[0,1,0,1],[0,1,2,2]] * ep)**2
#Ck = np.diag((distCoeffs.reshape(-1) * ep )**2)
#Cr = np.diag((rVecs[j].reshape(-1) * ep )**2)
#Ct = np.diag((tVecs[j].reshape(-1) * ep )**2)
#
#
#Crt = [Cr, Ct]
#
#
## %%
#xpp, ypp, Cpp = cl.ccd2hom(imagePoints[j,0], cameraMatrix, Cccd, Cf)
#
#
## %% undistort
#xp, yp, Cp = cl.homDist2homUndist(xpp, ypp, distCoeffs, model, Cpp, Ck)
#
#
## %% project to plane z=0 from homogenous
#xm, ym, Cm = cl.xypToZplane(xp, yp, rVecs[j], tVecs[j], Cp, Crt)
#
#Caux = Cm
## %%
#xm, ym, Cm = cl.inverse(imagePoints[j,0], rVecs[j], tVecs[j],                                        
#                        cameraMatrix, distCoeffs, model,
#                        Cccd, Cf, Ck, Crt)
#
#fig = plt.figure()
#ax = fig.gca()
#ax.scatter(chessboardModel[0,:,0], chessboardModel[0,:,1])
#cl.plotPointsUncert(ax, Cm, xm, ym, 'k')
#
#
#
#er = [xm, ym] - chessboardModel[0,:,:2].T
##
## %%
#statement1 = '''
#p1 = np.tensordot(er, Cm, axes=(0,1))[range(54),range(54)]
#p2 = p1.dot(er)[range(54),range(54)]
#'''
#
## %%
#statement2 = '''
#Er = np.empty_like(xp)
#for i in range(len(xp)):
#    Er[i] = er[:,i].dot(Cm[i]).dot(er[:,i])
#'''
#
## %%
#statement3 = '''
#q1 = [np.sum(Cm[:,:,0]*er.T,1), np.sum(Cm[:,:,1]*er.T,1)];
#q2 = np.sum(q1*er,0)
#'''
## %%
#
#t1 = timeit.timeit(statement1, globals=globals(), number=10000) / 1e4
#
#t2 = timeit.timeit(statement2, globals=globals(), number=10000) / 1e4
#
#t3 = timeit.timeit(statement3, globals=globals(), number=10000) / 1e4
#
#print(t1/t3, t2/t3)


## %%
#ep = 0.0001  # relative standard deviation in parameters
#
## matriz de incerteza de deteccion en pixeles
#Cccd = np.repeat([np.eye(2,2)*0.1**2],imagePoints[j,0].shape[0], axis=0)
#Cf = np.diag(cameraMatrix[[0,1,0,1],[0,1,2,2]] * ep)**2
#Ck = np.diag((distCoeffs.reshape(-1) * ep )**2)
#
#
#fig = plt.figure()
#ax = fig.gca()
#
#ax.scatter(chessboardModel[0,:,0], chessboardModel[0,:,1],
#            marker='+', c='k', s=100)
#
#for j in range(0,n,3):
#    Cr = np.diag((rVecs[j].reshape(-1) * ep )**2)
#    Ct = np.diag((tVecs[j].reshape(-1) * ep )**2)
#    
#    Crt = [Cr, Ct]
#    
#    xm, ym, Cm = cl.inverse(imagePoints[j,0], rVecs[j], tVecs[j],                                        
#                                         cameraMatrix, distCoeffs, model,
#                                         Cccd, Cf, Ck, Crt)
#    
#    cl.plotPointsUncert(ax, Cm, xm, ym, 'k')



# %% poniendo ruido

def sacoParmams(pars):
    r = pars[4:7]
    t = pars[7:10]
    d = pars[10:]
    
    if model is 'poly' or model is 'rational':
        d = np.concatenate([d[:2],np.zeros_like(d[:2]), d[2].reshape((1,-1))])
    
    if len(pars.shape) > 1:
        k = np.zeros((3, 3, pars.shape[1]), dtype=float)
    else:
        k = np.zeros((3, 3), dtype=float)
    k[[0,1,2,2],[0,1,0,1]] = pars[:4]
    k[2,2] = 1
    
    return r.T, t.T, k.T, d.T

def sdt2covs(covAll):
    Cf = np.diag(covAll[:4])
    Crt = np.diag(covAll[4:10])
    
    Ck = np.diag(covAll[10:])
    return Cf, Ck, Crt


j = 0 # elijo una imagen para trabajar

import glob
imagenFile = glob.glob(imagesFolder+'*.png')[j]
plt.figure()
plt.imshow(plt.imread(imagenFile), origin='lower')
plt.scatter(imagePoints[j,0,:,0], imagePoints[j,0,:,1])

nSampl = int(3e4)  # cantidad de samples

ep = 1e-5 # relative standard deviation in parameters
stdIm = 1e-2

sacaCeros = {'poly' : [0,1,4],
          'rational' : [0,1,4,5,6],
          'fisheye' : range(4)}

# apilo todos los parametros juntos
parsAll = np.hstack([cameraMatrix[[0,1,0,1],[0,1,2,2]],
                    rVecs[j].reshape(-1),
                    tVecs[j].reshape(-1),
                    distCoeffs.reshape(-1)[sacaCeros[model]]])

# standard deviations
sAll = ep * parsAll

Cccd = np.repeat([np.eye(2) * stdIm**2], imagePoints[j,0].shape[0], axis=0)
Cf, Ck, Crt = sdt2covs(sAll**2)

# genero todo el ruido
noiseParam = np.random.randn(nSampl, sAll.shape[0])
noisePos = np.random.randn(nSampl, imagePoints[j,0].shape[0], imagePoints[j,0].shape[1])

# dejo los valores preparados
parsGen = parsAll.reshape((1, -1)) + noiseParam * sAll.reshape((1, -1))
posIgen = imagePoints[j,0].reshape((1,-1,2)) + noisePos * stdIm
posMap = np.zeros_like(posIgen)


rG, tG, kG, dG = sacoParmams(parsGen.T)


# %% TEST EACH STEP. STEP 1: CCD TO DISTORTED HOMOGENOUS ===================

#### COMPARACION DE JACOBIANOS ####
# parte ANALITICA jacobianos
Jd_i, Jd_k = cl.ccd2homJacobian(imagePoints[j,0], cameraMatrix)

# parte JACOBIANOS NUMERICOS
def step1vsX(X, cameraMatrix):
    imagePoints = X.T
    xpp, ypp, Cpp = cl.ccd2hom(imagePoints, cameraMatrix)
    
    return np.array([xpp, ypp])

def step1vsParams(params, imagePoints):
    cameraMatrix = np.zeros((3,3))
    cameraMatrix[[0 , 1, 0, 1], [0, 1, 2, 2]] = params
    cameraMatrix[2,2] = 1
    
    xpp, ypp, Cpp = cl.ccd2hom(imagePoints, cameraMatrix)
    
    return np.array([xpp, ypp])

# calculo con derivada numerica
Jd_inumeric = ndf.Jacobian(step1vsX, order=4)(imagePoints[j,0].T, cameraMatrix).T

Jd_knumeric = ndf.Jacobian(step1vsParams, order=4,
                           method='central')(cameraMatrix[[0 , 1, 0, 1],
                                                       [0, 1, 2, 2]],
                                          imagePoints[j,0]).transpose((2,0,1))

indJacNon0 = [[0,1,2,3], [0,1,0,1]]
jkanal = Jd_k.T[indJacNon0].reshape(-1)
jknumDif = np.abs(Jd_knumeric.T[indJacNon0].reshape(-1) - jkanal)
jianal = np.diag(Jd_i)
jinum = np.abs(Jd_inumeric[:,[0,1],[0,1]] - jianal)

# COMPARO JACOBIANOS
plt.figure()
plt.title('Jacobianos relative error')
plt.plot(jkanal, jknumDif / np.abs(jkanal), '+', label='params')
plt.plot(jianal.reshape((-1,1)), (jinum / jianal).T, 'xk', label='posicion')
plt.yscale('log')
plt.legend(loc=0)

#### COMPARACION DE COVARIANZAS ####

# %% montecarlo vs analitico comparo covarianzas
mahalanobis = np.zeros([n, m])

for j in range(n):
    # parte ANALITICA propagacion de incerteza
    xpp, ypp, Cpp = cl.ccd2hom(imagePoints[j,0], cameraMatrix, Cccd=Cccd, Cf=Cf)
    
    posIgen = imagePoints[j,0].reshape((1,-1,2)) + noisePos * stdIm
    
    xyp = np.zeros((nSampl,m,2))
    # parte MONTE CARLO
    for i in range(nSampl):
        # posMap[i,:,0], posMap[i,:,1], _ = cl.ccd2hom(posIgen[i], cameraMatrix)
        xyp[i,:,0], xyp[i,:,1], _ = cl.ccd2hom(posIgen[i], kG[i])
#    # COMPARO VARIANZAS
#    # medias y varianzas de las nubes de puntos
#    posIMean = np.mean(posIgen, axis=0)
#    dif = (posIgen - posIMean).T
#    posIVar = np.sum([dif[0] * dif, dif[1] * dif], axis=-1) / (nSampl - 1)
#    posIVar = posIVar.transpose((2,0,1))
    
    xypMean = np.mean(xyp, axis=0)
    dif = (xyp - xypMean).T
    xypVar = np.sum([dif[0] * dif, dif[1] * dif], axis=-1).T / (nSampl - 1)
#    posMapVar = posMapVar.transpose((2,0,1))

    
    # mido la distancia mahalanobis
    mahalanobis[j] = [bl.varMahal(xypVar[i], nSampl, Cpp[i]) for i in range(m)]

plt.figure()
nh, bins, patches = plt.hist(np.reshape(mahalanobis,-1), 100, normed=True)
ch2pdf = bl.chi2.pdf(bins,3)
plt.plot(bins, ch2pdf)

# %%

#fig = plt.figure()
#ax = fig.gca()
#ax.set_title('pos generadas')
#ax.scatter(posIgen[:,:,0], posIgen[:,:,1], marker='.', c='b', s=1)
#cl.plotPointsUncert(ax, Cccd, imagePoints[j,0,:,0], imagePoints[j,0,:,1], 'k')
#cl.plotPointsUncert(ax, posIVar, posIMean[:,0], posIMean[:,1], 'b')

cAnal = Cpp[:, [0,1,0],[0,1,1]]
cDif = np.abs(xypVar[:, [0,1,0],[0,1,1]] - cAnal)
plt.plot(cAnal.T, cDif.T / np.linalg.norm(Cpp, axis=(1,2)), '+b')


fig = plt.figure()
ax = fig.gca()
ax.set_title('propagacion')
ax.scatter(posMap[:,:,0], posMap[:,:,1], marker='.', c='b', s=1)
cl.plotPointsUncert(ax, Cpp, xpp, ypp, 'k')
cl.plotPointsUncert(ax, xypVar, xypMean[:,0], xypMean[:,1], 'b')

xypVar / Cpp
xypMean[:,0] / xpp
xypMean[:,1] / ypp




# %% TEST EACH STEP. STEP 2: HOMOGENOUS UNDISTORTION =======================
reload(cl)
# parte ANALITICA propagacion de incerteza
xp, yp, Cp = cl.homDist2homUndist(xpp, ypp, distCoeffs, model, Cpp=Cpp, Ck=Ck)

# parte ANALITICA jacobianos
_, _, Jh_d, Jh_k =  cl.homDist2homUndist_ratioJacobians(xpp, ypp, distCoeffs, model)

# parte MONTE CARLO
# Genero los valores a usar
# matriz para rotoescalear el ruido
convEllip2 = np.array([cl.unit2CovTransf(c) for c in Cpp])
# aplico rotoescaleo
xypertub2 = (convEllip2.reshape((1,-1,2,2)) *
             noisePos.reshape((nSampl,-1,1,2))).sum(-1)
# aplico la perturbacion
xPPgen = xpp.reshape((1, -1)) + xypertub2[:,:,0]
yPPgen = ypp.reshape((1, -1)) + xypertub2[:,:,1]



xP = np.zeros_like(xPPgen)
yP = np.zeros_like(yPPgen)

# hago todos los mapeos de Monte Carlo, una iteracion por sample
for i in range(nSampl):
    xP[i], yP[i], _ = cl.homDist2homUndist(xPPgen[i], yPPgen[i], dG[i], model)


# parte JACOBIANOS NUMERICOS
def step2vsX(X, distCoeffs, model):
    xpp, ypp = X
    xp, yp, Cpp = cl.homDist2homUndist(xpp, ypp, distCoeffs, model)
    return np.array([xp, yp])

def step2vsParams(distCoeffs, X, model):
    xpp, ypp = X
    xp, yp, Cpp = cl.homDist2homUndist(xpp, ypp, distCoeffs, model)
    return np.array([xp, yp])

X = np.array([xpp, ypp])
Jh_dnumeric = ndf.Jacobian(step2vsX)(X, distCoeffs, model).T
Jh_knumeric = ndf.Jacobian(step2vsParams)(distCoeffs, X, model).transpose((2,0,1))


# COMPARO JACOBIANOS
plt.figure()
plt.title('Jacobianos')
plt.plot(Jh_d.flat, np.abs(Jh_dnumeric - Jh_d).reshape(-1), '+', label='posicion')
plt.plot(Jd_knumeric.flat, np.abs(Jd_knumeric - Jd_k).reshape(-1), 'x', label='params')
plt.legend(loc=0)
plt.yscale('log')

#plt.figure()
#plt.title('Jacobianos 2')
#plt.plot(Jh_knumeric.flat, Jh_k.flat, '+', label='posicion')


# COMPARO VARIANZAS
# saco media y varianza de cada nube de puntos
def mediaCovars(x, y):
    '''
    saco media y covarianza de x(n,m), y(n,m), el primer indice es de muestreo
    '''
    
    xm, ym = np.mean([x,y], axis=1)
    dx, dy = (x - xm), (y - ym)
    Cxy = dx*dy
    
    return xm, ym, np.array([[dx**2, Cxy],[Cxy, dy**2]]).mean(2).T

xPPm, yPPm, varPP = mediaCovars(xPPgen, yPPgen)
xPm, yPm, varP = mediaCovars(xP, yP)


#fig = plt.figure()
#ax = fig.gca()
#ax.set_title('pos generadas')
#ax.plot(xPPgen, yPPgen, '.b')
#cl.plotPointsUncert(ax, Cpp, xpp, ypp, 'k')
#cl.plotPointsUncert(ax, varPP, xPPm, yPPm, 'b')


fig = plt.figure()
ax = fig.gca()
ax.set_title('propagacion')
ax.scatter(xP, yP, marker='.', c='b', s=1)
cl.plotPointsUncert(ax, Cp, xp, yp, 'k')
cl.plotPointsUncert(ax, varP, xPm, yPm, 'b')


# %% TEST EACH STEP. STEP 3: PROJECT TO MAP, UNDO ROTOTRASLATION
# parte ANALITICA propagacion de incerteza
xm, ym, Cm = cl.xypToZplane(xp, yp, rVecs[j], tVecs[j], Cp=Cp, Crt=Crt)

# parte ANALITICA jacobianos
JXm_Xp, JXm_rtV =  cl.jacobianosHom2Map(xp, yp, rVecs[j], tVecs[j])

# parte MONTE CARLO
# dejo los valores preparados
# matriz para rotoescalear el ruido
convEllip3 = np.array([cl.unit2CovTransf(c) for c in Cp])
# aplico rotoescaleo
xypertub3 = (convEllip3.reshape((1,-1,2,2)) *
             noisePos.reshape((nSampl,-1,1,2))).sum(-1)

xPgen = xp.reshape((1, -1)) + xypertub3[:,:,0]
yPgen = yp.reshape((1, -1)) + xypertub3[:,:,1]
xM = np.zeros_like(xPgen)
yM = np.zeros_like(yPgen)

# hago todos los mapeos de Monte Carlo, una iteracion por sample
for i in range(nSampl):
    xM[i], yM[i], _ = cl.xypToZplane(xPgen[i], yPgen[i], rG[i], tG[i])


# parte JACOBIANOS NUMERICOS
def step3vsX(X, rt):
    xp, yp = X
    xm, ym, Cm= cl.xypToZplane(xp, yp, rt[:3], rt[3:])
    return np.array([xm, ym])

def step3vsParams(rt, X):
    return step3vsX(X, rt)

X = np.array([xp, yp])
rt = np.concatenate([rVecs[j], tVecs[j]])
JXm_Xpnumeric = ndf.Jacobian(step3vsX)(X, rt)
JXm_rtVnumeric = ndf.Jacobian(step3vsParams)(rt, X)


# COMPARO JACOBIANOS
plt.figure()
plt.title('Jacobianos')
plt.plot(JXm_Xp.flat, np.abs(JXm_Xpnumeric - JXm_Xp).reshape(-1), '+', label='posicion')
plt.plot(JXm_rtVnumeric.flat, np.abs(JXm_rtVnumeric - JXm_rtV).reshape(-1), 'x', label='params')
plt.legend(loc=0)
plt.yscale('log')



# COMPARO VARIANZAS
# saco media y varianza de cada nube de puntos
xPm, yPm, varP = mediaCovars(xPgen, yPgen)
xMm, yMm, varM = mediaCovars(xM, yM)


#fig = plt.figure()
#ax = fig.gca()
#ax.set_title('pos generadas')
#ax.plot(xPgen, yPgen, '.b')
#cl.plotPointsUncert(ax, Cp, xp, yp, 'k')
#cl.plotPointsUncert(ax, varP, xPm, yPm, 'b')


fig = plt.figure()
ax = fig.gca()
ax.set_title('propagacion')
ax.scatter(xM, yM, marker='.', c='b', s=1)
cl.plotPointsUncert(ax, Cm, xm, ym, 'k')
cl.plotPointsUncert(ax, varM, xMm, yMm, 'b')



# %% TESTEO LOS TRES PASOS JUNTOS 
# hago todos los mapeos de Monte Carlo
for posM, posI, pars in zip(posMap, posIgen, parsGen):
    r, t, k, d = sacoParmams(pars)
    posM[:,0], posM[:,1], _ = cl.inverse(posI, r, t, k, d, model)


# %% saco media y varianza de cada nube de puntos
posMapMean = np.mean(posMap, axis=0)
dif = (posMap - posMapMean).T
posMapVar = np.mean([dif[0] * dif, dif[1] * dif], axis=-1).T



# %% mapeo propagando incerteza los valores con los que comparar
xmJ, ymJ, CmJ = cl.inverse(imagePoints[j,0], rVecs[j], tVecs[j],                                        
                                     cameraMatrix, distCoeffs, model,
                                     Cccd, Cf, Ck, Crt)

XJ = np.array([xmJ, ymJ]).T

# %% grafico
xm, ym = posMap.reshape(-1,2).T

fig = plt.figure()
ax = fig.gca()

ax.scatter(chessboardModel[0,:,0], chessboardModel[0,:,1],
            marker='+', c='k', s=100)

cl.plotPointsUncert(ax, CmJ, xmJ, ymJ, 'k')

ax.scatter(xm, ym, marker='.', c='b', s=1)
cl.plotPointsUncert(ax, posMapVar, posMapMean[:,0], posMapMean[:,1], 'b')


# %% pruebo la funcion error
stdIm = 1e-6
Cccd = np.repeat([np.eye(2) * stdIm**2], imagePoints[j,0].shape[0], axis=0)

# hago el mapeo
xmJ, ymJ, CmJ = cl.inverse(imagePoints[j,0], rVecs[j], tVecs[j],                                        
                                     cameraMatrix, distCoeffs, model,
                                     Cccd, Cf, Ck, Crt*0.5e8)

fig = plt.figure()
ax = fig.gca()
ax.scatter(chessboardModel[0,:,0], chessboardModel[0,:,1],
            marker='+', c='k', s=100)
cl.plotPointsUncert(ax, CmJ, xmJ, ymJ, 'k')


XJ = np.array([xmJ, ymJ]).T






