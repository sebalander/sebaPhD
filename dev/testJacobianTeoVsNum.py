#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:51:43 2018

comparo jacobianos teoricos y numericos

@author: sebalander
"""




# %%
import numpy as np
import glob
from calibration import calibrator as cl
import matplotlib.pyplot as plt
from importlib import reload
import scipy.linalg as ln
from numpy import sqrt, cos, sin
from dev.bayesLib import flat2int
import dev.bayesLib as bl
import scipy.stats as sts
import scipy.special as spe
import numdifftools as nd


# %% LOAD DATA
# input
plotCorners = False
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
modelos = ['poly', 'rational', 'fisheye', 'stereographic']
model = modelos[3]
Ns = [2,3]
model

intrCalibFile = "/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/"
intrCalibFile +="extraDataSebaPhD/traces15IntrCalibResults.npy"

imagesFolder = "./resources/intrinsicCalib/" + camera + "/"
cornersFile =      imagesFolder + camera + "Corners.npy"
patternFile =      imagesFolder + camera + "ChessPattern.npy"
imgShapeFile =     imagesFolder + camera + "Shape.npy"

# load data
chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.png')
imagePointsAll = np.load(cornersFile)

intrCalib = np.load(intrCalibFile).all()

nIm, _, nPts, _ = imagePointsAll.shape  # cantidad de imagenes
# Parametros de entrada/salida de la calibracion
objpoints = np.array([chessboardModel]*nIm)

fkV = intrCalib['inMean']
cameraMatrix, distCoeffs = flat2int(fkV, Ns, model)
f = cameraMatrix[[0,1],[2,2]]
rtV = intrCalib['exMean']

imSel = 0
xi, yi = imagePointsAll[imSel, 0].T


# %% testear CCD a Dist
funcPts = lambda x: np.array(cl.ccd2dis(x[0], x[1], cameraMatrix)[:2])
funcMat = lambda x: np.array(cl.ccd2dis(xi, yi, bl.flat2CamMatrix(x, model))[:2])
derPts = nd.Jacobian(funcPts)
derMat = nd.Jacobian(funcMat)

xd = np.zeros_like(imagePointsAll[:, 0,:,0])
yd = np.zeros_like(xd)

for imSel in range(nIm):
    xi, yi = imagePointsAll[imSel, 0].T
    Jd_i, Jd_f  = cl.ccd2disJacobian(xi, yi, cameraMatrix)
    retVal = cl.ccd2dis(xi, yi, cameraMatrix, Cccd=False, Cf=False)
    xd[imSel], yd[imSel], _, _ = retVal

    print(imSel)
    print(np.allclose(derPts([xi, yi]).T, Jd_i))
    print(np.allclose(derMat(f).T, Jd_f[:,:,2:], rtol=1e-3))


# %%  de Dist a Homogeneas
funcPts = lambda x: np.array(cl.dis2hom(x[0], x[1], distCoeffs, model)[:2])
def funcCoe(x, xDis, yDis):
    return np.array(cl.dis2hom(xDis, yDis, x, model)[:2])

derPts = nd.Jacobian(funcPts)
derCoe = nd.Jacobian(funcCoe)
# derPts([xi[0],yi[0]])
# derCoe(distCoeffs, xi[0], yi[0])

xh = np.zeros_like(xd)
yh = np.zeros_like(xd)

for imSel in range(nIm):
    _, _, Jh_d, Jh_k = cl.dis2hom_ratioJacobians(xd[imSel], yd[imSel],
                                                 distCoeffs, model)
    xh[imSel], yh[imSel], _ = cl.dis2hom(xd[imSel], yd[imSel], distCoeffs,
                                         model)

    print(imSel)
    print(np.allclose(derPts([xd[imSel], yd[imSel]]).T, Jh_d))
    aux = derCoe(distCoeffs, xd[imSel], yd[imSel])
    print(np.allclose(aux, Jh_k[:,:,0]))
    plt.plot(Jh_k[:,0,0], aux[:,0], '+')

plt.plot([-0.2, 0.2],[-0.2,0.2])

# %% de homogeneas a mundo
def funcPts(x, rV, tV):
    return np.array(cl.xyhToZplane(x[0], x[1],rV,tV )[:2])

def funcCoe(x, xDis, yDis):
    return np.array(cl.xyhToZplane(xDis, yDis, x[:3], x[3:])[:2])


derPts = nd.Jacobian(funcPts)
derCoe = nd.Jacobian(funcCoe)

derPts([xh[0,0],yh[0,0]], rtV[0,:3], rtV[0,3:])
derCoe(rtV[0], xh[0,0], yh[0,0])

xw = np.zeros_like(xh)
yw = np.zeros_like(xh)

jacXteo = list()
jacXnum = list()
jacTteo = list()
jacTnum = list()

for imSel in range(nIm):
    JXm_Xp, JXm_rtV = cl.jacobianosHom2Map(np.float64(xh[imSel]),
                                           np.float64(yh[imSel]),
                                           rtV[imSel,:3], rtV[imSel,3:])
    xw[imSel], yw[imSel], _ = cl.xyhToZplane(xh[imSel], yh[imSel],
                                             rtV[imSel,:3], rtV[imSel,3:])

    print(imSel)
    aux1 = derPts([xh[imSel], yh[imSel]], rtV[imSel,:3], rtV[imSel, 3:])
    print(np.allclose(aux1, JXm_Xp))
    plt.figure(1)
    plt.plot(aux1.flat, JXm_Xp.flat, '+')
    jacXteo.append(JXm_Xp.reshape(-1))
    jacXnum.append(aux1.reshape(-1))

    aux2 = derCoe(rtV[imSel], xh[imSel], yh[imSel])
    print(np.allclose(aux2, JXm_rtV))
    jacTteo.append(JXm_rtV.reshape(-1))
    jacTnum.append(aux2.reshape(-1))

    print(np.abs(aux2 / JXm_rtV -1).max(), np.abs(aux2 - JXm_rtV).max())
    plt.figure(2)
    plt.plot(aux2.flat, JXm_rtV.flat, '+')


plt.figure(1)
plt.plot([-25,25], [-25,25])

plt.figure(2)
plt.plot([-12,12], [-12,12])

jacXteo = np.array(jacXteo).reshape(-1)
jacXnum = np.array(jacXnum).reshape(-1)
jacTteo = np.array(jacTteo).reshape(-1)
jacTnum = np.array(jacTnum).reshape(-1)

plt.figure()
plt.plot(np.abs(jacXteo), np.abs(jacXteo -jacXnum), '+')


plt.figure()
plt.plot(np.abs(jacTteo), np.abs(jacTteo -jacTnum), '+')












