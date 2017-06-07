#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 20:11:26 2017

test uncertanty propagation wrt montecarlo

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

def calculaCovarianza(xM, yM):
    '''
    lso inputs son de tamaño (Nsamples, Mpuntos)
    con Nsamples por cada uno de los Mpuntos puntos
    '''
    muXm = np.mean(xM, axis=0)
    muYm = np.mean(yM, axis=0)

    xMcen = xM.T - muXm.reshape(-1, 1)
    yMcen = yM.T - muYm.reshape(-1, 1)

    CmNum = np.empty_like(Cm)

    CmNum[:, 0, 0] = np.sum(xMcen**2, axis=1)
    CmNum[:, 1, 1] = np.sum(yMcen**2, axis=1)
    CmNum[:, 0, 1] = CmNum[:, 1, 0] = np.sum(xMcen*yMcen, axis=1)
    CmNum /= (xM.shape[0] - 1)

    return CmNum


# %% LOAD DATA
# input
plotCorners = False
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
modelos = ['poly', 'rational', 'fisheye']
model = modelos[2]

model

imagesFolder = "./resources/intrinsicCalib/" + camera + "/"
cornersFile =      imagesFolder + camera + "Corners.npy"
patternFile =      imagesFolder + camera + "ChessPattern.npy"
imgShapeFile =     imagesFolder + camera + "Shape.npy"

# load data
imagePoints = np.load(cornersFile)
chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.png')

nIm, _, nPts, _ = imagePoints.shape  # cantidad de imagenes
# Parametros de entrada/salida de la calibracion
objpoints = np.array([chessboardModel]*nIm)

distCoeffsFile =   imagesFolder + camera + model + "DistCoeffs.npy"
linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
tVecsFile =        imagesFolder + camera + model + "Tvecs.npy"
rVecsFile =        imagesFolder + camera + model + "Rvecs.npy"

# load model specific data
distCoeffs = np.load(distCoeffsFile).reshape(-1)
cameraMatrix = np.load(linearCoeffsFile)
rVecs = np.load(rVecsFile)
tVecs = np.load(tVecsFile)

# covarianza de los parametros intrinsecos de la camara
Cf = np.eye(4)*(0.1)**2
if model is 'poly':
    Ck = np.diag((distCoeffs[[0, 1, 4]]*0.001)**2)  # 0.1% error distorsion
if model is 'rational':
    Ck = np.diag((distCoeffs[[0, 1, 4, 5, 6, 7]]*0.001)**2)  # 0.1% error distorsion
if model is 'fisheye':
    Ck = np.diag((distCoeffs*0.001)**2)  # 0.1% error distorsion

# %% choose 

imSearch = '21h58m38s'
for i in range(len(images)):
    if images[i].find(imSearch) is not -1:
        j = i

print('\t imagen', j)
imagePoints = np.load(cornersFile)
imagePoints = imagePoints[j, 0]
npts = imagePoints.shape[0]

img = plt.imread(images[j])

# cargo la rototraslacion
rV = rVecs[j].reshape(-1)
tV = tVecs[j].reshape(-1)

# invento unas covarianzas
Ci = (1.0**2) * np.array([np.eye(2)]*imagePoints.shape[0])
Cr = np.diag((rV*0.001)**2)
Ct = np.diag((tV*0.001)**2)
Crt = [Cr, Ct]

# propagate to homogemous
xpp, ypp, Cpp = cl.ccd2hom(imagePoints, cameraMatrix, Cccd=Ci, Cf=Cf)

# go to undistorted homogenous
xp, yp, Cp = cl.homDist2homUndist(xpp, ypp, distCoeffs, model, Cpp=Cpp, Ck=Ck)

# project to map
xm, ym, Cm = cl.xypToZplane(xp, yp, rV, tV, Cp=Cp, Crt=[Cr, Ct])


# %% simulate many
N = 1000

# por suerte todas las pdfs son indep
xI = np.random.randn(N, npts, 2).dot(np.sqrt(Ci[0])) + imagePoints
rots = np.random.randn(N, 3).dot(np.sqrt(Cr)) + rV
tras = np.random.randn(N, 3).dot(np.sqrt(Ct)) + tV
kD = np.random.randn(N, Ck.shape[0]).dot(np.sqrt(Ck)) + distCoeffs  # disorsion
kL = np.zeros((N, 3, 3), dtype=float)  # lineal
kL[:, 2, 2] = 1
kL[:, :2, 2] = np.random.randn(N, 2).dot(np.sqrt(Cf[2:, 2:])) + cameraMatrix[:2, 2]
kL[:, [0, 1], [0, 1]] = np.random.randn(N, 2).dot(np.sqrt(Cf[:2, :2]))
kL[:, [0, 1], [0, 1]] += cameraMatrix[[0, 1], [0, 1]]

xPP, yPP, xP, yP, xM, yM = np.empty((6, N, npts), dtype=float)

for i in range(N):
    # % propagate to homogemous
    xPP[i], yPP[i] = cl.ccd2hom(xI[i], kL[i])

    # % go to undistorted homogenous
    xP[i], yP[i] = cl.homDist2homUndist(xPP[i], yPP[i], kD[i], model)

    # % project to map
    xM[i], yM[i] = cl.xypToZplane(xP[i], yP[i], rots[i], tras[i])


# %% plot everything

# plot initial uncertanties
fig1 = plt.figure(1)
ax1 = fig1.gca()
ax1.imshow(img)
#for i in range(nPts):
#    x, y = imagePoints[i]
#    cl.plotEllipse(ax1, Ci[i], x, y, 'b')

# propagate to homogemous
fig2 = plt.figure(2)
ax2 = fig2.gca()
#for i in range(nPts):
#    cl.plotEllipse(ax2, Cpp[i], xpp[i], ypp[i], 'b')

# go to undistorted homogenous
fig3 = plt.figure(3)
ax3 = fig3.gca()
#for i in range(nPts):
#    cl.plotEllipse(ax3, Cp[i], xp[i], yp[i], 'b')

# project to map
fig4 = plt.figure(4)
ax4 = fig4.gca()
ax4.plot(xm, ym, '+', markersize=2)
#for i in range(nPts):
#    cl.plotEllipse(ax4, Cm[i], xm[i], ym[i], 'b')


for i in range(N):
    # plot corners
    ax1.plot(xI[i, :, 0], xI[i, :, 1], '.k', markersize=0.5)

    # % propagate to homogemous
    ax2.plot(xPP[i], yPP[i], '.k', markersize=0.5)

    # % go to undistorted homogenous
    ax3.plot(xP[i], yP[i], '.k', markersize=0.5)

    # % project to map
    ax4.plot(xM[i], yM[i], '.k', markersize=0.5)


# %% comparo numericamente
CiNum = calculaCovarianza(xI[:, :, 0], xI[:, :, 1])
CppNum = calculaCovarianza(xPP, yPP)
CpNum = calculaCovarianza(xP, yP)
CmNum = calculaCovarianza(xM, yM)

# calculo las normas de frobenius de cada matriz y de la diferencia
CiF, CppF, CpF, CmF = [ln.norm(C, axis=(1,2)) for C in [Ci, Cpp, Cp, Cm]]
CiNF, CppNF, CpNF, CmNF = [ln.norm(C, axis=(1,2)) for C in [CiNum, CppNum, CpNum, CmNum]]

CiDF, CppDF, CpDF, CmDF = [ln.norm(C, axis=(1,2))
    for C in [Ci - CiNum, Cpp - CppNum, Cp - CpNum, Cm - CmNum]]

# %% grafico las normas de las covarianzas
# radios respecto al centro de distorsion
rads = ln.norm(imagePoints - cameraMatrix[:2, 2], axis=1)

plt.figure()
plt.subplot(221)
plt.scatter(rads, CiDF / CiF)

plt.subplot(222)
plt.scatter(rads, CppDF / CppF)

plt.subplot(223)
plt.scatter(rads, CpDF / CpF)

plt.subplot(224)
plt.scatter(rads, CmDF / CmF)

'''
ver que la diferencia entre las covarianzas depende del radio y esta diferencia
puede atribuirse a la nolinealidad de distorsion porque tiene una curva parecida

1- meter los 54 ptos de todas las imagenes a ver si siguen la misma tendencia
2- analizar por separado la covarianza numerica y la teorica y la diferencia

es muy loco que la dependencia en el radio parece volver a difuminarse al hacer
el ultimo paso del mapeo.
'''

plt.figure()
plt.subplot(221)
plt.scatter(rads, CiF)

plt.subplot(222)
plt.scatter(rads, CppF)

plt.subplot(223)
plt.scatter(rads, CpF)

plt.subplot(224)
plt.scatter(rads, CmF)


# %%
from scipy.special import chdtri

def ptosAdentro(x, y, muX, muY, C, p):
    '''
    calcular la cantidad de puntos que caen adentro para una dada probablidad
    '''
    # calculo ma raiz cuadrada de C. tal que A.dot(A.T) = C
    l, v = ln.eig(ln.inv(C))
    A = np.sqrt(l.real) * v

    # llevo los vectores a la forma linealizada
    X = np.array([x - muX, y - muY]).T.dot(A)

    # radio^2 para 2D y probabiliad p de que esten adentro
    r2 = chdtri(2, 1 - p)

    adentro = np.sum(ln.norm(X, axis=1) <= np.sqrt(r2)) / x.shape[0]

    return adentro


i = 9
ptosAdentro(xI[:, i, 0], xI[:, i, 1], imagePoints[i, 0], imagePoints[i, 1], Ci[i],  0.7)
# x, y, muX, muY, C, p = (xI[:, 9, 0], xI[:, 9, 1], imagePoints[9, 0], imagePoints[9, 1], Ci[0], 0.7)

ptosAdentro(xM[:, i], yM[:, i], xm[i], ym[i], Cm[i], 0.7)



# %% este ejemplo anda. algo falta con la diagonalizacion
from scipy.special import chdtri
import numpy as np

# %%
rho = -0.6
N = 100000
D = 2
S = np.diag(np.arange(D)+1)
S[[0, 1], [1, 0]] = np.sqrt(S[0, 0] + S[1, 1]) * rho

p = 0.74

x = np.random.randn(N, D)

r2 = chdtri(D, 1 - p)
pNum = np.sum(np.sum(x**2, axis=1) <= r2) / N

# print(r)
print(p)
print(pNum)








