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


def calculaCovarianza(xM, yM,):
    '''
    lso inputs son de tamaño (Nsamples, Mpuntos)
    con Nsamples por cada uno de los Mpuntos puntos
    '''
    muXm = np.mean(xM, axis=0)
    muYm = np.mean(yM, axis=0)

    xMcen = xM.T - muXm.reshape(-1, 1)
    yMcen = yM.T - muYm.reshape(-1, 1)

    CmNum = np.empty((xM.shape[1], 2, 2), dtype=float)

    CmNum[:, 0, 0] = np.sum(xMcen**2, axis=1)
    CmNum[:, 1, 1] = np.sum(yMcen**2, axis=1)
    CmNum[:, 0, 1] = CmNum[:, 1, 0] = np.sum(xMcen*yMcen, axis=1)
    CmNum /= (xM.shape[0] - 1)

    return [muXm, muYm], CmNum


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
imagePointsAll = np.load(cornersFile)

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

# %% invento unas covarianzas
Ci = (1.0**2) * np.array([np.eye(2)]*imagePointsAll.shape[2])
Cr = np.eye(3) * (np.pi / 180)**2
Ct = np.eye(3) * 0.1**2
Crt = [Cr, Ct]
# covarianza de los parametros intrinsecos de la camara
Cf = np.eye(4) * 0.1**2
if model is 'poly':
    Ck = np.diag((distCoeffs[[0, 1, 4]]*0.001)**2)  # 0.1% error distorsion
if model is 'rational':
    Ck = np.diag((distCoeffs[[0, 1, 4, 5, 6, 7]]*0.001)**2)  # 0.1% dist er
if model is 'fisheye':
    Ck = np.diag((distCoeffs*0.001)**2)  # 0.1% error distorsion

N = 1000  # cantidad de realizaciones 


# %% choose an image
imSearch = '22h22m11s'
for i in range(len(images)):
    if images[i].find(imSearch) is not -1:
        j = i

print('\t imagen', j)
imagePoints = imagePointsAll[j, 0]
npts = imagePoints.shape[0]

img = plt.imread(images[j])

# cargo la rototraslacion
rV = rVecs[j].reshape(-1)
tV = tVecs[j].reshape(-1)


# %% funcion que hace todas las cuentas
def analyticVsMC(imagePoints, Ci, cameraMatrix, Cf, distCoeffs, Ck, rtV, Crt):
    rV, tV = rtV
    
    # propagate to homogemous
    xpp, ypp, Cpp = cl.ccd2hom(imagePoints, cameraMatrix, Cccd=Ci, Cf=Cf)
    
    # go to undistorted homogenous
    xp, yp, Cp = cl.homDist2homUndist(xpp, ypp, distCoeffs, model, Cpp=Cpp, Ck=Ck)
    
    # project to map
    xm, ym, Cm = cl.xypToZplane(xp, yp, rV, tV, Cp=Cp, Crt=Crt)
    
    # generar puntos y parametros
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

    muI, CiNum = calculaCovarianza(xI[:, :, 0], xI[:, :, 1])
    muPP, CppNum = calculaCovarianza(xPP, yPP)
    muP, CpNum = calculaCovarianza(xP, yP)
    muM, CmNum = calculaCovarianza(xM, yM)

    ptsTeo = [imagePoints, [xpp, ypp], [xp, yp], [xm, ym]]
    ptsNum = [muI, muPP, muP, muM]

    covTeo = [Ci, Cpp, Cp, Cm]
    covNum = [CiNum, CppNum, CpNum, CmNum]

    return ptsTeo, ptsNum, covTeo, covNum

# %%
rtV = [rV, tV]
ptsTeo, ptsNum, covTeo, covNum = analyticVsMC(imagePoints, Ci, cameraMatrix, Cf, distCoeffs, Ck, rtV, Crt)

xI, [xpp, ypp], [xp, yp], [xm, ym] = ptsTeo
muI, muPP, muP, muM = ptsNum
Ci, Cpp, Cp, Cm = covTeo
CiNum, CppNum, CpNum, CmNum = covNum

## %% plot everything
#
## plot initial uncertanties
#fig1 = plt.figure(1)
#ax1 = fig1.gca()
#ax1.imshow(img)
##for i in range(nPts):
##    x, y = imagePoints[i]
##    cl.plotEllipse(ax1, Ci[i], x, y, 'b')
#
## propagate to homogemous
#fig2 = plt.figure(2)
#ax2 = fig2.gca()
##for i in range(nPts):
##    cl.plotEllipse(ax2, Cpp[i], xpp[i], ypp[i], 'b')
#
## go to undistorted homogenous
#fig3 = plt.figure(3)
#ax3 = fig3.gca()
##for i in range(nPts):
##    cl.plotEllipse(ax3, Cp[i], xp[i], yp[i], 'b')
#
## project to map
#fig4 = plt.figure(4)
#ax4 = fig4.gca()
#ax4.plot(xm, ym, '+', markersize=2)
##for i in range(nPts):
##    cl.plotEllipse(ax4, Cm[i], xm[i], ym[i], 'b')
#
#
#for i in range(N):
#    # plot corners
#    ax1.plot(xI[i, :, 0], xI[i, :, 1], '.k', markersize=0.5)
#
#    # % propagate to homogemous
#    ax2.plot(xPP[i], yPP[i], '.k', markersize=0.5)
#
#    # % go to undistorted homogenous
#    ax3.plot(xP[i], yP[i], '.k', markersize=0.5)
#
#    # % project to map
#    ax4.plot(xM[i], yM[i], '.k', markersize=0.5)


# %% comparo numericamente

# calculo las normas de frobenius de cada matriz y de la diferencia
CiF, CppF, CpF, CmF = [ln.norm(C, axis=(1,2)) for C in [Ci, Cpp, Cp, Cm]]
CiNF, CppNF, CpNF, CmNF = [ln.norm(C, axis=(1,2)) for C in [CiNum, CppNum, CpNum, CmNum]]

CiDF, CppDF, CpDF, CmDF = [ln.norm(C, axis=(1,2))
    for C in [Ci - CiNum, Cpp - CppNum, Cp - CpNum, Cm - CmNum]]

li, lpp, lp, lm = np.empty((4,npts,3), dtype=float)
liN, lppN, lpN, lmN = np.empty((4,npts,3), dtype=float)

# %%
def sacoValsAng(C):
    npts = C.shape[0]
    L = np.empty((npts, 3), dtype=float)
    # calculo valores singulares y angulo
    for i in range(npts):
        l, v = ln.eig(C[i])
        L[i] = [np.sqrt(l[0].real), np.sqrt(l[1].real), np.arctan(v[0, 1] / v[0, 0])]
    return L

Li, Lpp, Lp, Lm = [sacoValsAng(C) for C in [Ci, Cpp, Cp, Cm]]
LiN, LppN, LpN, LmN = [sacoValsAng(C) for C in [CiNum, CppNum, CpNum, CmNum]]


# %% "indicadores" de covarianza
rads = ln.norm(imagePoints - cameraMatrix[:2, 2], axis=1)

tVm = cl.rotateRodrigues(tV,-rV) # traslation vector in maps frame of ref
factorVista = ln.norm(chessboardModel[0].T + tVm.reshape(-1,1), axis=0) / tVm[2]



# %% grafico las normas de las covarianzas
# radios respecto al centro de distorsion

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
plt.scatter(rads, CiNF)

plt.subplot(222)
plt.scatter(rads, CppF)
plt.scatter(rads, CppNF)

plt.subplot(223)
plt.scatter(rads, CpF)
plt.scatter(rads, CpNF)

plt.subplot(224)
plt.scatter(rads, CmF)
plt.scatter(rads, CmNF)


# %%
rads = factorVista
plt.figure()

var = 0
plt.subplot(341)
plt.scatter(rads, Li[:, var])
plt.scatter(rads, LiN[:, var])

plt.subplot(342)
plt.scatter(rads, Lpp[:, var])
plt.scatter(rads, LppN[:, var])

plt.subplot(343)
plt.scatter(rads, Lp[:, var])
plt.scatter(rads, LpN[:, var])

plt.subplot(344)
plt.scatter(rads, Lm[:, var])
plt.scatter(rads, LmN[:, var])

var = 1
plt.subplot(345)
plt.scatter(rads, Li[:, var])
plt.scatter(rads, LiN[:, var])

plt.subplot(346)
plt.scatter(rads, Lpp[:, var])
plt.scatter(rads, LppN[:, var])

plt.subplot(347)
plt.scatter(rads, Lp[:, var])
plt.scatter(rads, LpN[:, var])

plt.subplot(348)
plt.scatter(rads, Lm[:, var])
plt.scatter(rads, LmN[:, var])


var = 2
plt.subplot(349)
plt.scatter(rads, Li[:, var])
plt.scatter(rads, LiN[:, var])

plt.subplot(3,4,10)
plt.scatter(rads, Lpp[:, var])
plt.scatter(rads, LppN[:, var])

plt.subplot(3,4,11)
plt.scatter(rads, Lp[:, var])
plt.scatter(rads, LpN[:, var])

plt.subplot(3,4,12)
plt.scatter(rads, Lm[:, var])
plt.scatter(rads, LmN[:, var])

# %%
plt.figure()

var = 0
plt.subplot(341)
plt.scatter(Li[:, var], LiN[:, var])
plt.plot([np.min(Li[:, var]), np.max(Li[:, var])],
          [np.min(LiN[:, var]), np.max(LiN[:, var])], '-k')

plt.subplot(342)
plt.scatter(Lpp[:, var], LppN[:, var])
plt.plot([np.min(Lpp[:, var]), np.max(Lpp[:, var])],
          [np.min(LppN[:, var]), np.max(LppN[:, var])], '-k')

plt.subplot(343)
plt.scatter(Lp[:, var], LpN[:, var])
plt.plot([np.min(Lp[:, var]), np.max(Lp[:, var])],
          [np.min(LpN[:, var]), np.max(LpN[:, var])], '-k')

plt.subplot(344)
plt.scatter(Lm[:, var], LmN[:, var])
plt.plot([np.min(Lm[:, var]), np.max(Lm[:, var])],
          [np.min(LmN[:, var]), np.max(LmN[:, var])], '-k')

var = 1
plt.subplot(345)
plt.scatter(Li[:, var], LiN[:, var])
plt.plot([np.min(Li[:, var]), np.max(Li[:, var])],
          [np.min(LiN[:, var]), np.max(LiN[:, var])], '-k')

plt.subplot(346)
plt.scatter(Lpp[:, var], LppN[:, var])
plt.plot([np.min(Lpp[:, var]), np.max(Lpp[:, var])],
          [np.min(LppN[:, var]), np.max(LppN[:, var])], '-k')

plt.subplot(347)
plt.scatter(Lp[:, var], LpN[:, var])
plt.plot([np.min(Lp[:, var]), np.max(Lp[:, var])],
          [np.min(LpN[:, var]), np.max(LpN[:, var])], '-k')

plt.subplot(348)
plt.scatter(Lm[:, var], LmN[:, var])
plt.plot([np.min(Lm[:, var]), np.max(Lm[:, var])],
          [np.min(LmN[:, var]), np.max(LmN[:, var])], '-k')

var = 2
plt.subplot(349)
plt.scatter(rads, Li[:, var], LiN[:, var])
plt.plot([np.min(Li[:, var]), np.max(Li[:, var])],
          [np.min(LiN[:, var]), np.max(LiN[:, var])], '-k')

plt.subplot(3,4,10)
plt.scatter(Lpp[:, var], LppN[:, var])
plt.plot([np.min(Lpp[:, var]), np.max(Lpp[:, var])],
          [np.min(LppN[:, var]), np.max(LppN[:, var])], '-k')

plt.subplot(3,4,11)
plt.scatter(Lp[:, var], LpN[:, var])
plt.plot([np.min(Lp[:, var]), np.max(Lp[:, var])],
          [np.min(LpN[:, var]), np.max(LpN[:, var])], '-k')

plt.subplot(3,4,12)
plt.scatter(Lm[:, var], LmN[:, var])
plt.plot([np.min(Lm[:, var]), np.max(Lm[:, var])],
          [np.min(LmN[:, var]), np.max(LmN[:, var])], '-k')


# %% saco de todos los puntos las desvests y angulo

L = list()
LN = list()
Rads = list()

for i in range(len(images)):
    print('\t imagen', i)
    imagePoints = imagePointsAll[i, 0]
    npts = imagePoints.shape[0]

    # cargo la rototraslacion
    rV = rVecs[i].reshape(-1)
    tV = tVecs[i].reshape(-1)

    rtV = [rV, tV]
    ptsTeo, ptsNum, covTeo, covNum = analyticVsMC(imagePoints, Ci,
                                                  cameraMatrix, Cf, distCoeffs,
                                                  Ck, rtV, Crt)

    L.append(sacoValsAng(covTeo[3]))
    LN.append(sacoValsAng(covNum[3]))
    Rads.append(ln.norm(imagePoints, axis=1))


# %%

for i in range(len(L)):
    for j in range(3):
        plt.subplot(2,3,j+1)
        plt.scatter(L[i][:,j], LN[i][:,j])

        plt.subplot(2,3,j+4)
        plt.scatter(Rads[i], L[i][:,j])
        plt.scatter(Rads[i], LN[i][:,j])


for j in range(2):
    plt.subplot(2,3,j+1)
    plt.plot([0, 0.5], [0, 0.5], 'k-')

plt.subplot(2,3,3)
plt.plot([-1, 1], [-1, 1], 'k-')

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








