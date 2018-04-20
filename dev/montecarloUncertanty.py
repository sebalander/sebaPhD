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
from dev.bayesLib import flat2int
import dev.bayesLib as bl
import scipy.stats as sts
import scipy.special as spe

def calculaCovarianza(xM, yM):
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

# %%
from scipy.stats import multivariate_normal

N = 10000
M = 23
C = np.array([[[2,1],[1,3]]] * M)

if np.all(np.linalg.eigvals(C) > 0):
    T = cl.unit2CovTransf(C)  # calculate transform matriz
    X = np.random.randn(N, M, 2)  # gen rndn points unitary normal
    X = (X.reshape((N, M, 1, 2)) *  # transform
         T.reshape((1, M, 2, 2))
         ).sum(-1)
else:
    print('Error: la covarianza no es definida positiva')

#X2 = multivariate_normal.rvs(mean=None, cov=C, size=n).T

[muX, muY], CNum = calculaCovarianza(X[:,:,0], X[:,:,1])
#[muX2, muY2], CNum2 = calculaCovarianza(X2[0].reshape((-1,1)), X2[1].reshape((-1,1)))

print(C)
print(CNum)

plt.figure()
plt.plot(C.flat, CNum.flat, '.')

plt.figure()
ax = plt.gca()
xx = X[:,:,0].reshape(-1)
yy = X[:,:,1].reshape(-1)
ax.scatter(xx, yy, s=1, alpha=0.03)
cl.plotPointsUncert(ax, [C[0]], [0], [0], col='k')


#CNum2
print(np.allclose(C, CNum, rtol=0.05))

# %% funcion que hace todas las cuentas
def analyticVsMC(imgPts, Ci, F, K, Cintr, rtV, Crt, retPts=True):
    '''
    Parameters and shapes
    -------
    imagePoints: coordinates in image (N,2)
    Ci: uncertainty of imegePoints (N,2,2)
    F: Camera Matrix (3,3)
    Cf: uncertainty on the 4 parameters of camera matrix (nF,nF), tipically
        nF=4.
    K: intrinsic parameters, a vector (nK,)
    Ck: uncertainty of intrinsic parameters (nK, nK)
    Cfk: covariance of cross intrinsic parameters (nF,nK)
    rtV: vector of 6 pose params (6,)
    Crt: covariance on pose (6,6)
    '''
    Cf = np.zeros((4, 4))  # param de CCD, dist focal, centro
    Cf[2:, 2:] += Cintr[:Ns[0], :Ns[0]]
    Ck = Cintr[Ns[0]:, Ns[0]:]  # k de distorsion
    Cfk = Cintr[:Ns[0], Ns[0]:]

    rV, tV = rtV.reshape((2, -1))

    # propagate to homogemous
    xd, yd, Cd, _ = cl.ccd2dis(imgPts[:, 0], imgPts[:, 1], F, Cccd=Ci, Cf=Cf)

    # go to undistorted homogenous
    xh, yh, Ch = cl.dis2hom(xd, yd, K, model, Cd=Cd, Ck=Ck, Cfk=Cfk)

    # project to map
    xm, ym, Cm = cl.xyhToZplane(xh, yh, rV, tV, Ch=Ch, Crt=Crt)

    # generar puntos y parametros
    # por suerte todas las pdfs son indep
    Ti = cl.unit2CovTransf(Ci)
    xI = (np.random.randn(N, nPts, 1, 2) *
          Ti.reshape((1, -1, 2, 2))).sum(-1) + imgPts

    Trt = cl.unit2CovTransf(Crt)
    rtVsamples = (np.random.randn(N, 1, 6) *
                  Trt.reshape((1, 6, 6))
                  ).sum(-1) + rtV
    rots = rtVsamples[:, :3]  # np.random.randn(N, 3).dot(np.sqrt(Cr)) + rV
    tras = rtVsamples[:, 3:]  # np.random.randn(N, 3).dot(np.sqrt(Ct)) + tV

#    # para chequear que esta bien multiplicado esto
#    ertest = (rtVsamples[0] - rtV).dot(Crt).dot((rtVsamples[0] - rtV).T)
#    np.isclose(ertest, np.sum(((rtVsamples[0] - rtV).dot(Dextr))**2))
#    np.isclose(ertest, np.sum(((rtVsamples[0] - rtV).dot(Dextr.T))**2))

    Tintr = cl.unit2CovTransf(Cintr)
    fkVsamples = (np.random.randn(N, 1, Ns[1]) *
                  Tintr.reshape((1, Ns[1], Ns[1]))
                  ).sum(-1) + fkV
    kL = fkVsamples[:, :Ns[0]]
    kD = fkVsamples[:, Ns[0]:]
#    kD = np.random.randn(N, Cf.shape[0]).dot(np.sqrt(Cf)) + K  # distorsion
#    kL = np.zeros((N, 3, 3), dtype=float)  # lineal
#    kL[:, 2, 2] = 1
#    kL[:, :2, 2] = np.random.randn(N, 2).dot(np.sqrt(Ck[2:, 2:])) + F[:2, 2]
#    kL[:, [0, 1], [0, 1]] = np.random.randn(N, 2).dot(np.sqrt(Ck[:2, :2]))
#    kL[:, [0, 1], [0, 1]] += F[[0, 1], [0, 1]]

    # estos son los puntos sampleados por montecarlo, despues de
    xD, yD, xH, yH, xM, yM = np.empty((6, N, npts), dtype=float)

    for i in range(N):
        # F, K = flat2int(fkVsamples[i], Ns, model)
        # % propagate to homogemous
        camMat = bl.flat2CamMatrix(kL[i], model)
        xD[i], yD[i], _, _ = cl.ccd2dis(xI[i, :, 0], xI[i, :, 1], camMat)

        # % go to undistorted homogenous
        xH[i], yH[i], _ = cl.dis2hom(xD[i], yD[i], kD[i], model)

        # % project to map
        xM[i], yM[i], _ = cl.xyhToZplane(xH[i], yH[i], rots[i], tras[i])

    muI, CiNum = calculaCovarianza(xI[:, :, 0], xI[:, :, 1])
    muD, CdNum = calculaCovarianza(xD, yD)
    muH, ChNum = calculaCovarianza(xH, yH)
    muM, CmNum = calculaCovarianza(xM, yM)

    ptsTeo = [[xd, yd], [xh, yh], [xm, ym]]
    ptsNum = [muI, muD, muH, muM]
    ptsMC = [xI, xD, yD, xH, yH, xM, yM]

    covTeo = [Ci, Cd, Ch, Cm]
    covNum = [CiNum, CdNum, ChNum, CmNum]

    if retPts:
        return ptsTeo, ptsNum, covTeo, covNum, ptsMC
    else:
        return covTeo, covNum


# %% LOAD DATA
np.random.seed(0)
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
imagePoints = np.load(cornersFile)
chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.png')
imagePointsAll = np.load(cornersFile)

intrCalib = np.load(intrCalibFile).all()

nIm, _, nPts, _ = imagePoints.shape  # cantidad de imagenes
# Parametros de entrada/salida de la calibracion
objpoints = np.array([chessboardModel]*nIm)

#distCoeffsFile =   imagesFolder + camera + model + "DistCoeffs.npy"
#linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
#tVecsFile =        imagesFolder + camera + model + "Tvecs.npy"
#rVecsFile =        imagesFolder + camera + model + "Rvecs.npy"

# load model specific data
imSel = 4 # ELIJO UNA DE LAS IMAGENES

fkV = intrCalib['inMean']
cameraMatrix, distCoeffs = flat2int(fkV, Ns, model)
rtV = intrCalib['exMean'][imSel]
imgPts = imagePoints[imSel,0]

Cintr = intrCalib['inCov'] / 1
#Cf = np.zeros((4,4))  # param de CCD, dist focal, centro
#Cf[2:,2:] += Cintr[:Ns[0], :Ns[0]]
#Ck = Cintr[Ns[0]:, Ns[0]:]  # k de distorsion
#Cfk = Cintr[:Ns[0], Ns[0]:]
#Dintr = cl.unit2CovTransf(Cintr)

covLim0 = 6 * imSel
covLim1 = covLim0 + 6
Crt = intrCalib['exCov'][covLim0:covLim1,covLim0:covLim1] / 1
#Dextr = cl.unit2CovTransf(Crt)

#rtSigmas = np.sqrt(np.diag(Crt))
#CorrRT = Crt / rtSigmas.reshape((-1,1)) / rtSigmas.reshape((1,-1))
#plt.matshow(CorrRT, cmap='coolwarm', vmax=1, vmin=-1)

## %% invento unas covarianzas
Ci = (1.0**2) * np.array([np.eye(2)]*nPts) / 1
#Di = np.sqrt(Ci) # a estas las supongo diagonales y listo
#Cr = np.eye(3) * (np.pi / 180)**2
#Ct = np.eye(3) * 0.1**2
#Crt = [Cr, Ct]
## covarianza de los parametros intrinsecos de la camara
#Cf = np.eye(4) * 0.1**2
#if model is 'poly':
#    Ck = np.diag((distCoeffs[[0, 1, 4]]*0.001)**2)  # 0.1% error distorsion
#if model is 'rational':
#    Ck = np.diag((distCoeffs[[0, 1, 4, 5, 6, 7]]*0.001)**2)  # 0.1% dist er
#if model is 'fisheye':
#    Ck = np.diag((distCoeffs*0.001)**2)  # 0.1% error distorsion
#
N = 5000  # cantidad de realizaciones
#




## %% choose an image
#imSearch = '22h22m11s'
#for i in range(len(images)):
#    if images[i].find(imSearch) is not -1:
#        j = i
#
#print('\t imagen', j)
#imagePoints = imagePointsAll[j, 0]
npts = imgPts.shape[0]
#
#img = plt.imread(images[j])
#
## cargo la rototraslacion
#rV = rVecs[j].reshape(-1)
#tV = tVecs[j].reshape(-1)


# %%
reload(cl)
np.random.seed(0)

retAll = analyticVsMC(imgPts, Ci, cameraMatrix, distCoeffs, Cintr, rtV, Crt)
ptsTeo, ptsNum, covTeo, covNum, ptsMC = retAll

[xd, yd], [xh, yh], [xm, ym] = ptsTeo
Ci, Cd, Ch, Cm = covTeo

xI, xD, yD, xH, yH, xM, yM = ptsMC

muI, muD, muH, muM = ptsNum
CiNum, CdNum, ChNum, CmNum = covNum

# %% plot everything

ptSelected = 0  # de lso 54 puntos este es el seleccionado para acercamiento


fig = plt.figure()
#from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

# plot initial uncertanties
#figI = plt.figure(1)
axI = plt.subplot(241)  # figI.gca()
cl.plotPointsUncert(axI, Ci, imgPts[:,0], imgPts[:,1], 'b')
cl.plotPointsUncert(axI, CiNum, muI[0], muI[1], 'k')
axI.plot(xI[:, :, 0].flat, xI[:, :, 1].flat, '.k', markersize=0.5)
axI.axis('equal')

# propagate to homogemous
axD = plt.subplot(242)  # figD.gca()
cl.plotPointsUncert(axD, Cd, xd, yd, 'b')
cl.plotPointsUncert(axD, CdNum, muD[0], muD[1], 'k')
axD.plot(xD.flat, yD.flat, '.k', markersize=0.5)
#axD.plot(xd[0], yd[0], 'xr', markersize=5)
axD.axis('equal')

# go to undistorted homogenous
#figH = plt.figure(3)
axH = plt.subplot(243)  # figH.gca()
cl.plotPointsUncert(axH, Ch, xh, yh, 'b')
cl.plotPointsUncert(axH, ChNum, muH[0], muH[1], 'k')
axH.plot(xH.flat, yH.flat, '.k', markersize=0.5)
#axH.plot(xh[0], yh[0], 'xr', markersize=5)
axH.axis('equal')

# project to map
#figM = plt.figure(4)
axM = plt.subplot(244)  # figM.gca()
axM.plot(xm, ym, '+', markersize=2)
cl.plotPointsUncert(axM, Cm, xm, ym, 'b')
cl.plotPointsUncert(axM, CmNum, muM[0], muM[1], 'k')
axM.plot(xM.flat, yM.flat, '.k', markersize=0.5)
#axM.plot(xm[0], ym[0], 'xr', markersize=5)
axM.axis('equal')


# inset image
axIins = plt.subplot(245)
#axIins = zoomed_inset_axes(axI, 30.0, loc=2) # zoom-factor: 2.5, location: upper-left
#mark_inset(axI, axIins, loc1=1, loc2=4, fc="none", ec="0.5")
cl.plotPointsUncert(axIins, [Ci[ptSelected]], [imgPts[ptSelected,0]], [imgPts[ptSelected,1]], 'b')
cl.plotPointsUncert(axIins, [CiNum[ptSelected]], [muI[0][ptSelected]], [muI[1][ptSelected]], 'k')
axIins.plot(xI[:, ptSelected, 0], xI[:, ptSelected, 1], '.k', markersize=0.5)
axIins.axis('equal')
#axIins.set_xticks([False])
#axIins.set_yticks([False])


#inset distorted
axDins = plt.subplot(246)
#axDins = zoomed_inset_axes(axD, 30.0, loc=2) # zoom-factor: 2.5, location: upper-left
#mark_inset(axD, axDins, loc1=1, loc2=4, fc="none", ec="0.5")
cl.plotPointsUncert(axDins, [Cd[ptSelected]], [xd[ptSelected]], [yd[ptSelected]], 'b')
cl.plotPointsUncert(axDins, [CdNum[ptSelected]], [muD[0][ptSelected]], [muD[1][ptSelected]], 'k')
axDins.plot(xD[:, ptSelected], yD[:, ptSelected], '.k', markersize=0.5)
axDins.axis('equal')


#inset homogenous
axHins = plt.subplot(247)
#axDins = zoomed_inset_axes(axD, 30.0, loc=2) # zoom-factor: 2.5, location: upper-left
#mark_inset(axD, axDins, loc1=1, loc2=4, fc="none", ec="0.5")
cl.plotPointsUncert(axHins, [Ch[ptSelected]], [xh[ptSelected]], [yh[ptSelected]], 'b')
cl.plotPointsUncert(axHins, [ChNum[ptSelected]], [muH[0][ptSelected]], [muH[1][ptSelected]], 'k')
axHins.plot(xH[:, ptSelected], yH[:, ptSelected], '.k', markersize=0.5)
axHins.axis('equal')


#inset homogenous
axMins = plt.subplot(248)
#axDins = zoomed_inset_axes(axD, 30.0, loc=2) # zoom-factor: 2.5, location: upper-left
#mark_inset(axD, axDins, loc1=1, loc2=4, fc="none", ec="0.5")
cl.plotPointsUncert(axMins, [Cm[ptSelected]], [xm[ptSelected]], [ym[ptSelected]], 'b')
cl.plotPointsUncert(axMins, [CmNum[ptSelected]], [muM[0][ptSelected]], [muM[1][ptSelected]], 'k')
axMins.plot(xM[:, ptSelected], yM[:, ptSelected], '.k', markersize=0.5)
axMins.axis('equal')

# dibujo los rectangulos
import matplotlib.patches as patches

# caja en imagen
boxIxy = np.array([axIins.get_xlim()[0], axIins.get_ylim()[0]]) # get caja
boxIwh = np.array([axIins.get_xlim()[1] - boxIxy[0], axIins.get_ylim()[1] - boxIxy[1]])
boxIxy -= 0.5 * boxIwh # agrando la caja
boxIwh*= 2
axI.add_patch( patches.Rectangle(boxIxy, boxIwh[0], boxIwh[1], fill=False))

# caja en distprsionado
boxDxy = np.array([axDins.get_xlim()[0], axDins.get_ylim()[0]]) # get caja
boxDwh = np.array([axDins.get_xlim()[1] - boxDxy[0], axDins.get_ylim()[1] - boxDxy[1]])
boxDxy -= 0.5 * boxDwh # agrando la caja
boxDwh*= 2
axD.add_patch( patches.Rectangle(boxDxy, boxDwh[0], boxDwh[1], fill=False))

# caja en homogeneas
boxHxy = np.array([axHins.get_xlim()[0], axHins.get_ylim()[0]]) # get caja
boxHwh = np.array([axHins.get_xlim()[1] - boxHxy[0], axHins.get_ylim()[1] - boxHxy[1]])
boxHxy -= 0.5 * boxHwh # agrando la caja
boxHwh*= 2
axH.add_patch( patches.Rectangle(boxHxy, boxHwh[0], boxHwh[1], fill=False))

# caja en mapa
boxMxy = np.array([axMins.get_xlim()[0], axMins.get_ylim()[0]]) # get caja
boxMwh = np.array([axMins.get_xlim()[1] - boxMxy[0], axMins.get_ylim()[1] - boxMxy[1]])
boxMxy -= 0.5 * boxMwh # agrando la caja
boxMwh*= 2
axM.add_patch( patches.Rectangle(boxMxy, boxMwh[0], boxMwh[1], fill=False))


plt.tight_layout()


# %% comparo numericamente

# calculo las normas de frobenius de cada matriz y de la diferencia
CiF, CdF, ChF, CmF = [ln.norm(C, axis=(1,2)) for C in [Ci, Cd, Ch, Cm]]
CiNF, CdNF, ChNF, CmNF = [ln.norm(C, axis=(1,2)) for C in [CiNum, CdNum, ChNum, CmNum]]

CiDF, CdDF, ChDF, CmDF = [ln.norm(C, axis=(1,2))
    for C in [Ci - CiNum, Cd - CdNum, Ch - ChNum, Cm - CmNum]]

li, ld, lh, lm = np.empty((4,npts,3), dtype=float)
liN, ldN, lhN, lmN = np.empty((4,npts,3), dtype=float)

# %%
def sacoValsAng(C):
    npts = C.shape[0]
    L = np.empty((npts, 3), dtype=float)
    # calculo valores singulares y angulo
    for i in range(npts):
        l, v = ln.eig(C[i])
        L[i] = [np.sqrt(l[0].real), np.sqrt(l[1].real), np.arctan(v[0, 1] / v[0, 0])]
    return L

Li, Ld, Lh, Lm = [sacoValsAng(C) for C in [Ci, Cd, Ch, Cm]]
LiN, LdN, LhN, LmN = [sacoValsAng(C) for C in [CiNum, CdNum, ChNum, CmNum]]


# %% "indicadores" de covarianza
# radio desde centro optico
rads = ln.norm(imgPts - cameraMatrix[:2, 2], axis=1)

tVm = cl.rotateRodrigues(rtV[3:],-rtV[:3]) # traslation vector in maps frame of ref
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
plt.scatter(rads, CdF)
plt.scatter(rads, CdNF)

plt.subplot(223)
plt.scatter(rads, ChF)
plt.scatter(rads, ChNF)

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
plt.scatter(rads, Ld[:, var])
plt.scatter(rads, LdN[:, var])

plt.subplot(343)
plt.scatter(rads, Lh[:, var])
plt.scatter(rads, LhN[:, var])

plt.subplot(344)
plt.scatter(rads, Lm[:, var])
plt.scatter(rads, LmN[:, var])

var = 1
plt.subplot(345)
plt.scatter(rads, Li[:, var])
plt.scatter(rads, LiN[:, var])

plt.subplot(346)
plt.scatter(rads, Ld[:, var])
plt.scatter(rads, LdN[:, var])

plt.subplot(347)
plt.scatter(rads, Lh[:, var])
plt.scatter(rads, LhN[:, var])

plt.subplot(348)
plt.scatter(rads, Lm[:, var])
plt.scatter(rads, LmN[:, var])


var = 2
plt.subplot(349)
plt.scatter(rads, Li[:, var])
plt.scatter(rads, LiN[:, var])

plt.subplot(3,4,10)
plt.scatter(rads, Ld[:, var])
plt.scatter(rads, LdN[:, var])

plt.subplot(3,4,11)
plt.scatter(rads, Lh[:, var])
plt.scatter(rads, LhN[:, var])

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
plt.scatter(Ld[:, var], LdN[:, var])
plt.plot([np.min(Ld[:, var]), np.max(Ld[:, var])],
          [np.min(LdN[:, var]), np.max(LdN[:, var])], '-k')

plt.subplot(343)
plt.scatter(Lh[:, var], LhN[:, var])
plt.plot([np.min(Lh[:, var]), np.max(Lh[:, var])],
          [np.min(LhN[:, var]), np.max(LhN[:, var])], '-k')

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
plt.scatter(Ld[:, var], LdN[:, var])
plt.plot([np.min(Ld[:, var]), np.max(Ld[:, var])],
          [np.min(LdN[:, var]), np.max(LdN[:, var])], '-k')

plt.subplot(347)
plt.scatter(Lh[:, var], LhN[:, var])
plt.plot([np.min(Lh[:, var]), np.max(Lh[:, var])],
          [np.min(LhN[:, var]), np.max(LhN[:, var])], '-k')

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
plt.scatter(Ld[:, var], LdN[:, var])
plt.plot([np.min(Ld[:, var]), np.max(Ld[:, var])],
          [np.min(LdN[:, var]), np.max(LdN[:, var])], '-k')

plt.subplot(3,4,11)
plt.scatter(Lh[:, var], LhN[:, var])
plt.plot([np.min(Lh[:, var]), np.max(Lh[:, var])],
          [np.min(LhN[:, var]), np.max(LhN[:, var])], '-k')

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



# %% saco la norma de frobenius de todas y comparo
N = 1000  # cantidad de realizaciones
imSel = 30  # ELIJO UNA DE LAS IMAGENES

covSuperList = list()

for imSel in range(0,33,4):
    print(imSel)
    rtV = intrCalib['exMean'][imSel]
    imgPts = imagePoints[imSel,0]

    covLim0 = 6 * imSel
    covLim1 = covLim0 + 6
    Crt = intrCalib['exCov'][covLim0:covLim1,covLim0:covLim1]
    Dextr = cl.unit2CovTransf(Crt)

    npts = imgPts.shape[0]


    retAll = analyticVsMC(imgPts, Ci, cameraMatrix, Cf, distCoeffs, Ck, Cfk,
                          rtV, Crt, retPts=False)
#    covTeo, covNum = retAll
#    Ci, Cd, Ch, Cm = covTeo
#    CiNum, CdNum, ChNum, CmNum = covNum

    covSuperList.append(retAll)

'''
los indices de la supermatriz:
(a,b,c,d,e)
a: imagen seleccionada
b: 0 para teorico, 1 par anumerico
c: imagen, disotrted, homogeneus, world
d: los puntos en una imagen
e, f: son los 2 indices de la covarianza
'''

covsTeo, covsNum = np.array(covSuperList).transpose((1, 2, 0, 3, 4, 5)).reshape((2, 4, -1, 2, 2))

scaleNum = np.linalg.inv(covsNum)


# saco la norma de frobenius de cada matriz
covSuperFrob = np.linalg.norm(covSuperList, axis=(4, 5))

# %%
p = 2
matt = np.eye(p)

np.exp(-p/2) / spe.gamma(N/2) / 2**(N*p/2) / np.linalg.det(matt)**(p/2+0.5)


sts.wishart.pdf(matt, df=N-1, scale=matt)



# %%
rv = sts.wishart()
rv.pdf()


frobQuotList = (covSuperFrob[:, 0] / covSuperFrob[:,1]).transpose((1, 0, 2)).reshape((4, -1))


plt.plot(frobQuotList)
plt.hist(frobQuotList[3])
plt.violinplot(frobQuotList.T, showmeans=True, showextrema=False)
