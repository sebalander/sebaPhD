# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:30:53 2016

calibrates intrinsic with diff distortion model

@author: sebalander
"""

# %%
import glob
import numpy as np
from calibration import calibrator as cl
import matplotlib.pyplot as plt


# %% LOAD DATA
# input
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
modelos = ['poly', 'rational', 'fisheye', 'stereographic']
model = modelos[3]
plotCorners3D = False
plotCornersDirect = False
plotCornersInverse = False

imagesFolder = "./resources/intrinsicCalib/" + camera + "/"
cornersFile =      imagesFolder + camera + "Corners.npy"
patternFile =      imagesFolder + camera + "ChessPattern.npy"
imgShapeFile =     imagesFolder + camera + "Shape.npy"

# output
distCoeffsFile =   imagesFolder + camera + model + "DistCoeffs.npy"
linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
tVecsFile =        imagesFolder + camera + model + "Tvecs.npy"
rVecsFile =        imagesFolder + camera + model + "Rvecs.npy"

# load data
imagePoints = np.load(cornersFile)
chessboardModel = np.load(patternFile)
imgSize = np.load(imgShapeFile)
images = glob.glob(imagesFolder+'*.png')

n = len(imagePoints)  # cantidad de imagenes
# Parametros de entrada/salida de la calibracion
objpoints = np.array([chessboardModel]*n)
m = chessboardModel.shape[1]  # cantidad de puntos

## %% para que calibrar ocmo fisheye no de error
##    flags=flags, criteria=criteria)
##cv2.error: /build/opencv/src/opencv-3.2.0/modules/calib3d/src/fisheye.cpp:1427: error: (-215) svd.w.at<double>(0) / svd.w.at<double>((int)svd.w.total() - 1) < thresh_cond in function CalibrateExtrinsics
##http://answers.opencv.org/question/102750/fisheye-calibration-assertion-error/
## saco algunas captiras de calibracion
#indSelect = np.arange(n)
#np.random.shuffle(indSelect)
#indSelect = indSelect<10

# %% OPTIMIZAR
#rms, K, D, rVecs, tVecs = cl.calibrateIntrinsic(objpoints[indSelect], imgpoints[indSelect], imgSize, model)

# saco condiciones iniciales para los parametros uso opencv y heuristica

#reload(cl)
if model is modelos[3]:
    # saco rVecs, tVecs
    rms, K, D, rVecs, tVecs = cl.calibrateIntrinsic(objpoints, imagePoints,
                                                    tuple(imgSize), modelos[2])
    
    # la matriz de lineal al CCD, esta no se toca, no se optimiza
    K = np.eye(3)
    K[[0,1],2] = imgSize / 2
    
    # la constante de distorsion es tal que ebarca toda la imagen
    D =  np.array([K[0,2]])
    
else:
    rms, K, D, rVecs, tVecs = cl.calibrateIntrinsic(objpoints, imagePoints,
                                                    tuple(imgSize), model)



# %% plot fiducial 
#points and corners to ensure the calibration data is ok
if plotCorners3D:
    for i in range(n): # [9,15]:
        rVec = rVecs[i]
        tVec = tVecs[i]
        fiducial1 = chessboardModel
        
        cl.fiducialComparison3D(rVec, tVec, fiducial1)
#

# %% TEST DIRECT MAPPING (DISTORTION MODEL)
# pruebo con la imagen j-esima

if plotCornersDirect:
    for j in range(n):  # range(len(imgpoints)):
        imagePntsX = imagePoints[j, 0, :, 0]
        imagePntsY = imagePoints[j, 0, :, 1]
    
        rvec = rVecs[j]
        tvec = tVecs[j]
    
        imagePointsProjected = cl.direct(chessboardModel, rvec, tvec, K, D, model)
        imagePointsProjected = imagePointsProjected.reshape((-1,2))
    
        xPos = imagePointsProjected[:, 0]
        yPos = imagePointsProjected[:, 1]
    
        plt.figure()
        im = plt.imread(images[j])
        plt.imshow(im)
        plt.plot(imagePntsX, imagePntsY, 'xr', markersize=10)
        plt.plot(xPos, yPos, '+b', markersize=10)
        #fig.savefig("distortedPoints3.png")
#


# %% TEST INVERSE MAPPING (DISTORTION MODEL)
# pruebo con la imagen j-esima

if plotCornersInverse:
    plt.figure()
    plt.plot(chessboardModel[0,:,0], chessboardModel[0,:,1], 'xr', markersize=10)

    for j in range(n):  # range(len(imgpoints)):
        rvec = rVecs[j]
        tvec = tVecs[j]
    
        xPos, yPos, cc = cl.inverse(imagePoints[j, 0], rvec, tvec, K, D, model)
    
        
        im = plt.imread(images[j])
        print(j, cc, images[j])
        plt.plot(xPos, yPos, '+b', markersize=10)













# %% START BAYESIAN CALIBRATION
from dev import bayesLib as bl
from importlib import reload
import scipy.stats as sts
import sys
sys.path.append("/home/sebalander/Code/sebaPhD")
reload(cl)
reload(bl)

# standar deviation from subpixel epsilon used
std = 1.0

# output file
intrinsicParamsOutFile = imagesFolder + camera + model + "intrinsicParamsML"
intrinsicParamsOutFile = intrinsicParamsOutFile + str(std) + ".npy"

# pongo en forma flat los valores iniciales
XextList = [bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)]
Xint, Ns = bl.int2flat(K, D, model)
covar0 = np.diag((Xint*1e-3)**2)

Ci = np.repeat([ std**2 * np.eye(2)],n*m, axis=0).reshape(n,m,2,2)
params = dict()
params["n"] = n
params["m"] = m
params["imagePoints"] = imagePoints
params["model"] = model
params["chessboardModel"] = chessboardModel
params["Cccd"] = Ci
params["Cf"] = False
params["Ck"] = False
params["Crt"] = False
params["model"] = model
#params = [n, m, imagePoints, model, chessboardModel, Ci]

# %%
# pruebo con una imagen

j = 0
ErIm = bl.errorCuadraticoImagen(XextList[j], Xint, Ns, params, j, mahDist=False)
print(ErIm.sum())

# %% pruebo el error total


Erto = bl.errorCuadraticoInt(Xint, Ns, XextList, params, mahDist=False)
E0 = bl.etotalInt(Xint, Ns, XextList, params)
print(Erto.sum(), E0)

# saco distancia mahalanobis de cada proyeccion
mahDistance = bl.errorCuadraticoInt(Xint, Ns, XextList, params, mahDist=True)

plt.figure()
nhist, bins, _ = plt.hist(mahDistance, 50, normed=True)
chi2pdf = sts.chi2.pdf(bins, 2)
plt.plot(bins, chi2pdf)
plt.yscale('log')


# %% primera prueba de MH la hago medio adaptativa
reload(bl)
from copy import deepcopy as dc
import time
import numpy.linalg as ln

# primera propuesta de pdf
sampleador = sts.multivariate_normal(Xint, covar0)

# objeto que se encarga de una iteracion
metropolis = bl.metHas(Ns, XextList, params, sampleador)

#Nmuestras = int(1e2)
Nini = 4**Xint.shape[0]
#frac = np.arange(Nmuestras) / Nmuestras

paraMuest = list() # np.zeros((Nmuestras, Xint.shape[0]))
errorMuestras = list() # np.zeros(Nmuestras)

paraMuest.append(Xint)
errorMuestras.append(E0)

# primero saco muestras solo cambiando el mean
for i in range(1, Nini):
    paM, erM = metropolis.nuevo(paraMuest[i-1], errorMuestras[i-1])
    paraMuest.append(paM)
    errorMuestras.append(erM)
    
    sampleador.mean = paraMuest[-1]


# muestras cambiando el mean y covarianza
covMuestras = list()
covMuestras.append(covar0)
while True:
    paM, erM = metropolis.nuevo(paraMuest[-1], errorMuestras[-1])
    paraMuest.append(paM)
    errorMuestras.append(erM)
    
    sampleador.mean = paraMuest[-1]
    # discard outliers and burn-in for covariance
    indexes = np.argsort(errorMuestras)[:int(len(errorMuestras)*0.95)]
    sampleador.cov = np.cov(np.array(paraMuest)[indexes].T)
    
    covMuestras.append(sampleador.cov)
    covMuestrasMean = np.mean(covMuestras[-indexes.shape[0]:],0)
    
    eps = ln.norm(covMuestras[-1] - covMuestrasMean) / ln.norm(covMuestrasMean )
    print(eps)
    
    if eps < 1e-2:
        break

covarianzas = np.array(covMuestras).reshape((-1,Xint.shape[0]**2))
medias = np.array(paraMuest)

plt.figure()
plt.plot(covarianzas - covarianzas[-1])

plt.figure()
plt.plot(medias - medias[-1])

# %%
reload(bl)
from copy import deepcopy as dc
import time

sampleador = sts.multivariate_normal(Xint, covar0)

metropolis = bl.metHas(Ns, XextList, params, sampleador)

Nmuestras = int(1e4)
Mmuestras = int(50)
nTot = Nmuestras * Mmuestras

paraMuest = np.zeros((Nmuestras,8))
errorMuestras = np.zeros(Nmuestras)

paraMuest[0], errorMuestras[0] = (Xint, E0)

tiempoIni = time.time()

for j in range(Mmuestras):

    for i in range(1, Nmuestras):
        paraMuest[i], errorMuestras[i] = metropolis.nuevo(paraMuest[i-1], errorMuestras[i-1])
        sampleador.mean = paraMuest[i]

        if i < 500: # no repito estas cuentas despues de cierto tiempo
            tiempoNow = time.time()
            Dt = tiempoNow - tiempoIni
            frac = (i  + Nmuestras * j)/ nTot
            DtEstimeted = (tiempoNow - tiempoIni) / frac
            stringTimeEst = time.asctime(time.localtime(tiempoIni + DtEstimeted))

        print('Epoch: %d/%d-%d/%d. Transcurrido: %.2fmin. Avance %.4f. Tfin: %s'
              %(j,Mmuestras,i,Nmuestras,Dt/60, frac, stringTimeEst) )
    # guardo estos datos
    np.save("/home/sebalander/Documents/datosMHintrinsic2-%d"%j, paraMuest)
    # para la proxima realizacion pongo la primera donde dejamos esta
    paraMuest[0], errorMuestras[0] = (paraMuest[-1], errorMuestras[-1])



# %% SAVE CALIBRATION
np.save(distCoeffsFile, D)
np.save(linearCoeffsFile, K)
np.save(tVecsFile, tVecs)
np.save(rVecsFile, rVecs)