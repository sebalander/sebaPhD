#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 19:16:43 2017

hacer la calibracion de los datos tomados en nov 2016

@author: sebalander
"""


# %%
import cv2
from copy import deepcopy as dc
from calibration import calibrator as cl
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload



# %% LOAD DATA
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
model = 'rational'

# model files
modelFile = "./resources/intrinsicCalib/" + camera + "/"
distCoeffsFile =   modelFile + camera + model + "DistCoeffs.npy"
cameraMatrixFile = modelFile + camera + model + "LinearCoeffs.npy"
imgShapeFile =     modelFile + camera + "Shape.npy"

# load data
cameraMatrix = np.load(cameraMatrixFile) # coef intrinsecos
distCoeffs = np.load(distCoeffsFile)
imgShape = np.load(imgShapeFile)

# %%
# data files
rawDataFile = '/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/2016-11-13 medicion/calibrExtr/'
imgFile = rawDataFile + 'vcaSnapShot.png'
dawCalibTxt = rawDataFile + 'puntosCalibracion.txt'

ptsCalib = np.loadtxt(dawCalibTxt)
img = cv2.imread(imgFile)

# corners in image must have shape (N,1,2)
imagePoints = ptsCalib[:, :2].reshape((-1,2))
# pongo longitud como X y latitud como Y
# points in 3D wolrd must have shape
objectPoints = np.concatenate((ptsCalib[:, 3:1:-1],
                               np.zeros((len(ptsCalib),1)) ),
                               axis=1).reshape((-1,3))

# hago deepcopy para que no chille solvePnP
# http://answers.opencv.org/question/1073/what-format-does-cv2solvepnp-use-for-points-in-python/
imagePoints = dc(imagePoints)
objectPoints = dc(objectPoints)
N = imagePoints.shape[0]  # cantidad de puntos

# %% POSE INICIAL
# height calculated from picture
h_pix = np.linalg.norm([544-530,145-689]) # balcon

pixs = np.array([np.linalg.norm([541-545,319-299]), # parabolica 1
                 np.linalg.norm([533-552,310-307]), # parabolica 2
                 np.linalg.norm([459-456,691-624]), # persona
                 np.linalg.norm([798-756,652-651]), # 5xancho de cajon cerveza
                 np.linalg.norm([767-766,668-613])]) # 4xalto de cajon cerveza

# corresponding values in meters
# usando las medidas de cajon de cerveza de ulises
mets = np.array([0.6, 0.6, 1.8, 5*0.297, 4*0.342]) # 

h_m = h_pix*mets/pixs
h_m

# according to google earth, a good a priori position is
# -34.629344, -58.370350
y0, x0 = -34.629344, -58.370350
z0 = 15.7 # metros, as measured by oliva
# and a rough conversion to height is using the radius of the earth
# initial height 
z0 = z0 * 180.0 / np.pi / 6400000.0 # now in degrees


tVecIni = np.array([x0, y0, z0])

# el versos de direccion Z es el mas facil y fiable dado tVecIni y el lugar a
# donde apunta el centro de la imagen en el mapa, hay que sacarlo por aparte
plt.imshow(img)
centerPoint = np.array(img.shape)[1::-1] / 2
plt.plot(centerPoint[0], centerPoint[1],'+r', markersize=50)
plt.plot(cameraMatrix[0,2], cameraMatrix[1,2],'+g', markersize=50)
# calculo la posicion del centro del eje optico con dos referencias
ref0 = np.array([-58.370435, -34.629379])
ref1 = np.array([-58.370369, -34.629318])
s0 = np.array([11.4, 0])
s1 = np.array([11.4 - 2.4, 1.1])  # marque en papel sobre la pantalla
                                  # son referencias sobre la foto
S0 = ref1 - ref0

# direccion X de la imagen como se ve en el mapa
# desde aca -34.629489, -58.370598
# hacia -34.628578, -58.369666
# ahora para el versor Y
# desde -34.629360, -58.370463
# hacia -34.627959, -58.372363
# uso los versores dela camara descriptos respecto al mapa
Xc = np.array([-58.369666 - (-58.370598), -34.628578 - (-34.629489), 0])
Yc = - np.array([-58.372363 - (-58.370463), -34.627959 - (-34.629360), 0])

Xc /= np.linalg.norm(Xc)
Yc /= np.linalg.norm(Yc)
Zc = np.cross(Xc, Yc)

R = np.array([Xc, Yc, Zc])

rVecIni = cv2.Rodrigues(R)[0].T

# condiciones iniciales, veamos de plotear con esto
tVecIni, rVecIni

cl.cornerComparison(img, imagePoints)

cl.fiducialComparison3D(rVecIni, tVecIni, objectPoints)

# %% propongo el versor z apuntando hacia donde es el centro de la imagen
# deduzco la posicion de este punto con un modelo lineal, es una regla de tres 
# simple a partir de los otros puntos de calibracion

# elijo los 5 puntos mas cercanos para no tomar en cuenta 
#
## calculo una transf lineal entre los puntos
#unos = np.ones((N,1))
#
#
#Xi = np.concatenate((imagePoints, unos),axis=1).T
#Xo = np.concatenate((objectPoints[:,:2], unos),axis=1).T
#
#xxInv = np.linalg.inv(np.dot(Xi, Xi.T))
#A = np.dot(Xo, Xi.T).dot(xxInv)
#
## coords de centro de la imagen +o-
#cenI = np.concatenate((cameraMatrix[:2,2],[1]))
#
#cenO = A.dot(cenI)
#plt.figure()
#plt.plot(cenO[0], cenO[1],'+', markersize=4)
#plt.plot(objectPoints[:,0], objectPoints[:,1],'o')
# LO MAS FACIL ES PONER A MANO EL CENTRO ESE, NO VALE LA PENA ANDAR 
# ESTIMANDOLO POR AHORA

# %%
reload(cl)

cl.fiducialComparison(rVecIni, tVecIni,objectPoints)

# %% ver que tan bien funcan las cond iniciales
reload(cl)

imagePointsProjected = cl.direct(objectPoints, rVecIni, tVecIni,
                                 cameraMatrix, distCoeffs, model)

objectPointsProjected = cl.inverse(imagePoints, rVecIni, tVecIni,
                                   cameraMatrix, distCoeffs, model)

# chequear que caigan donde deben
cl.cornerComparison(img, imagePoints, imagePointsProjected)

cl.fiducialComparison(rVecIni, tVecIni, objectPoints, objectPointsProjected)
cl.fiducialComparison3D(rVecIni, tVecIni, objectPoints, objectPointsProjected)

# %%


























## %% uso la funcion dada por opencv
#
#retval, rVec, tVec = cv2.solvePnP(objectPoints, imagePoints,
#                                  cameraMatrix, distCoeffs)
#
## %%
#objectPointsProjected = cl.inverse(imagePoints, rVec, tVec, cameraMatrix,
#                                   distCoeffs, model)
#
#cl.fiducialComparison3D(rVec, tVec, objectPoints, objectPointsProjected)
#cl.fiducialComparison(objectPoints, objectPointsProjected)
#
#
#
#
## %% USAR OPENCV PERO CON COND INICIALES CONOCIDAS
#
#retval, rVec, tVec = cv2.solvePnP(objectPoints, imagePoints,
#                                  cameraMatrix, distCoeffs,
#                                  rVecIni, tVecIni, useExtrinsicGuess = True)
## NOT WORKING. TRY ANOTHER ALGORITHM
#
## %%
##cv2.SOLVEPNP_EPNP
##cv2.SOLVEPNP_DLS
##cv2.SOLVEPNP_UPNP
#retval, rVec, tVec = cv2.solvePnP(objectPoints, imagePoints,
#                                  cameraMatrix, distCoeffs,
#                                  flags=cv2.SOLVEPNP_EPNP)
#
#retval, rVec, tVec = cv2.solvePnP(objectPoints, imagePoints,
#                                  cameraMatrix, distCoeffs,
#                                  flags=cv2.SOLVEPNP_DLS)
#
#retval, rVec, tVec = cv2.solvePnP(objectPoints, imagePoints,
#                                  cameraMatrix, distCoeffs,
#                                  flags=cv2.SOLVEPNP_UPNP)



## %%
#objectPointsProjected = cl.inverse(imagePoints, rVec, tVec, cameraMatrix,
#                                   distCoeffs, model)
#
#cl.fiducialComparison3D(rVec, tVec, objectPoints, objectPointsProjected)
#cl.fiducialComparison(objectPoints, objectPointsProjected)





## %% PARAMTER HANDLING
## tweaking linear params so we get correct units
#linearCoeffs = load(linearCoeffsFile) # coef intrinsecos
#distCoeffs = load(distCoeffsFile)
#
#
## giving paramters the appropiate format for the optimisation function
#paramsIni = pc.formatParameters(rVecIni, tVecIni,
#                                linearCoeffs, distCoeffs, model)
## also, retrieving the numerical values
#pc.retrieveParameters(paramsIni, model)
#
## %% CASE 1: use direct mapping to test initial parameters
## project
#cornersCase1 = pc.direct(fiducialPoints, rVecIni, tVecIni,
#                         linearCoeffs, distCoeffs, model)
## plot to compare
#pc.cornerComparison(img, imageCorners, cornersCase1)
## calculate residual
#sum(pc.residualDirect(paramsIni, fiducialPoints, imageCorners, model)**2)

