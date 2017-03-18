#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 19:16:43 2017

hacer la calibracion de los datos tomados en nov 2016

@author: sebalander
"""


# %%
from cv2 import imread
from numpy import load, sum
import poseCalibration as pc
import numpy as np
import matplotlib.pyplot as plt

# %% FILES
# input files
dataFolder = "/home/sebalander/Code/VisionUNQextra/"
dataFolder += "Videos y Mediciones/2016-11-13 medicion/calibrExtr/"

imageFile = dataFolder + "vcaSnapShot.png"
ptosCalibFile = dataFolder + 'puntosCalibracion.txt'

# intrinsic parameters (input)
distCoeffsFile = "/home/sebalander/Code/sebaPhD/resources/fishWideDistCoeffs.npy"
linearCoeffsFile = "/home/sebalander/Code/sebaPhD/resources/fishWideLinearCoeffs.npy"

model= 'rational'  # tiene que ser compatible con los parametros provistos
# others: 'stereographic' 'fisheye', 'unified'


linearCoeffs = load(linearCoeffsFile) # coef intrinsecos
distCoeffs = load(distCoeffsFile)

# %% leer los puntos de calibracion
ptsCalib = np.loadtxt(ptosCalibFile)
img = plt.imread(imageFile)


imageCorners = ptsCalib[:, :2].reshape((-1,1,2))

# pongo longitud como X y latitud como Y
fiducialPoints = np.concatenate((ptsCalib[:, 3:1:-1],
                           np.zeros((len(ptsCalib),1)) ), axis=1)
fiducialPoints = fiducialPoints.reshape((1,-1,3))

# chequear que caigan donde deben
pc.cornerComparison(img, imageCorners, imageCorners)

# %% POSE INICIAL
# lat lon de lugar de exprimento -34.629352, -58.370357
# tomo 
h = 15.0
R = 6.4e6
tVecIni = [-58.370357, -34.629352, h * 90 / R / np.pi]

# direccion X de la imagen como se ve en el mapa
# desde aca -34.629489, -58.370598
# hacia -34.628578, -58.369666
# ahora para el versor Y
# desde -34.629360, -58.370463
# hacia -34.627959, -58.372363
# propongo homografia que lleva de la camara al mapa
# uso los versores dela camara descriptos respecto al mapa
Xc = np.array([-58.369666 - (-58.370598), -34.628578 - (-34.629489), 0])
Yc = np.array([-58.372363 - (-58.370463), -34.627959 - (-34.629360), 0])

Xc /= np.linalg.norm(Xc)
Yc /= np.linalg.norm(Yc)

H = np.array([[Xc[0], Yc[0], 0, 0],
              [Xc[1], Yc[1], 0, 0],
              [Xc[2], Yc[2], 1, 0]])

rVecIni, tVecIni = pc.homogr2pose(H)


# %% PARAMTER HANDLING
# giving paramters the appropiate format for the optimisation function
paramsIni = pc.formatParameters(rVecIni, tVecIni,
                                linearCoeffs, distCoeffs, model)
# also, retrieving the numerical values
pc.retrieveParameters(paramsIni, model)

# %% CASE 1: use direct mapping to test initial parameters
# project
cornersCase1 = pc.direct(fiducialPoints, rVecIni, tVecIni,
                         linearCoeffs, distCoeffs, model)
# plot to compare
pc.cornerComparison(img, imageCorners, cornersCase1)
# calculate residual
sum(pc.residualDirect(paramsIni, fiducialPoints, imageCorners, model)**2)

