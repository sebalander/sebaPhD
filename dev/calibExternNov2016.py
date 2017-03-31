#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 19:16:43 2017

hacer la calibracion de los datos tomados en nov 2016

@author: sebalander
"""


# %%
from cv2 import imread

from alibration import calibrator as pc
import numpy as np
import matplotlib.pyplot as plt

# %% FILES
# input files
dataFolder = "/home/sebalander/Code/VisionUNQextra/"
dataFolder += "Videos y Mediciones/2016-11-13 medicion/calibrExtr/"

imageFile = dataFolder + "vcaSnapShot.png"
ptosCalibFile = dataFolder + 'puntosCalibracion.txt'

# intrinsic parameters (input)
distCoeffsFile = "resources/fishWideChessboard/fishWideDistCoeffs.npy"
linearCoeffsFile = "resources/fishWideChessboard/fishWideLinearCoeffs.npy"

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


tVecIni = [x0, y0, z0]

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
# tweaking linear params so we get correct units
linearCoeffs = load(linearCoeffsFile) # coef intrinsecos
distCoeffs = load(distCoeffsFile)


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

