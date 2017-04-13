#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:51:26 2017

@author: sebalander
"""



# %%
import cv2
from cv2 import Rodrigues
from copy import deepcopy as dc
from calibration import calibrator as cl
import numpy as np
from numpy import shape
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

# data files
rawDataFile = '/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/2016-11-13 medicion/calibrExtr/'
imgFile = rawDataFile + 'vcaSnapShot.png'
dawCalibTxt = rawDataFile + 'puntosCalibracion.txt'

# output files
tVecIniFile = rawDataFile + 'tVecIni.npy'
rVecIniFile = rawDataFile + 'rVecIni.npy'


# %% CALCULANDO LA ROTOTRASLACION DE MANERA LINEAL COMO EL LIBRO

# ================= DES DISTORSIONO A COORD HOMOGÉNEAS
imageCorners = imagePoints.reshape((-1,2))
distCoeffs = distCoeffs.reshape(14)

plt.figure()
plt.scatter(imageCorners[:,0],imageCorners[:,1])


xpp = (imageCorners[:,0] - linearCoeffs[0,2]) / linearCoeffs[0,0]
ypp = (imageCorners[:,1] - linearCoeffs[1,2]) / linearCoeffs[1,1]
rpp = sqrt(xpp**2 + ypp**2)

plt.figure()
plt.scatter(xpp,ypp)


# polynomial coeffs, grade 7
# # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
poly = [[distCoeffs[4], # k3
         -r*distCoeffs[7], # k6
         distCoeffs[1], # k2
         -r*distCoeffs[6], # k5
         distCoeffs[0], # k1
         -r*distCoeffs[5], # k4
         1,
         -r] for r in rpp]

# choose the maximum of the real roots, hopefully it will be positive
rootsPoly = [roots(p) for p in poly]
rp_rpp = array([max(roo[isreal(roo)].real) for roo in rootsPoly]) / rpp

xp = xpp * rp_rpp
yp = ypp * rp_rpp

# ================= COORD   en  3D
x, y, _ = objectPoints.T


# ================= resulevo el problema lineal

ceros = np.zeros_like(x)
unos = np.ones_like(x)

mitadX = np.array([x, y, unos, ceros, ceros, ceros, -x*xpp, -y*xpp, - xpp])
mitadY = np.array([ceros, ceros, ceros, x, y, unos, -x*ypp, -y*ypp, - ypp])
PP = np.concatenate((mitadX,mitadY), axis=1)

plt.imshow(PP)

PPsquare = np.dot(PP,PP.T)

L, V = np.linalg.eig(PPsquare)

# le doy forma
tVec = V[-1,[2,5,8]]
R1 = V[-1,[0,3,6]]
R2 = V[-1,[1,4,7]]
R1 /= np.linalg.norm(R1) # normalizo los versores
R2 /= np.linalg.norm(R2)
# los versores no dan normales quiza por overfitting??

R3 = np.cross(R1,R2) # completo los que faltan
rVec = np.array([R1,R2,R3]).T

# veo como dan las normas de cada solucion
for i in range(9):
    tVec = V[i,[2,5,8]]
    R1 = V[-i,[0,3,6]]
    R2 = V[i,[1,4,7]]
    r1 = np.linalg.norm(R1)
    r2 =np.linalg.norm(R2)
    (r1, r2, np.linalg.norm(tVec))
    r1/r2
    
    R1 /= r1 # normalizo los versores
    R2 /= r2
    # los versores no dan normales quiza por overfitting??
    
    R3 = np.cross(R1, R2) # completo los que faltan
    rVec = np.array([R1,R2,R3]).T


