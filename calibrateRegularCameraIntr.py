# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:30:53 2016

calibrates using fisheye distortion model (polynomial in theta)

help in
http://docs.opencv.org/ref/master/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d&gsc.tab=0

@author: sebalander
"""

# %%
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# %% LOAD DATA
#imagesFolder = "./resources/fishChessboardImg/"
#cornersFile = "/home/sebalander/code/sebaPhD/resources/fishCorners.npy"
#patternFile = "/home/sebalander/code/sebaPhD/resources/chessPattern.npy"
#imgShapeFile = "/home/sebalander/code/sebaPhD/resources/fishShape.npy"

imagesFolder = "./resources/PTZchessboard/zoom 0.0/"
cornersFile = "/home/sebalander/Code/sebaPhD/resources/PTZchessboard/zoom 0.0/ptzCorners.npy"
patternFile = "/home/sebalander/Code/sebaPhD/resources/chessPattern.npy"
imgShapeFile = "/home/sebalander/Code/sebaPhD/resources/ptzImgShape.npy"

corners = np.load(cornersFile)
fiducialPoints = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.png')

# output files
distCoeffsFile = "/home/sebalander/Code/sebaPhD/resources/PTZchessboard/zoom 0.0/ptzDistCoeffs.npy"
linearCoeffsFile = "/home/sebalander/Code/sebaPhD/resources/PTZchessboard/zoom 0.0/ptzLinearCoeffs.npy"
rvecsFile = "/home/sebalander/Code/sebaPhD/resources/PTZchessboard/zoom 0.0/ptzRvecs.npy"
tvecsFile = "/home/sebalander/Code/sebaPhD/resources/PTZchessboard/zoom 0.0/ptzTvecs.npy"

# %% LINEAR APPROACH TO CAMERA CALIBRATION
n = corners.shape[0] # number of images
m = corners.shape[2] # points per image

#i = 3
#j = 31
#x = corners[i,0,j]
#p = fiducialPoints[0,j]
cer = np.zeros(4)



def linea1(x,y,xp,yp):
    return [x, 0, -x*xp, y, 0, -y*xp, 1, 0, -xp]

def linea2(x,y,xp,yp):
    return [0, x, -x*yp, 0, y, -y*yp, 0, 1, -yp]

def lineas(x,y,xp,yp):
    
    [[x, 0, -x*xp, y, 0, -y*xp, 1, 0, -xp],
     [0, x, -x*yp, 0, y, -y*yp, 0, 1, -yp]]
    
    return np.array([linea1(x,y,xp,yp),linea2(x,y,xp,yp)])

i = 0 # indice de imagen, hasta n
j = 0 # indice de punto, hasta m

x = corners[i,0,j,0]
y = corners[i,0,j,1]
xp = fiducialPoints[0,j,0]
yp = fiducialPoints[0,j,1]

lineas(x,y,xp,yp)

def H(fiducialPoints,corners):
    m = fiducialPoints.shape[1]
    h = [ lineas(fiducialPoints[0,j,0], fiducialPoints[0,j,1],
                 corners[0,j,0], corners[0,j,1])
     for j in range(m)]
    
    h = np.array(h)
    h = np.reshape(h,(m*2,9))
    return h

h = H(corners[0],fiducialPoints)

h2 = np.matmul(h.T,h)

if np.linalg.matrix_rank(h2)==9:
    "tiene rango 9, signifique lo que signifique"

L, V = np.linalg.eigh(h2)

U, S, V = np.linalg.svd(h)
# elijo el menor
v = V[8]
# chequear que no este repetido?
r1 = v[0:3]
r2 = v[3:6]
t = v[6:9]

np.dot(r1, r2) # no dan perpendiculares
r3 = np.cross(r1,r2)


n1 = np.linalg.norm(r1)
n2 = np.linalg.norm(r2) # las normas dan cualquier cosa
n3 = np.linalg.norm(r3)
nt = np.linalg.norm(t)

# componer matriz de rotacion cruda
# ortonormalizarla

# normalizar los vectores, sacar el de rodrigues

# %% 
# hacer optimizacion para terminar de ajustar el pinhole y asi sacar la pose para cada imagen. usar las funciones de las librerias extrinsecas
# es para cada imagen por separado

# %%
# para todas las imagenes juntas dejar la pose fija y optimizar los parametros de distorsion 

# %% optimizar todo junto a la vez tomando lo anterior como condiciones iniciales.