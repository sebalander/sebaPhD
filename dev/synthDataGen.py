#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:19:55 2018

simular las posese extrinsicas y los datos de calibracion

@author: sebalander
"""
# %% imports
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import collections as clt


'''
Fields with default values:

    model = 'stereographic' string indicating camera intrinsic model
            ['poly', 'rational', 'fisheye', 'stereographic']

    camera = 'vcaWide' string camera model

    s = np.array([1600, 900]) is the image size

    k = 800 is the sintehtic stereographic parameter

    uv= np.array([800, 450]) is the stereographic optical center

    cornersIn = None numpy array that holds the detected corners in
                chessboard images

    chessModel = None numpy array of the chessboard model, 3D

    cornersEx = None numpy array of the real world corners in urban setting

    objExtr = None numpy array of corresponding lat-lon (or whatever unit)
              of world calibration points.

    poseRvecs = [] los rVecs asociados a los angulos de las poses
            sinteticas para las calibraciones extrinsecas simuladas

    poseTvecs = [] los tVecs asociados a angulos y alturas de las poses
            sinteticas para las calibraciones extrinseas
            poseTvecs[i,j] es el tVec del angulo i-esimo y altura j-esima

Data
    Synt
        Intr         # listo: synthintr
            k
            uv
            camera
            model
            s
        Ches         # listo: synthches
            nIm
            nPt
            objPt
            imgPt
            rVecs
            tVecs
        Extr
            ang
            h
            rVecs
            tVecs
            objPt
            imgPt
            index10
    Real
        Ches         # listo: realches
            nIm
            nPt
            objPt
            imgPt
            imgFiles
        Balk
            objPt
            imgPt
        Dete
'''

SynthIntr = clt.namedtuple('Intr', ['k', 'uv', 's', 'camera', 'model'])
SynthChes = clt.namedtuple('Ches', ['nIm', 'nPt', 'objPt', 'imgPt',
                                      'rVecs', 'tVecs'])
SynthExtr = clt.namedtuple('Extr', ['ang', 'h', 'rVecs', 'tVecs', 'objPt',
                                    'imgPt', 'index10'])
Synt = clt.namedtuple('Synt', ['Intr', 'Ches', 'Extr'])

RealChes = clt.namedtuple('Ches', ['nIm', 'nPt', 'objPt', 'imgPt', 'imgFiles'])
RealBalk = clt.namedtuple('Balk', ['objPt', 'imgPt'])
RealDete = clt.namedtuple('Dete', [])
Real = clt.namedtuple('Real', ['Ches', 'Balk', 'Dete'])

Data = clt.namedtuple('Data', ['Synt', 'Real'])

# %% Sinteticos intrínsecos
camera = 'vcaWide'
model='stereographic'
s = np.array((1600, 900))
uv = s / 2.0
k = 800.0  # a proposed k for the camera, en realidad estimamos que va a ser #815

synthintr = SynthIntr(k, uv, s, camera, model)

# %% chessboard data, la separo para real y sintetico

imagesFolder = "./resources/intrinsicCalib/" + camera + "/"
cornersFile = imagesFolder + camera + "Corners.npy"
patternFile = imagesFolder + camera + "ChessPattern.npy"
imgShapeFile = imagesFolder + camera + "Shape.npy"

# model data files opencv
# distCoeffsFileOCV = imagesFolder + camera + 'fisheye' + "DistCoeffs.npy"
# linearCoeffsFileOCV = imagesFolder + camera + 'fisheye' + "LinearCoeffs.npy"
tVecsFileOCV = imagesFolder + camera + 'fisheye' + "Tvecs.npy"
rVecsFileOCV = imagesFolder + camera + 'fisheye' + "Rvecs.npy"


# ## load data
RealChesImgPt = np.load(cornersFile)
ChesObjPt = np.load(patternFile)[0]
#s = np.load(imgShapeFile)
RealChesImgFiles = glob.glob(imagesFolder + '*.png')

ChesnIm = RealChesImgPt.shape[0]
ChesnPt = RealChesImgPt.shape[2]  # cantidad de puntos por imagen

# load model specific data from opencv Calibration
# distCoeffsOCV = np.load(distCoeffsFileOCV)
# cameraMatrixOCV = np.load(linearCoeffsFileOCV)
SyntChesRvecs = np.load(rVecsFileOCV)
SyntChesTvecs = np.load(tVecsFileOCV)

realches = RealChes(ChesnIm, ChesnPt, ChesObjPt, RealChesImgPt,
                    RealChesImgFiles)

# ahora hago la proyección hacia la imagen

from calibration import calibrator as cl
cameraMatrix = np.eye(3)
cameraMatrix[[0,1],2] = uv
distCoeff = np.array([k])

SynthChesImPts = np.array([cl.direct(ChesObjPt, SyntChesRvecs[i],
                                     SyntChesTvecs[i], cameraMatrix, distCoeff,
                                     model) for i in range(ChesnIm)])

synthches = SynthChes(ChesnIm, ChesnPt, ChesObjPt, SynthChesImPts,
                      SyntChesRvecs, SyntChesTvecs)








# %% table of poses sintéticas
fov = np.deg2rad(183 / 2)  # angulo hasta donde ve la camara

hs = np.array([7.5, 15])
angs = np.deg2rad([0, 30, 60])

# para las orientaciones consideroq ue el versor x es igual al canónico
# primero los tres versores en sus tres casos. CADA COLUMNA es un caso
xC = np.array([[1,0,0]]*3).T
zC = np.array([np.zeros(3), np.sin(angs), -np.cos(angs)])
yC = np.array([zC[0], zC[2], -zC[1]])

# serían tres matrices de rotación, las columnas son los versores
Rmats = np.concatenate([[xC], [yC], [zC]]).transpose((2,1,0))

# calculos los rVecs
rVecs = np.array([cv2.Rodrigues(R)[0] for R in Rmats]).reshape((-1, 3))

# los tVecs los saco de las alturas
T0 = np.zeros((3,2))
T0[2] = hs

# son 6 vectores tVecs[i, j] es el correspondiente al angulo i y altura j
tVecs = np.transpose(- Rmats.dot(T0), (0, 2, 1))

allData.poseRvecs = rVecs
allData.poseTvecs = tVecs

# %%
Npts = np.array([10, 20])
totPts = np.sum(Npts)

# radios en el piso
np.tan(fov - angs).reshape((1, -1)) * hs.reshape((-1, 1))
# los radios dan muy chicos. voya  definir por default que el radio de los
# puntos de calbracion es 50m
rMaxW = 50

# genero puntos equiprobables en un radio unitario:
xy = []

while len(xy) < totPts:
    xyInSquare = np.random.rand(2 * totPts, 2)
    areInCircle = np.linalg.norm(xyInSquare, axis=1) <= 1

    xy = xyInSquare[areInCircle]

    if len(xy) >= totPts:
        xy = xy[:totPts]
        break

plt.scatter(xy[:,0], xy[:,1])
plt.axis('equal')
# no me gusta.

# K-means
# puntos en un radio unitario, en grilla cuadrada de unidad 1/100
x, y = np.array(np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200)))
areInCircle = (x**2 + y**2) <=1
plt.matshow(areInCircle)

x = x[areInCircle]
y = y[areInCircle]

plt.scatter(x, y)
plt.axis('equal')

xy = np.concatenate((x,y)).reshape((2, -1))

plt.scatter(xy[0], xy[1])
plt.axis('equal')

from scipy.cluster.vq import kmeans2 as km
data = xy.T

#mu1, labels1 = km(data, Npts[0], iter=100, thresh=1e-07, minit='points',)
mu2, labels2 = km(data, Npts[1], iter=100, thresh=1e-07, minit='points',)
#
#plt.subplot(121)
#plt.scatter(xy[0], xy[1], c=labels1, cmap='tab20')
#plt.axis('equal')
#plt.scatter(mu1[:, 0], mu1[:, 1], marker='x', s=20, c='k')

#plt.subplot(122)
plt.scatter(xy[0], xy[1], c=labels2, cmap='tab20')
plt.axis('equal')
plt.scatter(mu2[:, 0], mu2[:, 1], marker='x', s=10, c='k')
for i in range(Npts[1]):
    plt.text(mu2[i, 0], mu2[i, 1], i)

# indexes to select from 20 to 10
indSel = [2, 3, 7, 8, 18, 6, 4, 11, 16, 19]
mu1 = mu2[indSel]
plt.scatter(mu1[:, 0], mu1[:, 1], marker='<', s=30, c='k')

# ahora los convierto al plano del mundo
# rMaxW
thMaxW = np.arctan2(rMaxW, hs[0])
rMaxIm = k * np.tan(thMaxW / 2)
# este es el radio en la imagen. para escalear los mu
mu2 *= rMaxIm
#fiIm = np.arctan2(mu2[:, 1], mu2[:, 0])
rIm = np.linalg.norm(mu2, axis=1)
thW = 2 * np.arctan(rIm / k)
rW = hs[0] * np.tan(thW)

muW = (mu2.T * rW / rIm).T
muW2 = muW[indSel]

allData.cornersEx = [muW2, muW]

# %%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(muW[:, 0], muW[:, 1], 0, c='r')
ax.scatter(muW2[:, 0], muW2[:, 1], 0, marker='>', c='k')

ax.scatter([0,0], [0, 0], hs, marker='X')


for i in range(Npts[1]):
    ax.text(muW[i, 0], muW[i, 1], 0, s=i)

# %%






