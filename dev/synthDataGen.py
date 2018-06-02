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
from scipy.cluster.vq import kmeans2 as km
from scipy.cluster.vq import vq
from scipy.spatial import Delaunay



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
        Intr         # listo: SynthIntr
            k
            uv
            camera
            model
            s
        Ches         # listo: SynthChes
            nIm
            nPt
            objPt
            imgPt
            rVecs
            tVecs
        Extr         # listo: SynthExtr
            ang
            h
            rVecs
            tVecs
            objPt
            imgPt
            index10
    Real
        Ches         # listo: RealChes
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

synthintr = clt.namedtuple('synthintr', ['k', 'uv', 's', 'camera', 'model'])
synthches = clt.namedtuple('synthches', ['nIm', 'nPt', 'objPt', 'imgPt',
                                      'rVecs', 'tVecs'])
synthextr = clt.namedtuple('synthextr', ['ang', 'h', 'rVecs', 'tVecs', 'objPt',
                                    'imgPt', 'index10'])
synt = clt.namedtuple('synt', ['Intr', 'Ches', 'Extr'])

realches = clt.namedtuple('realches', ['nIm', 'nPt', 'objPt', 'imgPt', 'imgFiles'])
realbalk = clt.namedtuple('realbalk', ['objPt', 'imgPt'])
realdete = clt.namedtuple('realdete', [])
real = clt.namedtuple('real', ['Ches', 'Balk', 'Dete'])

datafull = clt.namedtuple('datafull', ['Synt', 'Real'])


# %% =========================================================================
# SYNTHETIC INTRINSIC

camera = 'vcaWide'
model='stereographic'
s = np.array((1600, 900))
uv = s / 2.0
k = 800.0  # a proposed k for the camera, en realidad estimamos que va a ser #815

SynthIntr = synthintr(k, uv, s, camera, model)



# %% =========================================================================
# CHESSBOARD DATA, la separo para real y sintetico

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

RealChes = realches(ChesnIm, ChesnPt, ChesObjPt, RealChesImgPt,
                    RealChesImgFiles)

# ahora hago la proyección hacia la imagen

from calibration import calibrator as cl
cameraMatrix = np.eye(3)
cameraMatrix[[0,1],2] = uv
distCoeff = np.array([k])

SynthChesImPts = np.array([cl.direct(ChesObjPt, SyntChesRvecs[i],
                                     SyntChesTvecs[i], cameraMatrix, distCoeff,
                                     model) for i in range(ChesnIm)])

SynthChes = synthches(ChesnIm, ChesnPt, ChesObjPt, SynthChesImPts,
                      SyntChesRvecs, SyntChesTvecs)

plt.figure()
plt.scatter(SynthChes.imgPt[:, :, 0], SynthChes.imgPt[:, :, 1])



# %% =========================================================================
#  SYNTHETIC EXTRINSIC table of poses sintéticas
angs = np.deg2rad([0, 30, 60])
hs = np.array([7.5, 15])
Npts = np.array([10, 20])

totPts = np.sum(Npts)
thMax = 2 * np.arctan(uv[0] / k)  # angulo hasta donde ve la camara wrt z axis
print(np.rad2deg(thMax))


# para las orientaciones consideroq ue el versor x es igual al canónico
# primero los tres versores en sus tres casos. CADA COLUMNA es un caso
xC = np.array([[1, 0, 0]]*3).T
zC = np.array([np.zeros(3), np.sin(angs), -np.cos(angs)])
yC = np.array([zC[0], zC[2], - zC[1]])

# serían tres matrices de rotación, las columnas son los versores
Rmats = np.concatenate([[xC], [yC], [zC]]).transpose((2,1,0))

# calculos los rVecs
SynthExtrRvecs = np.array([cv2.Rodrigues(R)[0] for R in Rmats]).reshape((-1, 3))

# los tVecs los saco de las alturas
T0 = np.zeros((3,2))
T0[2] = hs

# son 6 vectores tVecs[i, j] es el correspondiente al angulo i y altura j
SynthExtrTvecs = np.transpose(- Rmats.dot(T0), (0, 2, 1))


# %%


# radios en el piso
np.tan(thMax - angs).reshape((1, -1)) * hs.reshape((-1, 1))
# los radios dan muy chicos. voya  definir por default que el radio de los
# puntos de calbracion es 50m
rMaxW = 50

## genero puntos equiprobables en un radio unitario:
#xy = []
#
#while len(xy) < totPts:
#    xyInSquare = np.random.rand(2 * totPts, 2)
#    areInCircle = np.linalg.norm(xyInSquare, axis=1) <= 1
#
#    xy = xyInSquare[areInCircle]
#
#    if len(xy) >= totPts:
#        xy = xy[:totPts]
#        break
#
#plt.scatter(xy[:,0], xy[:,1])
#plt.axis('equal')
## no me gusta.

# K-means
# puntos en un radio unitario, en grilla cuadrada de unidad 1/100
x, y = np.array(np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)))
areInCircle = (x**2 + y**2) <=1

x = x[areInCircle]
y = y[areInCircle]

xy = np.concatenate((x,y)).reshape((2, -1))
data = xy.T


#mu1, labels1 = km(data, Npts[0], iter=100, thresh=1e-07, minit='points',)
mu, labels = km(data, Npts[1], iter=100, thresh=1e-07, minit='points')

tri = Delaunay(mu)
#
#plt.subplot(121)
#plt.scatter(xy[0], xy[1], c=labels1, cmap='tab20')
#plt.axis('equal')
#plt.scatter(mu1[:, 0], mu1[:, 1], marker='x', s=20, c='k')



plt.subplot(131)
plt.imshow(areInCircle)

plt.subplot(132)
plt.scatter(xy[0], xy[1], c=labels, cmap='tab20')
plt.axis('equal')
plt.scatter(mu[:, 0], mu[:, 1], marker='x', s=10, c='k')
plt.triplot(mu[:,0], mu[:,1], tri.simplices)
for i in range(Npts[1]):
    plt.text(mu[i, 0], mu[i, 1], i)


# para imprimir los vecinos del nodo 6
indptr, indices = tri.vertex_neighbor_vertices
vecinos = lambda nod: indices[indptr[nod]:indptr[nod + 1]]


def distortionNodes(indSel):
    mu10 = mu[indSel]
    code, dist = vq(data, mu10)
    return mu10, dist.sum()

# indexes to select from 20 to 10
indSel = np.sort([1,  3,  4,  5,  8,  9, 10, 11, 12, 16])
mu10, distor = distortionNodes(indSel)


# recorro los indices buscando en cada caso el cambio mejor a un vecino 1o
nCambs = -1
loops = -1
loopsMax = 100

distorEvol = list()
distorEvol.append(distor)

while nCambs is not 0 and loops < loopsMax:
    nCambs = 0
    loops += 1
    print('Empezamos un loop nuevo ', loops)
    print('==========================')
    for ii, ind in enumerate(indSel):
        vec = vecinos(ind)
        vec = vec[[v not in indSel for v in vec]]

        indSelList = np.array([indSel]*(len(vec)))
        indSelList[:, ii] = vec

        distorList = [distortionNodes(indVec)[1] for indVec in indSelList]

        if np.all(distor < distorList):
            # me quedo con la que estaba
            print('\tno hay cambio', ii)
        else:
            imin = np.argmin(distorList)
            indSel = np.sort(np.copy(indSelList[imin]))
            distor = distorList[imin]
            distorEvol.append(distor)
            nCambs += 1
            print('\thay cambio, es el ', nCambs, 'elemento', ii)


print(indSel)
mu10, distor = distortionNodes(indSel)



plt.subplot(133)
code, dist = vq(data, mu10)
plt.scatter(xy[0], xy[1], c=code, cmap='tab20')
plt.scatter(mu10[:, 0], mu10[:, 1], marker='<', s=50, c='k')
plt.axis('equal')
plt.triplot(mu[:,0], mu[:,1], tri.simplices)
for i in range(Npts[1]):
    plt.text(mu[i, 0], mu[i, 1], i)


# ahora los convierto al plano del mundo
# rMaxW
thMaxW = np.arctan2(rMaxW, hs[0])
rMaxIm = k * np.tan(thMaxW / 2)
# este es el radio en la imagen. para escalear los mu
muIm = mu * rMaxIm
#fiIm = np.arctan2(mu2[:, 1], mu2[:, 0])
rIm = np.linalg.norm(muIm, axis=1)
thW = 2 * np.arctan(rIm / k)
rW = hs[0] * np.tan(thW)

muW = (muIm.T * rW / rIm).T
muW10 = muW[indSel]

SynthExtrObjPt = np.concatenate([muW, np.zeros((muW.shape[0],1))], axis=1)


# proyecto a la imagen
SynthExtrImPt = np.zeros((len(angs), len(hs), muW.shape[0], 2))
for i in range(len(angs)):
    rv = SynthExtrRvecs[i]
    for j in range(len(hs)):
        tv = SynthExtrTvecs[i, j]
        SynthExtrImPt[i, j] = cl.direct(SynthExtrObjPt, rv, tv, cameraMatrix,
                                        distCoeff, model)

SynthExtr = synthextr(angs, hs, SynthExtrRvecs, SynthExtrTvecs, muW,
                      SynthExtrImPt, indSel)




# %%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(muW[:, 0], muW[:, 1], 0, c='r')
ax.scatter(muW10[:, 0], muW10[:, 1], 0, marker='>', c='k', s=200)

ax.scatter([0,0], [0, 0], hs, marker='X')


for i in range(Npts[1]):
    ax.text(muW[i, 0], muW[i, 1], 0, s=i)

# %%






