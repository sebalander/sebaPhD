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



header = '''Nested namedtuples that hold the data for the paper

Data
    Synt
        Intr         # listo: SyntIntr
            camera  'vcaWide' string camera model
            model   string indicating camera intrinsic model
                    ['poly', 'rational', 'fisheye', 'stereographic']
            s       is the image size
            k       sintehtic stereographic parameter
            uv      = s / 2 is the stereographic optical center
        Ches         # listo: SyntChes
            nIm     number of images
            nPt     number of point in image
            objPt   chessboard model grid
            rVecs   synth rotation vectors
            tVecs   synth tVecs
            imgPt   synth corners projected from objPt with synth params
            imgNse  noise of 1 sigma for the image
        Extr         # listo: SyntExtr
            ang     angles of synth pose tables
            h       heights  of synth pose tables
            rVecs   rotation vectors associated to angles
            tVecs   tVecs associated to angles and h
            objPt   distributed 3D points on the floor
            imgPt   projected to image
            imgNse  noise for image detection, sigma 1
            index10 indexes to select 10 points well distributed
    Real
        Ches         # listo: RealChes
            nIm     number of chess images
            nPt     number of chess points per image
            objPt   chessboard model, 3D
            imgPt   detected corners in chessboard images
            imgFls  list of paths to the chessboard images
        Balk
            objPt   calibration world points, lat lon
            imgPt   image points for calibration
            priLLA  prior lat-lon-altura
            imgFl   camera snapshot file
        Dete
            carGps  car gps coordinates
            carIm   car image detection traces
'''


#syntintr = clt.namedtuple('syntintr', ['k', 'uv', 's', 'camera', 'model'])
#syntches = clt.namedtuple('syntches', ['nIm', 'nPt', 'rVecs', 'tVecs',
#                                       'objPt', 'imgPt', 'imgNse'])
#syntextr = clt.namedtuple('syntextr', ['ang', 'h', 'rVecs', 'tVecs', 'objPt',
#                                       'imgPt', 'index10', 'imgNse'])
#synt = clt.namedtuple('synt', ['Intr', 'Ches', 'Extr'])
#
#realches = clt.namedtuple('realches', ['nIm', 'nPt', 'objPt', 'imgPt', 'imgFls'])
#realbalk = clt.namedtuple('realbalk', ['objPt', 'imgPt', 'priorLLA', 'imgFl'])
#realdete = clt.namedtuple('realdete', ['carGPS', 'carIm'])
#real = clt.namedtuple('real', ['Ches', 'Balk', 'Dete'])
#
#datafull = clt.namedtuple('datafull', ['Synt', 'Real'])

from calibration.calibrator import datafull, real, realdete, realbalk, realches
from calibration.calibrator import synt, syntextr, syntches, syntintr


# %% =========================================================================
# SYNTHETIC INTRINSIC

camera = 'vcaWide'
model='stereographic'
s = np.array((1600, 904))
uv = s / 2.0
k = 800.0  # a proposed k for the camera, en realidad estimamos que va a ser #815

SyntIntr = syntintr(k, uv, s, camera, model)



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

SyntChesImPts = np.array([cl.direct(ChesObjPt, SyntChesRvecs[i],
                                     SyntChesTvecs[i], cameraMatrix, distCoeff,
                                     model) for i in range(ChesnIm)])

SyntChesImPtsNoise = np.random.randn(np.prod(SyntChesImPts.shape)
                                            ).reshape(SyntChesImPts.shape)

SyntChes = syntches(ChesnIm, ChesnPt, SyntChesRvecs, SyntChesTvecs,
                    ChesObjPt, SyntChesImPts, SyntChesImPtsNoise)


plt.figure()
plt.title('corners de todas las imagenes')
for crnr in SyntChes.imgPt:
    plt.scatter(crnr[:, 0], crnr[:, 1], marker='x', s=5)
plt.axis('equal')

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
SyntExtrRvecs = np.array([cv2.Rodrigues(R)[0] for R in Rmats]).reshape((-1, 3))

# los tVecs los saco de las alturas
T0 = np.zeros((3,2))
T0[2] = hs

# son 6 vectores tVecs[i, j] es el correspondiente al angulo i y altura j
SyntExtrTvecs = np.transpose(- Rmats.dot(T0), (0, 2, 1))


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



#plt.subplot(131)
#plt.imshow(areInCircle)
#
#plt.subplot(132)
#plt.scatter(xy[0], xy[1], c=labels, cmap='tab20')
#plt.axis('equal')
#plt.scatter(mu[:, 0], mu[:, 1], marker='x', s=10, c='k')
#plt.triplot(mu[:,0], mu[:,1], tri.simplices)
#for i in range(Npts[1]):
#    plt.text(mu[i, 0], mu[i, 1], i)


# para imprimir los vecinos del nodo 6
indptr, indices = tri.vertex_neighbor_vertices
vecinos = lambda nod: indices[indptr[nod]:indptr[nod + 1]]


def distortionNodes(indSel):
    mu10 = mu[indSel]
    code, dist = vq(data, mu10)
    return mu10, dist.sum()

# indexes to select from 20 to 10
indSel = np.sort([1,  3,  4,  5,  8,  9, 10, 11, 12, 16])

def optimizeIndSel(indSel, verbose=False):
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
        if verbose:
            print('Empezamos un loop nuevo ', loops)
            print('==========================')
        for ii, ind in enumerate(indSel):
            vec = vecinos(ind)
            vec = vec[[v not in indSel for v in vec]]

            if len(vec) is 0:
                continue

            indSelList = np.array([indSel] * len(vec))
            indSelList[:, ii] = vec

            distorList = [distortionNodes(indVec)[1] for indVec in indSelList]

            if np.all(distor < distorList):
                # me quedo con la que estaba
                if verbose:
                    print('\tno hay cambio', ii)
            else:
                imin = np.argmin(distorList)
                indSel = np.sort(np.copy(indSelList[imin]))
                distor = distorList[imin]
                distorEvol.append(distor)
                nCambs += 1
                if verbose:
                    print('\thay cambio, es el ', nCambs, 'elemento', ii)

    if verbose:
        print(indSel)
    mu10, distor = distortionNodes(indSel)
    return indSel, mu10, distor

nIter = 50
indList20 = np.arange(0, 20)

retOpt = list()
for i in range(nIter):
    indSel = np.random.choice(indList20, 10, replace=False)
    retOpt.append(optimizeIndSel(indSel, False))

argBest = np.argmin([ret[2] for ret in retOpt])

indSel, mu10, distor = retOpt[argBest]

#plt.subplot(133)
plt.figure()
plt.title('distribución en el espacio imagen')
code, dist = vq(data, mu10)
plt.scatter(xy[0], xy[1], c=code, cmap='tab20')
plt.axis('equal')
for i in range(Npts[1]):
    plt.text(mu[i, 0] + 0.05, mu[i, 1] + 0.05, i)
plt.triplot(mu[:,0], mu[:,1], tri.simplices, c='k')
plt.scatter(mu10[:, 0], mu10[:, 1], marker='<', s=100, c='w')


# %%

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

SyntExtrObjPt = np.concatenate([muW, np.zeros((muW.shape[0],1))], axis=1)


# proyecto a la imagen
SyntExtrImPt = np.zeros((len(angs), len(hs), muW.shape[0], 2))
for i in range(len(angs)):
    rv = SyntExtrRvecs[i]
    for j in range(len(hs)):
        tv = SyntExtrTvecs[i, j]
        SyntExtrImPt[i, j] = cl.direct(SyntExtrObjPt, rv, tv, cameraMatrix,
                                        distCoeff, model)

SyntExtrImPtsNoise = np.random.randn(np.prod(SyntExtrImPt.shape))
SyntExtrImPtsNoise = SyntExtrImPtsNoise.reshape(SyntExtrImPt.shape)

SyntExtr = syntextr(angs, hs, SyntExtrRvecs, SyntExtrTvecs, muW,
                      SyntExtrImPt, indSel, SyntExtrImPtsNoise)


# %%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(SyntExtrObjPt[:, 0], SyntExtrObjPt[:, 1], 0, c='r', s=100)
ax.scatter(SyntExtrObjPt[SyntExtr.index10, 0], SyntExtrObjPt[SyntExtr.index10, 1],
           0, c='k', s=200)

cols = ['k', 'b', 'm', 'g', 'r', 'c', 'y',]

for i in range(len(SyntExtr.h)):
    for j in range(len(SyntExtr.ang)):
        rMat = cv2.Rodrigues(SyntExtr.rVecs[j])[0]
        tVw = - rMat.T.dot(SyntExtr.tVecs[j, i])

        print(tVw, np.rad2deg(np.arccos(rMat[:,2].dot([0,0,-1]))))
        for ve in 3 * rMat.T:
            ax.plot([tVw[0], tVw[0] + ve[0]],
                    [tVw[1], tVw[1] + ve[1]],
                    [tVw[2], tVw[2] + ve[2]], c=cols[j])


for i in range(Npts[1]):
    ax.text(muW[i, 0], muW[i, 1], 0, s=i)

# %% =========================================================================
# DATOS EXTRINSECOS

# puntos da calibracion sacadas
calibPointsFile = "./resources/nov16/puntosCalibracion.txt"
imageFile = "./resources/nov16/vlcsnap.png"
calibPts = np.loadtxt(calibPointsFile)

priorLLA = np.array([-34.629344, -58.370350, 15.7])

RealBalk = realbalk(calibPts[:,2:], calibPts[:,:2], priorLLA, imageFile)

plt.figure()
plt.scatter(RealBalk.objPt[:, 1], RealBalk.objPt[:, 0])
plt.scatter(RealBalk.priorLLA[1], RealBalk.priorLLA[0])
plt.axis('equal')



# %% detecciones del auto
gpsCelFile = "/home/sebalander/Code/sebaPhD/resources/encoderGPS/"
gpsCelFile += "20161113192738.txt"

import pandas as pd

gps = pd.read_csv(gpsCelFile)

carGPS = gps.loc[:, ['time', 'lat', 'lon', 'accuracy']].as_matrix()

carIm = []
RealDete = realdete(carGPS, carIm)

# %% =========================================================================
# JUNTO TODO Y GUARDO
Synt = synt(SyntIntr, SyntChes, SyntExtr)
Real = real(RealChes, RealBalk, RealDete)
DataFull = datafull(Synt, Real)

fullDataFile = "./resources/fullDataIntrExtr.npy"

#text_file = open(fullDataFile + "README.txt", "w")
#text_file.write(header)
#text_file.close()

import pickle

file = open(fullDataFile, "wb")
pickle.dump(DataFull, file)
file.close()
