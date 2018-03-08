#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 2018

do metropolis sampling to estimate PDF of chessboard calibration. this involves
intrinsic and extrinsic parameters, so it's a very high dimensional search
space (before it was only intrinsic)

@author: sebalander
"""

# %%
# import glob
import os
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
from copy import deepcopy as dc
import corner
import time

# %env THEANO_FLAGS='device=cuda, floatX=float32'
import theano
import theano. tensor as T
import pymc3 as pm
import scipy as sc
import seaborn as sns

import sys
sys.path.append("/home/sebalander/Code/sebaPhD")
from calibration import calibrator as cl
from dev import bayesLib as bl

print('libraries imported')

# %% LOAD DATA
# input
plotCorners = False
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
modelos = ['poly', 'rational', 'fisheye', 'stereographic']
model = modelos[3]

imagesFolder = "./resources/intrinsicCalib/" + camera + "/"
cornersFile = imagesFolder + camera + "Corners.npy"
patternFile = imagesFolder + camera + "ChessPattern.npy"
imgShapeFile = imagesFolder + camera + "Shape.npy"

# model data files
distCoeffsFile = imagesFolder + camera + model + "DistCoeffs.npy"
linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
tVecsFile = imagesFolder + camera + model + "Tvecs.npy"
rVecsFile = imagesFolder + camera + model + "Rvecs.npy"

imageSelection = np.arange(0, 33)  # selecciono con que imagenes trabajar
n = len(imageSelection)  # cantidad de imagenes

# ## load data
imagePoints = np.load(cornersFile)[imageSelection]

chessboardModel = np.load(patternFile)
m = chessboardModel.shape[1]  # cantidad de puntos por imagen
imgSize = tuple(np.load(imgShapeFile))
# images = glob.glob(imagesFolder+'*.png')

# Parametros de entrada/salida de la calibracion
objpoints2D = np.array([chessboardModel[0, :, :2]]*n).reshape((n, m, 2))

# load model specific data from opencv Calibration
distCoeffs = np.load(distCoeffsFile)
cameraMatrix = np.load(linearCoeffsFile)
rVecs = np.load(rVecsFile)[imageSelection]
tVecs = np.load(tVecsFile)[imageSelection]

print('raw data loaded')

# pongo en forma flat los valores iniciales
Xint, Ns = bl.int2flat(cameraMatrix, distCoeffs, model)
XextList = np.array([bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)])

NintrParams = Xint.shape[0]
NextrParams = n * 6
NfreeParams = NextrParams + NintrParams
NdataPoints = n*m

# 0.1pix as image std
# https://stackoverflow.com/questions/12102318/opencv-findcornersubpix-precision
# increase to 1pix porque la posterior da demasiado rara
stdPix = 1.0
Ci = np.repeat([stdPix**2 * np.eye(2)], n*m, axis=0).reshape(n, m, 2, 2)

Crt = np.repeat([False], n)  # no RT error

# output file
intrinsicParamsOutFile = imagesFolder + camera + model + "intrinsicParamsML"
intrinsicParamsOutFile = intrinsicParamsOutFile + str(stdPix) + ".npy"

## pruebo con un par de imagenes
#for j in range(0, n, 3):
#    xm, ym, Cm = cl.inverse(imagePoints[j, 0], rVecs[j], tVecs[j],
#                            cameraMatrix,
#                            distCoeffs, model, Cccd=Ci[j], Cf=False, Ck=False,
#                            Crt=False, Cfk=False)
#    print(xm, ym, Cm)


# datos medidos, observados, experimentalmente en el mundo y en la imagen
yObs = objpoints2D.reshape(-1)

print('data formated')


# %%
'''
aca defino la proyeccion para que la pueda usar thean0
http://deeplearning.net/software/theano/extending/extending_theano.html
'''


class ProjectionT(theano.Op):
    # itypes and otypes attributes are
    # compulsory if make_node method is not defined.
    # They're the type of input and output respectively
    # xInt, xExternal
    itypes = [T.dvector, T.dmatrix]
    # xm, ym, cM
    otypes = [T.dtensor3]  # , T.dtensor4]

    # Python implementation:
    def perform(self, node, inputs_storage, output_storage):
        # print('IDX %d projection %d, global %d' %
        #       (self.idx, self.count, projCount))
        Xint, Xext = inputs_storage
#        xyM, cM = output_storage

        # saco los parametros de flat para que los use la func de projection
        # print(Xint)
        cameraMatrix, distCoeffs = bl.flat2int(Xint, Ns, model)
        # print(cameraMatrix, distCoeffs)
        xy = np.zeros((n, m, 2))
        cm = np.zeros((n, m, 2, 2))

        for j in range(n):
            rVec, tVec = bl.flat2ext(Xext[j])
            xy[j, :, 0], xy[j, :, 1], cm[j] = cl.inverse(
                    imagePoints.reshape((n, m, 2))[j], rVec, tVec,
                    cameraMatrix, distCoeffs, model, Cccd=Ci[j])

        xy -= objpoints2D

        S = np.linalg.inv(cm)

        u, s, v = np.linalg.svd(S)

        sdiag = np.zeros_like(u)
        sdiag[:, :, [0, 1], [0, 1]] = np.sqrt(s)

        A = (u.reshape((n, m, 2, 2, 1, 1)) *
             sdiag.reshape((n, m, 1, 2, 2, 1)) *
             v.transpose((0, 1, 3, 2)).reshape((n, m, 1, 1, 2, 2))
             ).sum(axis=(3, 4))

        xy = np.sum(xy.reshape((n, m, 2, 1)) * A, axis=3)

        output_storage[0][0] = xy

    # optional:
    check_input = True


print('projection defined for theano')

# %% pruebo si esta bien como OP

#try:
#    del XintOP, XextOP, projTheanoWrap, projTfunction
#except:
#    pass
#else:
#    pass

XintOP = T.dvector('XintOP')
XextOP = T.dmatrix('XextOP')

projTheanoWrap = ProjectionT()

projTfunction = theano.function([XintOP, XextOP],
                                projTheanoWrap(XintOP, XextOP))

try:
    out = projTfunction(Xint, XextList)
except:
    sys.exit("no anduvo el wrapper de la funciona  theano")
else:
    pass
    # print(out)

plt.scatter(out[:, :, 0], out[:, :, 1])


# %%
# from importlib import reload
# reload(cl)
# reload(bl)


# indexes to read diagona 2x2 blocks of a matrix
# nTot = 2 * n * m
# xInxs = [[[i, i], [i+1, i+1]] for i in range(0, nTot, 2)]
# yInxs = [[[i, i+1], [i, i+1]] for i in range(0, nTot, 2)]

projectionModel = pm.Model()

projTheanoWrap = ProjectionT()

# set lower and upper bounds for uniform prior distributions
# for camera matrix is important to avoid division by zero

# intrDelta = np.abs([100, 100, 200, 100,
#                    Xint[4] / 2, Xint[5] / 2, Xint[6] / 2, Xint[7] / 2])
intrDelta = np.abs([100, 100, 100])
intrLow = Xint - intrDelta  # intrinsic
intrUpp = Xint + intrDelta

extrDelta = np.array([[0.3, 0.3, 0.3, 3, 3, 3]]*n)
extrLow = XextList - extrDelta
extrUpp = XextList + extrDelta


allLow = np.concatenate([intrLow, extrLow.reshape(-1)])
allUpp = np.concatenate([intrUpp, extrUpp.reshape(-1)])

xAll0 = np.concatenate([Xint, XextList.reshape(-1)], 0)

observedNormed = np.zeros((n * m * 2))


with projectionModel:

    # Priors for unknown model parameters
    xIn = pm.Uniform('xIn', lower=intrLow, upper=intrUpp, shape=(NintrParams,),
                     transform=None)
    xEx = pm.Uniform('xEx', lower=extrLow, upper=extrUpp, shape=(n, 6),
                     transform=None)

    # apply numpy based function
    xyMNor = projTheanoWrap(xIn, xEx)

    mu = T.reshape(xyMNor, (-1, ))
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=1, observed=observedNormed)

print('model defined')

#
# # %% saco de las sampleadas anteriores
#
# pathFiles = "/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/extraDataSebaPhD/"
#
# Smean = np.load(pathFiles + "Smean.npy")
# Scov = np.load(pathFiles + "Scov.npy")
#
# Sintr = np.sqrt(np.diag(Scov[:NintrParams,:NintrParams]))
# Sextr = np.sqrt(np.diag(Scov)[NintrParams:])


# %% aca harcodeo buenas condiciones iniciales
## esto es para el modelo wide fisheye
#inMean = np.array(
#      [ 3.98204193e+02,  4.11166644e+02,  8.08151855e+02,  4.67128941e+02,
#        9.58414611e-02, -1.79782629e-02,  1.71555867e-02, -4.14991611e-03])
#
#Sin = np.array(
#      [1.10881031e-01, 1.02037362e-01, 4.34863593e-02, 4.69344339e-02,
#       7.05596265e-05, 2.74114158e-06, 3.53176045e-06, 2.32011924e-07])

#inCov = np.array(
#      [[ 9.03278324e-02,  8.01184408e-02,  5.93501910e-03,
#        -3.28437427e-03, -1.37613801e-04, -2.13500585e-06,
#         1.37155732e-05,  3.80320535e-07],
#       [ 8.01184408e-02,  9.03987874e-02,  9.05846356e-03,
#        -4.98390308e-03, -1.34137229e-04, -1.73441031e-06,
#         1.69912087e-05,  3.51041632e-07],
#       [ 5.93501910e-03,  9.05846356e-03,  9.05394824e-03,
#        -7.48250515e-04, -3.81539942e-07, -8.56656518e-07,
#         1.44193878e-06,  4.07269304e-08],
#       [-3.28437427e-03, -4.98390308e-03, -7.48250515e-04,
#         1.01802146e-02,  4.58709928e-06, -1.55095197e-06,
#        -1.32518335e-06,  1.01157804e-07],
#       [-1.37613801e-04, -1.34137229e-04, -3.81539942e-07,
#         4.58709928e-06,  4.99190949e-07, -1.04398026e-08,
#        -7.71458599e-08,  2.34150723e-09],
#       [-2.13500585e-06, -1.73441031e-06, -8.56656518e-07,
#        -1.55095197e-06, -1.04398026e-08,  3.49035741e-09,
#         1.30677751e-09, -2.78472504e-10],
#       [ 1.37155732e-05,  1.69912087e-05,  1.44193878e-06,
#        -1.32518335e-06, -7.71458599e-08,  1.30677751e-09,
#         1.71681579e-08, -6.80709826e-10],
#       [ 3.80320535e-07,  3.51041632e-07,  4.07269304e-08,
#         1.01157804e-07,  2.34150723e-09, -2.78472504e-10,
#        -6.80709826e-10,  1.22395226e-10]])

inMean = np.array([809.30185034, 468.47943396, 803.95311444])

Sin = np.array([0.12922189, 0.11908904, 0.52271924])

inCov = np.array(
      [[ 0.0166983 ,  0.00027483, -0.02344957],
       [ 0.00027483,  0.0141822 , -0.03783155],
       [-0.02344957, -0.03783155,  0.27323541]])

exMean = np.array(
      [[-2.02900500e-01,  9.25537630e-02, -8.14360504e-02,
        -3.20281763e+00, -5.49314862e+00,  6.19169297e+00],
       [ 9.23834843e-01,  8.05854969e-02, -7.50334368e-02,
        -3.05558934e+00, -4.20798532e+00,  4.02127694e+00],
       [ 9.16549702e-01, -3.44500609e-01,  9.75640329e-01,
         9.62113959e+00, -5.94970738e+00,  4.03319310e+00],
       [ 1.46112337e-01, -5.24183650e-01,  6.45232087e-01,
        -6.44909323e+00, -4.06099958e+00,  3.78554670e+00],
       [-3.27525387e-02,  8.63881899e-01, -6.78215598e-04,
        -1.75798602e-01, -2.45663021e+00,  7.66089994e+00],
       [ 8.52834805e-01,  5.27537686e-01,  8.95318834e-01,
         7.84115177e+00, -6.87750026e+00,  5.30007509e+00],
       [ 7.17500056e-01,  9.69240935e-01,  1.77926988e+00,
         1.84242465e+01,  6.09000951e+00,  7.46251627e+00],
       [-7.70341274e-01,  1.02344491e-01,  1.10849035e+00,
        -4.84078919e+00, -3.62527346e+00,  1.13076158e+01],
       [ 1.74844844e-02,  1.48729410e-02, -1.58285577e+00,
        -2.38826271e+00,  4.90076009e+00,  5.29722932e+00],
       [-6.50141511e-02, -3.23483371e-02, -3.09970003e+00,
         4.00829176e+00,  4.91086156e+00,  5.54717360e+00],
       [ 1.83322538e-01,  6.87823603e-02,  2.28079820e-02,
        -3.81470276e+00,  2.06902738e+00,  5.18498779e+00],
       [ 6.00646640e-01,  1.93417482e-01,  2.22281700e-01,
        -5.68431661e-01, -2.02142755e+01,  1.46221938e+01],
       [-5.40246651e-02, -2.11416475e-02, -3.46167052e-02,
        -4.64614371e+00, -6.10521670e-01,  1.97420798e+01],
       [-1.29085724e+00,  3.38850937e-02,  2.86479602e+00,
        -1.71855968e+01, -3.08509499e+00,  1.22469318e+01],
       [-1.19643867e-01, -1.03032120e+00,  1.69608012e+00,
        -1.66315194e+00, -5.16443395e+00,  7.83821379e+00],
       [-4.11945939e-01, -4.93823676e-01,  1.38214030e+00,
        -4.90789664e+00, -5.48551830e+00,  1.25222984e+01],
       [-2.48968404e-01,  8.12261110e-01,  1.71485247e+00,
         1.42569779e+00,  1.62324346e+01,  2.05873641e+01],
       [-2.60641385e-01,  1.03297098e+00,  6.34638796e-01,
         1.66267689e+01,  6.87308555e+00,  1.74012455e+01],
       [-3.58361931e-02,  8.85904462e-02, -1.42242336e-01,
        -2.28331352e+00, -1.03515875e+01,  1.85797135e+01],
       [-7.46337080e-03, -2.31705739e-01, -1.65967353e-01,
        -9.53505587e+00, -7.72944180e-01,  2.54809527e+00],
       [ 3.31512333e-01,  9.94144601e-01,  1.43126397e+00,
         1.20317350e+01, -4.16889736e+00,  7.56476727e+00],
       [ 6.69736580e-01, -3.85378259e-03, -2.98763261e+00,
         3.60961811e+00,  4.58737087e-01,  1.45480781e+01],
       [-4.18210917e-02, -5.51931465e-01,  2.98756682e-01,
        -1.12870948e+01,  1.76878170e+00,  3.42917339e+00],
       [ 1.63747969e-02, -1.25657362e+00,  6.87570265e-03,
        -5.15171266e+00, -2.42541991e+00,  1.44296422e+00],
       [-1.09297417e+00,  2.18058512e-01,  2.59279787e+00,
        -1.69360549e+01,  7.18385625e+00,  1.21823376e+01],
       [-9.39109252e-01,  6.32381201e-01,  2.06790553e+00,
        -1.07440343e+01,  1.42858283e+01,  1.52089400e+01],
       [-1.24097228e-01, -9.09715866e-01, -2.92556195e+00,
         1.45284452e+00,  1.63514037e+01,  1.21710293e+01],
       [ 7.21540140e-01,  4.53978514e-01,  1.34881949e+00,
         9.23677484e+00, -6.61865917e+00,  4.40814476e+00],
       [-6.74234890e-01, -1.13367190e-01, -3.01580072e+00,
         1.87882990e+00,  6.97763629e+00,  9.88923284e+00],
       [ 1.81284625e-02, -2.97165890e-02, -6.30578294e-02,
        -3.90798997e+00, -2.73066603e+00,  4.30922692e+00],
       [ 3.41445501e-01,  5.68740081e-01, -3.02838759e-01,
         1.36736316e+01, -9.31810618e+00,  1.55915990e+01],
       [ 3.82671084e-02,  7.39319537e-01,  9.78552251e-04,
         1.65080339e+01, -2.83679638e+00,  1.05353317e+01],
       [-3.29204302e-02,  9.12735610e-01,  5.42426360e-01,
         1.73778406e+01,  1.89384698e+00,  1.01839145e+01]])

Sex = np.array(
      [[1.27530323e-03, 4.61496190e-04, 6.17624002e-05, 2.64167986e-03,
        4.06590691e-03, 6.96282946e-03],
       [1.35843570e-03, 1.79694082e-04, 2.39294451e-04, 2.46379225e-03,
        3.58693625e-03, 4.49898830e-03],
       [2.32338162e-03, 1.68299109e-03, 1.54269987e-03, 1.35668786e-02,
        9.24758098e-03, 1.24990840e-02],
       [4.53146829e-04, 2.89060477e-03, 8.83822950e-04, 5.23888965e-03,
        3.63430037e-03, 7.66343735e-03],
       [4.96534461e-05, 8.67338632e-04, 6.24528154e-08, 4.01021275e-03,
        1.87784245e-03, 4.79765194e-03],
       [5.87680568e-03, 6.78744934e-03, 3.08703493e-03, 2.81537779e-02,
        1.82938300e-02, 2.76630944e-02],
       [4.21309285e-03, 1.49557237e-02, 3.59630689e-03, 5.06876146e-02,
        2.61425518e-02, 3.20083010e-02],
       [4.20935460e-03, 1.44937570e-04, 1.61716207e-03, 9.94731165e-03,
        1.03901335e-02, 1.27805223e-02],
       [2.36014560e-05, 1.22763892e-05, 6.90275535e-04, 4.19398531e-03,
        1.91265659e-03, 4.74281049e-03],
       [1.65440181e-04, 1.02943722e-04, 4.95108232e-04, 2.99871758e-03,
        4.21499907e-03, 7.74795852e-03],
       [1.18327242e-03, 2.56307645e-04, 2.76878010e-05, 3.38846305e-03,
        5.80309100e-03, 9.94147696e-03],
       [7.06931660e-03, 5.63815987e-04, 4.01030484e-04, 9.30700715e-03,
        4.95542024e-02, 4.46503059e-02],
       [1.22748679e-04, 1.17529397e-05, 2.27023068e-05, 9.76140407e-03,
        7.28874896e-03, 3.18146519e-02],
       [1.26075654e-02, 1.22441066e-04, 7.72107825e-03, 7.30853645e-02,
        2.00119532e-02, 5.42588621e-02],
       [9.69910348e-04, 3.60264548e-03, 1.27429789e-03, 1.03072213e-02,
        8.87562757e-03, 1.51296272e-02],
       [4.19959814e-03, 3.01106058e-03, 1.11812325e-03, 1.12594651e-02,
        5.83877593e-03, 2.56891671e-02],
       [5.61690019e-03, 1.17757900e-02, 5.33904434e-03, 9.66027837e-03,
        4.89172719e-02, 5.86029318e-02],
       [7.33013186e-03, 7.57060191e-03, 4.48325457e-03, 3.70556944e-02,
        2.10520053e-02, 2.71239973e-02],
       [6.13170880e-05, 1.41950521e-04, 6.71835640e-04, 7.37182778e-03,
        1.83021644e-02, 3.54792818e-02],
       [1.75100977e-06, 1.06694396e-03, 6.16752735e-04, 4.86903510e-03,
        3.74962055e-03, 6.55862290e-03],
       [1.63501422e-03, 4.32187309e-03, 1.29785354e-03, 1.46388204e-02,
        5.93853365e-03, 1.30657561e-02],
       [7.38336449e-03, 2.62381099e-07, 1.87348094e-03, 7.26648603e-03,
        1.19799503e-02, 2.97337118e-02],
       [1.77206300e-05, 2.11540771e-03, 1.29369330e-03, 7.39306870e-03,
        5.24392593e-03, 9.02557086e-03],
       [1.18019397e-05, 1.16121417e-03, 2.96600157e-06, 4.32993425e-03,
        2.06334293e-03, 5.76236667e-03],
       [1.27000652e-02, 9.80669988e-04, 5.01767396e-03, 8.21875239e-02,
        2.74069575e-02, 5.09904049e-02],
       [1.38904601e-02, 1.29778491e-02, 8.45081195e-03, 4.13958084e-02,
        5.54787194e-02, 5.56717191e-02],
       [3.58669948e-04, 1.31092333e-02, 4.42702516e-03, 9.74733786e-03,
        2.58419497e-02, 2.46060877e-02],
       [6.25257490e-03, 2.50690445e-03, 2.57496833e-03, 2.09548152e-02,
        7.71735278e-03, 2.13891543e-02],
       [6.57576596e-03, 1.04827863e-03, 1.09088852e-03, 6.45272030e-03,
        1.38782199e-02, 3.56914703e-02],
       [2.61166969e-05, 1.03725148e-05, 2.12454697e-04, 2.69632797e-03,
        1.92431546e-03, 4.40232470e-03],
       [5.14253430e-03, 6.28147338e-03, 2.55781969e-03, 4.56242502e-02,
        2.60156448e-02, 4.75348607e-02],
       [2.20239646e-05, 5.04380402e-03, 3.33848752e-08, 4.46334047e-02,
        7.50863391e-03, 3.08903317e-02],
       [4.75858435e-05, 9.08161061e-03, 1.50584491e-03, 5.84477978e-02,
        1.72236169e-02, 3.56940114e-02]]).reshape(-1)


print('defined harcoded initial conditions')


# %% calculate estimated radius step
'''
para ver que tan disperso es el paso de cada propuesta. como las propuestas se
sacan de una pdf gaussiana n-dimendional pasa que empieza a haber mucho volumen
de muestras que se acumulan mucho a un cierto radio. hay un compromiso entre
que el volumen aumenta
'''


def radiusStepsNdim(n):
    '''
    retorna moda, la media y la desv est del radio  de pasos al samplear
    de gaussianas hiperdimensionales de norma 1
    '''
    # https://www.wolframalpha.com/input/?i=integrate+x%5E(n-1)+exp(-x%5E2%2F2)+from+0+to+infinity
    # integral_0^∞ x^(n - 1) exp(-x^2/2) dx = 2^(n/2 - 1) Γ(n/2) for Re(n)>0
    Inorm = 2**(n/2 - 1) * sc.special.gamma(n/2)

    # https://www.wolframalpha.com/input/?i=integrate+x%5En+exp(-x%5E2%2F2)+from+0+to+infinity
    # integral_0^∞ x^n exp(-x^2/2) dx = 2^((n - 1)/2) Γ((n + 1)/2) for Re(n)>-1
    ExpectedR = 2**((n-1)/2) * sc.special.gamma((n+1)/2)

    # https://www.wolframalpha.com/input/?i=integrate+x%5E(n%2B1)+exp(-x%5E2%2F2)+from+0+to+infinity
    # integral_0^∞ x^(n + 1) exp(-x^2/2) dx = 2^(n/2) Γ(n/2 + 1) for Re(n)>-2
    ExpectedR2 = 2**(n/2) * sc.special.gamma(n/2 + 1)

    ModeR = np.sqrt(n - 1)

    # normalizo las integrales:
    ExpectedR /= Inorm
    ExpectedR2 /= Inorm

    DesvEstR = np.sqrt(ExpectedR2 - ExpectedR**2)

    return np.array([ModeR, ExpectedR, DesvEstR])


ModeR, ExpectedR, DesvEstR = radiusStepsNdim(NfreeParams)


print("moda, la media y la desv est del radio\n", ModeR, ExpectedR, DesvEstR)


# %% metropolis desde MAP. a ver si zafo de la proposal dist
'''
http://docs.pymc.io/api/inference.html
'''
nDraws = 125000
nTune = 0
tuneBool = nTune != 0
nChains = 8

# escalas características del tipical set
scaIn = 1 / radiusStepsNdim(NintrParams)[1]
scaEx = 1 / radiusStepsNdim(6)[1]

InSeed = pm.MvNormal.dist(mu=inMean, cov=inCov*scaIn**2).random(size=nChains)
exSeed = np.random.randn(n, 6, nChains) * Sex.reshape((n, 6, 1)) * scaEx
exSeed += exMean.reshape((n, 6, 1))

start = [dict({'xIn': InSeed[i], 'xEx': exSeed[:,:,i]}) for i in range(nChains)]

scaIn /= 10 # achico la escala un orden de magnitud mas
scaEx /= 20

print("starting metropolis",
      "|nnDraws =", nDraws, " nChains = ", nChains,
      "|nscaIn = ", scaIn, "scaEx = ", scaEx)

with projectionModel:
    stepInt = pm.Metropolis(vars=[xIn], S=Sin, tune=tuneBool)
    stepInt.scaling = scaIn

    stepExt = pm.Metropolis(vars=[xEx], S=Sex, tune=tuneBool)
    stepExt.scaling = scaEx

    step = [stepInt, stepExt]

    trace = pm.sample(draws=nDraws, step=step, njobs=nChains, start=start,
                      tune=nTune, chains=nChains, progressbar=True,
                      discard_tuned_samples=False,
                      compute_convergence_checks=False)

repsIn = np.diff(trace['xIn'], axis=0) == 0
repsEx = np.diff(trace['xEx'], axis=0) == 0

repsInRate = np.sum(repsIn) / np.prod(repsIn.shape)
repsExRate = np.sum(repsEx) / np.prod(repsEx.shape)

print("las repeticiones: intrin=", repsInRate, " y extrin=", repsExRate)


# las 5 y 6 posrian considerarse cuasi definitivas
# la 7 le pongo un paso un poco mas grande a ver que onda pero sigue siendo
# con tuneo
# la 8 le saco el tuneo y el paso es fijo en
# 1 / 10 / radiusStepsNdim(NintrParams)[1]
# en la 9 pongo 1 / 20 para extrinseco. la proxima tengo que cambiar el inicio?
# no el inicio esta bien pero en todas las cadenas es igual y eso hace que
# tarde en hacer una exploracion hasta que se separan
# el 10 tiene starts aleatoriso distribuidos
# el 11 tiene actualizado covarianza y media con start distribuido
# el 12 ademas escaleado apra evitar cond iniciales demasiado lejanas y es
# larga

pathFiles = "/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/"
pathFiles += "extraDataSebaPhD/traces" + str(12)

np.save(pathFiles + "Int", trace['xIn'])
np.save(pathFiles + "Ext", trace['xEx'])

print("saved data to")
print(pathFiles)

