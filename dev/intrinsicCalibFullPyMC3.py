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
from calibration import calibrator as cl
import corner
import time

# %env THEANO_FLAGS='device=cuda, floatX=float32'
import theano
import theano. tensor as T
import pymc3 as pm

import sys
sys.path.append("/home/sebalander/Code/sebaPhD")
from dev import bayesLib as bl

# %% LOAD DATA
# input
plotCorners = False
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
modelos = ['poly', 'rational', 'fisheye']
model = modelos[2]

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


# pongo en forma flat los valores iniciales
Xint, Ns = bl.int2flat(cameraMatrix, distCoeffs, model)
XextList = np.array([bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)])

NintrParams = distCoeffs.shape[0] + 4
NfreeParams = n*6 + NintrParams
NdataPoints = n*m

# 0.3pix as image std
stdPix = 0.1
Ci = np.repeat([stdPix**2 * np.eye(2)], n*m, axis=0).reshape(n, m, 2, 2)

# Cf = np.eye(distCoeffs.shape[0])
# Ck = np.eye(4)
# Cfk = np.eye(distCoeffs.shape[0], 4)  # rows for distortion coeffs

# Crt = np.eye(6) # 6x6 covariance of pose
# Crt[[0,1,2],[0,1,2]] *= np.deg2rad(1)**2 # 5 deg stdev in every angle
# Crt[[3,4,5],[3,4,5]] *= 0.01**2 # 1/100 of the unit length as std for pos
# Crt = np.repeat([Crt] , n, axis=0)
Crt = np.repeat([False], n)  # no RT error

# output file
intrinsicParamsOutFile = imagesFolder + camera + model + "intrinsicParamsML"
intrinsicParamsOutFile = intrinsicParamsOutFile + str(stdPix) + ".npy"

# pruebo con un par de imagenes
for j in range(0, n, 3):
    xm, ym, Cm = cl.inverse(imagePoints[j, 0], rVecs[j], tVecs[j],
                            cameraMatrix,
                            distCoeffs, model, Cccd=Ci[j], Cf=False, Ck=False,
                            Crt=False, Cfk=False)
    print(xm, ym, Cm)


# datos medidos, observados, experimentalmente en el mundo y en la imagen
yObs = objpoints2D.reshape(-1)


# no usar las constantes como tensores porque da error...
# # diccionario de parametros, constantes de calculo
# xObsConst = T.as_tensor_variable(imagePoints.reshape((n,m,2)), 'xObsConst',
#                                  ndim=3)
# CiConst = T.as_tensor_variable(Ci, 'cIConst', ndim=4)


# %%
'''
aca defino la proyeccion para que la pueda usar thean0
http://deeplearning.net/software/theano/extending/extending_theano.html
'''
global projIDXcounter
projIDXcounter = 0
global projCount
projCount = 0

class ProjectionT(theano.Op):
    # itypes and otypes attributes are
    # compulsory if make_node method is not defined.
    # They're the type of input and output respectively
    # xInt, xExternal
    itypes = [T.dvector, T.dmatrix]
    # xm, ym, cM
    otypes = [T.dtensor3, T.dtensor4]

#    def __init__(self):
#        global projIDXcounter
#        self.idx = projIDXcounter
#        projIDXcounter += 1
#
#        self.count = 0

    # Python implementation:
    def perform(self, node, inputs_storage, output_storage):
        # global projCount
        # projCount += 1
        # self.count += 1
        # print('IDX %d projection %d, global %d' %
        #       (self.idx, self.count, projCount))
        Xint, Xext = inputs_storage
        xyM, cM = output_storage

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

        # print(xy)

        xyM[0] = xy
        cM[0] = cm

    # optional:
    check_input = True


# %% pruebo si esta bien como OP

try:
    del XintOP, XextOP, projTheanoWrap, projTfunction
except:
    pass
else:
    pass

XintOP = T.dvector('XintOP')
XextOP = T.dmatrix('XextOP')

projTheanoWrap = ProjectionT()

projTfunction = theano.function([XintOP, XextOP],
                                projTheanoWrap(XintOP, XextOP))

out = projTfunction(Xint, XextList)

print(out)

plt.scatter(out[0][:, :, 0], out[0][:, :, 1])


# %%
# from importlib import reload
# reload(cl)
# reload(bl)


# indexes to read diagona 2x2 blocks of a matrix
nTot = 2 * n * m
xInxs = [[[i, i], [i+1, i+1]] for i in range(0, nTot, 2)]
yInxs = [[[i, i+1], [i, i+1]] for i in range(0, nTot, 2)]

projectionModel = pm.Model()

projTheanoWrap = ProjectionT()

# set lower and upper bounds for uniform prior distributions
# for camera matrix is important to avoid division by zero

intrDelta = np.abs([100, 100, 200, 100,
                    Xint[4] / 2, Xint[5] / 2, Xint[6] / 2, Xint[7] / 2])
intrLow = Xint - intrDelta  # intrinsic
intrUpp = Xint + intrDelta
# extrLow = np.array([[-3.2, -3.2, -3.2, -25, -25, -25]]*n)
# extrUpp = np.array( [[3.2,  3.2,  3.2,  25,  25,  25]]*n)

extrLow = XextList - [0.3, 0.3, 0.3, 3, 3, 3]
extrUpp = XextList + [0.3, 0.3, 0.3, 3, 3, 3]


allLow = np.concatenate([intrLow, extrLow.reshape(-1)])
allUpp = np.concatenate([intrUpp, extrUpp.reshape(-1)])

xAll0 = np.concatenate([Xint, XextList.reshape(-1)], 0)

with projectionModel:
    # Priors for unknown model parameters
    # xIn = pm.Uniform('xIn', lower=intrLow, upper=intrUpp, shape=Xint.shape)
    # xEx = pm.Uniform('xEx', lower=extrLow, upper=extrUpp, shape=XextList.shape)

    xAll = pm.Uniform('xAll', lower=allLow, upper=allUpp, shape=allLow.shape)

    xIn = xAll[:Ns[-1]]
    xEx = xAll[Ns[-1]:].reshape((n, 6))

    xyM, cM = projTheanoWrap(xIn, xEx)

    mu = T.reshape(xyM, (-1,))

    # sp.block_diag(out[1].reshape((-1,2,2))) # for sparse
    bigC = T.zeros((nTot, nTot))
    c3Diag = T.reshape(cM, (-1, 2, 2))  # list of 2x2 covariance blocks
    bigC = T.set_subtensor(bigC[xInxs, yInxs], c3Diag)

    Y_obs = pm.MvNormal('Y_obs', mu=mu, cov=bigC, observed=yObs)


# %% avoid map estimate as reccomended in
# https://discourse.pymc.io/t/frequently-asked-questions/74
# # aca saco el maximo a posteriori, bastante util para hacer montecarlo
# despues
# import scipy.optimize as opt
#
# #
# #try:
# #    map_estimate
# #except:
# #    print('set initial state arbitrarily')
# #    start = {'xIn': Xint,
# #             'xEx': XextList}
# #else:
# #    print('set initial state with previous map_estiamte')
# #    start=map_estimate
#
# start = {'xIn': Xint, 'xEx': XextList}
#
#
#
# #niter = 1000
# #map_estimate = pm.find_MAP(model=projectionModel, start=start,
#                             maxeval=int(niter * 10), maxiter=niter,
#                             fmin=opt.fmin, xtol=1e-2,ftol=1e-3)
#
#
# map_estimate = pm.find_MAP(model=projectionModel, start=start)
#
#
#
# print(map_estimate['xIn']- Xint)
# #print(map_estimate['x0'], x0True)


# %% metropolis desde MAP. a ver si zafo de la proposal dist
'''
http://docs.pymc.io/api/inference.html
'''

# start = map_estimate
# start = {'xIn': Xint, 'xEx': XextList}
start = {'xAll' : xAll0}
# start = {'xAll' : Smean}

# start = {'x0': map_estimate['x0']}#,
#          'alfa': map_estimate['alfa'],
#          'rtV': map_estimate['rtV']}

# scale = {'x0': map_estimate['x0_interval__'],
#          'alfa': map_estimate['alfa_interval__'],
#          'rtV': map_estimate['rtV_interval__']}
#
# scale = [map_estimate['x0_interval__'], map_estimate['alfa_interval__'], map_estimate['rtV_interval__']]

nDraws = 50
nChains = 6

# Sproposal = np.concatenate([XextList.reshape((-1)), Xint.reshape((-1))])
# Sproposal = np.abs(Sproposal) * 1e-3
#
# Sproposal = {'xIn_interval__' : np.abs(Xint) * 1e-3,
#             'xEx_interval__' : np.abs(XextList) * 1e-3}

Scov = allUpp - allLow

with projectionModel:
    step = pm.Metropolis(S=Scov, scaling=1e-8, tune_interval=10)
#    step = pm.Metropolis(vars=basic_model.x0, # basic_model.alfa,basic_model.rtV],
#                         S=np.abs(map_estimate['x0_interval__'])),
#                         scaling=1e-1,
#                         tune=True,
#                         tune_interval=50)

#    step = pm.Metropolis()

    trace = pm.sample(draws=nDraws, step=step, start=start, njobs=nChains,
                      tune=20, chains=nChains, progressbar=True,
                      random_seed=123)
                      # , live_plot=True,
                      # compute_convergence_checks=True) #,
                      # init='auto', n_init=200,)

# %%
plt.figure()
Smin = np.min(trace['xAll'], axis=0)
plt.plot(np.abs(trace['xAll'] - Smin),'x-', markersize=1)


# %%
corner.corner(trace['xAll'][:,:8])


# %%
# Scov = (Scov  + np.cov(trace['xAll'].T)) / 2
Scov = (np.cov(trace3['xAll'].T) + Scov) / 2
Smean = (np.mean(trace3['xAll'], axis=0) + Smean) / 2

# saco el promedio de las matrices de covarianza de los bloques 6x6 extrinsecos
xBlk = [[0, 1, 2, 3, 4, 5]] * 6
xBlkAll = np.arange(NintrParams,NfreeParams,6).reshape((-1,1,1)) + [xBlk]*n
yBlkAll = xBlkAll.transpose((0,2,1))
extrS = np.mean(Scov[xBlkAll, yBlkAll], axis=0)

'''
extrS = np.array([[ 2.06434771e-21,  3.63405709e-23,  2.82540609e-22,
         8.42567425e-20,  3.12203522e-19, -3.27077383e-19],
       [ 3.63405709e-23,  2.57327282e-21, -4.97619001e-25,
        -3.18702380e-20,  4.34245103e-19, -2.92461577e-19],
       [ 2.82540609e-22, -4.97619001e-25,  2.52469500e-21,
         3.16066090e-19,  9.91440179e-20,  2.96264872e-19],
       [ 8.42567425e-20, -3.18702380e-20,  3.16066090e-19,
         2.31953016e-15,  1.22331437e-16,  4.82367034e-16],
       [ 3.12203522e-19,  4.34245103e-19,  9.91440179e-20,
         1.22331437e-16,  2.60254016e-15,  6.01781418e-17],
       [-3.27077383e-19, -2.92461577e-19,  2.96264872e-19,
         4.82367034e-16,  6.01781418e-17,  1.90129713e-15]])

ScovNew = np.zeros((NfreeParams, NfreeParams))

ScovNew[:NintrParams,:NintrParams] = np.array([[ 1.45205446e-06,  5.08615812e-07, -6.11360747e-07,
        -7.04102352e-07, -5.66300357e-16,  5.23622145e-17,
        -2.44955646e-16,  5.52125195e-17],
       [ 5.08615812e-07,  2.97716419e-06,  2.47639870e-07,
        -1.48342349e-06, -1.21334427e-15, -8.19524694e-17,
        -6.63393217e-17,  3.90450297e-17],
       [-6.11360747e-07,  2.47639870e-07,  4.49166710e-06,
         5.18123791e-08,  2.47022954e-16, -5.07212506e-17,
         1.73436122e-16, -2.57886547e-17],
       [-7.04102352e-07, -1.48342349e-06,  5.18123791e-08,
         1.26699975e-06,  9.79266370e-16, -1.06333076e-17,
         1.51288858e-16, -3.72320589e-17],
       [-5.66300357e-16, -1.21334427e-15,  2.47022954e-16,
         9.79266370e-16,  1.55048385e-24,  7.56859543e-26,
         1.43433646e-25, -2.43754256e-26],
       [ 5.23622145e-17, -8.19524694e-17, -5.07212506e-17,
        -1.06333076e-17,  7.56859543e-26,  3.39533868e-26,
        -1.77669257e-26,  3.41487441e-27],
       [-2.44955646e-16, -6.63393217e-17,  1.73436122e-16,
         1.51288858e-16,  1.43433646e-25, -1.77669257e-26,
         1.09026439e-25, -1.03544226e-26],
       [ 5.52125195e-17,  3.90450297e-17, -2.57886547e-17,
        -3.72320589e-17, -2.43754256e-26,  3.41487441e-27,
        -1.03544226e-26,  4.00566400e-27]])
'''


'''
In [36]: Smean
Out[36]:
array([ 3.98213521e+02,  4.11224775e+02,  8.08169564e+02,  4.67122046e+02,
        9.58412207e-02, -1.79782432e-02,  1.71556081e-02, -4.14991879e-03,
       -2.28322462e-01,  9.20027735e-02, -8.14613082e-02, -3.17308934e+00,
       -5.38585657e+00,  6.26164383e+00,  9.45077936e-01,  8.08304282e-02,
       -7.51600297e-02, -3.03507465e+00, -4.06491057e+00,  4.00096601e+00,
        9.50916102e-01, -3.38361474e-01,  9.60425885e-01,  9.60555438e+00,
       -5.72674887e+00,  3.88570300e+00,  1.46053354e-01, -5.57198700e-01,
        6.43733030e-01, -6.43676977e+00, -4.00413471e+00,  3.72567547e+00,
       -3.27605373e-02,  8.62886207e-01, -6.78201432e-04, -1.27359412e-01,
       -2.44355042e+00,  7.63205909e+00,  8.88361698e-01,  5.26028732e-01,
        8.64146244e-01,  7.80177035e+00, -6.70277531e+00,  5.20695030e+00,
        7.30336876e-01,  8.99827967e-01,  1.78243857e+00,  1.86150936e+01,
        5.98526378e+00,  7.29298765e+00, -8.08738428e-01,  1.02303559e-01,
        1.09629125e+00, -4.95598455e+00, -3.51410147e+00,  1.14528430e+01,
        1.75068619e-02,  1.48805221e-02, -1.58488521e+00, -2.35995566e+00,
        4.91849779e+00,  5.37940157e+00, -6.50590283e-02, -3.22829044e-02,
       -3.09914039e+00,  4.02494559e+00,  4.89104703e+00,  5.56949282e+00,
        1.97581406e-01,  6.88105109e-02,  2.27910698e-02, -3.79923073e+00,
        1.99756603e+00,  5.16839394e+00,  5.68235709e-01,  1.93106014e-01,
        2.22484490e-01, -4.74288721e-01, -1.97951364e+01,  1.48902312e+01,
       -5.40230272e-02, -2.11437124e-02, -3.46421590e-02, -4.59416797e+00,
       -5.82225412e-01,  1.97710238e+01, -1.41653099e+00,  3.38020448e-02,
        2.78678950e+00, -1.75695022e+01, -2.96368188e+00,  1.23532714e+01,
       -1.18460309e-01, -1.05137374e+00,  1.69773008e+00, -1.69412489e+00,
       -5.06532137e+00,  7.90455051e+00, -4.14649903e-01, -4.94426169e-01,
        1.38043596e+00, -5.03135277e+00, -5.43547007e+00,  1.26991966e+01,
       -2.50042201e-01,  8.02786932e-01,  1.72645413e+00,  1.47947989e+00,
        1.60842477e+01,  2.09283726e+01, -2.68805247e-01,  1.03164360e+00,
        6.40659057e-01,  1.68264894e+01,  6.68865788e+00,  1.73046827e+01,
       -3.58352735e-02,  8.85232394e-02, -1.41836937e-01, -2.20225564e+00,
       -1.01215599e+01,  1.87058555e+01, -7.46207061e-03, -2.65485197e-01,
       -1.61748892e-01, -9.54394554e+00, -7.97594735e-01,  2.41651826e+00,
        3.29768724e-01,  1.04935133e+00,  1.43090193e+00,  1.22238773e+01,
       -4.11740103e+00,  7.54633038e+00,  6.08319706e-01, -3.85403982e-03,
       -2.99504577e+00,  3.68058002e+00,  5.48597577e-01,  1.45748882e+01,
       -4.18170179e-02, -5.97130394e-01,  3.06790855e-01, -1.12050748e+01,
        1.66399217e+00,  3.24935875e+00,  1.63730986e-02, -1.23804169e+00,
        6.87577835e-03, -5.26068417e+00, -2.40616638e+00,  1.44066156e+00,
       -1.21411378e+00,  2.17619172e-01,  2.55174591e+00, -1.73766332e+01,
        7.20953855e+00,  1.22812383e+01, -9.95886791e-01,  6.40191151e-01,
        2.05514621e+00, -1.08960679e+01,  1.41197214e+01,  1.53498848e+01,
       -1.24198097e-01, -8.39867265e-01, -2.94574793e+00,  1.48699765e+00,
        1.61442015e+01,  1.23622436e+01,  7.38455019e-01,  4.61661711e-01,
        1.33651323e+00,  9.35040071e+00, -6.55144173e+00,  4.39687080e+00,
       -6.33742277e-01, -1.13585852e-01, -3.01844512e+00,  1.92689848e+00,
        6.99976307e+00,  1.01504128e+01,  1.81275476e-02, -2.97138175e-02,
       -6.27459196e-02, -3.89081176e+00, -2.70593243e+00,  4.32909703e+00,
        3.42741379e-01,  6.03378276e-01, -3.02368589e-01,  1.38806225e+01,
       -9.07390614e+00,  1.55793088e+01,  3.82664071e-02,  8.75231097e-01,
        9.78528144e-04,  1.74233332e+01, -2.79635047e+00,  1.07892644e+01,
       -3.29153925e-02,  1.08510732e+00,  5.22249671e-01,  1.81473165e+01,
        1.98678517e+00,  1.03291327e+01])


extrS
Out[33]:
array([[ 4.47647315e-24, -5.93088996e-26,  7.69379298e-25,
         7.25920194e-22,  2.68583955e-21, -7.56695994e-21],
       [-5.93088996e-26,  5.47386029e-24, -2.98541274e-25,
        -4.45535901e-21,  1.09036465e-20, -5.69907951e-21],
       [ 7.69379298e-25, -2.98541274e-25,  5.06238138e-24,
         7.56605179e-21,  2.06712555e-22,  8.76807154e-21],
       [ 7.25920194e-22, -4.45535901e-21,  7.56605179e-21,
         4.80619414e-16,  2.00626792e-17,  1.43968266e-16],
       [ 2.68583955e-21,  1.09036465e-20,  2.06712555e-22,
         2.00626792e-17,  6.66085932e-16,  5.25124140e-17],
       [-7.56695994e-21, -5.69907951e-21,  8.76807154e-21,
         1.43968266e-16,  5.25124140e-17,  4.19908999e-16]])

Scov[:NintrParams,:NintrParams]
Out[34]:
array([[ 6.54106691e-06,  9.65007005e-08, -4.35522383e-07,
        -9.03292919e-07, -7.39431720e-19, -7.98840628e-21,
        -1.04698092e-19,  8.39372531e-21],
       [ 9.65007005e-08,  2.00224285e-05, -1.14723434e-06,
        -1.42846843e-06,  3.25649084e-18, -7.66737386e-20,
         2.65875004e-19,  5.82518820e-21],
       [-4.35522383e-07, -1.14723434e-06,  5.01292212e-06,
         2.44863622e-07, -9.56544643e-19, -1.25481253e-20,
         4.78139241e-20, -3.12105133e-21],
       [-9.03292919e-07, -1.42846843e-06,  2.44863622e-07,
         3.51182368e-06,  1.26587976e-18, -4.20140346e-20,
        -6.49554399e-20, -1.00108293e-21],
       [-7.39431720e-19,  3.25649084e-18, -9.56544643e-19,
         1.26587976e-18,  4.09786489e-29, -1.79946158e-30,
         1.79719221e-30,  1.16737172e-31],
       [-7.98840628e-21, -7.66737386e-20, -1.25481253e-20,
        -4.20140346e-20, -1.79946158e-30,  1.81335193e-31,
        -1.57864445e-31, -1.36453216e-32],
       [-1.04698092e-19,  2.65875004e-19,  4.78139241e-20,
        -6.49554399e-20,  1.79719221e-30, -1.57864445e-31,
         2.84779080e-31,  1.31094872e-32],
       [ 8.39372531e-21,  5.82518820e-21, -3.12105133e-21,
        -1.00108293e-21,  1.16737172e-31, -1.36453216e-32,
         1.31094872e-32,  1.33080253e-33]])
'''


plt.matshow(np.log(np.abs(extrS)), cmap='inferno')
plt.matshow(np.log(np.abs(Scov)), cmap='inferno')


ScovNew = np.zeros_like(Scov)
ScovNew[:NintrParams,:NintrParams] = Scov[:NintrParams,:NintrParams]
ScovNew[xBlkAll, yBlkAll] = extrS


plt.matshow(np.log(np.abs(ScovNew)), cmap='inferno')

sigmas = np.sqrt(np.diag(Scov))
Scorr = Scov / ( sigmas.reshape((-1,1)) * sigmas.reshape((1,-1)))

plt.matshow(Scorr, cmap='inferno')

plt.matshow(np.log(np.abs(Scorr[8::6,8::6])), cmap='inferno')





# %% metropolis
'''
http://docs.pymc.io/api/inference.html
'''


start = {'xAll' : Smean}
nDraws = 100
nChains = 8


with projectionModel:
    step = pm.Metropolis(S=ScovNew, scaling=4e-3, tune_interval=10)

    trace4 = pm.sample(draws=nDraws, step=step, start=start, njobs=nChains,
                      tune=0, chains=nChains, progressbar=True)


plt.figure()
plt.plot(np.abs(trace4['xAll'] - trace4['xAll'][-1]), '-x', markersize=1.5)


# probabilidad de repeticion

nRepet = np.sum(trace4['xAll'][1:,0] == trace4['xAll'][:-1,0])
pRep = nRepet / (trace4['xAll'].shape[0] - 1)

print('tasa de repeticion es', pRep)

# %%

tracesAll = [trace, trace1, trace2, trace3, trace4]

np.save("/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/extraDataSebaPhD/intrinsicTracesAll", tracesAll)


# %% ========== initial approach based on iterative importance sampling
# in each iteration sample points from search space assuming gaussian pdf
# calculate total error and use that as log of probability

class sampleadorExtrinsecoNormal:
    '''
    manages the sampling of extrinsic parameters
    '''
    def __init__(self, muList, covarList):
        '''
        receive a list of rtvectors and define mu and covar and start sampler
        '''
        self.mean = np.array(muList)
        # gaussian normalizing factor
        self.piConst = (np.pi*2)**(np.prod(self.mean.shape)/2)
        self.setCov(covarList)

    def rvs(self, retPdf=False):
        x = sts.multivariate_normal.rvs(size=(len(self.matrixTransf),6))
        x2 = (x.reshape(-1,6,1) * self.matrixTransf).sum(2)

        if retPdf:
            pdf = np.exp(- np.sum(x**2) / 2) / self.piConst
            return x2 + self.mean, pdf

        return x2 + self.mean

    def setCov(self, covarList):
        self.cov = np.array(covarList)
        self.matrixTransf = np.array([cl.unit2CovTransf(x) for x in self.cov])


# %%
'''
se propone una pdf de donde samplear. se trata de visualizar y de hacer un
ajuste grueso
'''

# propongo covarianzas de donde samplear el espacio de busqueda
Cfk = np.diag((Xint/1000)**2 + 1e-6)  # 1/1000 de desv est
# regularizado para que no sea singular

Crt = np.eye(6) # 6x6 covariance of pose
Crt[[0,1,2],[0,1,2]] *= np.deg2rad(1)**2 # 1 deg stdev in every angle
Crt[[3,4,5],[3,4,5]] *= 0.1**2 # 1/10 of the unit length as std for pos

# reduzco la covarianza en un factor (por la dimensionalidad??)
fkFactor = 1e-6
rtFactor = 1e-6
Cfk *= fkFactor
Crt = np.repeat([Crt * rtFactor] , n, axis=0)

# instancio los sampleadres
sampleadorInt = sts.multivariate_normal(Xint, dc(Cfk))
sampleadorExt = sampleadorExtrinsecoNormal(XextList, dc(Crt))
# evaluo
intSamp = sampleadorInt.rvs()
extSamp, pdfExtSamp = sampleadorExt.rvs(retPdf=True)

errSamp = etotal(intSamp, Ns, extSamp, params)

# trayectoria de medias
meanList = list()





# %% trato de hacer una especie de gradiente estocástico con momentum
# cond iniciales
Xint0 = dc(Xint)
Xext0 = dc(np.array(XextList))
Xerr0 = etotal(Xint0, Ns, Xext0, params)

beta = 0.9
beta1 = 1 - beta + 1e-2 # un poco mas grande para que no se achique el paso

deltaInt = np.zeros_like(Xint0)
deltaExt = np.zeros_like(Xext0)

sampleIntList = list([Xint0])
sampleExtList = list([Xext0])
sampleErrList = list([Xerr0])

# %% loop


for i in range(5000):
    print(i, "%.20f"%sampleErrList[-1])
    Xint1 = sampleadorInt.rvs()
    Xext1 = sampleadorExt.rvs()
    Xerr1 = etotal(Xint1, Ns, Xext1, params)

    if Xerr0 > Xerr1: # caso de que dé mejor
        deltaInt = deltaInt * beta + beta1 * (Xint1 - Xint0)
        deltaExt = deltaExt * beta + beta1 * (Xext1 - Xext0)
        print('a la primera', np.linalg.norm(deltaInt), np.linalg.norm(deltaExt))

        # salto a ese lugar
        Xint0 = dc(Xint1)
        Xext0 = dc(Xext1)
        Xerr0 = dc(Xerr1)

        sampleadorInt.mean = Xint0  + deltaInt
        sampleadorExt.mean = Xext0 + deltaExt
        sampleIntList.append(Xint0)
        sampleExtList.append(Xext0)
        sampleErrList.append(Xerr0)

    else: # busco para el otro lado a ver que dá
        Xint2 = 2 * Xint0 - Xint1
        Xext2 = 2 * Xext0 - Xext1
        Xerr2 = etotal(Xint2, Ns, Xext2, params)

        if Xerr0 > Xerr2: # caso de que dé mejor la segunda opcion
            deltaInt = deltaInt * beta + beta1 * (Xint2 - Xint0)
            deltaExt = deltaExt * beta + beta1 * (Xext2 - Xext0)
            print('a la segunda', np.linalg.norm(deltaInt), np.linalg.norm(deltaExt))

            # salto a ese lugar
            Xint0 = dc(Xint2)
            Xext0 = dc(Xext2)
            Xerr0 = dc(Xerr2)

            sampleadorInt.mean = Xint0 + deltaInt
            sampleadorExt.mean = Xext0 + deltaExt
            sampleIntList.append(Xint0)
            sampleExtList.append(Xext0)
            sampleErrList.append(Xerr0)
        else: # las dos de los costados dan peor
            ## mido la distancia hacia la primera opcion
            #dist = np.sqrt(np.sum((Xint1 - Xint0)**2) + np.sum((Xext1 - Xext0)**2))
            # distancia al vertice de la parabola
            r = (Xerr2 - Xerr1) / 2 / (Xerr1 + Xerr2 - 2 * Xerr0) #* dist / dist
            if np.isnan(r) or np.isinf(r):
                print('r is nan inf')
                sampleadorInt.cov *= 1.5  # agrando covarianzas
                sampleadorExt.setCov(sampleadorExt.cov * 1.5)
                continue # empiezo loop nuevo

            # calculo un nuevo lugar como vertice de la parabola en 1D
            Xint3 = Xint0 + (Xint1 - Xint0) * r
            Xext3 = Xext0 + (Xext1 - Xext0) * r
            Xerr3 = etotal(Xint3, Ns, Xext3, params)

            # trato de usar el dato de curvatura para actualizar la covarianza
            diffVectIntr = Xint1 - Xint0
            distInt = np.linalg.norm(diffVectIntr)
            XtX = diffVectIntr.reshape((-1,1)) * diffVectIntr.reshape((1,-1))
            XtX = XtX + np.eye(NintrParams) * distInt * 1e-2 # regularizo

            # saco la curvatura forzando que sea positiva (en gral es)
            a = np.abs((Xerr1 + Xerr2 - 2 * Xerr0)) / distInt**2
            Hintr = a * np.linalg.inv(XtX) # hessiano intrinseco


            diffVectExtr = Xext1 - Xext0
            distExt = np.linalg.norm(diffVectExtr, axis=1)
            # saco la curvatura forzando que sea positiva (en gral es)
            a = np.abs((Xerr1 + Xerr2 - 2 * Xerr0)) / distExt**2
            XtX = diffVectExtr.reshape((n,6,1)) * diffVectExtr.reshape((n,1,6))
            XtX[:,[0,1,2,3,4,5],[0,1,2,3,4,5]] += distExt.reshape((-1,1)) * 1e-3
            Hextr = a.reshape((-1,1,1)) * np.linalg.inv(XtX) # hessiano intrinseco

            # actualizo las covarianzas
            sampleadorInt.cov = sampleadorInt.cov * beta + beta1 * Hintr
            sampleadorExt.setCov(sampleadorExt.cov * beta + beta1 * Hextr)



            if Xerr0 > Xerr3:
                deltaInt = deltaInt * beta + beta1 * (Xint3 - Xint0)
                deltaExt = deltaExt * beta + beta1 * (Xext3 - Xext0)
                print('a la tercera', np.linalg.norm(deltaInt), np.linalg.norm(deltaExt))

                # salto a ese lugar
                Xint0 = dc(Xint3)
                Xext0 = dc(Xext3)
                Xerr0 = dc(Xerr3)

                sampleadorInt.mean = Xint0 + deltaInt
                sampleadorExt.mean = Xext0 + deltaExt
                sampleIntList.append(Xint0)
                sampleExtList.append(Xext0)
                sampleErrList.append(Xerr0)
            else:
                print('no anduvo, probar de nuevo corrigiendo', r)
                sampleadorInt.cov *= 0.9  # achico covarianzas
                sampleadorExt.setCov(sampleadorExt.cov * 0.9)

                deltaInt *= 0.7 # achico el salto para buscar mas cerca
                deltaExt *= 0.7

                sampleadorInt.mean = Xint0 + deltaInt
                sampleadorExt.mean = Xext0 + deltaExt

# %%

intrinsicGradientList = np.array(sampleIntList)
extrinsicGradientList = np.array(sampleExtList)
errorGradientList = np.array(sampleErrList)


plt.figure(1)
plt.plot(errorGradientList - errorGradientList[-1])


sampleadorIntBkp = dc(sampleadorInt)
sampleadorExtBkp = dc(sampleadorExt)

# %%
plt.figure(2)
minErr = np.min(errorGradientList)
for i in range(NintrParams):
    plt.plot(intrinsicGradientList[:,i] - intrinsicGradientList[-1,i],
             errorGradientList - minErr, '-x')

plt.figure(3)
for i in range(n):
    for j in range(6):
        plt.plot(extrinsicGradientList[:,i,j] - extrinsicGradientList[-1,i,j],
                 errorGradientList - minErr, '-x')
plt.semilogy()



    # %% metropolis hastings
from numpy.random import rand as rn

def nuevo(oldInt, oldExt, oldErr, retPb=False):
    global generados
    global gradPos
    global gradNeg
    global mismo
    global sampleador

    # genero nuevo
    newInt = sampleadorInt.rvs() # rn(8) * intervalo + cotas[:,0]
    newExt = sampleadorExt.rvs()
    generados += 1

    # cambio de error
    newErr = etotal(newInt, Ns, newExt, params)
    deltaErr = newErr - oldErr

    if deltaErr < 0:
        gradPos += 1
        print(generados, gradPos, gradNeg, mismo, "Gradiente Positivo")
        if retPb:
            return newInt, newExt, newErr, 1.0
        return newInt, newExt, newErr # tiene menor error, listo
    else:
        # nueva oportunidad, sampleo contra rand
        pb = np.exp(- deltaErr / 2)

        if pb > rn():
            gradNeg += 1
            print(generados, gradPos, gradNeg, mismo, "Gradiente Negativo, pb=", pb)
            if retPb:
                return newInt, newExt, newErr, pb
            return newInt, newExt, newErr  # aceptado a la segunda oportunidad
        else:
#            # vuelvo recursivamente al paso2 hasta aceptar
#            print('rechazado, pb=', pb)
#            new, newE = nuevo(old, oldE)
            mismo +=1
            print(generados, gradPos, gradNeg, mismo, "Mismo punto,        pb=", pb)
            if retPb:
                return oldInt, oldExt, oldErr, pb
            return oldInt, oldExt, oldErr

#    return newInt, newExt, newErr

generados = 0
gradPos = 0
gradNeg = 0
mismo = 0


intSamp = sampleadorInt.rvs()
extSamp = sampleadorExt.rvs()
errSamp = etotal(intSamp, Ns, extSamp, params)

# pruebo una iteracion
newInt, newExt, newErr = nuevo(intSamp, extSamp, errSamp)


# %% metropolis para hacer burnin y sacar una covarianza para q

Nmuestras = 1000
#Mmuestras = int(50)
#nTot = Nmuestras * Mmuestras

generados = 0
gradPos = 0
gradNeg = 0
mismo = 0

sampleIntList = np.zeros((Nmuestras, distCoeffs.shape[0] + 4))
sampleExtList = np.zeros((Nmuestras,n,6))
sampleErrList = np.zeros(Nmuestras)

# cargo samplers
sampleadorInt = dc(sampleadorIntBkp)
sampleadorExt = dc(sampleadorExtBkp)

sampleadorInt.cov = Cfk
sampleadorExt.setCov(Crt / 1000) # no se porque hay que dividir por mil

# primera
sampleIntList[0] = sampleadorInt.rvs()
sampleExtList[0] = sampleadorExt.rvs()
sampleErrList[0] = etotal(intSamp, Ns, extSamp, params)

tiempoIni = time.time()

for i in range(1, Nmuestras):
    sampleIntList[i], sampleExtList[i], sampleErrList[i], pb = nuevo(sampleIntList[i-1], sampleExtList[i-1], sampleErrList[i-1], retPb=True)

    if np.isnan(pb):
        break

    # actualizo centroide
    sampleadorInt.mean = sampleIntList[i]
    sampleadorExt.mean = sampleExtList[i]

    tiempoNow = time.time()
    Dt = tiempoNow - tiempoIni
    frac = i / Nmuestras # (i  + Nmuestras * j)/ nTot
    DtEstimeted = (tiempoNow - tiempoIni) / frac
    stringTimeEst = time.asctime(time.localtime(tiempoIni + DtEstimeted))

    print('Epoch: %d/%d. Tfin: %s'
          %(i, Nmuestras, stringTimeEst),
          np.linalg.norm(sampleadorInt.cov),
          np.linalg.norm(sampleadorExt.cov))




corner.corner(sampleIntList)


os.system("speak 'aadfafañfañieñiweh'")



# %% usando PYMC3
import pymc3 as pm

projection_model = pm.Model()

with projection_model:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)
    sigma = pm.HalfNormal('sigma', sd=1)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)


