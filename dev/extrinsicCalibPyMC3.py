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
import cv2

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

# puntos da calibracion sacadas
calibPointsFile = "./resources/nov16/puntosCalibracion.txt"

pathFiles = "/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/"
pathFiles += "extraDataSebaPhD/traces" + str(15)

imageFile = "/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/"
imageFile += "2016-11-13 medicion/vcaSnapShot.png"

# ## load data
img = plt.imread(imageFile) # imagen
calibPoints = np.loadtxt(calibPointsFile) # puntos de calibracion extrinseca
imagePoints = calibPoints[:,:2]
objecPoints = calibPoints[:,2:]
m = calibPoints.shape[0]
intrCalibResults = np.load(pathFiles + "IntrCalibResults.npy").item() # param intrinsecos

imgSize = img.shape
Xint = intrCalibResults['inMean']
Ns = np.array([2,3])
cameraMatrix, distCoeffs = bl.flat2int(intrCalibResults['inMean'], Ns, model)

Cint = intrCalibResults['inCov']
Cf = np.zeros((4,4))
Cf[2:,2:] = Cint[:2,:2]
Ck = Cint[2, 2]
Cfk = np.zeros((4,1))
Cfk[2:] = Cint[2:,2]

print('raw data loaded')

# https://stackoverflow.com/questions/12102318/opencv-findcornersubpix-precision
# increase to 1pix porque la posterior da demasiado rara
stdPix = 1.0
Cccd = np.repeat([stdPix**2 * np.eye(2)], m, axis=0)

Crt = np.repeat([False], m)  # no RT error




# camera position prior
camPrior = np.array([-34.629344, -58.370350])

aEarth = 6378137.0#Equatorial radius in m
bEarth = 6356752.3#Polar radius in m
def ll2m(lat, lon, lat0, lon0):
    '''
    lat0 en grados
    '''
    lat0 = np.deg2rad(lat0)
    lat = np.deg2rad(lat)
    dlat = lat - lat0

    lon0 = np.deg2rad(lon0)
    dlon = np.deg2rad(lon) - lon0

    a_cos_cuad = (aEarth * np.cos(lat))**2
    b_sin_cuad = (bEarth * np.sin(lat))**2
    Rm = (aEarth * bEarth)**2 / np.power(a_cos_cuad + b_sin_cuad, 3 / 2)
    Rn = aEarth**2 / np.sqrt(a_cos_cuad + b_sin_cuad)

    dy = dlat * Rm
    dx = dlon * Rn * np.cos(lat)

    return dx, dy

objM = np.array(ll2m(objecPoints[:,0], objecPoints[:,1], camPrior[0], camPrior[1])).T

plt.figure()
plt.scatter(objM[:,0],objM[:,1])
plt.scatter(0,0) # la camara esta en el cero


plt.figure()
plt.imshow(img)
plt.scatter(imagePoints[:,0], imagePoints[:,1])

h0cam = 15.7  # altura medida en metros
x0cam = np.array([0, 0, h0cam])


# %% saco rvec, t vec con solve pnp. nopuedo porque no hay para este modelo
# tiro a ojo las condicioenes iniciales
versors0 = cl.euler(np.pi / 3, np.pi / 10, np.pi)

rV0 = cv2.Rodrigues(versors0)[0][:,0]
tV0 = versors0.dot(- np.array([0, 0, h0cam]))

rVinterval = np.array([(np.pi / 3)**2] * 3) # 60 grados para cada lado
tVcov = np.diag([16, 16, 1])  # le pongo una std de 4m en horizontal y 1m en vertical
tVcov0 = versors0.dot(tVcov).dot(versors0.T)

xm, ym, cm = cl.inverse(imagePoints, rV0, tV0, cameraMatrix, distCoeffs, model,
                        Cccd=Cccd, Cf=Cf, Ck=Ck, Crt=False, Cfk=Cfk)


def mahalanobosDiff(xm, ym, cm, objM):
    '''
    calcula la proyeccion de las diferencias de predicciones pero proyectadas
    sobre las bases que diagonalizan las covarianzas
    '''
    xy = np.array([xm, ym]).T - objM

    S = np.linalg.inv(cm)

    u, s, v = np.linalg.svd(S)

    sdiag = np.zeros_like(u)
    sdiag[:, [0, 1], [0, 1]] = np.sqrt(s)

    A = (u.reshape((m, 2, 2, 1, 1)) *
         sdiag.reshape((m, 1, 2, 2, 1)) *
         v.transpose((0, 2, 1)).reshape((m, 1, 1, 2, 2))
         ).sum(axis=(2, 3))

    return np.sum(xy.reshape((m, 2, 1)) * A, axis=2)

mahalanobosDiff(xm, ym, cm, objM)


plt.figure()
ax = plt.gca()
ax.scatter(objM[:,0],objM[:,1])
ax.scatter(0,0) # la camara esta en el cero
cl.plotPointsUncert(ax, cm, xm, ym, 'k')



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
    itypes = [T.dvector]
    # xm, ym, cM
    otypes = [T.dvector]  # , T.dtensor4]

    # Python implementation:
    def perform(self, node, inputs_storage, output_storage):
        # print('IDX %d projection %d, global %d' %
        #       (self.idx, self.count, projCount))
        Xext = inputs_storage[0]
#        xyM, cM = output_storage

        # saco los parametros de flat para que los use la func de projection
        rVec, tVec = bl.flat2ext(Xext)

        xm, ym, cm = cl.inverse(imagePoints, rV0, tV0, cameraMatrix,
                                distCoeffs, model, Cccd=Cccd, Cf=Cf, Ck=Ck,
                                Crt=False, Cfk=Cfk)

        xy = mahalanobosDiff(xm, ym, cm, objM)

        output_storage[0] = xy.reshape(-1)

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
inMean = np.array([810.36355832, 470.00982051, 798.4825905 ])


inCov = np.array(
      [[ 0.1336706 ,  0.04361427, -0.17582661],
       [ 0.04361427,  0.33718186, -0.22929851],
       [-0.17582661, -0.22929851,  1.36442735]])


Sin = np.array([0.3656099 , 0.58067362, 1.16808705])


exMean = np.array(
      [[-1.87655907e-01,  8.68920468e-02, -8.75026635e-02,
        -3.23239768e+00, -5.50789861e+00,  6.08491932e+00],
       [ 9.24989200e-01,  7.30526552e-02, -7.33215817e-02,
        -3.05891314e+00, -4.22428798e+00,  3.94351921e+00],
       [ 9.18611356e-01, -3.44844751e-01,  9.74298118e-01,
         9.61129019e+00, -5.98895198e+00,  3.96471691e+00],
       [ 1.37578846e-01, -5.28738609e-01,  6.42504512e-01,
        -6.47816476e+00, -4.09608043e+00,  3.74021899e+00],
       [-2.83139008e-02,  8.62871422e-01, -6.66602428e-04,
        -1.98225060e-01, -2.48329497e+00,  7.61484145e+00],
       [ 8.51179121e-01,  5.20407517e-01,  8.95260047e-01,
         7.81925472e+00, -6.90796750e+00,  5.21594297e+00],
       [ 6.33803482e-01,  9.20642809e-01,  1.83533206e+00,
         1.87385098e+01,  6.18614167e+00,  7.56164478e+00],
       [-7.75052454e-01,  8.29911229e-02,  1.10970260e+00,
        -4.89649577e+00, -3.67205125e+00,  1.12440016e+01],
       [ 1.67071964e-02,  7.90824852e-03, -1.58290397e+00,
        -2.40141020e+00,  4.87432605e+00,  5.22993152e+00],
       [-6.15599826e-02, -7.25710543e-02, -3.09841916e+00,
         3.99057505e+00,  4.88009783e+00,  5.46007247e+00],
       [ 1.66943147e-01,  6.64381516e-02,  2.38464407e-02,
        -3.82834616e+00,  2.05737042e+00,  5.19274240e+00],
       [ 6.59321371e-01,  2.00840142e-01,  2.26504389e-01,
        -6.19048807e-01, -2.02913290e+01,  1.43624968e+01],
       [-7.22857554e-02, -2.35668796e-02, -3.60178148e-02,
        -4.70131493e+00, -6.79413603e-01,  1.96555426e+01],
       [-1.31518458e+00, -9.54254989e-02,  2.89742448e+00,
        -1.75277481e+01, -3.15729132e+00,  1.22604394e+01],
       [-1.49070028e-01, -1.04481417e+00,  1.68433846e+00,
        -1.74785456e+00, -5.26113707e+00,  7.86813963e+00],
       [-4.36530432e-01, -5.44654019e-01,  1.37413918e+00,
        -4.96522891e+00, -5.53284989e+00,  1.24435652e+01],
       [-2.07652706e-01,  8.45708204e-01,  1.69280948e+00,
         1.33145873e+00,  1.59175406e+01,  2.01936043e+01],
       [-1.99477981e-01,  1.11044198e+00,  5.87726903e-01,
         1.65018520e+01,  6.74021490e+00,  1.72115220e+01],
       [ 4.83712332e-02,  8.69832245e-02, -1.47586712e-01,
        -2.33155227e+00, -1.04947273e+01,  1.84239433e+01],
       [-8.35713961e-03, -2.37365325e-01, -1.66612621e-01,
        -9.54090572e+00, -7.92107516e-01,  2.47446520e+00],
       [ 3.78599511e-01,  1.01083238e+00,  1.41162660e+00,
         1.19174033e+01, -4.25437323e+00,  7.40109141e+00],
       [ 6.73658295e-01, -3.79832272e-03, -2.98824504e+00,
         3.57734053e+00,  4.03361243e-01,  1.44633186e+01],
       [-6.99784934e-02, -5.53692806e-01,  2.84772171e-01,
        -1.14057600e+01,  1.76575403e+00,  3.42620469e+00],
       [ 2.09150790e-02, -1.25886507e+00,  2.12822462e-03,
        -5.16681821e+00, -2.42976511e+00,  1.38399100e+00],
       [-1.04750258e+00,  2.20986272e-01,  2.60457473e+00,
        -1.66057168e+01,  7.00084660e+00,  1.17682223e+01],
       [-8.85598159e-01,  6.25207666e-01,  2.07847136e+00,
        -1.05897920e+01,  1.39703242e+01,  1.48073678e+01],
       [-1.76341463e-01, -9.15797290e-01, -2.93856729e+00,
         1.41019424e+00,  1.61347948e+01,  1.19022451e+01],
       [ 6.87842965e-01,  4.03261010e-01,  1.34855826e+00,
         9.24237394e+00, -6.65169744e+00,  4.35754630e+00],
       [-6.74867559e-01, -1.40488895e-01, -3.01353575e+00,
         1.83247947e+00,  6.91446116e+00,  9.76887846e+00],
       [ 2.72652665e-02, -3.42597717e-02, -6.62741952e-02,
        -3.92400685e+00, -2.73140195e+00,  4.23019720e+00],
       [ 3.37172270e-01,  5.16833509e-01, -3.09618941e-01,
         1.32221369e+01, -9.17145550e+00,  1.50017446e+01],
       [ 3.98065561e-02,  7.45372380e-01,  9.74924806e-04,
         1.65011729e+01, -2.89626159e+00,  1.04386689e+01],
       [-7.28669134e-02,  9.14371258e-01,  5.66606399e-01,
         1.74782269e+01,  1.86578543e+00,  1.01670645e+01]])


Sex = np.array(
      [[2.01622735e-03, 1.51799439e-03, 8.71692153e-04, 6.69387996e-03,
        1.12358272e-02, 1.48248371e-02],
       [2.09190241e-03, 1.76806446e-03, 8.61060229e-04, 6.87035203e-03,
        1.26383395e-02, 1.30862217e-02],
       [2.73188134e-03, 2.33926314e-03, 2.16153263e-03, 2.30205653e-02,
        1.80691856e-02, 2.63338888e-02],
       [3.00492951e-03, 2.60962283e-03, 1.16930878e-03, 9.53907263e-03,
        9.36040080e-03, 1.81638840e-02],
       [1.78922438e-03, 1.34003636e-03, 1.88929691e-05, 8.79349881e-03,
        9.43112366e-03, 1.20093728e-02],
       [6.09865239e-03, 5.58607301e-03, 2.40224408e-03, 1.94666116e-02,
        1.59712736e-02, 2.26515869e-02],
       [1.51299085e-02, 1.73762427e-02, 9.96884625e-03, 5.99043060e-02,
        3.58021976e-02, 3.83587346e-02],
       [3.37564790e-03, 2.98857620e-03, 1.26389022e-03, 1.31575389e-02,
        1.83175305e-02, 2.25340628e-02],
       [2.22098239e-03, 2.49718346e-03, 5.87040432e-04, 5.18851282e-03,
        7.82261131e-03, 1.49216424e-02],
       [2.73086755e-03, 3.45411445e-03, 8.19080662e-04, 6.53206320e-03,
        1.06979110e-02, 1.25970914e-02],
       [1.92847694e-03, 1.41758012e-03, 1.09818876e-03, 6.10437650e-03,
        1.16608773e-02, 1.36638530e-02],
       [1.63788430e-02, 1.30197847e-02, 7.05469679e-03, 2.46193264e-02,
        4.98141424e-02, 5.44780699e-02],
       [2.08970564e-02, 7.02409774e-03, 1.76151708e-03, 2.06861096e-02,
        3.15448631e-02, 8.17667007e-02],
       [1.07457202e-02, 1.38046555e-02, 7.52839036e-03, 6.13058555e-02,
        2.62365250e-02, 6.13300882e-02],
       [3.58525705e-03, 3.87923149e-03, 1.61893400e-03, 1.28780774e-02,
        1.41745364e-02, 2.20180180e-02],
       [1.28146234e-02, 1.01971171e-02, 4.10749350e-03, 3.13639416e-02,
        2.02339540e-02, 5.04092927e-02],
       [1.40093783e-02, 1.55289648e-02, 7.72237491e-03, 2.51592558e-02,
        8.40493286e-02, 9.32630673e-02],
       [1.52124069e-02, 2.06413406e-02, 1.08592784e-02, 4.98144307e-02,
        4.56297311e-02, 4.74906592e-02],
       [9.59424454e-03, 8.37777525e-03, 2.89881237e-03, 1.93904975e-02,
        3.99214924e-02, 6.82252648e-02],
       [8.72435061e-04, 1.68156750e-03, 7.56408366e-04, 4.33410106e-03,
        9.22202663e-03, 1.77212155e-02],
       [5.67844696e-03, 6.15528306e-03, 2.87445958e-03, 1.67568970e-02,
        1.63174254e-02, 2.43371687e-02],
       [7.30380062e-03, 1.73041899e-04, 1.60555998e-03, 1.43879496e-02,
        2.17426763e-02, 2.50415055e-02],
       [2.29167427e-03, 2.29110364e-03, 1.82475220e-03, 1.31318332e-02,
        1.42240745e-02, 2.45435653e-02],
       [2.03955682e-03, 1.81980747e-03, 1.19456907e-03, 9.36305588e-03,
        6.23218747e-03, 1.15164725e-02],
       [1.02605683e-02, 1.06412509e-02, 6.20083558e-03, 6.57853983e-02,
        2.84223133e-02, 6.19105522e-02],
       [1.61128793e-02, 1.25333786e-02, 8.55634849e-03, 5.97927633e-02,
        6.99316807e-02, 6.85914428e-02],
       [1.65682103e-02, 1.62941974e-02, 6.80746406e-03, 1.63031509e-02,
        5.55983154e-02, 5.86464621e-02],
       [6.13010086e-03, 4.57919294e-03, 2.34832968e-03, 1.80635937e-02,
        1.30840601e-02, 2.74213734e-02],
       [5.84576657e-03, 6.24866180e-03, 1.39727111e-03, 1.13106557e-02,
        2.07973926e-02, 3.13759074e-02],
       [1.80958546e-03, 1.34689098e-03, 6.28822121e-04, 5.28920086e-03,
        7.83934464e-03, 1.21832086e-02],
       [8.49275397e-03, 1.47709054e-02, 4.61308762e-03, 1.12845628e-01,
        7.94452318e-02, 1.19291243e-01],
       [3.39868590e-03, 6.72615462e-03, 4.60190698e-05, 6.34898549e-02,
        2.34450303e-02, 4.22995863e-02],
       [7.70492266e-03, 1.02269515e-02, 4.43320407e-03, 7.34887931e-02,
        2.68215026e-02, 3.40593443e-02]]).reshape(-1)


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
nDraws = 200000
nTune = 0
tuneBool = nTune != 0
nChains = 6

# %%
# escalas características del tipical set
scaIn = 1 / radiusStepsNdim(NintrParams)[1]
scaEx = 1 / radiusStepsNdim(6)[1]

InSeed = pm.MvNormal.dist(mu=inMean, cov=inCov).random(size=nChains)
exSeed = np.random.randn(n, 6, nChains) * Sex.reshape((n, 6, 1))
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
# como veo que la 12 va mejorando pero falta converger le mando la 13 con cond actualizadas
# la 14 la largo con cond iniciales mas dispersas y parece que es casi
# la definitiva si le saco el burnin!!!
# la 15 ya tiene que se rla ultima. nop. todavia es transitoria
# va la 16 con lo ultimo de los resultados de la 15

pathFiles = "/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/"
pathFiles += "extraDataSebaPhD/traces" + str(16)

np.save(pathFiles + "Int", trace['xIn'])
np.save(pathFiles + "Ext", trace['xEx'])

print("saved data to")
print(pathFiles)

