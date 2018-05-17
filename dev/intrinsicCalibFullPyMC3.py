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
from importlib import reload
import corner
import time

# %env THEANO_FLAGS='device=cuda, floatX=float32'
import theano
import theano. tensor as T
import pymc3 as pm
import scipy as sc
import seaborn as sns

import scipy.optimize as opt

import sys
sys.path.append("/home/sebalander/Code/sebaPhD")
from calibration import calibrator as cl
from dev import bayesLib as bl

print('libraries imported')

# %%

def stereoFromFisheye(distcoeffs):
    '''
    takes 4 distcoeffs of the opencv fisheye model and returns the
    corresponding k stereografic parameter suche on average they are equal
    (integrating over the angle 0 to pi/2)
    from wolfram site:
    https://www.wolframalpha.com/input/?i=integral+of+K*tan(x%2F2)+-+(x%2Bk1*x%5E3%2Bk2*x%5E5%2Bk3*x%5E7%2Bk4*x%5E9)+from+0+to+pi%2F2
    '''
    piPow = np.pi**(np.arange(1,6)*2)
    numAux = np.array([3840, 480, 80, 15, 3])
    fisheyeIntegral = np.sum(piPow * numAux) / 30720
    return fisheyeIntegral / np.log(2)


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
nIm = len(imageSelection)  # cantidad de imagenes

# ## load data
imagePoints = np.load(cornersFile)[imageSelection]

chessboardModel = np.load(patternFile)
nPt = chessboardModel.shape[1]  # cantidad de puntos por imagen
imgSize = tuple(np.load(imgShapeFile))
# images = glob.glob(imagesFolder+'*.png')

# Parametros de entrada/salida de la calibracion
objpoints2D = np.array([chessboardModel[0, :, :2]]*nIm).reshape((nIm, nPt, 2))

# load model specific data from opencv Calibration
distCoeffs = np.load(distCoeffsFile)
cameraMatrix = np.load(linearCoeffsFile)
rVecsOCV = np.load(rVecsFile)[imageSelection]
tVecsOCV = np.load(tVecsFile)[imageSelection]

# comparo con opencv
distCoeffsFileOCV = imagesFolder + camera + modelos[2] + "DistCoeffs.npy"
linearCoeffsFileOCV = imagesFolder + camera + modelos[2] + "LinearCoeffs.npy"
distCoeffsOCV = np.load(distCoeffsFileOCV)
cameraMatrixOCV = np.load(linearCoeffsFileOCV)

stereoFromFisheye(distCoeffsOCV) * np.mean(cameraMatrixOCV[[0, 1], [0, 1]])
#esto da muy mal, habría que pesar mejor en diferentes regiones de la imagen

print('raw data loaded')

# pongo en forma flat los valores iniciales
XintOCV, Ns = bl.int2flat(cameraMatrix, distCoeffs, model)
XextListOCV = np.array([bl.ext2flat(rVecsOCV[i], tVecsOCV[i])for i in range(nIm)])

NintrParams = XintOCV.shape[0]
NextrParams = nIm * 6
NfreeParams = NextrParams + NintrParams
NdataPoints = nIm * nPt

# 0.1pix as image std
# https://stackoverflow.com/questions/12102318/opencv-findcornersubpix-precision
# increase to 1pix porque la posterior da demasiado rara
stdPix = 1.0
Ci = np.repeat([stdPix**2 * np.eye(2)], NdataPoints, axis=0).reshape(nIm, nPt, 2, 2)

Crt = np.repeat([False], nIm)  # no RT error

# output file
intrinsicParamsOutFile = imagesFolder + camera + model + "intrinsicParamsML"
intrinsicParamsOutFile = intrinsicParamsOutFile + str(stdPix) + ".npy"

## pruebo con un par de imagenes
#for j in range(0, nIm, 3):
#    xm, ym, Cm = cl.inverse(imagePoints[j, 0], rVecs[j], tVecs[j],
#                            cameraMatrix,
#                            distCoeffs, model, Cccd=Ci[j], Cf=False, Ck=False,
#                            Crt=False, Cfk=False)
#    print(xm, ym, Cm)


# datos medidos, observados, experimentalmente en el mundo y en la imagen
yObs = objpoints2D.reshape(-1)

print('data formated')

# %% testeo el error calculado como antes
params = dict()

params["imagePoints"] = imagePoints
params["model"] = model
params["chessboardModel"] = chessboardModel
params["Cccd"] = Ci
params["Cf"] = False
params["Ck"] = False
params["Crt"] = [False] * nIm
params["Cfk"] = False

reload(bl)
Eint = bl.errorCuadraticoInt(XintOCV, Ns, XextListOCV, params)

# %% defino la funcion a minimizar


def objective(xAll):
    Xint = xAll[:Ns[1]]
    XextList = xAll[Ns[1]:].reshape((-1, 6))

    Eint = bl.errorCuadraticoInt(Xint, Ns, XextList, params)
    return np.sum(Eint)


xAllOCV = np.concatenate([XintOCV, XextListOCV.reshape(-1)])


# %% result of optimisation:

xAllOpt = np.array([
        8.16472244e+02,  4.72646126e+02,  7.96435717e+02, -1.77272668e-01,
        7.67281179e-02, -9.03548594e-02, -3.33339378e+00, -5.54790259e+00,
        5.99651705e+00,  9.30295123e-01,  6.21785570e-02, -6.96743493e-02,
       -3.13558219e+00, -4.25053069e+00,  3.87610434e+00,  9.19901975e-01,
       -3.59066370e-01,  9.74042501e-01,  9.47115482e+00, -6.02396770e+00,
        3.96904837e+00,  1.36935625e-01, -5.42713201e-01,  6.43889150e-01,
       -6.55503634e+00, -4.13393364e+00,  3.64817246e+00, -2.13080979e-02,
        8.53474025e-01,  1.11909981e-03, -3.06900646e-01, -2.53776109e+00,
        7.57360018e+00,  8.52154654e-01,  5.11584688e-01,  8.97896445e-01,
        7.68107905e+00, -6.95803581e+00,  5.21527990e+00,  6.29435124e-01,
        9.12272513e-01,  1.83799323e+00,  1.85151971e+01,  6.08915024e+00,
        7.66699884e+00, -7.78003463e-01,  6.73711760e-02,  1.10490676e+00,
       -5.06829679e+00, -3.74958656e+00,  1.11505467e+01,  2.54343546e-02,
        4.05416594e-03, -1.58361597e+00, -2.48236025e+00,  4.84091669e+00,
        5.20689047e+00, -4.43952330e-02, -7.01522701e-02, -3.09646881e+00,
        3.89079512e+00,  4.85194798e+00,  5.49819233e+00,  1.69406316e-01,
        5.40372043e-02,  3.00215179e-02, -3.90828973e+00,  1.99912411e+00,
        5.13066284e+00,  6.62491305e-01,  1.92824964e-01,  2.29515447e-01,
       -9.10653837e-01, -2.04052824e+01,  1.42154590e+01, -6.37592106e-02,
       -2.53455685e-02, -3.58656047e-02, -5.00455800e+00, -8.12217250e-01,
        1.95699162e+01, -1.32713680e+00, -1.01306097e-01,  2.89174569e+00,
       -1.77829953e+01, -3.26660847e+00,  1.20506102e+01, -1.55204173e-01,
       -1.05546726e+00,  1.68382535e+00, -1.88353255e+00, -5.31860869e+00,
        7.82312257e+00, -4.37144137e-01, -5.52910372e-01,  1.37146393e+00,
       -5.16628075e+00, -5.62202475e+00,  1.23453259e+01, -2.16223315e-01,
        8.37239606e-01,  1.69334147e+00,  9.75590372e-01,  1.57674791e+01,
        2.01951559e+01, -1.99116907e-01,  1.10314769e+00,  5.88264305e-01,
        1.61820088e+01,  6.60128144e+00,  1.72941117e+01,  5.40902817e-02,
        7.62769730e-02, -1.49449032e-01, -2.63389072e+00, -1.06080724e+01,
        1.82899752e+01, -8.56471354e-03, -2.45078027e-01, -1.64320059e-01,
       -9.59370234e+00, -8.45154875e-01,  2.38798610e+00,  3.77548626e-01,
        9.98276457e-01,  1.41182263e+00,  1.17437175e+01, -4.32923051e+00,
        7.43836485e+00,  6.96268996e-01,  3.95929202e-02, -2.98359684e+00,
        3.33863256e+00,  2.87001819e-01,  1.45762698e+01, -6.78229876e-02,
       -5.67645552e-01,  2.88318506e-01, -1.14904508e+01,  1.70303908e+00,
        3.30095194e+00,  2.42190792e-02, -1.27367655e+00,  1.54813038e-03,
       -5.20891681e+00, -2.45320303e+00,  1.31695324e+00, -1.06274465e+00,
        2.08679680e-01,  2.60268127e+00, -1.68834508e+01,  6.90152958e+00,
        1.16182962e+01, -8.95894326e-01,  6.12672038e-01,  2.07821135e+00,
       -1.08799650e+01,  1.38492242e+01,  1.47120196e+01, -1.60848855e-01,
       -9.06361915e-01, -2.94268629e+00,  1.16721052e+00,  1.60323027e+01,
        1.19152148e+01,  6.91402436e-01,  3.94207470e-01,  1.34789930e+00,
        9.09838596e+00, -6.69821565e+00,  4.36094691e+00, -6.55471764e-01,
       -1.34010450e-01, -3.01528340e+00,  1.66529059e+00,  6.84657929e+00,
        9.77907952e+00,  3.24220688e-02, -4.44796297e-02, -6.42984860e-02,
       -3.98798128e+00, -2.77091915e+00,  4.16758939e+00,  3.48300864e-01,
        5.05710578e-01, -3.11771835e-01,  1.29033074e+01, -9.27263245e+00,
        1.49798210e+01,  4.11331979e-02,  7.36662257e-01,  2.92912667e-03,
        1.62603380e+01, -3.00052137e+00,  1.05005924e+01, -7.00610553e-02,
        9.00948402e-01,  5.66637445e-01,  1.72115713e+01,  1.75486781e+00,
        1.02325778e+01])


XintOpt = xAllOpt[:Ns[1]]
XextListOpt = xAllOpt[Ns[1]:].reshape((-1,6))

# %%
#ret = opt.minimize(objective, xAll0, method='Powell', tol=1e-1)
#ret2 = opt.minimize(objective, ret.x, method='Powell', tol=1e-3)
#ret3 = opt.minimize(objective, ret2.x, method='Powell', tol=1e-6)
#ret4 = opt.minimize(objective, ret3.x, method='Powell', tol=1e-6)
#ret5 = opt.minimize(objective, ret4.x, tol=1e-4)

#ret = opt.minimize(objective, xAllOCV, tol=1e-4)
if False:
    ret = opt.minimize(objective, xAllOpt, tol=1e-4)
    errAbs = np.abs(ret.x - xAllOpt)
    valAbs = np.abs(ret.x)
    plt.plot(valAbs, errAbs, '.')

    epAbs = np.sort(errAbs)
    epRel = np.sort(errAbs / valAbs)

    A, R = np.meshgrid(epAbs, epRel)

    validity = R.reshape((NfreeParams, NfreeParams,1)) * valAbs.reshape((1, 1, -1))
    validity += A.reshape((NfreeParams, NfreeParams, 1))
    validity = errAbs < validity
    validPoints = np.sum(validity, axis=2) / errAbs.shape[0]


    plt.pcolormesh(A, R, validPoints, cmap='gray')
    plt.loglog()
    plt.xlabel('Abs Error')
    plt.ylabel('Rel Error')


    C = np.linalg.inv(ret.hess_inv)
    sig = np.sqrt(np.diag(C))
    Corr = C / (sig.reshape((-1, 1)) * sig.reshape((1, -1)))

    plt.matshow(Corr, vmin=-1, vmax=1, cmap='coolwarm')


# %% trato de hacerle una derivada
#import numdifftools as ndf
#
#J = ndf.Jacobian(objective)



# %%
'''
aca defino la proyeccion para que la pueda usar thean0
http://deeplearning.net/software/theano/extending/extending_theano.html
'''


class ProjectionT(theano.Op):
    # itypes and otypes attributes are
    # compulsory if make_node method is not defined.
    # They're the type of input and output respectively
    # xAll # xInt, xExternal
    itypes = [T.dvector] # , T.dmatrix]
    # xm, ym, cM
    otypes = [T.dtensor3]  # , T.dtensor4]

    # Python implementation:
    def perform(self, node, inputs_storage, output_storage):
        # print('IDX %d projection %d, global %d' %
        #       (self.idx, self.count, projCount))
        Xint = inputs_storage[0][:Ns[1]]
        Xext = inputs_storage[0][Ns[1]:].reshape((-1,6))
#        xyM, cM = output_storage

        # saco los parametros de flat para que los use la func de projection
        # print(Xint)
        cameraMatrix, distCoeffs = bl.flat2int(Xint, Ns, model)
        # print(cameraMatrix, distCoeffs)
        xy = np.zeros((nIm, nPt, 2))
        cm = np.zeros((nIm, nPt, 2, 2))

        for j in range(nIm):
            rVec, tVec = bl.flat2ext(Xext[j])
            xi, yi = imagePoints.reshape((nIm, nPt, 2))[j].T
            xy[j, :, 0], xy[j, :, 1], cm[j] = cl.inverse(
                    xi, yi, rVec, tVec,
                    cameraMatrix, distCoeffs, model, Cccd=Ci[j])

        xy -= objpoints2D

        S = np.linalg.inv(cm)

        u, s, v = np.linalg.svd(S)

        sdiag = np.zeros_like(u)
        sdiag[:, :, [0, 1], [0, 1]] = np.sqrt(s)

        A = (u.reshape((nIm, nPt, 2, 2, 1, 1)) *
             sdiag.reshape((nIm, nPt, 1, 2, 2, 1)) *
             v.transpose((0, 1, 3, 2)).reshape((nIm, nPt, 1, 1, 2, 2))
             ).sum(axis=(3, 4))

        xy = np.sum(xy.reshape((nIm, nPt, 2, 1)) * A, axis=3)

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

#XintOper = T.dvector('XintOP')
#XextOper = T.dmatrix('XextOP')
xAllOper = T.dvector('xAllOper')

projTheanoWrap = ProjectionT()

projTfunction = theano.function([xAllOper],# XextOper],
                                projTheanoWrap(xAllOper))# , XextOper))

try:
#    outOpt = projTfunction(XintOpt, XextListOpt)
    outOpt = projTfunction(xAllOper)
except:
#    sys.exit("no anduvo el wrapper de la funciona  theano")
    print("no anduvo el wrapper de la funciona  theano")
else:
    pass
    # print(out)



# # no comparo con OCV para no meter un tema y despues tener que explicarlo
#outOCV = np.zeros_like(outOpt)
#for i in range(nIm):
#    xi, yi = imagePoints[i,0].T
#    outOCV[i,:,0], outOCV[i,:,1], _ = cl.inverse(xi, yi, rVecsOCV[i],
#              tVecsOCV[i], cameraMatrixOCV, distCoeffsOCV, modelos[2])
#
#outOCV -= objpoints2D

plotBool = False
if plotBool:
    plt.figure()
    plt.plot(outOpt[:, :, 0].flatten(), outOpt[:, :, 1].flatten(), '.k',
             markersize=0.7)

#plt.plot(outOCV[:, :, 0].flatten(), outOCV[:, :, 1].flatten(), '+r')


# %%
# from importlib import reload
# reload(cl)
# reload(bl)


# indexes to read diagona 2x2 blocks of a matrix
# nTot = 2 * nIm * nPt
# xInxs = [[[i, i], [i+1, i+1]] for i in range(0, nTot, 2)]
# yInxs = [[[i, i+1], [i, i+1]] for i in range(0, nTot, 2)]

projectionModel = pm.Model()

projTheanoWrap = ProjectionT()

# set lower and upper bounds for uniform prior distributions
# for camera matrix is important to avoid division by zero

# intrDelta = np.abs([100, 100, 200, 100,
#                    Xint[4] / 2, Xint[5] / 2, Xint[6] / 2, Xint[7] / 2])
intrDelta = np.abs([100, 100, 100])
intrLow = XintOpt - intrDelta  # intrinsic
intrUpp = XintOpt + intrDelta

extrDelta = np.array([[0.3, 0.3, 0.3, 3, 3, 3]]*nIm)
extrLow = XextListOpt - extrDelta
extrUpp = XextListOpt + extrDelta


allLow = np.concatenate([intrLow, extrLow.reshape(-1)])
allUpp = np.concatenate([intrUpp, extrUpp.reshape(-1)])

xAll0 = np.concatenate([XintOpt, XextListOpt.reshape(-1)], 0)

observedNormed = np.zeros((nIm * nPt * 2))


with projectionModel:

    # Priors for unknown model parameters
#    xIn = pm.Uniform('xIn', lower=intrLow, upper=intrUpp, shape=(NintrParams,),
#                     transform=None)
#    xEx = pm.Uniform('xEx', lower=extrLow, upper=extrUpp, shape=(nIm, 6),
#                     transform=None)
    xAl = pm.Uniform('xAl', lower=allLow, upper=allUpp, shape=allLow.shape,
                     transform=None)

    # apply numpy based function
    xyMNor = projTheanoWrap(xAl) # xIn, xEx)

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

try:
    covOPdiag = np.diag(ret.hess_inv)
except:
    covOPdiag = np.array(
      [0.99666917, 1.01725545, 0.99230996, 0.92448788, 0.81269367,
       0.46444881, 0.98598025, 0.98665314, 0.99619757, 0.97972878,
       0.96732291, 0.48881261, 1.02317142, 0.99738691, 1.02176324,
       0.98016844, 0.99053176, 0.98051997, 1.0045555 , 0.99957425,
       1.02122366, 0.87471028, 0.97719328, 0.55544213, 0.99778134,
       0.99827303, 1.00659944, 0.85216935, 0.98208789, 0.13934568,
       0.96463168, 1.01624222, 0.97889943, 0.97044974, 1.01869262,
       0.74040321, 1.01927562, 1.04108516, 1.14922666, 0.93030665,
       1.03253671, 0.97825876, 1.00254591, 1.0122422 , 1.00003419,
       1.01133444, 0.95523071, 0.84140185, 1.01104793, 1.04161458,
       1.13881759, 0.85826757, 0.60880472, 0.02773636, 0.99311584,
       1.02451849, 1.06699635, 0.99782994, 0.99631883, 0.8033285 ,
       0.94975035, 1.01521908, 1.01211061, 0.92451836, 0.81870015,
       0.82183431, 1.00520635, 0.99902623, 0.99371426, 1.09465475,
       1.06197124, 0.95492071, 0.99968204, 1.02553992, 1.02736384,
       1.11058998, 1.00785108, 0.91943086, 1.06795506, 1.03985036,
       0.99997106, 1.06452488, 0.99176444, 0.86202203, 1.01531346,
       1.03781383, 1.01318191, 0.90510674, 1.26167704, 0.87964094,
       1.02313256, 1.09463896, 1.0245659 , 1.0194543 , 1.00324881,
       0.99926508, 1.00268766, 1.01389343, 1.00373701, 1.01620944,
       1.0904048 , 1.05922911, 1.20156496, 1.00189132, 1.00447545,
       1.04580116, 1.01109723, 0.96776504, 1.00422124, 1.01554554,
       1.04860152, 1.00164859, 1.00463207, 0.90290399, 1.00116518,
       1.01461016, 1.01009007, 0.7869471 , 0.77256265, 0.13678561,
       0.93776702, 1.04072775, 1.1266469 , 0.99732331, 1.00230652,
       0.98210756, 1.02949382, 1.00271666, 1.02334042, 1.10331503,
       1.02724226, 0.99579062, 1.23285126, 1.06129086, 1.00075568,
       0.87217687, 0.94557873, 0.80048822, 1.05378651, 1.06405655,
       1.00725197, 0.81407575, 0.94466406, 0.58668429, 1.03874161,
       1.15654318, 1.00558437, 1.00568688, 1.22975702, 0.974743  ,
       1.03104617, 1.0591857 , 1.00597734, 1.05657699, 1.45644042,
       0.99690871, 1.20990415, 1.00970418, 1.30362057, 0.99993822,
       1.00001622, 0.99922863, 1.0002478 , 1.00094582, 1.00347633,
       0.96679911, 0.98195348, 0.8624708 , 1.00320789, 0.99922754,
       0.99894586, 0.99635016, 0.99876612, 0.94421186, 0.99904539,
       1.00533761, 1.00080393, 0.97327825, 0.9260509 , 0.05477286,
       0.98552962, 0.98541202, 1.00898964, 1.07033115, 0.98371253,
       0.99367689, 1.01665865, 1.0236957 , 1.05833444, 1.01003038,
       1.01508945, 0.955479  , 1.00029497, 1.00148514, 1.02546372,
       1.00120396, 0.99943828, 0.99830778, 1.01370469, 1.0046385 ,
       0.99987442])


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
#
inMean = np.array([816.52306533, 472.6509038 , 795.35757785])


#inCov = np.array(
#      [[ 0.1336706 ,  0.04361427, -0.17582661],
#       [ 0.04361427,  0.33718186, -0.22929851],
#       [-0.17582661, -0.22929851,  1.36442735]])


inCov = Sin = np.array(
      [[ 0.22201776,  0.02473017, -0.07705355],
       [ 0.02473017,  0.02946335, -0.02757416],
       [-0.07705355, -0.02757416,  0.90622708]])


exMean = np.array([[-1.80167178e-01,  7.70878291e-02, -9.11332085e-02,
        -3.33767486e+00, -5.53933063e+00,  5.99338849e+00],
       [ 9.30252815e-01,  6.29244768e-02, -7.00228922e-02,
        -3.13657634e+00, -4.25220832e+00,  3.87072930e+00],
       [ 9.18770862e-01, -3.59104432e-01,  9.72840572e-01,
         9.46331750e+00, -6.02505143e+00,  3.95175051e+00],
       [ 1.38368094e-01, -5.40701641e-01,  6.43722864e-01,
        -6.55540744e+00, -4.13410627e+00,  3.63578192e+00],
       [-2.38996202e-02,  8.51708791e-01,  1.10038464e-03,
        -3.13671548e-01, -2.53169888e+00,  7.56497691e+00],
       [ 8.54461633e-01,  5.12248498e-01,  8.97735395e-01,
         7.67730295e+00, -6.95649265e+00,  5.19809539e+00],
       [ 6.24022772e-01,  9.11974627e-01,  1.84133535e+00,
         1.85380532e+01,  6.10095731e+00,  7.65631705e+00],
       [-7.77273102e-01,  6.73312499e-02,  1.10478194e+00,
        -5.07150815e+00, -3.75016477e+00,  1.11403444e+01],
       [ 2.66400052e-02,  3.85463199e-03, -1.58203501e+00,
        -2.49082973e+00,  4.83656452e+00,  5.20168570e+00],
       [-4.49425065e-02, -6.68025381e-02, -3.09784857e+00,
         3.89631162e+00,  4.84452886e+00,  5.49198735e+00],
       [ 1.69530683e-01,  5.42945198e-02,  2.92943607e-02,
        -3.90946819e+00,  2.00034637e+00,  5.12276745e+00],
       [ 6.60246221e-01,  1.92566621e-01,  2.29654105e-01,
        -9.12700556e-01, -2.04042003e+01,  1.41847163e+01],
       [-6.00258060e-02, -2.68463338e-02, -3.60838138e-02,
        -5.00649319e+00, -8.11922768e-01,  1.95246215e+01],
       [-1.33185030e+00, -1.00501354e-01,  2.88778409e+00,
        -1.77998534e+01, -3.27404467e+00,  1.20315003e+01],
       [-1.56715071e-01, -1.05864768e+00,  1.68304519e+00,
        -1.88651720e+00, -5.31981199e+00,  7.81543480e+00],
       [-4.39015924e-01, -5.54065974e-01,  1.37152066e+00,
        -5.16910099e+00, -5.62044311e+00,  1.23354915e+01],
       [-2.12725179e-01,  8.36348015e-01,  1.69324731e+00,
         9.71432308e-01,  1.57485182e+01,  2.01341204e+01],
       [-2.00938217e-01,  1.10426149e+00,  5.88735787e-01,
         1.61808550e+01,  6.60226030e+00,  1.72631491e+01],
       [ 5.12075282e-02,  7.65048634e-02, -1.49076892e-01,
        -2.63534470e+00, -1.06133112e+01,  1.82756012e+01],
       [-8.07593457e-03, -2.48448650e-01, -1.65477009e-01,
        -9.59574683e+00, -8.35501930e-01,  2.36221967e+00],
       [ 3.79958987e-01,  9.99906946e-01,  1.41113308e+00,
         1.17419291e+01, -4.32946182e+00,  7.41637272e+00],
       [ 6.93547538e-01,  4.27537532e-02, -2.98380624e+00,
         3.33789720e+00,  2.84884758e-01,  1.45640980e+01],
       [-6.77337244e-02, -5.66907521e-01,  2.88282057e-01,
        -1.14902588e+01,  1.70107787e+00,  3.28529706e+00],
       [ 2.71373754e-02, -1.27192954e+00,  1.49621469e-03,
        -5.21038889e+00, -2.45187911e+00,  1.30818072e+00],
       [-1.06429943e+00,  2.12329072e-01,  2.59992594e+00,
        -1.68895300e+01,  6.89918253e+00,  1.15891882e+01],
       [-9.01672732e-01,  6.10882974e-01,  2.07802477e+00,
        -1.08991531e+01,  1.38690599e+01,  1.47006073e+01],
       [-1.61219868e-01, -9.08374670e-01, -2.94221168e+00,
         1.16533329e+00,  1.60280144e+01,  1.18848337e+01],
       [ 6.89585013e-01,  3.94733743e-01,  1.34967818e+00,
         9.10776785e+00, -6.69747467e+00,  4.34937548e+00],
       [-6.52008590e-01, -1.34253326e-01, -3.01545982e+00,
         1.66521547e+00,  6.85216957e+00,  9.77746025e+00],
       [ 3.18205199e-02, -4.38767512e-02, -6.64824468e-02,
        -3.99451213e+00, -2.76141393e+00,  4.15847115e+00],
       [ 3.47395592e-01,  5.07440498e-01, -3.11362863e-01,
         1.29073933e+01, -9.27749780e+00,  1.49588435e+01],
       [ 4.11161529e-02,  7.38634608e-01,  2.96523788e-03,
         1.62627796e+01, -3.00193706e+00,  1.04785773e+01],
       [-6.95117072e-02,  9.03717658e-01,  5.66261359e-01,
         1.72162297e+01,  1.75659780e+00,  1.02105909e+01]])


Sex = np.array([[1.52479303e-03, 1.68167445e-03, 8.98754252e-04, 8.72151718e-03,
        5.15530424e-03, 1.28739102e-02],
       [2.19472514e-03, 1.97445265e-03, 8.25030686e-04, 6.34301814e-03,
        7.98991959e-03, 1.25097042e-02],
       [2.51606013e-03, 2.08482813e-03, 1.94829836e-03, 1.87353192e-02,
        1.09968352e-02, 2.06112199e-02],
       [2.57611391e-03, 2.48998555e-03, 1.18997992e-03, 1.20328431e-02,
        4.14362996e-03, 1.58189060e-02],
       [1.26963587e-03, 1.35893135e-03, 7.06690137e-05, 9.22476480e-03,
        3.45503719e-03, 9.21037232e-03],
       [7.47604475e-03, 6.03579932e-03, 2.90351842e-03, 2.68711520e-02,
        1.65873650e-02, 2.55882018e-02],
       [1.69496695e-02, 1.81995874e-02, 1.11309217e-02, 6.44843305e-02,
        3.14112593e-02, 5.07653062e-02],
       [3.13279083e-03, 3.01720583e-03, 1.36930239e-03, 1.70649961e-02,
        8.55825145e-03, 1.72046343e-02],
       [2.20576561e-03, 1.19789546e-03, 6.52279865e-04, 7.83241475e-03,
        3.61458349e-03, 9.72823174e-03],
       [2.82787519e-03, 3.61501197e-03, 8.70681686e-04, 8.22622430e-03,
        5.38497796e-03, 1.53320458e-02],
       [1.58642973e-03, 1.79285300e-03, 1.20556659e-03, 6.80597694e-03,
        6.46813167e-03, 1.42327812e-02],
       [1.84009515e-02, 1.41779306e-02, 6.79883320e-03, 3.12334884e-02,
        4.48088419e-02, 6.95537796e-02],
       [7.98732941e-03, 5.79336575e-03, 2.07111388e-03, 2.51199486e-02,
        1.44062150e-02, 6.01547813e-02],
       [1.38209459e-02, 1.43769694e-02, 8.94933084e-03, 9.84376142e-02,
        2.58117457e-02, 6.11061713e-02],
       [4.18011113e-03, 4.62053996e-03, 2.18349714e-03, 1.41098763e-02,
        1.12261060e-02, 2.15771037e-02],
       [1.04489156e-02, 1.19281751e-02, 3.12076298e-03, 1.93509174e-02,
        7.66211865e-03, 3.51358863e-02],
       [1.52623685e-02, 1.45045138e-02, 7.69081751e-03, 3.13981700e-02,
        8.35139512e-02, 1.00461429e-01],
       [1.69990048e-02, 1.78319288e-02, 1.11578108e-02, 5.95182716e-02,
        3.45686089e-02, 5.34271482e-02],
       [5.80086828e-03, 8.31709423e-03, 3.13288245e-03, 2.40468109e-02,
        2.49542601e-02, 5.48008967e-02],
       [1.33222368e-03, 1.70249875e-03, 6.94106616e-04, 6.44673519e-03,
        4.07957995e-03, 1.55058279e-02],
       [5.36821821e-03, 5.70037415e-03, 3.06505616e-03, 2.38766239e-02,
        8.12687807e-03, 1.84587611e-02],
       [8.07127230e-03, 7.12536903e-03, 1.98864206e-03, 1.93546834e-02,
        1.15530610e-02, 2.81182943e-02],
       [2.28407405e-03, 2.61409513e-03, 1.90622518e-03, 1.34921667e-02,
        7.23600715e-03, 1.87651943e-02],
       [1.57477733e-03, 1.56273289e-03, 3.54651310e-04, 8.70370344e-03,
        3.68166414e-03, 9.08649200e-03],
       [1.01786601e-02, 9.81502545e-03, 5.21805738e-03, 7.38352975e-02,
        2.77284642e-02, 5.23798554e-02],
       [1.74118881e-02, 1.41258355e-02, 8.98176608e-03, 6.52864189e-02,
        6.99362974e-02, 7.41618321e-02],
       [1.45806489e-02, 1.85215233e-02, 6.75793917e-03, 2.12389346e-02,
        5.36688843e-02, 5.87534234e-02],
       [5.77746496e-03, 5.66282304e-03, 2.20897842e-03, 1.50999702e-02,
        8.42518896e-03, 2.07843349e-02],
       [6.02643031e-03, 6.72757792e-03, 1.42294416e-03, 1.36019522e-02,
        1.45002870e-02, 3.14657651e-02],
       [1.79992842e-03, 1.16831095e-03, 6.03507314e-04, 6.17583094e-03,
        3.81726175e-03, 9.42681515e-03],
       [8.13872308e-03, 7.98843619e-03, 3.53079514e-03, 6.31883706e-02,
        3.33131466e-02, 7.42287605e-02],
       [3.17841141e-03, 6.56346679e-03, 3.88077619e-04, 6.64711504e-02,
        9.47797196e-03, 4.63800137e-02],
       [8.60052456e-03, 1.01189668e-02, 5.30223279e-03, 7.01465438e-02,
        2.25246945e-02, 4.98055332e-02]]).reshape(-1)

exCov = np.diag(Sex**2)
#inMean = xAllOpt[:Ns[-1]]
#exMean = xAllOpt[Ns[-1]:].reshape((-1,6))
#Sin = np.abs(inMean) * 1e-4 # np.sqrt(covOPdiag[:Ns[-1]])
#inCov = np.diag(Sin**2)
#Sex = np.abs(exMean).reshape(-1) * 1e-3 # np.sqrt(covOPdiag[Ns[-1]:])

# del rejunte de todos en un solo vector
#alMean = np.concatenate((inMean, exMean.reshape(-1)))
#Sal = np.concatenate((np.sqrt(np.diag(inCov)), Sex.reshape(-1)))
alMean = np.array([ 8.16506629e+02,  4.72673059e+02,  7.95248657e+02, -1.80478563e-01,
        7.64854287e-02, -9.13167238e-02, -3.33719539e+00, -5.53946684e+00,
        5.99255238e+00,  9.29933477e-01,  6.19355734e-02, -7.00424639e-02,
       -3.13797135e+00, -4.25570589e+00,  3.86833239e+00,  9.17789946e-01,
       -3.58883359e-01,  9.73287512e-01,  9.46048391e+00, -6.02816511e+00,
        3.94251842e+00,  1.37838383e-01, -5.41364386e-01,  6.43248684e-01,
       -6.55482426e+00, -4.13328724e+00,  3.63596850e+00, -2.38513022e-02,
        8.51751654e-01,  1.10610313e-03, -3.14546183e-01, -2.53041905e+00,
        7.56580076e+00,  8.56005898e-01,  5.14018197e-01,  8.97799747e-01,
        7.69065670e+00, -6.95962313e+00,  5.20488180e+00,  6.23861542e-01,
        9.07658868e-01,  1.84114876e+00,  1.85393052e+01,  6.10918872e+00,
        7.65143246e+00, -7.77124379e-01,  6.73641024e-02,  1.10484213e+00,
       -5.06958316e+00, -3.74651708e+00,  1.11385232e+01,  2.62123326e-02,
        3.73425061e-03, -1.58203856e+00, -2.49044348e+00,  4.83634426e+00,
        5.20088990e+00, -4.53474123e-02, -6.70142923e-02, -3.09737531e+00,
        3.89492548e+00,  4.84259754e+00,  5.49126684e+00,  1.68757081e-01,
        5.51274089e-02,  3.01891954e-02, -3.90800567e+00,  1.99927315e+00,
        5.12494645e+00,  6.51909976e-01,  1.92633771e-01,  2.30285713e-01,
       -9.08773459e-01, -2.04135586e+01,  1.41970471e+01, -6.09647229e-02,
       -2.65710826e-02, -3.58392354e-02, -5.01073081e+00, -8.15060747e-01,
        1.95226844e+01, -1.33030011e+00, -1.00740285e-01,  2.88939332e+00,
       -1.78299160e+01, -3.28421749e+00,  1.20380072e+01, -1.57007396e-01,
       -1.05776631e+00,  1.68235217e+00, -1.88811121e+00, -5.32016635e+00,
        7.81677071e+00, -4.39487926e-01, -5.52374368e-01,  1.37220110e+00,
       -5.17298218e+00, -5.61705099e+00,  1.23312359e+01, -2.12694378e-01,
        8.32349680e-01,  1.69271964e+00,  9.70210098e-01,  1.57458590e+01,
        2.01492648e+01, -2.03789791e-01,  1.10557197e+00,  5.91258391e-01,
        1.61868046e+01,  6.59290887e+00,  1.72540554e+01,  4.96549370e-02,
        7.67211759e-02, -1.47759496e-01, -2.63552624e+00, -1.06133907e+01,
        1.82639239e+01, -7.96856171e-03, -2.48460077e-01, -1.65610423e-01,
       -9.59497165e+00, -8.36512749e-01,  2.36010278e+00,  3.79257271e-01,
        9.99197068e-01,  1.41127756e+00,  1.17297458e+01, -4.32981464e+00,
        7.40401101e+00,  6.94625060e-01,  4.57226834e-02, -2.98443871e+00,
        3.34134712e+00,  2.82935803e-01,  1.45668003e+01, -6.80804948e-02,
       -5.67228819e-01,  2.88547968e-01, -1.14894620e+01,  1.69994102e+00,
        3.28692344e+00,  2.67102999e-02, -1.27192764e+00,  1.46289441e-03,
       -5.20922383e+00, -2.45358811e+00,  1.30578398e+00, -1.06493140e+00,
        2.13265822e-01,  2.60029027e+00, -1.68824830e+01,  6.89455023e+00,
        1.15901814e+01, -8.95878272e-01,  6.12391234e-01,  2.07658464e+00,
       -1.09091479e+01,  1.38673819e+01,  1.46946244e+01, -1.60969752e-01,
       -9.06123611e-01, -2.94298417e+00,  1.16075333e+00,  1.60273796e+01,
        1.18827903e+01,  6.90335841e-01,  3.94638640e-01,  1.34933522e+00,
        9.10414512e+00, -6.69531155e+00,  4.34699026e+00, -6.52507041e-01,
       -1.34994918e-01, -3.01553671e+00,  1.66947579e+00,  6.85081848e+00,
        9.78286214e+00,  3.21234004e-02, -4.41984517e-02, -6.60370727e-02,
       -3.99407691e+00, -2.76264180e+00,  4.15719060e+00,  3.46736788e-01,
        5.07381453e-01, -3.11723058e-01,  1.28913800e+01, -9.27486012e+00,
        1.49524197e+01,  4.14732064e-02,  7.34576856e-01,  2.95779932e-03,
        1.62799390e+01, -3.00104746e+00,  1.04736719e+01, -7.08114403e-02,
        9.04428936e-01,  5.65636587e-01,  1.72047391e+01,  1.75827641e+00,
        1.02059229e+01])

Sal = np.array([0.73434837, 0.79484514, 1.94896611, 0.01064089, 0.007827  ,
       0.00540884, 0.03623457, 0.04738548, 0.07399149, 0.01152841,
       0.00817604, 0.00477024, 0.03336005, 0.04019474, 0.05191765,
       0.01041313, 0.01205988, 0.00968695, 0.09017543, 0.06120177,
       0.08552671, 0.01399467, 0.01140009, 0.00624284, 0.05411949,
       0.0363931 , 0.06303563, 0.00997974, 0.00737276, 0.00304918,
       0.04007265, 0.03372606, 0.03966415, 0.02800276, 0.0249353 ,
       0.01340348, 0.10421091, 0.0638665 , 0.10075593, 0.0578897 ,
       0.07776158, 0.03842367, 0.22659351, 0.16098488, 0.1522148 ,
       0.01820332, 0.01714404, 0.00770563, 0.07033052, 0.06160929,
       0.0906178 , 0.01096375, 0.0106074 , 0.00376827, 0.02925374,
       0.0328036 , 0.0532061 , 0.01264768, 0.02106561, 0.00406399,
       0.02964809, 0.03931641, 0.06214912, 0.00770163, 0.00691567,
       0.00461308, 0.03449499, 0.04312548, 0.04434128, 0.08213891,
       0.05449966, 0.02377892, 0.12394149, 0.22277477, 0.19684114,
       0.08910111, 0.09032757, 0.01132697, 0.09741995, 0.10641567,
       0.50621088, 0.04790375, 0.0651413 , 0.0292078 , 0.29515115,
       0.12894486, 0.1484297 , 0.0195152 , 0.02363329, 0.00996783,
       0.06149686, 0.04985875, 0.07990743, 0.04771264, 0.05066176,
       0.01432414, 0.10210573, 0.06544174, 0.17441317, 0.06893023,
       0.05240582, 0.03272606, 0.1217859 , 0.33839886, 0.34902672,
       0.07013438, 0.0874054 , 0.037837  , 0.25673323, 0.18909744,
       0.19009876, 0.04628884, 0.04072337, 0.01581049, 0.09900818,
       0.15728465, 0.31381459, 0.00879968, 0.00874784, 0.00429785,
       0.03016648, 0.03464576, 0.05176297, 0.02194274, 0.03070409,
       0.01388094, 0.10591455, 0.0597149 , 0.08732278, 0.03963004,
       0.04074681, 0.00884056, 0.06541606, 0.07777651, 0.13247497,
       0.01274164, 0.01389309, 0.00867428, 0.06625915, 0.06332858,
       0.05949822, 0.01046015, 0.0070262 , 0.00625293, 0.04065399,
       0.02948325, 0.03944661, 0.06160629, 0.05840838, 0.02992609,
       0.29674191, 0.11975091, 0.18906597, 0.07123335, 0.05896813,
       0.03562417, 0.23699893, 0.25788698, 0.22948624, 0.05912923,
       0.07876134, 0.0281562 , 0.09335165, 0.22654509, 0.21637966,
       0.02753915, 0.02783914, 0.01160911, 0.10604998, 0.05368515,
       0.07781311, 0.02616466, 0.04122096, 0.00716101, 0.05923833,
       0.07215889, 0.14929459, 0.01045039, 0.00690386, 0.00258632,
       0.0303048 , 0.02859566, 0.04636717, 0.04666154, 0.04400014,
       0.02046472, 0.26745975, 0.15747167, 0.27465553, 0.02591351,
       0.03357364, 0.01608692, 0.3493999 , 0.10970748, 0.14581197,
       0.0325685 , 0.04981978, 0.02198706, 0.3115405 , 0.11843307,
       0.16055191])

#Sal *= 4

print('defined harcoded initial conditions')


# %% calculate estimated radius step

ModeR, ExpectedR, DesvEstR = radiusStepsNdim(NfreeParams)

print("moda, la media y la desv est del radio\n", ModeR, ExpectedR, DesvEstR)


# %% metropolis desde MAP. a ver si zafo de la proposal dist
'''
http://docs.pymc.io/api/inference.html
'''

nDraws = 6000
nTune = 0
nTuneInter = 0
tuneBool = nTune != 0
tune_thr = False

nChains = 500
nCores = 1

tallyBool = False
convChecksBool = False
rndSeedBool = True

# escalas características del tipical set
#scaIn = 1 / radiusStepsNdim(NintrParams)[1] / 5
#scaEx = 1 / radiusStepsNdim(NextrParams)[1] / 2
# escala para compensar el radio del set tipico
scaNdim = 1 / radiusStepsNdim(NfreeParams)[1]

# lamb = 2.38 / np.sqrt(2 * Sal.size)
# scaIn = scaEx = 1 / radiusStepsNdim(NfreeParams)[1]

# determino la escala y el lambda para las propuestas
scaAl = scaNdim / np.sqrt(3)


# %%
if rndSeedBool:
#    InSeed = pm.MvNormal.dist(mu=inMean, cov=inCov).random(size=nChains)
#    exSeed = np.random.randn(nIm, 6, nChains) * Sex.reshape((nIm, 6, 1))
#    exSeed += exMean.reshape((nIm, 6, 1))
    alSeed = np.random.randn(nChains, NfreeParams) * Sal# .reshape((-1, 1))
    alSeed += alMean# .reshape((-1, 1))

    if nChains is not 1:
#        start = [dict({'xIn': InSeed[i], 'xEx': exSeed[:,:,i]})
#        for i in range(nChains)]
        start = [dict({'xAl': alSeed[i]}) for i in range(nChains)]

    else:
        start = dict({'xAl': alSeed})
else:
    start = dict({'xAl': alMean})


#zIn = np.zeros_like(inMean)
#def propIn(S, n=1):
#    return pm.MvNormal.dist(mu=zIn, cov=S).random(size=n)
#
#def propEx(S, n=1):
#    return  np.random.randn(n, nIm, 6) * Sex.reshape((nIm, 6))

#propIn = pm.step_methods.MultivariateNormalProposal  # (Sin)
#propEx = pm.step_methods.MultivariateNormalProposal  # (np.diag(Sex**2))
propAl = pm.step_methods.MultivariateNormalProposal # (np.diag(Sex**2))
#propAl = sc.stats.norm(scale=Sal).rvs

#print("starting metropolis",
#      "|nnDraws =", nDraws, " nChains = ", nChains,
#      "|nscaIn = ", scaIn, "scaEx = ", scaEx)

'''
aca documentacion copada
https://pymc-devs.github.io/pymc/modelfitting.html

https://github.com/pymc-devs/pymc3/blob/75f64e9517a059ce678c6b4d45b7c64d77242ab6/pymc3/step_methods/metropolis.py

'''

simulBool = False
    if simulBool:
    with projectionModel:
    #    stepInt = pm.DEMetropolis(vars=[xIn], S=Sin, tune=tuneBool,
    #                            tune_interval=nTuneInter, tune_throughout=tune_thr,
    #                            tally=tallyBool, scaling=scaIn, proposal_dist=propIn)
    #    stepInt.scaling = scaIn
    #
    #    stepExt = pm.DEMetropolis(vars=[xEx], S=exCov, tune=tuneBool,
    #                              tune_interval=nTuneInter, tune_throughout=tune_thr,
    #                              tally=tallyBool, scaling=scaEx, proposal_dist=propEx)
    #    stepExt.scaling = scaEx
    #
    #    step = [stepInt, stepExt]

        step = pm.DEMetropolis(vars=[xAl], S=np.diag(Sal**2), tune=tuneBool,
                                  tune_interval=nTuneInter, tune_throughout=tune_thr,
                                  tally=tallyBool, scaling=scaAl, proposal_dist=propAl)
        step.tune = tuneBool
        step.lamb = scaAl
        step.scaling = scaAl

        trace = pm.sample(draws=nDraws, step=step, njobs=nChains, start=start,
                          tune=nTune, chains=nChains, progressbar=True,
                          discard_tuned_samples=False, cores=nCores,
                          compute_convergence_checks=convChecksBool,
                          parallelize=True)


saveBool = True
if saveBool:
    pathFiles = "/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/"
    pathFiles += "extraDataSebaPhD/traces" + str(33)

#    np.save(pathFiles + "Int", trace['xIn'])
#    np.save(pathFiles + "Ext", trace['xEx'])
    np.save(pathFiles + "All", trace['xAl'])

    print("saved data to")
    print(pathFiles)

loadBool = True
if loadBool:
    pathFiles = "/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/"
    pathFiles += "extraDataSebaPhD/traces" + str(33)

    trace = dict()

    trace['xAl'] = np.load(pathFiles + "All.npy")


#corner.corner(trace['xIn'])
##corner.corner(trace['xEx'][:,2])
#corner.corner(trace['xAl'][:,:3])

#concDat = np.concatenate([trace['xIn'], trace['xEx'].reshape((-1, nIm*6))], axis=1)

traceArray = trace['xAl'].reshape((nChains, -1, NfreeParams)).transpose((2, 1, 0))
# queda con los indices (parametro, iteracion, cadena)

plt.figure()
plt.plot(traceArray[3], alpha=0.2)
plt.xlabel('iteración')
plt.ylabel('parámetro k')
plt.title('%d trazas de DEMetropolis'% traceArray.shape[2])

difAll = np.diff(traceArray, axis=1)

repeats = np.zeros_like(traceArray)
repeats[:, 1:] = difAll == 0
repsRate = repeats.sum() / np.prod(repeats.shape)

print("tasa de repetidos", repsRate)
#print("tasa de repetidos Intrinsico",
#      repeats[:,:NintrParams].sum() / (repeats.shape[0] * NintrParams))
#print("tasa de repetidos extrinseco",
#      repeats[:,NintrParams:].sum() / (repeats.shape[0] * nIm * 6))
repTrace = repeats.sum(axis=(0, 1))

# como veo que hay una traza que es constante, la elimino
plt.hist(repTrace, 30)
indRepsMax = np.argwhere(repTrace > np.prod(traceArray.shape[:2]) * 0.98).reshape(-1)[0]

traceArray = np.delete(traceArray, indRepsMax, axis=2)

indexCut = 2000
traceCut = traceArray[:, indexCut:]
del traceArray
del repeats
del difAll

_, nDraws, nTrac = traceCut.shape

# % proyecto sobre los dos autovectores ppales
traceMean = np.mean(traceCut, axis=(1, 2))
traceDif = traceCut - traceMean.reshape((-1, 1, 1))

#traceCov = np.sum(traceDif.reshape(NfreeParams, -1, 1) *
#                  traceDif.T.reshape(1, -1, NfreeParams), axis=1)

U, S, Vh = np.linalg.svd(traceDif.reshape((NfreeParams, -1)).T,
                        full_matrices=False)

traceCov = np.cov(traceDif.reshape((NfreeParams, -1)))
traceSig = np.sqrt(np.diag(traceCov))
traceCrr = traceCov / (traceSig.reshape((-1, 1)) * traceSig.reshape((1, -1)))

plt.matshow(traceCrr, vmin=-1, vmax=1, cmap='coolwarm')

plt.figure()
plt.plot(np.abs(traceMean), traceSig, '+')

print(traceMean[:3], traceCov[:3,:3])

versors = Vh[:2].T / S[:2]
ppalXY = traceDif.T.dot(versors)
ppalX, ppalY = ppalXY.T


#nChains = 8
#nDraws = 100000
#nTune = 0

plt.figure()
plt.plot(ppalX, ppalY, linewidth=0.5)
plt.axis('equal')

plt.figure()
plt.plot(ppalX, linewidth=0.5)
plt.plot(ppalX[:,::50], linewidth=2)


plt.figure()
plt.plot(ppalY, linewidth=0.5)
plt.plot(ppalY[:,::50], linewidth=2)


# corner de los intirnsecos
corner.corner(traceCut[:3].T.reshape((-1,3)))

# %%



#plt.figure()
#plt.plot((trace['xIn'][:, 2] - trace['xIn'][-1, 2]).reshape((nChains,
#         nDraws + nTune)).T)


# %%

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

# 16: datos con la corrección. todavia no converge. 50mil en 6 procesos
# 17: usnado las desv est de 16, mil en 6 procesos
# 18: tentaiva de definitivo, no adaptivo
# 19: sets posteado a PyMC discourse
# 20: sacando los factores 2 y 5. no da muy bonito
# https://discourse.pymc.io/t/metropolis-ideal-balance-between-slow-mixing-and-high-rejection-rate/1205
# 21: con lo que salio de la corrida 19 que parecia buena
# 22: corrida con DEMetropolis
# 24: DEMet pero dejando que tunee todo lo que quiera
# 25: tune y 16 cadenas
# 26: reduzco la escala de intrinseco (x10) para que no haya tantos repetidos

# 27: metropolis adaptativo
# 28: DEmetropolis adaptativo

# 27: DEmet bien largo
# 28: otro de igual longitud
# 29: pruebas para ver si cambio la acntidad de cadenas. UNI LOS xIn Y xEx
# PARA QUE SEA TODA UNA SOLA VARIABLE. MUCHO MEJOR!!
# 30: un DEM largo ahora que las variables estan unificadas y parece que
# converge a algo.
# 31: ya con la Sal en su valor masomenos de convergencia! muchas cadenas.
# 33: un DEM largo con mi invento del lambda y el escaleo de la propuesta