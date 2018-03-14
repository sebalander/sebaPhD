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


# para tres dimensiones
ModeR3, ExpectedR3, DesvEstR3 = radiusStepsNdim(3)


print("moda, la media y la desv est del radio\n", ModeR3, ExpectedR3, DesvEstR3)




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
imageFile += "2016-11-13 medicion/vcaSnapShot.pngrVtrace"

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
tV0 = versors0.dot(- x0cam)

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

calcula la posicion normalizada tipo mahalanobis
'''


class ProjectionT(theano.Op):
    # itypes and otypes attributes are
    # compulsory if make_node method is not defined.
    # They're the type of input and output respectively
    # xInt, xExternal
    itypes = [T.dvector, T.dvector]
    # xm, ym, cM
    otypes = [T.dmatrix]  # , T.dtensor4]

    # Python implementation:
    def perform(self, node, inputs_storage, output_storage):
        # print('IDX %d projection %d, global %d' %
        #       (self.idx, self.count, projCount))
        rVec, tVecMap = inputs_storage

        tVec = cv2.Rodrigues(rVec)[0].dot(-tVecMap)
#        print(inputs_storage, Xext.shape)
#        xyM, cM = output_storage

        # saco los parametros de flat para que los use la func de projection
#        rVec, tVec = bl.flat2ext(Xext)

        xm, ym, cm = cl.inverse(imagePoints, rVec, tVec, cameraMatrix,
                                distCoeffs, model, Cccd=Cccd, Cf=Cf, Ck=Ck,
                                Crt=False, Cfk=Cfk)

        xy = mahalanobosDiff(xm, ym, cm, objM)
        output_storage[0][0] = xy
#        print(output_storage, xy.shape)


    # optional:
    check_input = True


print('projection defined for theano')

# %% pruebo si esta bien como OP

rVop = T.dvector('rVop')
tVop = T.dvector('tVop')
projTheanoWrap = ProjectionT()
projTfunction = theano.function([rVop, tVop], projTheanoWrap(rVop, tVop))

out = projTfunction(rV0, x0cam)

plt.figure()
plt.scatter(out[:, 0], out[:, 1])


# %%

projectionModel = pm.Model()

projTheanoWrap = ProjectionT()

rVinterval = np.array([(np.pi / 3)**2] * 3) # 60 grados para cada lado
rVlow = rV0 - rVinterval
rVupp = rV0 + rVinterval
# le pongo una std de 4m en horizontal y 1m en vertical
tVmapCov = np.diag([16, 16, 1])


observedNormed = np.zeros((m * 2))


with projectionModel:
    # Pri ors for unknown model parameters
    rV = pm.Uniform('rV', lower=rVlow, upper=rVupp, transform=None, shape=(3,))
    tVmap =pm.MvNormal('tVmap', mu=x0cam, cov=tVmapCov, transform=None, shape=(3,))

    # apply numpy based function
    xyMNor = projTheanoWrap(rV, tVmap)

    mu = T.reshape(xyMNor, (-1, ))
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=1, observed=observedNormed)

print('model defined')

#
## %% busco el MAP para tener un mejr punto de partida # no funcoooo
#
#with projectionModel:
#    map_estimate = pm.find_MAP(progressbar=True, maxeval=1000)



# %%
#rVmean = rV0
#rVcov = np.diag(rVinterval**2)
#
#tVmapMean = x0cam
#tVmapCov = tVmapCov


rVmean = np.array([-3.01249448, -1.33999947,  0.19189521])

rVcov = np.array(
      [[ 8.65423040e-06, -4.79061847e-06,  3.61370999e-06],
       [-4.79061847e-06,  5.07453890e-06, -1.35143104e-06],
       [ 3.61370999e-06, -1.35143104e-06,  1.39671893e-05]])

tVmapMean = np.array([-0.83660535,  0.66920183, 15.72752568])

tVmapCov = np.array(
      [[0.00166999, 0.00124925, 0.00038034],
       [0.00124925, 0.00319138, 0.0014012 ],
       [0.00038034, 0.0014012 , 0.00345138]])



print('defined harcoded initial conditions')


# %% metropolis desde MAP. a ver si zafo de la proposal dist
'''
http://docs.pymc.io/api/inference.html
'''
nDraws = 125000
nTune = 0
tuneBool = nTune != 0
nChains = 8

# escalas características del tipical set
scaRv = 1 / ExpectedR3 / 10
scaTv = 1 / ExpectedR3 / 10

rVseed = pm.MvNormal.dist(mu=rVmean, cov=rVcov).random(size=nChains)
tVseed = pm.MvNormal.dist(mu=tVmapMean, cov=tVmapCov).random(size=nChains)

start = [dict({'rV': rVseed[i], 'tVmap': tVseed[i]}) for i in range(nChains)]

print("starting metropolis",
      "|nnDraws =", nDraws, " nChains = ", nChains,
      "|nscaIn = ", scaRv, "scaEx = ", scaTv)


# %%
with projectionModel:
    stepRv = pm.Metropolis(vars=[rV], S=rVcov, tune=tuneBool)
    stepRv.scaling = scaRv

    stepTv = pm.Metropolis(vars=[tVmap], S=tVmapCov, tune=tuneBool)
    stepTv.scaling = scaTv

    step = [stepRv, stepTv]

    trace = pm.sample(draws=nDraws, step=step, njobs=nChains, start=start,
                      tune=nTune, chains=nChains, progressbar=True,
                      discard_tuned_samples=False,
                      compute_convergence_checks=False)

rVtrace = trace['rV'].reshape((nChains, nDraws, 3)).transpose((1,0,2))
tVtrace = trace['tVmap'].reshape((nChains, nDraws, 3)).transpose((1,0,2))

concatenatedTrace = np.concatenate([trace['rV'], trace['tVmap']], axis=1)

repsRv = np.diff(rVtrace, axis=0) == 0
repsTv = np.diff(tVtrace, axis=0) == 0

repsRvRate = np.sum(repsRv) / np.prod(repsRv.shape)
repsTvRate = np.sum(repsTv) / np.prod(repsTv.shape)

print("las repeticiones: intrin=", repsRvRate, " y extrin=", repsTvRate)

# %%
corner.corner(concatenatedTrace)

fig1 = plt.figure()
ax1 = fig1.gca()
fig2 = plt.figure()
ax2 = fig2.gca()

for i in range(3):
    ax1.plot(rVtrace[:,:,i])

    ax2.plot(tVtrace[:,:,i])

### ya no es necesario descartar cadenas
## %% descarto las cadenas que tienen pocos samples y saco el promedio con eso
#rVrepsPchain = repsRv.sum(0)
#tVrepsPchain = repsTv.sum(0)
#
#rVgoodChains = ~ np.all(rVrepsPchain == np.max(rVrepsPchain), axis=1)
#tVgoodChains = ~ np.all(tVrepsPchain == np.max(tVrepsPchain), axis=1)
#
#
#rVgoodChains = rVtrace[:,rVgoodChains]
#tVgoodChains = tVtrace[:,tVgoodChains]
#
#
#rVmean = np.mean(rVgoodChains, axis=(0,1))
#rVcov = np.cov(rVgoodChains.reshape((-1,3)).T)
#
#tVmapMean = np.mean(tVgoodChains, axis=(0,1))
#tVmapCov = np.cov(tVgoodChains.reshape((-1,3)).T)

# %%
pathFiles = "/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/"
pathFiles += "extraDataSebaPhD/tracesExtrinsicRTvec"

saveBool = False
loadBool = True

if saveBool:
    np.save(pathFiles, concatenatedTrace)

    print("saved data to")
    print(pathFiles)

if loadBool:
    concatenatedTrace = np.load(pathFiles + ".npy")
    print('loaded data')


# %%
rVtrace = concatenatedTrace[:,:3]#.reshape((nChains,-1,3)).transpose((1,0,2))
tVmapTrace = concatenatedTrace[:,3:]#.reshape((nChains,-1,3)).transpose((1,0,2))

# convierto el vector de tV a opencv
tVtrace = np.zeros_like(tVmapTrace)
for i in range(rVtrace.shape[0]):
    rot = -cv2.Rodrigues(rVtrace[i])[0]
    tVtrace[i] = rot.dot(tVmapTrace[i])

concatenatedTraceRTocv = np.concatenate([rVtrace,tVtrace], axis=1)

rtVmean = np.mean(concatenatedTraceRTocv, axis=0)
rtCov = np.cov(concatenatedTraceRTocv.T)

tVmapMean = np.mean(tVtrace, axis=0)
tVmapCov = np.cov(tVtrace.T)

# %% test figures here
corner.corner(concatenatedTraceRTocv)

xm, ym, cm = cl.inverse(imagePoints, rtVmean[:3], rtVmean[3:], cameraMatrix,
                        distCoeffs, model, Cccd=Cccd, Cf=Cf, Ck=Ck, Crt=rtCov,
                        Cfk=Cfk)


from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes


# %%
fig = plt.figure()
ax = fig.gca()

## tratr de que quede girada la figura
## https://stackoverflow.com/questions/21652631/how-to-rotate-a-simple-matplotlib-axes#
#plot_extents = -400, 150, -40, 150
#transform = Affine2D().rotate_deg(-60)
#helper = floating_axes.GridHelperCurveLinear(transform, plot_extents)
#ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=helper)
#fig.add_subplot(ax)


ax.scatter(objM[:,0],objM[:,1])
ax.scatter(tVmapMean[0], tVmapMean[1]) # posicion de la camara

cl.plotPointsUncert(ax, cm, xm, ym, 'k')
cl.plotPointsUncert(ax, [tVmapCov[:2,:2]], [tVmapMean[0]], [tVmapMean[1]], 'b')
ax.set_aspect('equal')
plt.show()








