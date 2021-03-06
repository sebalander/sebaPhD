#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 2018

hacer calib extrisneca con pymc3

@author: sebalander
"""

# %%
# import glob
import os
import corner
import time
import seaborn as sns
import scipy as sc
import scipy.stats as sts
import matplotlib.pyplot as plt
from copy import deepcopy as dc
import numpy as np
from importlib import reload
from glob import glob

# %env THEANO_FLAGS='device=cuda, floatX=float32'
import theano
import theano.tensor as T
import pymc3 as pm
import cv2
import scipy.optimize as opt

import sys
sys.path.append("/home/sebalander/Code/sebaPhD")
from calibration import calibrator as cl
from dev import bayesLib as bl

import pickle
from calibration.calibrator import datafull, real, realdete, realbalk, realches
from calibration.calibrator import synt, syntextr, syntches, syntintr
import numdifftools as ndft

from time import sleep
print('libraries imported')

# compute_test_value is 'off' by default, meaning this feature is inactive
theano.config.compute_test_value = 'off' # Use 'warn' to activate this feature

# %%
def radiusStepsNdim(n):
    '''
    retorna moda, la media y la desv est del radio  de pasos al samplear
    de gaussianas hiperdimensionales de sigma 1
    '''
    # https://www.wolframalpha.com/input/?i=integrate+x%5E(n-1)+exp(-x%5E2%2F2)+from+0+to+infinity
    # integral_0^∞ x^(n - 1) exp(-x^2/2) dx = 2^(n/2 - 1) Γ(n/2) for Re(n)>0
    Inorm = 2**(n / 2 - 1) * sc.special.gamma(n / 2)

    # https://www.wolframalpha.com/input/?i=integrate+x%5En+exp(-x%5E2%2F2)+from+0+to+infinity
    # integral_0^∞ x^n exp(-x^2/2) dx = 2^((n - 1)/2) Γ((n + 1)/2) for Re(n)>-1
    ExpectedR = 2**((n - 1) / 2) * sc.special.gamma((n + 1) / 2)

    # https://www.wolframalpha.com/input/?i=integrate+x%5E(n%2B1)+exp(-x%5E2%2F2)+from+0+to+infinity
    # integral_0^∞ x^(n + 1) exp(-x^2/2) dx = 2^(n/2) Γ(n/2 + 1) for Re(n)>-2
    ExpectedR2 = 2**(n / 2) * sc.special.gamma(n / 2 + 1)

    ModeR = np.sqrt(n - 1)

    # normalizo las integrales:
    ExpectedR /= Inorm
    ExpectedR2 /= Inorm

    DesvEstR = np.sqrt(ExpectedR2 - ExpectedR**2)

    return np.array([ModeR, ExpectedR, DesvEstR])



def extractCaseData(exCase):
    objpoints = fullData.Synt.Extr.objPt
    imagePoints = fullData.Synt.Extr.imgPt[exCase[0], exCase[1]]
    imagePoints += stdPix * fullData.Synt.Extr.imgNse[exCase[0], exCase[1]]
    if not exCase[2]:
        objpoints = objpoints[fullData.Synt.Extr.index10]
        imagePoints = imagePoints[fullData.Synt.Extr.index10]
    xi, yi = imagePoints.T  # lo dejo listo para el modelo theano

    # load true rotation traslation values
    rVecsT = fullData.Synt.Extr.rVecs[exCase[0]]
    tVecsT = fullData.Synt.Extr.tVecs[exCase[0], exCase[1]]

    # select wich points are in the camera FOV, con la coordenada z
    xCam = (cv2.Rodrigues(rVecsT)[0][2, :2].dot(objpoints.T)).T + tVecsT[2]
    inFOV = xCam > 0

    return imagePoints[inFOV], objpoints[inFOV], rVecsT, tVecsT






# %%
pi2 = 2 * np.pi
pi2sq = np.sqrt(pi2)


def prob1vs0(t, x, adaptPrior=True):
    '''
    calculo el logaritmo del cociente de las probabilidades de que una serie
    de datos siga
    un modelo cnstante o lineal, si < 1 es que es mas probable el modelo
    constante
    '''
    m0, c0 = np.polyfit(t, x, 0, cov=True)
    m1, c1 = np.polyfit(t, x, 1, cov=True)

    dif0 = x - m0[0]
    var0 = np.mean((dif0)**2)
    if np.isclose(var0, 0):  # si varianza da cero es que sampleo mal
        return 1  # devuelve para que se considere que no convergió

    dif1 = x - m1[0] * t - m1[1]
    var1 = np.mean((dif1)**2)
    if np.allclose(var1, 0):
        return 1

    # defino los priors
    if adaptPrior:
        pConst0 = 1 / (np.max(dif0) - np.min(dif0))  # prior de la constante
        deltaDif1 = np.max(dif1) - np.min(dif1)
        pConst1 = 1 / deltaDif1
        penDelta = deltaDif1 / (t[-1] - t[0])
        pPendi1 = 1 / penDelta / 2  # prior de la pendiente

        pWgH0 = np.log(pConst0)
        pWgH1 = np.log(pConst1 * pPendi1)
    else:
        pWgH0 = 1.0
        pWgH1 = 1.0

    pDagWH0 = sc.stats.multivariate_normal.logpdf(dif0, cov=var0)
    pDagWH1 = sc.stats.multivariate_normal.logpdf(dif1, cov=var1)

    deltaW0 = np.log(pi2sq * np.sqrt(c0)[0, 0])
    deltaW1 = np.log(pi2 * np.sqrt(np.linalg.det(c1)))

    prob1_0 = np.sum(pDagWH1 - pDagWH0)
    prob1_0 += pWgH1 * deltaW1 - pWgH0 - deltaW0

    return prob1_0


def funcExp(x, a, b, c):
    return a * np.exp(- x / np.abs(b)) + c


# %%

def logPerror(xAll, case):
    rvec = xAll[:3]
    tvec = xAll[3:]

    Eint = cl.errorCuadraticoImagen(case.imagePoints, case.objpoints,
           rvec, tvec, case.cameraMatrix, case.distCoeffs, case.model,
           case.Ci, case.Cf, case.Ck, case.Crt, case.Cfk)
    return Eint

def objective(xAll, case):
    return np.sum(logPerror(xAll, case))

hessNum = ndft.Hessian(objective)

def optimizar(xAll, case):
    '''
    guarda en el objeto la posicion optimizada y una covarianza calculada como
    derivada numerica
    '''
    ret = opt.minimize(objective, xAll, args=case)
    case.xAllOpt = ret.x
    case.covOpt = np.linalg.inv(hessNum(case.xAllOpt, case))


class casoCalibExtr:
    def __init__(self, fullData, intrCalibResults, case, stdPix, allDelta,
                 nCores, nTune, nTuneInter, tuneBool, tune_thr, tallyBool,
                 convChecksBool, rndSeedBool, scaNdim, scaAl, nDraws, nChains,
                 indexSave, pathFiles):
        self.case = case
        self.nCores = nCores
        self.nTune = nTune
        self.nTuneInter = nTuneInter
        self.tuneBool = tuneBool
        self.tune_thr = tune_thr
        self.tallyBool = tallyBool
        self.convChecksBool = convChecksBool
        self.rndSeedBool = rndSeedBool
        self.scaNdim = scaNdim
        self.scaAl = scaAl
        self.nDraws = nDraws
        self.nChains = nChains
        self.indexSave = indexSave
        self.pathFiles = pathFiles
        self.allDelta = allDelta

        self.camera = fullData.Synt.Intr.camera
        self.model = fullData.Synt.Intr.model
        self.imgSize = fullData.Synt.Intr.s
        Ns = [2, 3]
        Xint = intrCalibResults['mean'][:3]
        self.cameraMatrix, self.distCoeffs = bl.flat2int(Xint, Ns, self.model)

        caseData = extractCaseData(exCase)
        self.imagePoints = caseData[0]
        self.xi, self.yi = self.imagePoints.T
        self.objpoints = caseData[1]
        self.rVecsT = caseData[2]
        self.tVecsT = caseData[3]
        self.xAllT = np.concatenate([self.rVecsT, self.tVecsT])
        self.nPt = self.objpoints.shape[0]
        self.nFree = 6
        self.observedNormed = np.zeros((self.nPt * 2))

        self.Crt = False  # no RT error
        self.Cf = np.zeros((4, 4))
        self.Cf[2:, 2:] = intrCalibResults['cov'][:2, :2]
        self.Ck = intrCalibResults['cov'][2, 2]
        self.Cfk = np.zeros(4)
        self.Cfk[2:] = intrCalibResults['cov'][:2, 2]
        self.Ci = np.array([stdPix**2 * np.eye(2)] * self.nPt)

        # tau de 1/5 pra la ventana movil
        self.weiEsp = np.exp(- np.arange(nDraws) * 5 / nDraws)[::-1]
        self.weiEsp /= np.sum(self.weiEsp)




# %%
'''
funcion arbitraria para theano

https://docs.pymc.io/notebooks/getting_started.html

https://github.com/pymc-devs/pymc3/blob/master/pymc3/examples/disaster_model_theano_op.py
'''

from theano.compile.ops import as_op

@as_op(itypes=[T.dvector], otypes=[T.dvector])
def project2diagonalisedError(xAll):
    '''
    no me queda otra que leer case desde el main como una variable global
    '''
    c = case
    xm, ym, Cm = cl.inverse(c.xi, c.yi, xAll[:3], xAll[3:], c.cameraMatrix,
                            c.distCoeffs, c.model, c.Ci, c.Cf, c.Ck, c.covOpt, c.Cfk)

    xNorm, yNorm = cl.points2linearised(xm - c.objpoints[:, 0],
                                        ym - c.objpoints[:, 1], Cm).T

    return np.concatenate([xNorm, yNorm])



#class project2diagonalisedError(theano.Op):
#    # Properties attribute
#    __props__ = ("xi", "yi", "cameraMatrix", "distCoeffs", "model", "Ci", "Cf", "Ck", "covOpt", "Cfk", "objpoints")
#
#    #itypes and otypes attributes are
#    #compulsory if make_node method is not defined.
#    #They're the type of input and output respectively
#    itypes = [T.dvector]
#    otypes = [T.dvector]
#
#    def __init__(self, case):
#        self.xi, self.yi, self.cameraMatrix, self.distCoeffs, self.model, self.Ci, self.Cf, self.Ck, self.covOpt, self.Cfk, self.objpoints = [case.xi, case.yi, case.cameraMatrix, case.distCoeffs, case.model, case.Ci, case.Cf, case.Ck, case.covOpt, case.Cfk, case.objpoints]
#
#
#    # Python implementation:
#    def perform(self, node, inputs_storage, output_storage):
#        xAll = inputs_storage[0][0]
#        xm, ym, Cm = cl.inverse(self.xi, self.yi, xAll[:3], xAll[3:], self.cameraMatrix,
#                            self.distCoeffs, self.model, self.Ci, self.Cf, self.Ck, self.covOpt, self.Cfk)
#
#        xNorm, yNorm = cl.points2linearised(xm - self.objpoints[:, 0],
#                                            ym - self.objpoints[:, 1], Cm).T
#
#        output_storage[0] = np.float64(np.concatenate([xNorm, yNorm]))




def getTrace(alMean, Sal, case):
    # prior bounds
    allLow = case.xAllT - case.allDelta
    allUpp = case.xAllT + case.allDelta

    alSeed = np.random.randn(case.nChains, case.nFree) * Sal  # .reshape((-1, 1))
    alSeed += alMean  # .reshape((-1, 1))
    start = [dict({'xAl': alSeed[i]}) for i in range(nChains)]

    projectionModel = pm.Model()
    with projectionModel:
        # Priors for unknown model parameters
        xAl = pm.Uniform('xAl', lower=allLow, upper=allUpp, shape=allLow.shape,
                         transform=None)
        xAl.tag.test_value= case.xAllT
#
#        proj = project2diagonalisedError(case)
#        x = theano.tensor.vector()
#        x.tag.test_value= case.xAllT
#        xyMNor = project2diagonalisedError(xAl, theano.shared(case))

#        f = theano.function([xAl], project2diagonalisedError(case)(xAl))
#        xyMNor = f(xAl)

        xyMNor = project2diagonalisedError(xAl)

        Y_obs = pm.Normal('Y_obs', mu=xyMNor, sd=1, observed=case.observedNormed)

        step = pm.DEMetropolis(vars=[xAl], S=Sal, tune=tuneBool,
                               tune_interval=nTuneInter, tune_throughout=tune_thr,
                               tally=tallyBool, scaling=scaAl)
        step.tune = tuneBool
        step.lamb = scaAl
        step.scaling = scaAl

        trace = pm.sample(draws=nDraws, step=step, njobs=nChains,
                          start=start,
                          tune=nTune, chains=nChains, progressbar=True,
                          discard_tuned_samples=False, cores=nCores,
                          compute_convergence_checks=convChecksBool,
                          parallelize=True)

    return trace


#lor = np.array([-100]*6)
#upr = - lor
#xAl = pm.Uniform.dist(lor, upr, shape=lor.shape)
#
#f = theano.function([xAl], project2diagonalisedError(case)(xAl))
#xyMNor = f(xAl)

#getTrace(case.xAllOpt, np.sqrt(np.diag(case.covOpt)), case)
'''
ponele que anda, no upde hacer que la funcion acepte a "case" como argumento
o algo que no sea leerlo desde las variables globales del main.
incluso fallo definir como un objeto mas complicado que pudiera inicializarse
guardando los parametros que necesito
'''


# %%


def getStationaryTrace(exCase):
    imagePoints, objpoints, rVecsT, tVecsT = extractCaseData(exCase)
    xi, yi = imagePoints.T
    nPt = objpoints.shape[0]  # cantidad de puntos
    Ci = np.array([stdPix**2 * np.eye(2)] * nPt)
    nFree = 6  # nro de parametros libres

    # pongo en forma flat los valores iniciales
    xAllT = np.concatenate([rVecsT, tVecsT])

    # pongo en forma flat los valores iniciales
    xAllT = np.concatenate([rVecsT, tVecsT])
    print('data loaded and formated')

    ret = opt.minimize(objective, xAllT)
    xAllOpt = ret.x
    covNum = np.linalg.inv(hessNum(xAllOpt))  # la achico por las dudas?

    print('initial optimisation and covariance estimated')

    # for proposal distr
    alMean = xAllOpt
    Sal = np.sqrt(np.diag(covNum))

    print('defined parameters')

    means = list()
    stdes = list()
    tracesList = list()

    probList = list()

    for intento in range(50):
        print("\n\n")
        print("============================")
        print('intento nro', intento, ' caso ', exCase)
        trace = getTrace(alMean, Sal)
        sleep(5)  # espero un ratito ...
        traceArray = trace['xAl'].reshape((nChains, -1, nFree))
        traceArray = traceArray.transpose((2, 1, 0))
        tracesList.append(traceArray)

        traceMean = np.mean(traceArray, axis=2)
        traceStd = np.std(traceArray, axis=2)

        means.append(traceMean)
        stdes.append(traceStd)

        probMean = np.zeros(6)
        probStd = np.zeros(6)
        for i in range(6):
            probMean[i] = prob1vs0(t, traceMean[i], adaptPrior=True)
            probStd[i] = prob1vs0(t, traceStd[i], adaptPrior=True)

        probList.append(np.array([probMean, probStd]))
        print(probList[-1].T)

        convergedBool = np.all([probMean < 0, probStd < 0])

        alMean = np.average(traceMean, axis=1, weights=weiEsp)
        Sal = np.average(traceStd, axis=1, weights=weiEsp)
        for sa in Sal:  # si alguno da cero lo regularizo
            if np.isclose(sa, 0):
                sa = 1e-6

        if intento > 0 and convergedBool:
            print("parece que convergió")
            break

    return means, stdes, probList, tracesList



# %% LOAD DATA
# input
# import collections as clt
fullDataFile = "./resources/nov16/fullDataIntrExtr.npy"
dataFile = open(fullDataFile, "rb")
fullData = pickle.load(dataFile)
dataFile.close()

intrCalibResults = np.load("./resources/nov16/syntIntrCalib.npy").all()
stdPix = 1.0
allDelta = np.concatenate([[np.deg2rad(10)] * 3, [5] * 3])


# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
#camera = fullData.Synt.Intr.camera
## modelos = ['poly', 'rational', 'fisheye', 'stereographic']
#model = fullData.Synt.Intr.model
#imgSize = fullData.Synt.Intr.s
#Ns = [2, 3]
#Xint = intrCalibResults['mean'][:3]
#cameraMatrix, distCoeffs = bl.flat2int(Xint, Ns, model)

# caso de extrinseca a analizar, indices: [angulos, alturas, 20 puntos]
exCasesList = list()
for aa in range(3):
    for hh in range(2):
        for nBo in [False, True]:
            exCasesList.append([aa, hh, nBo])

exCase = exCasesList[0]

#
## cargo los puntos de calibracion
#imagePoints, objpoints, rVecsT, tVecsT = extractCaseData(exCase)
#xi, yi = imagePoints.T
#nPt = objpoints.shape[0]  # cantidad de puntos
#nFree = 6  # nro de parametros libres


## load intrisic calibrated
## 0.1pix as image std
## https://stackoverflow.com/questions/12102318/opencv-findcornersubpix-precision
## increase to 1pix porque la posterior da demasiado rara
#Crt = False  # no RT error
#Cf = np.zeros((4, 4))
#Cf[2:, 2:] = intrCalibResults['cov'][:2, :2]
#Ck = intrCalibResults['cov'][2, 2]
#Cfk = np.zeros(4)
#Cfk[2:] = intrCalibResults['cov'][:2, 2]
#stdPix = 1.0
#Ci = np.array([stdPix**2 * np.eye(2)] * nPt)




# pongo en forma flat los valores iniciales
#xAllT = np.concatenate([rVecsT, tVecsT])

print('data loaded and formated')

# %% metropolis desde MAP. a ver si zafo de la proposal dist
'''
http://docs.pymc.io/api/inference.html

aca documentacion copada
https://pymc-devs.github.io/pymc/modelfitting.html

https://github.com/pymc-devs/pymc3/blob/75f64e9517a059ce678c6b4d45b7c64d77242ab6/pymc3/step_methods/metropolis.py

'''


nCores = 1
nTune = 0
nTuneInter = 0
tuneBool = nTune != 0
tune_thr = False
tallyBool = False
convChecksBool = False
rndSeedBool = True

# escalas características del tipical set
nFree = 6
scaNdim = 1 / radiusStepsNdim(nFree)[1]
# determino la escala y el lambda para las propuestas
scaAl = scaNdim / np.sqrt(3)

# lamb = 2.38 / np.sqrt(2 * Sal.size)
# scaIn = scaEx = 1 / radiusStepsNdim(NfreeParams)[1]
nDraws = 100
nChains = 10 * int(1.1 * nFree)
indexSave = 0

#header = "nDraws %d, nChains %d" % (nDraws, nChains)
pathFiles = "/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/"
pathFiles += "extraDataSebaPhD/tracesSynt" + str(indexSave)

# %%
case = casoCalibExtr(fullData, intrCalibResults, exCase, stdPix, allDelta,
                 nCores, nTune, nTuneInter, tuneBool, tune_thr, tallyBool,
                 convChecksBool, rndSeedBool, scaNdim, scaAl, nDraws, nChains,
                 indexSave, pathFiles)


# uso las
optimizar(case.xAllT, case)

getTrace(case.xAllOpt, np.sqrt(np.diag(case.covOpt)), case)







# %% defino la funcion a minimizar
erCuaIm = case.logPerror(case.xAllT)
erCua = objective(case.xAllT)

print(erCuaIm, np.exp(- erCuaIm / 2))

print(erCua, np.exp(- erCua / 2))

covNum = np.linalg.inv(hessNum(xAllT))

# %%

ret = opt.minimize(objective, case.xAllT)

xAllOpt = ret.x
covOpt = np.exp(ret.fun / 2) * ret.hess_inv

sigOpt = np.sqrt(np.diag(covOpt))

crrOpt = covOpt / (sigOpt.reshape((-1, 1)) * sigOpt.reshape((1, -1)))

# plt.matshow(crrOpt, vmin=-1, vmax=1, cmap='coolwarm')



xm, ym, Cm = cl.inverse(xi, yi, xAllOpt[:3], xAllOpt[3:], cameraMatrix,
                        distCoeffs, model, Ci, Cf, Ck, covOpt, Cfk)

plt.figure()
ax = plt.gca()
ax.scatter(xm, ym)
ax.scatter(objpoints[:, 0], objpoints[:, 1])
for i in range(len(xm)):
    cl.plotEllipse(ax, Cm[i], xm[i], ym[i], 'k')
ax.axis('equal')


reload(cl)

xNorm, yNorm = cl.points2linearised(xm - objpoints[:, 0],
                                  ym - objpoints[:, 1], Cm).T

plt.scatter(xNorm, yNorm)
plt.axis('equal')


plt.scatter(fullData.Synt.Extr.imgNse[exCase[0], exCase[1], :, 0],
            fullData.Synt.Extr.imgNse[exCase[0], exCase[1], :, 1])




# %%
# chequeo la funcion arbitraria que va ausar theano


xAllTheano = theano.shared(xAllT)
parameters = [xi, yi, cameraMatrix, distCoeffs, model, Ci, Cf, Ck, covOpt,
              Cfk, objpoints]

#projObj = diagonalisedError(parameters)

project2diagonalisedError(xAllTheano).eval()


# %%
# 10 grados de error de rotacion
# intervalo de 5 unidades de distancia
allDelta = np.concatenate([[np.deg2rad(10)] * 3, [5] * 3])
observedNormed = np.zeros((nPt * 2))

# calculate estimated radius step
ModeR, ExpectedR, DesvEstR = radiusStepsNdim(nFree)
print("moda, la media y la desv est del radio\n", ModeR, ExpectedR, DesvEstR)




#exCaseList = [[1, 0, 1]]


# %%
saveBool = True

#exCase = [2, 1, True]

for exCase in exCasesList[:2]:
    file = pathFiles + "ext-%d-%d-%d" % tuple(exCase)

    print("\n\n")
    print("========================================================")
    print("========================================================")
    print(pathFiles)

    means, stdes, probList, tracesList = getStationaryTrace(exCase)

    intento = -1
    for m, s, p in zip(means, stdes, probList):
        intento += 1
        plt.figure(1)
        plt.plot(t + intento * nDraws, m.T - xAllOpt)
        plt.figure(2)
        plt.plot(t + intento * nDraws, s.T)

    if saveBool:
    #    np.save(pathFiles + "Int", trace['xIn'])
    #    np.save(pathFiles + "Ext", trace['xEx'])
    #    trace['docs'] = header
        np.save(file, tracesList)

        print("saved data to")
        print(file)
        print("exiting")
#        sys.exit()

# %%
loadBool = True
if loadBool:
    filesFound = glob(pathFiles + "ext-*.npy")
    traces = dict()

    for file in filesFound:
        print(file)
        trace = dict()
        trace = np.load(file).all()

        key = file[-9:-4]
        traces[key] = trace


# %%
traceArray = trace['xAl'].reshape((nChains, -1, nFree)).transpose((2, 1, 0))
# queda con los indices (parametro, iteracion, cadena)

fig, axs = plt.subplots(2, 3)
for i, ax in enumerate(axs.flatten()):
    ax.plot(traceArray[i], alpha=0.2)

#corner.corner(trace['xAl'])

difAll = np.diff(traceArray, axis=1)

repeats = np.zeros_like(traceArray)
repeats[:, 1:] = difAll == 0
repsRate = repeats.sum() / np.prod(repeats.shape)

print("tasa de repetidos", repsRate)

# %%
indexCut = 1000
traceCut = traceArray[:, indexCut:]
del traceArray
del repeats
del difAll


# saco la desvest y hago un ajuste
stdIte = np.std(traceCut, axis=2)
itime = np.arange(stdIte.shape[1])

plt.figure()
plt.plot(stdIte.T, alpha=0.2)
plt.xlabel('iteración')
plt.ylabel('std parámetro')
plt.plot(stdIte[0], 'k')


from scipy.optimize import curve_fit

def funcExp(x, a, b, c):
    return a * np.exp(- x / b) + c

# ajuste exponencial
linFitList = list()
expFitList = list()
for i in range(nFree):
    p0 = [stdIte[i, 0], 1e3, 5 * stdIte[i, 0]]
    popt, pcov = curve_fit(funcExp, itime, stdIte[i], p0)
    expFitList.append([popt, np.sqrt(np.diag(pcov))])

    popt, pcov = np.polyfit(itime, stdIte[i], 1, cov=True)
    linFitList.append([popt, np.sqrt(np.diag(pcov))])

linFitList = np.array(linFitList)
expFitList = np.array(expFitList)

Tnext = 3e4
stdNext = Tnext * linFitList[:, 0, 0] + linFitList[:, 0, 1]

plt.figure()
plt.plot(Sal, stdNext, '+')

_, nDraws, nTrac = traceCut.shape

# % proyecto sobre los dos autovectores ppales
traceMean = np.mean(traceCut, axis=(1, 2))
traceDif = traceCut - traceMean.reshape((-1, 1, 1))

U, S, Vh = np.linalg.svd(traceDif.reshape((nFree, -1)).T,
                        full_matrices=False)

traceCov = np.cov(traceDif.reshape((nFree, -1)))
traceSig = np.sqrt(np.diag(traceCov))
traceCrr = traceCov / (traceSig.reshape((-1, 1)) * traceSig.reshape((1, -1)))

saveIntrBool = False
if saveIntrBool:
    np.save("/home/sebalander/Code/sebaPhD/resources/nov16/syntIntrCalib.npy",
            {'mean': traceMean,
             'cov': traceCov,
             "camera": camera,
             "model": model,
             "trueVals": xAllT,
             "datafile": fullDataFile})


plt.matshow(traceCrr, vmin=-1, vmax=1, cmap='coolwarm')

cols = np.array([[1] * 3, [2] * 3]).reshape(-1)
cols = np.concatenate([[0] * 3, cols])

plt.figure()
plt.scatter(np.abs(traceMean), traceSig)

print(traceMean[:3], traceCov[:3,:3])

versors = Vh[:2].T / S[:2]
ppalXY = traceDif.T.dot(versors)
ppalX, ppalY = ppalXY.T


plt.figure()
plt.plot(ppalX, ppalY, linewidth=0.5)
plt.axis('equal')

plt.figure()
plt.plot(ppalX, linewidth=0.5)
plt.plot(ppalX[:,::50], linewidth=2)


plt.figure()
plt.plot(ppalY, linewidth=0.5)
plt.plot(ppalY[:,::50], linewidth=2)



pathFiles = "/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/"
pathFiles += "extraDataSebaPhD/tracesSynt" + str(0)

trace = dict()

trace['xAl'] = np.load(pathFiles + "All.npy")


