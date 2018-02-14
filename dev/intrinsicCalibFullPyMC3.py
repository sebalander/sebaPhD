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
#import glob
import os
import numpy as np
#import scipy.linalg as ln
import scipy.stats as sts
import matplotlib.pyplot as plt
#from importlib import reload
from copy import deepcopy as dc
#import numdifftools as ndf
from calibration import calibrator as cl
import corner
import time
import theano
import theano. tensor as T
import pymc3 as pm

import scipy.sparse as sp
import scipy as sc

import sys
sys.path.append("/home/sebalander/Code/sebaPhD")
#from dev import multipolyfit as mpf
from dev import bayesLib as bl
#from multiprocess import Process, Queue, Value
# https://github.com/uqfoundation/multiprocess/tree/master/py3.6/examples

# %% LOAD DATA
# input
plotCorners = False
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
modelos = ['poly', 'rational', 'fisheye']
model = modelos[2]

imagesFolder = "./resources/intrinsicCalib/" + camera + "/"
cornersFile =      imagesFolder + camera + "Corners.npy"
patternFile =      imagesFolder + camera + "ChessPattern.npy"
imgShapeFile =     imagesFolder + camera + "Shape.npy"

# model data files
distCoeffsFile =   imagesFolder + camera + model + "DistCoeffs.npy"
linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
tVecsFile =        imagesFolder + camera + model + "Tvecs.npy"
rVecsFile =        imagesFolder + camera + model + "Rvecs.npy"

imageSelection = np.arange(0,33) # selecciono con que imagenes trabajar
n = len(imageSelection)  # cantidad de imagenes

# ## load data
imagePoints = np.load(cornersFile)[imageSelection]

chessboardModel = np.load(patternFile)
m = chessboardModel.shape[1]  # cantidad de puntos por imagen
imgSize = tuple(np.load(imgShapeFile))
# images = glob.glob(imagesFolder+'*.png')

# Parametros de entrada/salida de la calibracion
objpoints2D = np.array([chessboardModel[0,:,:2]]*n).reshape((n,m,2))

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
Ci = np.repeat([ stdPix**2 * np.eye(2)],n*m, axis=0).reshape(n,m,2,2)

#Cf = np.eye(distCoeffs.shape[0])
#Ck = np.eye(4)
#Cfk = np.eye(distCoeffs.shape[0], 4)  # rows for distortion coeffs
#
#Crt = np.eye(6) # 6x6 covariance of pose
#Crt[[0,1,2],[0,1,2]] *= np.deg2rad(1)**2 # 5 deg stdev in every angle
#Crt[[3,4,5],[3,4,5]] *= 0.01**2 # 1/100 of the unit length as std for pos
#Crt = np.repeat([Crt] , n, axis=0)
Crt = np.repeat([False], n) # no RT error

# output file
intrinsicParamsOutFile = imagesFolder + camera + model + "intrinsicParamsML"
intrinsicParamsOutFile = intrinsicParamsOutFile + str(stdPix) + ".npy"

# pruebo con un par de imagenes
for j in range(0,n,3):
    
    xm, ym, Cm = cl.inverse(imagePoints[j,0], rVecs[j], tVecs[j], cameraMatrix,
                            distCoeffs, model, Cccd=Ci[j], Cf=False, Ck=False,
                            Crt=False, Cfk=False)
    print(xm, ym, Cm)


# datos medidos, observados, experimentalmente en el mundo y en la imagen
yObs = objpoints2D.reshape(-1)


# no usar las constantes como tensores porque da error...
## diccionario de parametros, constantes de calculo
#xObsConst = T.as_tensor_variable(imagePoints.reshape((n,m,2)), 'xObsConst', ndim=3)
#CiConst = T.as_tensor_variable(Ci, 'cIConst', ndim=4)


# %%
'''
aca defino la proyeccion para que la pueda usar thean0
http://deeplearning.net/software/theano/extending/extending_theano.html
'''


class ProjectionT(theano.Op):
    #itypes and otypes attributes are
    #compulsory if make_node method is not defined.
    #They're the type of input and output respectively
    # xInt, xExternal
    itypes = [T.dvector, T.dmatrix]
    # xm, ym, cM
    otypes = [T.dtensor3, T.dtensor4]



    # Python implementation:
    def perform(self, node, inputs_storage, output_storage):
        Xint, Xext = inputs_storage
        xyM, cM = output_storage
        
        # saco los parametros de flat para que los use la func de projection
        # print(Xint)
        cameraMatrix, distCoeffs = bl.flat2int(Xint, Ns, model)
        # print(cameraMatrix, distCoeffs)
        xy = np.zeros((n,m,2))
        cm = np.zeros((n,m,2,2))
        
        for j in range(n):
            rVec, tVec = bl.flat2ext(Xext[j])
            xy[j,:,0], xy[j,:,1], cm[j] = cl.inverse(imagePoints.reshape((n,m,2))[j], rVec, tVec,
                                    cameraMatrix, distCoeffs, model,
                                    Cccd=Ci[j])
            
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


XintOP = T.dvector('XintOP')
XextOP = T.dmatrix('XextOP')

projTheanoWrap = ProjectionT()

projTfunction = theano.function([XintOP, XextOP],
                                projTheanoWrap(XintOP, XextOP))

out = projTfunction(Xint, XextList)

print(out)

plt.scatter(out[0][:,:,0], out[0][:,:,1])


# %%
#from importlib import reload
#reload(cl)
#reload(bl)


# indexes to read diagona 2x2 blocks of a matrix
nTot = 2 * n * m
xInxs = [ [[i,i],[i+1,i+1]] for i in range(0, nTot, 2)]
yInxs = [ [[i,i+1],[i,i+1]] for i in range(0, nTot, 2)]

projectionModel = pm.Model()

projTheanoWrap = ProjectionT()

# set lower and upper bounds for uniform prior distributions
# for camera matrix is important to avoid division by zero
intrLow = [300, 300, 600,  300, -0.1, -0.1, -0.1, -0.1] # intrinsic
intrUpp = [500, 500, 1000, 600,  0.1,  0.1,  0.1,  0.1]
extrLow = np.array([[-3.2, -3.2, -3.2, -25, -25, -25]]*n)
extrUpp = np.array([[ 3.2,  3.2,  3.2,  25,  25,  25]]*n)

with projectionModel:
    # Priors for unknown model parameters
    xIn = pm.Uniform('xIn', lower=intrLow, upper=intrUpp, shape=Xint.shape)
    xEx = pm.Uniform('xEx', lower=extrLow, upper=extrUpp, shape=XextList.shape)
    
    xyM, cM = projTheanoWrap(xIn, xEx)
    
    mu = T.reshape(xyM, (-1,))
    
    # sp.block_diag(out[1].reshape((-1,2,2))) # for sparse
    bigC = T.zeros((nTot, nTot))
    c3Diag = T.reshape(cM, (-1,2,2)) # list of 2x2 covariance blocks
    bigC = T.set_subtensor(bigC[xInxs, yInxs], c3Diag)
    
    Y_obs = pm.MvNormal('Y_obs', mu=mu, cov=bigC, observed=yObs)







# %%
# aca saco el maximo a posteriori, bastante util para hacer montecarlo despues
import scipy.optimize as opt

#
#try:
#    map_estimate
#except:
#    print('set initial state arbitrarily')
#    start = {'xIn': Xint,
#             'xEx': XextList}
#else:
#    print('set initial state with previous map_estiamte')
#    start=map_estimate

start = {'xIn': Xint, 'xEx': XextList}



niter = 10
#map_estimate = pm.find_MAP(model=projectionModel, start=start, maxeval=int(niter * 10), maxiter=niter, fmin=opt.fmin, xtol=1e-2,ftol=1e-3)


map_estimate = pm.find_MAP(model=projectionModel, start=start, maxeval=niter)



print(map_estimate['xIn'], Xint)
#print(map_estimate['x0'], x0True)


# %%























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


