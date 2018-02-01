#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:50:19 2018

@author: sebalander
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import pymc3 as pm

# %%
# Initialize random number generator
np.random.seed(123)

# True parameter values
sigma = 0.02
cov = sigma**2
alpha, beta = 0.2, 3

# funciones del modelo
f = lambda x, a: 1 / (1 + np.exp(-x / a))
g = lambda x, b: b * x

# funciones de propagacion de incerteza
def fDif(x, a, c):
    '''
    calcula la salida d ela funcion y ademas propaga la incerteza
    '''
    ep = np.exp(-x/a)
    fu = 1 / (1 + ep)
    
    er = c * (ep * fu**2 / a)**2  # propago covarianza
    return fu, er

gDif = lambda x, b, c: b**2 * c

def fullPropa(x, a, b, c):
    x2, c2 = fDif(x, a, c)
    x3, c3 = [g(x2, b), gDif(x2, b, c2)]
    
    return x3, c3



# Size of dataset
size = 20

# Predictor variable
X = np.linspace(0,1,size)
X1 = X + np.random.randn(size) * sigma
# Simulate outcome variable

Y, C = fullPropa(X1, alpha, beta, cov)


plt.errorbar(X, Y, yerr=np.sqrt(C))



# %%
import pymc3 as pm

basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Uniform('alpha', lower=0, upper=1)
    beta = pm.Uniform('beta', lower=0, upper=10) # shape=2)

    # Expected value of outcome
    mu, c = fullPropa(X, alpha, beta, cov)

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=np.sqrt(c), observed=Y)

# %%
map_estimate = pm.find_MAP(model=basic_model)
map_estimate

# %%
from scipy import optimize

map_estimate = pm.find_MAP(model=basic_model, fmin=optimize.fmin_powell)
map_estimate

# %%
from scipy import optimize

nDraws = 1000
nChains = 10
nSampels = nChains * nDraws

with basic_model:
    # draw 500 posterior samples
    trace = pm.sample(draws=nDraws, chains=nChains)


[np.array(trace[varNm]).shape for varNm in trace.varnames]

samples = np.hstack([trace[varNm].reshape((nSampels,-1)) for varNm in trace.varnames])

corner(samples,bins=100)

# %%

nDraws = 1000
nChains = 10
nSampels = nChains * nDraws

with basic_model:

    # obtain starting values via MAP
    start = pm.find_MAP(fmin=optimize.fmin_powell)

    # instantiate sampler
    step = pm.Slice()

    # draw 5000 posterior samples
    trace = pm.sample(nDraws, step=step, start=start, chains=nChains)

[np.array(trace[varNm]).shape for varNm in trace.varnames]

samples = np.hstack([trace[varNm].reshape((nSampels,-1)) for varNm in trace.varnames])

corner(samples[:,2:],bins=100)


# %% testeo que haya muchas covarianzas
# =======================================
nDim = 3
nvars = 11

mu = np.random.randn(nvars,nDim)
cov = np.random.randn(nvars,nDim, nDim)

cov = (cov.reshape((nvars,nDim, nDim, 1, 1)) *
       cov.transpose((0,2,1)).reshape((nvars, 1, 1, nDim, nDim))
       )

cov = np.sum(cov, axis=(2,3))

np.linalg.inv(cov[0])


yNor = pm.MvNormal.dist(mu=mu,cov=cov)

# %% caso analogo a las imagenes y la calibracion
# ===============================================

def projection(x, param, cI=None):
    '''
    calcula las coordenadas no distorsionadas y propaga incertezas
    '''
    # recupero los parametros en forma original
    X0 = param[:2]
    alpha = param[2]
    rtV = param[3:].reshape((nIm,2,2))
    
    # proyecto la distorsion optica
    x1 = x - X0
    r1 = np.linalg.norm(x1,axis=2)
    r2 = np.tan(r1 / alpha)
    q = r2 / r1
    x2 = x1 * q.reshape((nIm, nPts, 1))
    
    if cI is not None:
        c2 = (q**2).reshape((nIm, nPts,1,1)) * cI
    
    # aca debería poner la dependencia de la incerteza a través de q, pero por ahora no
    # cq = 
    
    # parte extrinseca
    x3 = np.sum(x2.reshape((nIm, nPts,2,1)) * rtV.reshape((nIm,1,2,2)), axis=3)
    

    if cI is not None:
        c3 = (rtV.reshape((nIm,1,2,2,1,1)) *
              c2.reshape((nIm,nPts,1,2,2,1)) *
              rtV.transpose((0,2,1)).reshape((nIm,1,1,1,2,2))).sum((3,4))
        return x3, c3
    else:
        return x3


def matSqrt(C):
    '''
    devuelve la lista de matrices tal que T.dot(T.T) = C
    '''
    sh = C.shape
    u, s, v = np.linalg.svd(C)
    
    T = u * np.sqrt(s).reshape((sh[0],sh[1],1,sh[2]))
    T = T.reshape((sh[0],sh[1],sh[2],sh[2],1))
    T = T * v.reshape((sh[0],sh[1],1,sh[2],sh[2]))
    
    return  np.sum(T, axis=3)


#(T.reshape((nIm,nPts,2,2,1))
#* T.transpose((0,1,3,2)).reshape((nIm,nPts,1,2,2))
#).sum(3)  # funciona!!!


# %% paramatros y generar datos
nIm = 20
nPts = 50

# true values of parameters and positions and everything
X0 = np.array([0.5, 0.5])
alpha = np.array(2)
rtV = np.random.rand(nIm,2,2)
# los pongo como un solo vector de parametros
param = np.concatenate([X0.reshape((-1,)),
                        alpha.reshape((-1,)),
                        rtV.reshape((-1,))], axis=0)
# posiciones reales en el mundo fisico
x = np.random.rand(nIm,nPts,2)
y = projection(x, param)
sI = 0.02 # sigma en la imagen
sM = 0.01 # sigma en el mapa

cI = np.zeros((nIm,nPts,2,2))
cI[:,:,[0,1],[0,1]] = sI**2

# datos medidos, observados, experimentalmente en el mundo y en la imagen
xObs = x + np.random.normal(size=x.shape, scale=sI)
yObs = y + np.random.normal(size=y.shape, scale=sM)


# %% modelo


basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    parametros = pm.Uniform('parametros', lower= np.zeros_like(param)-10, upper=np.zeros_like(param)+10)

    # nuestra prediccion desde la imagen al mapa basada en mediciones en la imagen
    yPred, cPred = projection(xObs, param, cI)
    
    dif = yObs - yPred
    S = np.linalg.inv(cPred)
    T = matSqrt(S) # matriz para transformar 
    
    difT = np.sum(dif.reshape((nIm, nPts, 2, 1)) * T, 3)
    
    # Likelihood (sampling distribution) of observations
    # aca no se que hacer, ya tengo todas las variables pero como lo expreso
    # una forma es definir una salida que sea como la posicion mahalanobis 
    # y poner que lo oservado es el verctor cero
    Y_obs = pm.Normal('Y_obs', mu=np.zeros_like(difT), sd=1, observed=difT)





# %%


T = matSqrt(S)






