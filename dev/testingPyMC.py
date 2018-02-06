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
import theano.tensor as T
import theano as the
import theano.tensor.nlinalg as lng


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


#
## %%
#basic_model = pm.Model()
#
#with basic_model:
#
#    # Priors for unknown model parameters
#    alpha = pm.Uniform('alpha', lower=0, upper=1)
#    beta = pm.Uniform('beta', lower=0, upper=10) # shape=2)
#
#    # Expected value of outcome
#    mu, c = fullPropa(X, alpha, beta, cov)
#
#    # Likelihood (sampling distribution) of observations
#    Y_obs = pm.Normal('Y_obs', mu=mu, sd=np.sqrt(c), observed=Y)
#
## %%
#map_estimate = pm.find_MAP(model=basic_model)
#map_estimate
#
## %%
#from scipy import optimize
#
#map_estimate = pm.find_MAP(model=basic_model, fmin=optimize.fmin_powell)
#map_estimate
#
## %%
#from scipy import optimize
#
#nDraws = 1000
#nChains = 10
#nSampels = nChains * nDraws
#
#with basic_model:
#    # draw 500 posterior samples
#    trace = pm.sample(draws=nDraws, chains=nChains)
#
#
#[np.array(trace[varNm]).shape for varNm in trace.varnames]
#
#samples = np.hstack([trace[varNm].reshape((nSampels,-1)) for varNm in trace.varnames])
#
#corner(samples,bins=100)
#
## %%
#
#nDraws = 1000
#nChains = 10
#nSampels = nChains * nDraws
#
#with basic_model:
#
#    # obtain starting values via MAP
#    start = pm.find_MAP(fmin=optimize.fmin_powell)
#
#    # instantiate sampler
#    step = pm.Slice()
#
#    # draw 5000 posterior samples
#    trace = pm.sample(nDraws, step=step, start=start, chains=nChains)
#
#[np.array(trace[varNm]).shape for varNm in trace.varnames]
#
#samples = np.hstack([trace[varNm].reshape((nSampels,-1)) for varNm in trace.varnames])
#
#corner(samples[:,2:],bins=100)
#
#
## %% testeo que haya muchas covarianzas
## =======================================
#nDim = 2
#nvars = 1000
#
#cov = np.random.randn(nvars,nDim, nDim)
#cov = (cov.reshape((nvars,nDim, nDim, 1, 1)) *
#       cov.transpose((0,2,1)).reshape((nvars, 1, 1, nDim, nDim))
#       )
#cov = np.sum(cov, axis=(2,3))
#
##mu = np.random.randn(nvars,nDim)
#
#np.linalg.inv(cov[0])
#
#
#yNor = pm.MvNormal.dist(mu=mu,cov=cov)

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
    print(x1.shape)
    r1 = np.linalg.norm(x1, axis=2)
    r2 = np.tan(r1 / alpha)
    q = r2 / r1
    x2 = x1 * q.reshape((nIm, nPts, 1))
    
    if cI is not None:
        c2 = (q**2).reshape((nIm, nPts,1,1)) * cI
    
    # aca debería poner la dependencia de la incerteza a través de q, pero por ahora no # cq = 
    
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
nIm = 10
nPts = 20

# true values of parameters and positions and everything
x0 = np.array([0.5, 0.5])
alfa = np.array([2])
rtV = np.random.rand(nIm,2,2)
# los pongo como un solo vector de parametros
param = np.concatenate([x0.reshape((-1,)),
                        alfa.reshape((-1,)),
                        rtV.reshape((-1,))], axis=0)
# posiciones reales en el mundo fisico
x = np.random.rand(nIm,nPts,2)
y = projection(x, param)
sI = 0.2 # sigma en la imagen
sM = 0.1 # sigma en el mapa

cI = np.zeros((nIm,nPts,2,2))
cI[:,:,[0,1],[0,1]] = sI**2

# datos medidos, observados, experimentalmente en el mundo y en la imagen
xObs = x + np.random.normal(size=x.shape, scale=sI)
yObs = y + np.random.normal(size=y.shape, scale=sM)

Yobs = yObs.reshape(-1)

# indexes to read diagona 2x2 blocks of a matrix
nTot = 2 * nIm * nPts
xInxs = [ [[i,i],[i+1,i+1]] for i in range(0, nTot, 2)]
yInxs = [ [[i,i+1],[i,i+1]] for i in range(0, nTot, 2)]

# %%

class proposalDist:
    def __init__(self, S, n=1):
        self.S = S
        self.n = n
    
    
    def __call__(self):
        Sx = self.S[:4].reshape((2,2))
        Sal = self.S[4]
        Srt = self.S[5:].reshape((4,4))
        
        xSam = pm.MvNormal.dist(mu=np.array([0,0]), cov=Sx).random(size=self.n)
        alfaSam = pm.Normal.dist(mu=0, sd=Sal).random(size=self.n)
        rtVsam = pm.MvNormal.dist(mu=np.array([0,0,0,0]),
                                  cov=Srt).random(size=(self.n, nIm))
        
        rtVsam = rtVsam.reshape((self.n, nIm*4))
        
#        return np.concatenate([xSam.reshape((self.n,-1)),
#                               alfaSam.reshape((self.n,-1)),
#                               rtVsam.reshape((self.n,-1))], axis=1)
        return rtVsam, xSam, alfaSam


Sx = np.eye(2)*0.1
Sal = np.array([0.1])
Srt = np.eye(4) * 0.1

S0 = np.concatenate([Sx.reshape(-1), Sal.reshape(-1), Srt.reshape(-1)])


sams = proposalDist(S0, n=1)
sams()
#xSams, alSams, rtSams = proposalDist(S0, n=5)


# %% modelo
niter= 100

try:
    del basic_model
except:
    pass

basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    x0 = pm.Uniform('x0', lower=0, upper=1, shape=2)
    alfa = pm.Uniform('alfa', lower=0, upper=5)
    rtV = pm.Uniform('rtV', lower=-3, upper=3, shape=nIm*4)
    
    rtV2 = T.reshape(rtV, (nIm,2,2))
    
    # nuestra prediccion desde la imagen al mapa basada en mediciones en la imagen
#    yPred, cPred = projection(xObs, parametros, cI)
    
    # proyecto la distorsion optica
    x1 = xObs - T.reshape(x0, (1,1,2))
    print(x.shape, T.shape(x1))
    
    r1 = T.sqrt(T.sum(x1**2, axis=2))
    r2 = T.tan(r1 / alfa)
    q = r2 / r1
    
    print(T.shape(x1),T.shape(q))
    r1 * r2 * q
    x2 = x1 * T.reshape(q, (nIm, nPts, 1))
    
    c2 = T.reshape(q**2, (nIm, nPts,1,1)) * cI
    
    # parte extrinseca
    x3 = T.reshape(x2, (nIm, nPts,2,1)) * T.reshape(rtV, (nIm,1,2,2))
    x3 = T.sum(x3, axis=3)
    
    c3 = T.sum(T.reshape(rtV2, (nIm,1,2,2,1,1)) *
               T.reshape(c2, (nIm,nPts,1,2,2,1)) *
               T.reshape(rtV2.transpose((0,2,1)), (nIm,1,1,1,2,2)),
               axis=(3,4))
    
    # Likelihood (sampling distribution) of observations
    # aca no se que hacer, ya tengo todas las variables pero como lo expreso
    # una forma es definir una salida que sea como la posicion mahalanobis 
    # y poner que lo oservado es el verctor cero
#    Y_obs = pm.MvNormal('Y_obs', mu=np.zeros_like(difT),
#                        cov=np.eye(2), observed=difT)
    mu = T.reshape(x3, (-1,))
    
    print(T.isnan(mu), T.isinf(mu))
    
    bigC = T.zeros((nTot, nTot))
    c3Diag = T.reshape(c3, (-1,2,2)) # list of blocks

    bigC = T.set_subtensor(bigC[xInxs, yInxs], c3Diag)
    
    Y_obs = pm.MvNormal('Y_obs', mu=mu, cov=bigC, observed=Yobs)



# %%

start = {'x0': np.array([0.5, 0.5]),
         'alfa': np.array(2),
         'rtV': np.random.rand(nIm,2,2)}

with basic_model:
    step = pm.Metropolis(vars=basic_model.vars,
                         S=S0,
                         proposal_dist=proposalDist,
                         scaling=1.0,
                         tune=True,
                         tune_interval=10,
                         )

    trace = pm.sample(niter, step=step, start=start, njobs=4, random_seed=123)
    
    
##    stepMeth = pm.Metropolis(vars=(x0, alfa, rtV), S=np.array(1), tune_interval=2)
#    step = pm.Metropolis(basic_model.vars, model=basic_model)
#    # draw 500 posterior samples
#    trace = pm.sample(draws=nDraws, step=step)

# %%
'''
vars : list
List of variables for sampler
S : standard deviation or covariance matrix
Some measure of variance to parameterize proposal distribution
proposal_dist : function
Function that returns zero-mean deviates when parameterized with S (and n). Defaults to normal.
scaling : scalar or array
Initial scale factor for proposal. Defaults to 1.
tune : bool
Flag for tuning. Defaults to True.
tune_interval : int
The frequency of tuning. Defaults to 100 iterations.
model : PyMC Model
Optional model for sampling step. Defaults to None (taken from context).
mode : string or Mode instance.
compilation mode passed to Theano functions
'''

# %%
from scipy import optimize

nDraws = 100
nChains = 3
nSampels = nChains * nDraws


with basic_model:
    stepMeth = pm.Metropolis()
    # draw 500 posterior samples
    trace = pm.iter_sample(draws=nDraws, step=stepMeth)

