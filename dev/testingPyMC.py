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
import theano
#import theano.tensor.nlinalg as lng
from theano.compile.debugmode import DebugMode
from scipy import optimize as opt

# %% caso analogo a las imagenes y la calibracion
# ===============================================

def projection(x, x0, alfa, rtV, cI):
    '''
    calcula las coordenadas no distorsionadas y propaga incertezas
    '''
#    # recupero los parametros en forma original
#    x0 = param[:2]
#    alpha = param[2]
#    rtV = param[3:].reshape((nIm,2,2))

    # proyecto la distorsion optica
    x1 = x - x0
    # print(x1.shape)
    r1 = np.linalg.norm(x1, axis=2)
    r2 = np.tan(r1 * alfa)
    q = r2 / r1
    x2 = x1 * q.reshape((nIm, nPts, 1))

    # if cI is not None:
    c2 = (q**2).reshape((nIm, nPts, 1, 1)) * cI

    # aca debería poner la dependencia de la incerteza a través de q, pero por ahora no # cq =

    # parte extrinseca
    x3 = np.sum(x2.reshape((nIm, nPts,2,1)) * rtV.reshape((nIm,1,2,2)), axis=3)

    # if cI is not None:
    c3 = (rtV.reshape((nIm,1,2,2,1,1)) *
          c2.reshape((nIm,nPts,1,2,2,1)) *
          rtV.transpose((0,2,1)).reshape((nIm,1,1,1,2,2))).sum((3,4))
    return x3, c3


def matSqrt(C):
    '''
    devuelve la lista de matrices tal que T.dot(T.T) = C

    **unused**

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
nPts = 50

# true values of parameters and positions and everything
x0True = np.array([0.5, 0.5])
alfaTrue = 0.2
np.random.seed(0)
rtVTrue = np.random.rand(nIm,2,2)

# posiciones reales en el mundo fisico
xTrue = np.random.rand(nIm,nPts,2)
sITrue = 0.02 # sigma en la imagen
cITrue = np.zeros((nIm,nPts,2,2))
cITrue[:,:,[0,1],[0,1]] = sITrue**2

yTrue, cMTrue = projection(xTrue, x0True, alfaTrue, rtVTrue, cITrue)
sMTrue = 0.01 # sigma en el mapa


# datos medidos, observados, experimentalmente en el mundo y en la imagen
xObs = xTrue + np.random.normal(size=xTrue.shape, scale=sITrue)
yObs = yTrue + np.random.normal(size=yTrue.shape, scale=sMTrue)

Yobs = yObs.reshape(-1)

# indexes to read diagona 2x2 blocks of a matrix
nTot = 2 * nIm * nPts
xInxs = [ [[i,i],[i+1,i+1]] for i in range(0, nTot, 2)]
yInxs = [ [[i,i+1],[i,i+1]] for i in range(0, nTot, 2)]


# %%
'''
aca defino la proyeccion para que la pueda usar thean0
http://deeplearning.net/software/theano/extending/extending_theano.html
'''


class ProjectionT(theano.Op):
    #itypes and otypes attributes are
    #compulsory if make_node method is not defined.
    #They're the type of input and output respectively
    # x, x0, alfa, rtV, cI
    itypes = [T.dtensor3, T.dvector, T.dscalar, T.dtensor3, T.dtensor4]
    # y, cM
    otypes = [T.dtensor3, T.dtensor4]

    # Python implementation:
    def perform(self, node, inputs_storage, output_storage):
        x, x0, alfa, rtV, cI = inputs_storage
        y, cM = output_storage

        y[0], cM[0] = projection(x, x0, alfa, rtV, cI)

    # optional:
    check_input = True

# %% pruebo si esta bien como OP

x = T.dtensor3('x')
x0 = T.dvector('x0')
alfa = T.dscalar('alfa')
rtV = T.dtensor3('rtV')
cI = T.dtensor4('cI')

projT = ProjectionT()

projTfunction = theano.function([x, x0, alfa, rtV, cI],                                 projT(x, x0, alfa, rtV, cI))

out = projTfunction(xTrue, x0True, alfaTrue, rtVTrue, cITrue)

print(out)



# %% modelo
niter= 100

try:
    basic_model
except NameError:
    pass
    # print "well, it WASN'T defined after all!"
else:
    del basic_model
    # print "sure, it was defined."

# instancia de la clase qeu envuelve mi funcion de python
projT = ProjectionT()

# constantes de calculo
xObsConst = T.as_tensor_variable(xObs, 'xObsConst', ndim=3)
cIConst = T.as_tensor_variable(cITrue, 'cIConst', ndim=4)

basic_model = pm.Model()

with basic_model:
    # Priors for unknown model parameters
    x0 = pm.Uniform('x0', lower=0, upper=1, shape=2)
    alfa = pm.Uniform('alfa', lower=0, upper=1)
    rtV = pm.Uniform('rtV', lower=0, upper=1, shape=(nIm,2,2))

    yM, cM = projT(xObsConst, x0, alfa, rtV, cIConst)

    mu = T.reshape(yM, (-1,))

    bigC = T.zeros((nTot, nTot))
    c3Diag = T.reshape(cM, (-1,2,2)) # list of blocks
    bigC = T.set_subtensor(bigC[xInxs, yInxs], c3Diag)

    Y_obs = pm.MvNormal('Y_obs', mu=mu, cov=bigC, observed=Yobs)

# %%
# aca saco el maximo a posteriori, bastante util para hacer montecarlo despues

try:
    map_estimate
except:
    print('set initial state arbitrarily')
    start = {'x0': np.array([0.5, 0.5]),
             'alfa': np.array(0.2),
             'rtV': rtVTrue }
#              'rtV': np.random.rand(nIm,2,2)}
else:
    print('set initial state with previous map_estiamte')
    start=map_estimate

niter = 30000

map_estimate = pm.find_MAP(model=basic_model, start=start, maxeval=int(niter * 1.3), maxiter=niter, fmin=opt.fmin, xtol=1e-2,ftol=1e-3)

print(map_estimate['alfa'], alfaTrue)
print(map_estimate['x0'], x0True)
print(map_estimate['rtV'] - rtVTrue)


# %% metropolis desde MAP. a ver si zafo de la proposal dist
'''
http://docs.pymc.io/api/inference.html
'''

start = map_estimate

#start = {'x0': map_estimate['x0']}#,
#         'alfa': map_estimate['alfa'],
#         'rtV': map_estimate['rtV']}

#scale = {'x0': map_estimate['x0_interval__'],
#         'alfa': map_estimate['alfa_interval__'],
#         'rtV': map_estimate['rtV_interval__']}
#
#scale = [map_estimate['x0_interval__'], map_estimate['alfa_interval__'], map_estimate['rtV_interval__']]

nDraws = 2000
nChains = 4

with basic_model:
    step = pm.Metropolis()
#    step = pm.Metropolis(vars=basic_model.x0, # basic_model.alfa,basic_model.rtV],
#                         S=np.abs(map_estimate['x0_interval__'])),
#                         scaling=1e-1,
#                         tune=True,
#                         tune_interval=50)

#    step = pm.Metropolis()

    trace = pm.sample(tune=1000, draws=nDraws, step=step, start=start,
                      njobs=4, chains=nChains, progressbar=True) #,
#                      init='auto', n_init=200,  random_seed=123)

# %%
plt.figure()
plt.plot(trace['x0'])
plt.figure()
plt.plot(trace['alfa'])

plt.hist(trace['alfa'], 20)
plt.hist(trace['x0'], 100)

plt.figure()
plt.plot(trace['rtV'].reshape(-1,nIm*4))

plt.figure()
plt.plot(trace['x0'][:,0], trace['x0'][:,1])

plt.hist2d(trace['x0'][:,0], trace['x0'][:,1], 100)





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


start = {'x0': np.array([0.5, 0.5]),
         'alfa': np.array(0.5),
         'rtV': np.random.rand(nIm,2,2)}

with basic_model:
    step = pm.Metropolis(vars=basic_model.vars,
                         S=S0,
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
METROPOLIS

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





STEP

draws : int
The number of samples to draw. Defaults to 500. The number of tuned samples are discarded by default. See discard_tuned_samples.
step : function or iterable of functions
A step function or collection of functions. If there are variables without a step methods, step methods for those variables will be assigned automatically.
init : str
Initialization method to use for auto-assigned NUTS samplers.

auto : Choose a default initialization method automatically. Currently, this is ‘jitter+adapt_diag’, but this can change in the future. If you depend on the exact behaviour, choose an initialization method explicitly.
adapt_diag : Start with a identity mass matrix and then adapt a diagonal based on the variance of the tuning samples. All chains use the test value (usually the prior mean) as starting point.
jitter+adapt_diag : Same as adapt_diag, but add uniform jitter in [-1, 1] to the starting point in each chain.
advi+adapt_diag : Run ADVI and then adapt the resulting diagonal mass matrix based on the sample variance of the tuning samples.
advi+adapt_diag_grad : Run ADVI and then adapt the resulting diagonal mass matrix based on the variance of the gradients during tuning. This is experimental and might be removed in a future release.
advi : Run ADVI to estimate posterior mean and diagonal mass matrix.
advi_map: Initialize ADVI with MAP and use MAP as starting point.
map : Use the MAP as starting point. This is discouraged.
nuts : Run NUTS and estimate posterior mean and mass matrix from the trace.
n_init : int
Number of iterations of initializer If ‘ADVI’, number of iterations, if ‘nuts’, number of draws.
start : dict, or array of dict
Starting point in parameter space (or partial point) Defaults to trace.point(-1)) if there is a trace provided and model.test_point if not (defaults to empty dict). Initialization methods for NUTS (see init keyword) can overwrite the default.
trace : backend, list, or MultiTrace
This should be a backend instance, a list of variables to track, or a MultiTrace object with past values. If a MultiTrace object is given, it must contain samples for the chain number chain. If None or a list of variables, the NDArray backend is used. Passing either “text” or “sqlite” is taken as a shortcut to set up the corresponding backend (with “mcmc” used as the base name).
chain_idx : int
Chain number used to store sample in backend. If chains is greater than one, chain numbers will start here.
chains : int
The number of chains to sample. Running independent chains is important for some convergence statistics and can also reveal multiple modes in the posterior. If None, then set to either njobs or 2, whichever is larger.
njobs : int
The number of chains to run in parallel. If None, set to the number of CPUs in the system, but at most 4. Keep in mind that some chains might themselves be multithreaded via openmp or BLAS. In those cases it might be faster to set this to one.
tune : int
Number of iterations to tune, if applicable (defaults to 500). Samplers adjust the step sizes, scalings or similar during tuning. These samples will be drawn in addition to samples and discarded unless discard_tuned_samples is set to True.
nuts_kwargs : dict
Options for the NUTS sampler. See the docstring of NUTS for a complete list of options. Common options are

target_accept: float in [0, 1]. The step size is tuned such that we approximate this acceptance rate. Higher values like 0.9 or 0.95 often work better for problematic posteriors.
max_treedepth: The maximum depth of the trajectory tree.
step_scale: float, default 0.25 The initial guess for the step size scaled down by 1/n**(1/4).
If you want to pass options to other step methods, please use step_kwargs.

step_kwargs : dict
Options for step methods. Keys are the lower case names of the step method, values are dicts of keyword arguments. You can find a full list of arguments in the docstring of the step methods. If you want to pass arguments only to nuts, you can use nuts_kwargs.
progressbar : bool
Whether or not to display a progress bar in the command line. The bar shows the percentage of completion, the sampling speed in samples per second (SPS), and the estimated remaining time until completion (“expected time of arrival”; ETA).
model : Model (optional if in with context) random_seed : int or list of ints

A list is accepted if njobs is greater than one.
live_plot : bool
Flag for live plotting the trace while sampling
live_plot_kwargs : dict
Options for traceplot. Example: live_plot_kwargs={‘varnames’: [‘x’]}
discard_tuned_samples : bool
Whether to discard posterior samples of the tune interval.
compute_convergence_checks : bool, default=True
Whether to compute sampler statistics like gelman-rubin and effective_n.



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



# %%
