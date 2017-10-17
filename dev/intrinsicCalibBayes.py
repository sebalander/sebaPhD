#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 23:38:35 2017

para UNLP

calibracion con incerteza:
1- calibracion intrinseca chessboard con ocv
2- tomo como condicion inicial y optimizo una funcion error custom
3- saco el hessiano en el optimo
4- sacar asi la covarianza de los parametros optimizados

teste0:
1- con imagenes no usadas para calibrar calcula la probabilidad de los
parámetros optimos dado los datos de test

@author: sebalander
"""

# %%
import glob
import numpy as np
import scipy.linalg as ln
import scipy.stats as sts
import matplotlib.pyplot as plt
from importlib import reload
from copy import deepcopy as dc
import numdifftools as ndf
from dev import multipolyfit as mpf

from dev import bayesLib as bl

from multiprocess import Process, Queue, Value
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

intrinsicParamsFile =   imagesFolder + camera + model + "intrinsicParamsML.npy"


intrinsicHessianFile = imagesFolder + camera + model + "intrinsicHessian.npy"

imageSelection = np.arange(33) # selecciono con que imagenes trabajar

# load data
imagePoints = np.load(cornersFile)[imageSelection]
n = len(imagePoints)  # cantidad de imagenes

chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
# images = glob.glob(imagesFolder+'*.png')

# Parametros de entrada/salida de la calibracion
objpoints = np.array([chessboardModel]*n)
m = chessboardModel.shape[1]  # cantidad de puntos


# load model specific data
distCoeffs = np.load(distCoeffsFile)
cameraMatrix = np.load(linearCoeffsFile)
rVecs = np.load(rVecsFile)[imageSelection]
tVecs = np.load(tVecsFile)[imageSelection]

# parametros auxiliares
Ci = np.repeat([ np.eye(2)],n*m, axis=0).reshape(n,m,2,2)  # 1px de std
params = [n, m, imagePoints, model, chessboardModel, Ci]

# pongo en forma flat los valores iniciales
Xint, Ns = bl.int2flat(cameraMatrix, distCoeffs)
XextList = [bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)]

# pruebo con una imagen
j = 0
ErIm = bl.errorCuadraticoImagen(XextList[j], Xint, Ns, params, j, mahDist=False)
print(ErIm.sum())

# pruebo el error total
def etotal(Xint, Ns, XextList, params):
    '''
    calcula el error total como la suma de los errore de cada punto en cada
    imagen
    '''
    return bl.errorCuadraticoInt(Xint, Ns, XextList, params).sum()

Erto = bl.errorCuadraticoInt(Xint, Ns, XextList, params, mahDist=False)
E0 = etotal(Xint, Ns, XextList, params)
print(Erto.sum(), E0)

# saco distancia mahalanobis de cada proyeccion
mahDistance = bl.errorCuadraticoInt(Xint, Ns, XextList, params, mahDist=True)

plt.figure()
nhist, bins, _ = plt.hist(mahDistance, 50, normed=True)
chi2pdf = sts.chi2.pdf(bins, 2)
plt.plot(bins, chi2pdf)
plt.yscale('log')

# %% grafico para saber en que rangos hayq ue tomar en cuenta el error
'''
me quedo con una region tal que la probabilidad no sea mucho menor a 10^3 veces
la probabilidad del optimo de OCV. o sea que el error tiene que ser como mucho
Emax = E0 - 2 * np.log(1e-3)
donde la diferencia de error Emax - E0 ~ 14
'''
difE = 6 * np.log(10)
Emax = E0 + difE


# %%muevo en cada dirección buscando esa cota de error
Xint2 = dc(Xint)
Xint2[7] = -4.21e-03
E2 = etotal(Xint2, Ns, XextList, params)
print(difE, E2 - E0)

# %%
'''
encontre por prueba y error estos rangos:

Xint2[0] = [397.9 : 398.6]
Xint2[1] = [410.5 : 411.7]
Xint2[2] = [807.9 : 808.4]
Xint2[3] = [466.8 : 467.4]
Xint2[4] = [9.52e-02 : 9.7e-02]
Xint2[5] = [-1.78e-02 : -1.815e-02]
Xint2[6] = [1.704e-02 : 1.726e-02]
Xint2[7] = [-4.08e-03 : -4.22e-03]

Xint = 
array([  398.213410,   411.227681,   808.169868,
         467.122082,   9.58412207e-02,  -1.79782432e-02,
         1.71556081e-02,  -4.14991879e-03])
    '''


cotas = np.array([[398.1, 398.34],          #  398.213410
                  [411.03, 411.42],         #  411.227681
                  [808.07, 808.27],         #  808.169868
                  [467.03, 467.21],         #  467.122082
                  [9.56e-02, 9.61e-02],     #  9.58412207e-02
                  [-1.815e-02, -1.78e-02],  #  -1.79782432e-02
                  [1.704e-02, 1.727e-02],   #  1.71556081e-02
                  [-4.08e-03, -4.22e-03]])  #  -4.14991879e-03

errores = np.zeros((8,2))
for i in range(8):
    Xint2 = dc(Xint)
    Xint2[i] = cotas[i,0]
    errores[i,0] = etotal(Xint2, Ns, XextList, params)

    Xint2[i] = cotas[i,1]
    errores[i,1] = etotal(Xint2, Ns, XextList, params)

print(errores - E0)





# %% ploteo en cada direccion, duplicando los intervalos si quiero
cotasDuplicadas = ((cotas.T - Xint) * 2 + Xint).T

Npts = 100
etas = np.linspace(0,1,Npts).reshape((1,-1))

intervalo = (cotasDuplicadas[:,1] - cotasDuplicadas[:,0])

barreParams = intervalo.reshape((-1,1)) * etas
barreParams += cotasDuplicadas[:,0].reshape((-1,1))

ers = np.zeros((8, Npts))

for i in range(8):
    Xint2 = dc(Xint)
    for j in range(Npts):
        Xint2[i] = barreParams[i,j]
        ers[i,j] = etotal(Xint2, Ns, XextList, params)
        print(i,j,ers[i,j])

paramsNormaliz = (barreParams.T - Xint) / intervalo

plt.plot(paramsNormaliz, ers.T, '-+')


'''
me quedo con las cotas que defini como la region de donde sacar samples
'''

# %% metropolis hastings
from numpy.random import rand as rn

def nuevo(old, oldE):
    global generados
    global aceptados
    global avance
    global retroceso
    
    # genero nuevo
    new = sampleador() # rn(8) * intervalo + cotas[:,0]
    newE = etotal(new, Ns, XextList, params)
    generados += 1
    print(generados, aceptados, avance, retroceso)
    
    # cambio de error
    deltaE = newE - oldE
    accept = deltaE < 0
    
    if accept:
        aceptados +=1
        avance += 1
        print("avance directo")
        return new, newE # tiene menor error, listo
    else:
        # nueva opoertunidad, sampleo contra rand
        accept = np.exp(- deltaE / 2) > rn()
        if accept:
            aceptados += 1
            retroceso += 1
            print("retroceso")
            return new, newE # aceptado a la segunda oportunidad
        else:
            # vuelvo recursivamente al paso2 hasta aceptar
            new, newE = nuevo(old, oldE)
    
    return new, newE

# %%
# matriz diagonal sacada brutamente de las cotas encontradas
covar = np.diag(np.mean(((cotas.T - Xint) / 10)**2, 0))
sampleador = sts.multivariate_normal(Xint, covar).rvs

Nmuestras = int(1e3)

generados = 0
aceptados = 0
avance = 0
retroceso = 0

paraMuest = np.zeros((Nmuestras,8))
errorMuestras = np.zeros(Nmuestras)

# primera
old = dc(Xint) # sampleador() # rn(8) * intervalo + cotas[:,0]
oldE = etotal(old, Ns, XextList, params)
paraMuest[0], errorMuestras[0] = (old, oldE)

for i in range(1, Nmuestras):
    paraMuest[i], errorMuestras[i] = nuevo(paraMuest[i-1], errorMuestras[i-1])


for i in range(8):
    plt.figure()
    plt.hist(paraMuest[:,i],30)

# %% new estimated covariance run Metropolis again
covar2 = np.cov(paraMuest.T)
sampleador = sts.multivariate_normal(Xint, covar2).rvs

Nmuestras = int(1e3)

generados = 0
aceptados = 0
avance = 0
retroceso = 0

paraMuest2 = np.zeros((Nmuestras,8))
errorMuestras2 = np.zeros(Nmuestras)

# primera
old = dc(Xint) # sampleador() # rn(8) * intervalo + cotas[:,0]
oldE = etotal(old, Ns, XextList, params)
paraMuest2[0], errorMuestras2[0] = (old, oldE)

for i in range(1, Nmuestras):
    paraMuest2[i], errorMuestras2[i] = nuevo(paraMuest2[i-1], errorMuestras2[i-1])


for i in range(8):
    plt.figure()
    plt.hist(paraMuest2[:,i],30)


# %% new estimated covariance run Metropolis again
covar3 = np.cov(paraMuest2.T)
sampleador = sts.multivariate_normal(Xint, covar3).rvs

Nmuestras = int(1e4)

generados = 0
aceptados = 0
avance = 0
retroceso = 0

paraMuest3 = np.zeros((Nmuestras,8))
errorMuestras3 = np.zeros(Nmuestras)

# primera
old = dc(Xint) # sampleador() # rn(8) * intervalo + cotas[:,0]
oldE = etotal(old, Ns, XextList, params)
paraMuest3[0], errorMuestras3[0] = (old, oldE)

for i in range(1, Nmuestras):
    paraMuest3[i], errorMuestras3[i] = nuevo(paraMuest3[i-1], errorMuestras3[i-1])


for i in range(8):
    plt.figure()
    plt.hist(paraMuest3[:,i],30)

mu4 = np.mean(paraMuest3, axis=0)
covar4 = np.cov(paraMuest3.T)

mu4Covar = covar4 / Nmuestras
covar4Covar = bl.varVarN(covar4, Nmuestras)

resultsML = dict()

resultsML['Nsamples'] = Nmuestras
resultsML['paramsMU'] = mu4
resultsML['paramsVAR'] = covar4
resultsML['paramsMUvar'] = mu4Covar
resultsML['paramsVARvar'] = covar4Covar

save = False
if save:
    np.save(intrinsicParamsFile, resultsML)


# el error relativo aproximadamente
np.sqrt(np.diag(covar4)) / mu4


# %% pruebo evaluar las funciones para los processos
#jacIntrin = bl.Jint(Xint, Ns, XextList, params)

hint = bl.ndf.Hessian(etotal, method='central')
hint.step.base_step = 1e-3 * Xint

hesIntrin = hint(Xint, Ns, XextList, params)

np.real(ln.eigvals(hesIntrin))


# %% intento optimizacion con leastsq
from scipy.optimize import minimize

#meths = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B',
#         'TNC', 'COBYLA', 'SLSQP', 'dogleg', 'trust-ncg']

meths = ["Nelder-Mead", "Powell", "L-BFGS-B", "COBYLA", "SLSQP"]
#meths = ['CG', 'BFGS', 'Newton-CG','TNC','dogleg', 'trust-ncg']



res = dict()
hInt = dict()

for me in meths:
    res[me] = minimize(etotal, Xint,
                        args=(Ns, XextList, params), method=me)
    print(me)
    print(res[me])
    print(np.abs((res[me].x - Xint)/ Xint))

    if res[me].success is True:
        hInt[me] = hint(res[me].x, Ns, XextList, params)  #  (Ns, Ns)

        print(np.real(ln.eigvals(hInt[me])))

        cov = ln.inv(hInt[me])
        plt.matshow(cov)
        #fun, hess_inv, jac, message, nfev, nit, njev, status, success, x = res

        sigs = np.sqrt(np.diag(cov))
        plt.figure()
        plt.plot(np.abs(sigs / res[me].x), 'x')



# %% trato de resolver el problema del hessiano no positivo en los parametros intrinsecos

Ci = np.repeat([np.eye(2)],n*m, axis=0).reshape(n,m,2,2)
params = [n, m, imagePoints, model, chessboardModel, Ci]

# pongo en forma flat los valores iniciales
Xint, Ns = bl.int2flat(cameraMatrix, distCoeffs)
XextList = [bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)]

jInt = bl.Jint(Xint, Ns, XextList, params)  # (Ns,)
hInt = bl.Hint(Xint, Ns, XextList, params)  #  (Ns, Ns)

# %%
deter = ln.det(hInt)

diagonal = np.diag(hInt)

u, s, v = ln.svd(hInt)


# %%
plt.figure()
plt.plot(Xint0, res.x, 'x')

(Xint0 - res.x) / Xint0

plt.figure()
plt.plot((Xint0 - res.x) / Xint0, 'x')


# %% ploteo en cada direccion
Npts = 10
etaM = 1e-4 # variacion total para cada lado
etas = 1 + np.linspace(-etaM, etaM, Npts)

XintFin = res.x
XmodAll = np.repeat([XintFin], Npts, axis=0) * etas.reshape((Npts, -1))
erAll = np.zeros_like(XmodAll)


for j in range(len(XintFin)):
    for i in range(Npts):
        print(j, i)
        X = dc(XintFin)
        X[j] = XmodAll[i, j]
        erAll[i, j] = bl.errorCuadraticoInt(X, Ns, XextList0, params)
        print(erAll[i, j])

plt.plot(etas, erAll, '-x')
plt.plot([etas[0], etas[-1]], [res.fun, res.fun])




# %%
# metodos que funcionan:
meths = ["Nelder-Mead", "Powell", "L-BFGS-B", "COBYLA", "SLSQP"]
resss = []

for me in meths:
    res = minimize(bl.errorCuadraticoInt, Xint0,
                   args=(Ns, XextList0, params), method=me)
    print(me)
    print(res)
    resss.append(res)

# %% como la libreria que hace ndiftools tiene problemas para calcularl el
# hessiano lo calculo por mi cuenta usando el resultado de los varias metodos


XX = np.array([r.x for r in resss])
YY = np.array([r.fun for r in resss])

armi = np.argmin(YY)
X0 = XX[armi]
Y0 = YY[armi] # bl.errorCuadraticoInt(X0, Ns, XextList0, params)


coefs, exps = mpf.multipolyfit(XX - X0, YY, 2, powers_out=True)

# asume first of 9 element in exponent is for 1
const = 0
jac = np.zeros((8))
hes = np.zeros((8,8))

const = coefs[0]
jac = coefs[1:9]
hes = np.zeros((8,8))
hes[np.tril_indices(8)] = hes[np.triu_indices(8)] = coefs[9:]


print(const)
print(jac)
print(hes)

# testeo
[const + jac.dot(x) + x.dot(hes).dot(x)/2 for x in (XX - X0)]

# %% ahora augmento las soluciones obtenidas
XXX = np.zeros((5,5,5,5,5,5,5,5,8))

for i in range(5):
    for j in range(5):
        for k in range(5):
            for l in range(5):
                for m in range(5):
                    for n in range(5):
                        for o in range(5):
                            for p in range(5):
                                XXX[i,j,k,l,m,n,o,p] = XX[[i,j,k,l,m,n,o,p],
                                                          [0,1,2,3,4,5,6,7]]

XXX.shape = (-1,8)

np.save('XXX.npy', XXX)

#import timeit
#statement = '''
#bl.errorCuadraticoInt(XX[0], Ns, XextList0, params)
#'''
#t1 = timeit.timeit(statement, globals=globals(), number=100) / 1e2
#print('horas que va atardar', t1 * XXX.shape[0] / 3600)

#indprueba = np.random.randint(0, XXX.shape[0], 500)
YYY = [bl.errorCuadraticoInt(x, Ns, XextList0, params) for x in XXX]
np.save('YYY.npy', YYY)
prob = np.exp(-np.array(YYY))  # peso proporcional a la probabilidad,
# YYY = np.load('YYY.npy')
# XXX = np.load('XXX.npy')

XXX0 = np.average(XXX,axis=0,weights=prob)


coefs, exps = mpf.multipolyfit(XXX - XXX0, YYY, 2, powers_out=True)

# asume first of 9 element in exponent is for 1
const = coefs[0]
jac = coefs[1:9]
hes = np.zeros((8,8))
hes.T[np.triu_indices(8)] = hes[np.triu_indices(8)] = coefs[9:]


print(const)
print(jac)
print(hes)
print(hes==hes)
# testeo
test = [const + jac.dot(x) + x.dot(hes).dot(x)/2 for x in (XXX - XXX0)]
# deberia dar la identidad +- error de taylor
plt.plot(YYY, test, 'x')


# %% saco el vertice de la hiperparabola calculada
vert = XXX0 - jac.dot(ln.inv(hes))

# ahora recalculo la hiperparabola
coefs, exps = mpf.multipolyfit(XXX - vert, YYY, 2, powers_out=True)

# asume first of 9 element in exponent is for 1
const = coefs[0]
jac = coefs[1:9]
hes = np.zeros((8,8))
hes.T[np.triu_indices(8)] = hes[np.triu_indices(8)] = coefs[9:]

cova = ln.inv(hes)  # esta seria la covarianza asociada

# %% un ultimo refinamiento, sampleo puntos de la gaussiana calculada
# y con esos puntos recalculo el vertice y todo eso

u,s,v = ln.svd(cova)  # para diagonalizar la elipse de la gaussiana

XXsamples = vert + np.random.randn(1000, 8).dot(s*v)  # sampleo

for i in range(6):
    print(i)
    plt.figure()
    plt.scatter(XXsamples.T[i], XXsamples.T[i+1])

# calculo los errores de esos samples
YYsamples = [bl.errorCuadraticoInt(x, Ns, XextList0, params) for x in XXsamples]
prob = np.exp(-np.array(YYsamples))  # peso proporcional a la probabilidad

# saco el valor esperado
XXsamples0 = np.average(XXsamples,axis=0,weights=prob)

coefs, exps = mpf.multipolyfit(XXsamples - XXsamples0, YYsamples, 2, powers_out=True)

constSam = coefs[0]
jacSam = coefs[1:9]
hesSam = np.zeros((8,8))
hesSam[np.triu_indices(8)] = hesSam.T[np.triu_indices(8)] = coefs[9:]

print(constSam)
print(jacSam)
print(hesSam)
print(ln.eig(hesSam)[0])

# testeo
testSam = [constSam + jacSam.dot(x) + x.dot(hesSam).dot(x)/2 for x in (XXsamples - XXsamples0)]

plt.plot(YYsamples, testSam, 'x')

# saco el vertice de la hiperparabola calculada
vert = XXsamples0 - jac.dot(ln.inv(hes))
cova = ln.inv(hes)  # esta seria la covarianza asociada

# %% grafico el error en cada direccion para ver que sea suave y en que
# escalas hacer el fiteo
Ci = np.repeat([np.eye(2)],n*m, axis=0).reshape(n,m,2,2)
params = [n, m, imagePoints, model, chessboardModel, Ci]

# pongo en forma flat los valores iniciales
Xint, Ns = bl.int2flat(cameraMatrix, distCoeffs)
XextList = [bl.ext2flat(rVecs[i], tVecs[i])for i in range(n)]

Nsam = 50
#                   0    1    2    3    4     5     6     7
anchos = np.array([7e0, 7e0, 2e1, 2e1, 6e-3, 4e-3, 2e-3, 1e-3])
rangos = np.array([Xint-anchos, Xint+anchos]).T


#np.linspace(0.5,1.5, Nsam)

xList = list()
yList = list()

fig, ax = plt.subplots(2,4)
ax = ax.flatten()

for i in range(8): # [1,2,3]:
    print(i)
    xSam = np.repeat([Xint], Nsam, axis=0)
    xSam[:, i] = np.linspace(rangos[i][0], rangos[i][1], Nsam)

    ySam = [bl.errorCuadraticoInt(x, Ns, XextList, params) for x in xSam]

    xList.append(xSam[:, i])
    yList.append(ySam)

#    plt.figure(i)
#    ax[i].title(i)
    ax[i].plot(xSam[:, i], ySam, 'b-')
#    plt.plot(xSam[:, i], ypoly, 'g-')
    ax[i].plot([Xint[i], Xint[i]], [np.min(ySam), np.max(ySam)], '-r')

#    # fiteo una parabola
#    p = np.polyfit(xSam[:, i], ySam, 2)
#    ypoly = np.polyval(p, xSam[:, i])

#    plt.figure(i*2+1)
#    plt.title(i)
#    ysqrt = np.sqrt(ySam)
#    ypolysqrt = np.sqrt(ypoly)
#    plt.plot(xSam[:, i], ysqrt, 'b-')
#    plt.plot(xSam[:, i], ypolysqrt, 'g-')
#    plt.plot([Xint[i], Xint[i]], [np.min(ysqrt), np.max(ysqrt)], '-r')

xList = np.array(xList).T - Xint
yList = np.array(yList).T

#xListInf = np.min(xList, axis=0)
#xListSup = np.max(xList, axis=0)
#
#yListInf = np.min(yList)
#yListSup = np.max(yList, axis=0)
#
#plt.plot((xList - xListInf) / (xListSup - xListInf),
#         (yList - yListInf) / (yListSup - yListInf), '-+')


# %%
# import funcion para ajustar hiperparabola
from dev import multipolyfit as mpf

def coefs2mats(coefs, n=8):
    const = coefs[0]
    jac = coefs[1:n+1]
    hes = np.zeros((n,n)) # los diagonales aparecen solo una vez
    hes[np.triu_indices(n)] = hes.T[np.triu_indices(n)] = coefs[n+1:]

    hes[np.diag_indices(n)] *= 2
    return const, jac, hes


# %% para hacer en muchos hilos
def mapListofParams(X, Ns, XextList, params):
    global mapCounter
    global npts
    ret = list()
    for  x in X:
        ret.append(bl.errorCuadraticoInt(x, Ns, XextList, params))
        mapCounter.value += 1
        print("%d de %d calculos: %2.3f por 100; error %g" %
              (mapCounter.value, npts, mapCounter.value/npts*100, ret[-1]))

    return np.array(ret)



def procMapParams(X, Ns, XextList, params, ret):
    ret.put(mapListofParams(X, Ns, XextList, params))



def mapManyParams(Xlist, Ns, XextList, params, nThre=4):
    '''
    function to map in many threads values of params to error function
    '''
    N = Xlist.shape[0] # cantidad de vectores a procesar
    procs = list()  # lista de procesos
    er = [Queue() for i in range(nThre)]  # donde comunicar el resultado
    retVals = np.zeros(N)  #

    inds = np.linspace(0, N, nThre + 1, endpoint=True, dtype=int)
    #print(inds)

    for i in range(nThre):
        #print('vecores', Xlist[inds[i]:inds[i+1]].shape)
        p = Process(target=procMapParams,
                    args=(Xlist[inds[i]:inds[i+1]], Ns, XextList, params,
                          er[i]))
        procs.append(p)
        p.start()

    [p.join() for p in procs]

    for i in range(nThre):
        ret = er[i].get()
        #print('loque dio', ret.shape)
        #print('donde meterlo', retVals[inds[i]:inds[i+1]].shape)
        retVals[inds[i]:inds[i+1]] = ret



    return retVals



# %% aca testeo que tan rapido es con multithreading para 4 nucleos
import timeit
mapCounter=0

statement = '''mapManyParams(Xrand, Ns, XextList, params, nThre=8)'''

timss = list()
nss = [50, 100, 200, 500, 1000, 2000]
mapCounter = Value('i', 0)

for npts in nss:
    print(npts)
    Xrand = (np.random.rand(npts, 8) * 2 - 1) * anchos + Xint
    timss.append(timeit.timeit(statement, globals=globals(), number=5) / 5)
    print(npts, timss[-1])

plt.plot(nss, timss, '-*')

p = np.polyfit(nss, timss, 1)
p = np.array([ 0.0338636, -0.2550772])
npts = int(1e5)
print('tiempo ', np.polyval(p, npts) / 60, 'hs; ptos por eje ', npts**0.125)
# tomaria como 10hs hacer un milon de puntos

# %%
# genero muchas perturbaciones de los params intrinsecos alrededor de las cond
# iniciales en un rango de +-10% que por las graficas parece ser bastante
# parabolico el comportamiento todavía
npts = int(1e4)
mapCounter = Value('i', 0)
Xrand = (np.random.rand(npts, 8) * 2 - 1) * anchos + Xint
Yrand = mapManyParams(Xrand, Ns, XextList, params,  nThre=8)

np.save('Xrand', Xrand)
np.save('Yrand', Yrand)

# prob = np.exp(-Yrand / np.mean(Yrand))  # peso proporcional a la probabilidad

# saco el valor esperado
# Xrand0 = np.average(Xrand, axis=0) # , weights=prob)
# centro la cuenta en Xint
DeltaX = Xrand - Xint
coefs, exps = mpf.multipolyfit(DeltaX, Yrand, 2, powers_out=True)

constRan, jacRan, hesRan = coefs2mats(coefs)
np.real(ln.eig(hesRan)[0])

nConsidered = np.arange(100, npts+100, 100, dtype=int)
autovalsConverg = np.zeros((nConsidered.shape[0], 8))

indShuff = np.arange(Yrand.shape[0])
np.random.shuffle(indShuff)

Xrand = Xrand[indShuff]
Yrand = Yrand[indShuff]


for i, ncon in enumerate(nConsidered):
    print(i)
    coefs, exps = mpf.multipolyfit(DeltaX[:ncon], Yrand[:ncon], 2, powers_out=True)
    constRan, jacRan, hesRan = coefs2mats(coefs)
    autovalsConverg[i] = np.real(ln.eig(hesRan)[0])
    print(autovalsConverg[i])

plt.figure()
plt.plot(nConsidered, autovalsConverg)


## autovectores son las columnas
#w, vr = ln.eig(hesRan, left=False, right=True)
#np.dot(vr, ln.diagsvd(np.real(w),8,8).dot(ln.inv(vr)))
#
## right singular vectors as rows
#u, s, v = ln.svd(hesRan)
#
#np.allclose(hesRan, np.dot(u, np.dot(ln.diagsvd(s,8,8), v)))
#
#plt.matshow(np.log(np.abs(hesRan)))
#
#print(l)

# testeo
testRan = [constRan + jacRan.dot(x) + x.dot(hesRan).dot(x) / 2 for x in DeltaX]

plt.figure()
plt.plot(Yrand, testRan, 'xk')
plt.plot(Yrand, testRan, 'xr')
emin, emax = [np.min(Yrand), np.max(Yrand)]
plt.plot([emin, emax],[emin, emax])

## saco el vertice de la hiperparabola calculada
#vert = Xint - jacRan.dot(ln.inv(hesRan))
#cova = ln.inv(hesRan)  # esta seria la covarianza asociada

#hesChancho = vr.dot(ln.diagsvd(s,8,8)).dot(ln.inv(vr))
#
#testChan = [constRan + jacRan.dot(x) + x.dot(hesChancho).dot(x) / 2 for x in DeltaX]
#
#plt.figure()
#plt.plot(Yrand, testChan, 'xk')
#plt.plot(Yrand, testChan, 'xr')
#emin, emax = [np.min(Yrand), np.max(Yrand)]
#plt.plot([emin, emax],[emin, emax])
# no le pega ni por asomo

# %%

#from scipy.stats import multivariate_normal
#rv = multivariate_normal(mean=vert, cov=cova)
#
## make positive semidefinite, asume simetry
#print(ln.eig(hesRan)[0])
## https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
#A = hesRan
#B = (A + A.T) / 2
#u, s, v = ln.svd(B)
##% Compute the symmetric polar factor of B. Call it H.
##% Clearly H is itself SPD.
#H = v.dot(s).dot(v.T)
#Ahat = (A + H) / 2
#Ahat = (Ahat + Ahat.T) /2
#
#ln.cho_factor(Ahat)
#
#ln.eig(Ahat)[0]



# %% ahora miro en los autovectores del hessiano
#u, s, v = np.linalg.svd(hesRan)
l, v = ln.eig(hesRan)
s = np.sqrt(np.abs(np.real(l)))
npts = int(2e1)
etasI = np.linspace(-1e3, 1e3, npts)

i = 0
for i in range(8):
    direc = v[:,i]
    perturb = np.repeat([direc], npts, axis=0) * etasI.reshape((-1,1)) / s[i]
    Xmod = vert + perturb  # modified parameters
    Ymod = np.array([bl.errorCuadraticoInt(x, Ns, XextList, params) for x in Xmod])
    Xdist = ln.norm(perturb, axis=1) * np.sign(etasI)

    plt.figure()
    plt.plot(Xdist, Ymod, '-*')



# %% pruebo de calcular el hessiano poniendo paso fijo sabiendo la escala

#                   0    1    2    3    4     5     6     7
anchos = np.array([7e0, 7e0, 2e1, 2e1, 6e-3, 4e-3, 2e-3, 1e-3])

autovalores = list() # aca guardar la lista de autovalores

# en cuánto reducir los anchos especificados a ojo
factores = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500,
            1e3, 2e3, 5e3, 1e4, 2e4, 5e4]

for fact in factores:
    step = anchos/fact
    Hint = ndf.Hessian(bl.errorCuadraticoInt, step=step)
    hes1step = Hint(Xint, Ns, XextList, params)

    autovalores.append(np.real(ln.eig(hes1step)[0]))
    print(fact, autovalores[-1])

autovalores = np.array(autovalores)

for i,aut in enumerate(autovalores.T):
    plt.figure(i)
    plt.plot(factores, aut, '-*')

# %% estos son los rangos de step dond eveo que hay estabilidad
# o algo razonable
rangoNunDiffStep = np.array([[1e-3, 1e-1],    # 0
                             [1e-5, 1e-1],  # 1
                             [1e-3, 1e-1],   # 2
                             [1e-3, 1e-1],   # 3
                             [1e-3, 1e-1],     # 4
                             [1e-3, 1e-1],      # 5
                             [1e-3, 1e-1],      # 6
                             [1e-3, 1e-1]]).T * anchos

nLogSteps = 100  # cantidad de pasos en los que samplear el diff step
logFactor = (rangoNunDiffStep[1] / rangoNunDiffStep[0])**(1 / (nLogSteps-1))

# calculate list of steps
diffSteps3 = (rangoNunDiffStep[0].reshape((1,-1))
             * np.cumproduct([logFactor]*nLogSteps, axis=0)
             / logFactor)

autovalores3 = np.zeros_like(diffSteps3)

for i in range(nLogSteps):
    print('paso', i)

    Hint = ndf.Hessian(bl.errorCuadraticoInt, step=diffSteps3[i])
    hes1step = Hint(Xint, Ns, XextList, params)

    autovalores3[i] = np.real(ln.eig(hes1step)[0])
    print('autovalores', autovalores3[i])


# %%
'''
falta explorar un poco mas para algunos steps, haciendolo mas chico.
se nota que no es facil determinar el paso optimo, es bastante complicado
y quiza haya que terminar haciendo algo iterativo tipo metropolis.
comparar con lo que de el hessiano de la hiperparabola. si da parecido ya
está.
'''

autovTotal = np.concatenate([autovalores[:73],
                             autovalores2,
                             autovalores3], axis=0)

diffStepsTotal =  np.concatenate([diffSteps[:73],
                             diffSteps2,
                             diffSteps3], axis=0)


np.save('autovaloresHess', autovTotal)
np.save('diffSteps', diffStepsTotal)

diffStepsTotal = np.load('diffSteps.npy')
autovTotal = np.load('autovaloresHess.npy')

for j in range(8):
    plt.figure()
    plt.title(j)
    plt.plot(diffStepsTotal[:,j], autovTotal[:,j], '-+')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid('on')

'''
en base a los graficos defino los steps con los que sacar el hessiano numerico
'''
stepsGraph = [0.3, 0.7, 2.5, 3.0, 1e-3, 4e-4, 4e-6, 3e-6]


Hint = ndf.Hessian(bl.errorCuadraticoInt, step=stepsGraph)
hesGraphStep = Hint(Xint, Ns, XextList, params)

np.real(ln.eig(hesGraphStep)[0])













