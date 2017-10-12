#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 19:01:13 2017

@author: sebalander
"""
# %%
import numpy as np

# %%
N = int(5e1)  # number of data, degrees of freedom
M = int(1e5)  # number of realisations
P = 2  # size of matrices


mu = np.array([7, 10])  # mean of data
c = np.array([[5, 3], [3, 7]])

# generate data
x = np.random.multivariate_normal(mu, c, (N, M))

# estimo los mu y c de cada relizacion
muest = np.mean(x, axis=0)
dif = x - muest
cest = dif.reshape((N,M,P,1)) * dif.reshape((N,M,1,P))
cest = np.sum(cest, axis=0) / (N - 1)

# saco la media y varianza entre todas las realizaciones
muExp = np.mean(muest, axis=0)
difmu = muest - muExp
muVar = np.sum(difmu.reshape((M,P,1)) * difmu.reshape((M,1,P)), axis=0) / (M - 1)

cExp = np.mean(cest, axis=0)
difc = cest - cExp

difcKron = np.array([np.kron(cc,cc) for cc  in difc])
cVarKron = np.sum(difcKron  / (M - 1), axis=0)

difc = difc.reshape((M,P*P,1))
cVarAux = np.reshape(difc * difc.transpose((0,2,1)), (M,P,P,P,P))
cVar = np.sum(cVarAux  / (M - 1), axis=0)  # VARIANZA NUMERICA


# saco las cuentas analiticas
expMu = mu
VarMu = c / N
expC = c
# no es necesario trasponer porque c es simetrica
cShaped = c.reshape((P,P,1,1))
VarCnn = cShaped * cShaped.transpose((3,2,1,0))
nn = (2 * N - 1) / (N - 1)**2

VarC = VarCnn * nn  ## VARIANZA TEORICA SEGUN SEBA
cVarnn = cVar / nn

print('media de media numerico: \n', muExp,
      '\n\n media de media analitico \n', expMu)

print('\n\n varianza de media numerico: \n', muVar,
      '\n\n varianza de media analitico \n', VarMu)

print('\n\n media de varianza numerico: \n', cExp,
      '\n\n media de varianza analitico \n', expC)

print('\n\n varianza de varianza numerico (sin normalizar):\n', cVarnn,
      '\n\n varianza de varianza analitico (sin normalizar)\n', VarCnn)


cKron = np.kron(c,c)

cVarnn
VarCnn
cKron.reshape((P,P,P,P))



np.reshape(c,(P*P,1), order='C') * np.reshape(c,(1, P*P), order='C')
cKron
cVarKron / nn
'''
no entiendo donde esta el problema
'''
#reldif = np.abs((cVar - VarC) / VarC)
#reldif > 1e-1

# %% re calculo las covarianzas pero con tamaños 4x4
# numerica
cVar2 = np.sum(difc * difc.transpose((0,2,1)) / (M - 1), axis=0)
# teorica
VarC2 = c.reshape((-1,1)) * c.reshape((1,-1)) * (2 * N - 1) / (N - 1)**2


# %% pruebo wishart
#  probabilidad de la covarianza estimada
import scipy.stats as sts
import scipy.special as spe

# N degrees of freedom
# inv(c) is precision matrix
wishRV = sts.wishart(N, c)
wpdf = wishRV.pdf
wishPDFofC = wpdf(c * (N-1))  # la scatter matrix de mayor probabilidad
eC = - np.log(wishPDFofC)
wpdf(wishRV.rvs())

[wpdf(m) for m in cest]


def wishConst(P, N):
    '''
    multivariate gamma
    '''
    aux1 = np.pi**(P * (P - 1) / 4)
    aux2 = spe.gamma( (N - np.arange(P)) / 2)
    
    multivariateGamma = aux1 * np.prod(aux2)
    return 1 / (multivariateGamma * 2**(N*P/2))

# constantes para calular la wishart pdf
wishartConstant = wishConst(P, N)
expW = (N - P - 1) / 2
expV = N / 2

def wishartPDF(W, V):
    '''
    implementation of wishart pdf as per muravchik notes and wikipedia
    '''
    detV = np.linalg.det(V)
    detW = np.linalg.det(W)
    exponent =  np.linalg.inv(V).dot(W)
    exponent =  - np.trace(exponent) / 2
    return wishartConstant * detW**expW * np.exp(exponent) / detV**expV



wishartPDF(cest[4] * (N - 1), c)
wpdf(cest[4] * (N - 1))


[wishartPDF(mat * (N - 1), c) for mat in cest[:10]]
[wpdf(mat * (N - 1)) for mat in cest[:10]]
wpdf(cest[:10].transpose((1,2,0)) * (N - 1))
'''
las dos dan igual asi que la libreria de scipy esta evaluando la pdf de la
matriz como corresponde.
ok, tengo una manera de medir que tan bien dan las covarianzas
'''

wishartPDFs = wpdf(cest.transpose((1,2,0)) * (N - 1))





# %% como llevar esto a una metrica
# cantidad de "grados de libertad" de una matriz de covarianza
# calculate error without including dims
# dimsErroConst = - np.log(np.pi * 2) - np.log(wishPDFofC)
dims = - np.log(wishPDFofC) / np.log(np.pi * 2) # 8.277  # P * (P + 1) / 2 
## elijo los grados de libertad tal que c tenga error cero, el minimo
#dims = (4 * P
#        - 2 * np.log(wishartConstant)
#        + (P+1) * np.log(np.linalg.det(c))) / np.log(np.pi * 2)

norLog =  dims * np.log(np.pi * 2)  # constante de la gaussiana
def E2(gaussPDF):
    return - np.log(gaussPDF) - norLog

e2 = E2(wishartPDFs)
import matplotlib.pyplot as plt
plt.hist(e2, 100)
er = np.sqrt(e2)
print(np.min(E), np.sqrt(E2(wishPDFofC)))

# %% covariance matrix
# https://www.statlect.com/probability-distributions/wishart-distribution
cKron = np.kron(c,c)


PvecA = np.array([[1,0,0,0],
                  [0,0,1,0],
                  [0,1,0,0],
                  [0,0,0,1]]) # como c es simetrica vec(c)=vec(c')

## testear PvecA
#A = np.array([[1,2],[3,4]])
#PvecA.dot(np.reshape(A,-1))
#np.reshape(A.T,-1)

const = (PvecA + np.eye(P*P)) / N

Var = const.dot(cKron)

print('teorico de wishart\n', Var)
print('numerico\n', cVar)
print('teorico seba\n', VarC)


print('\n\nteorico de wishart2\n', Var)
print('numerico2\n', cVar2)
print('teorico seba2\n', VarC2)

# %%
'''
quedo demostrado que la expected variance segun lateoria de wishart es la que
coincide con las simulaciones (se puede rastrear la diferencia con lo propuesto
por mi pero no vale la pena).

ahora ver como sacar una distancia mahalanobis
'''
import scipy.linalg as ln

PvecA = np.array([[1,0,0,0],
                  [0,0,1,0],
                  [0,1,0,0],
                  [0,0,0,1]]) # como c es simetrica vec(c)=vec(c')

const = (PvecA + np.eye(P*P))

def varVar(c, N):
    '''
    para c de 2x2
    '''
    cKron = np.kron(c,c)
    
    return const.dot(cKron) / N

# tomo dos de las matrices estimadas
i = 1234
j = 100

ci = cest[i]
cj = cest[j]


def varMahal(c1, n, c2, rank=False):
    '''
    calculate mahalanobis distance between two matrices, taking the first one
    as reference (order is important)
    if rank enabled, also returns the accumulated probability up to that 
    mahalanobis distance taking into account 3 degrees of freedom
    '''
    # se elimina la fina y columna redundante porque no aportan nada
    c1Var = varVar(c1, n)[[0,1,3]].T[[0,1,3]].T
    c1Pres = np.linalg.inv(c1Var) # precision matrix
    
    c1flat = c1[[0,0,1],[0,1,1]]
    c2flat = c2[[0,0,1],[0,1,1]]
    
    cFlatDif = c1flat - c2flat
    
    mahDist = cFlatDif.dot(c1Pres).dot(cFlatDif)
    
    if rank:
        ranking = sts.chi2.cdf(mahDist, 3)
        return mahDist, ranking
    else:
        return mahDist

# elimino uno de los componentes que es redundante # y lo multiplico
# mul = np.array([[1],[2],[1]])
ind = [0,1,3]
ciVar = varVar(ci, N)[ind] # * mul
cjVar = varVar(cj, N)[ind] # * mul
ciVar = ciVar.T[ind].T
cjVar = cjVar.T[ind].T

ciFlat = np.reshape(ci, -1)[ind]
cjFlat = np.reshape(cj, -1)[ind]
cFlatDif = ciFlat - cjFlat

A =  varVar(ci, N)
ln.svd(A)


ciPres = np.linalg.inv(ciVar)
cjPres = np.linalg.inv(cjVar)

dMahi = cFlatDif.dot(ciPres).dot(cFlatDif)
dMahj = cFlatDif.dot(cjPres).dot(cFlatDif)

varMahal(ci, N, cj)
varMahal(cj, N, ci)

from dev import bayesLib as bl
bl.varMahal(ci, N, cj, rank=True)

# intuyo que la manera de juntar las dos covarianzas es sumar las distancias al cuadrado:

dM = np.sqrt(dMahi + dMahj)
# on, poque en realidad voy a tener solo varianza asociada auna de las matrices
# la otra es teorica y no sale de MC. en todo caso habría que saber que error
# tiene acotando el error de taylosr

# %% testeo las distancias mahalanobis de las esperanzas wrt lo estimado

difMU = muest - mu # la varianza de esto tiene que ser VarMu
difC = cest - c # la varianza de esto tiene que ser Var
difC2 = difC[:,[0,0,1],[0,1,1]]

presMU = ln.inv(VarMu)
Var2 = Var[[0,1,3]].T[[0,1,3]].T # saco las dimensiones irrelevantes
presC = ln.inv(Var2)

mahMU = difMU.reshape((-1,2,1)) * presMU.reshape((1,2,2)) * difMU.reshape((-1,1,2))
mahMU = mahMU.sum(axis=(1,2))

mahC = difC2.reshape((-1,3,1)) * presC.reshape((1,3,3)) * difC2.reshape((-1,1,3))
mahC = mahC.sum(axis=(1,2))

rankings = sts.chi2.cdf(mahC,3)

# grafico y comparo con chi cuadrado
plt.figure()
nMU, binsMU, patchesMU = plt.hist(mahMU, 1000, normed=True)
chi2MU = sts.chi2.pdf(binsMU,2)
plt.plot(binsMU, chi2MU, label='chi2, df=2')

plt.figure()
nC, binsC, patchesC = plt.hist(mahC, 1000, normed=True)
chi2C = sts.chi2.pdf(binsC,3)
plt.plot(binsC, chi2C, label='chi2, df=3')

'''
como sigue la pdf chi cuadrado parece que esta todo bien, asi que la distancia
de mahalanobis es un buen indicador de la distancia.

falta poner loq ue vendría a ser el sigma segun los grados de libertad. para
cada distancia de mahalanobis pouedo calcular en que elipse esta, o sea indicar
"el volumen de la elipse" donde esta, mientras mas chico mejor.
'''

muEllVol = sts.chi2.cdf(mahMU, 2)
cEllVol = sts.chi2.cdf(mahC, 3)

plt.figure()
plt.plot(mahMU, muEllVol, '.', label='mu')
plt.plot(mahC, cEllVol, '.', label='C')
plt.xlabel('distancia mahalanobis al cuadrado')
plt.ylabel('acumulado de probabiliad')