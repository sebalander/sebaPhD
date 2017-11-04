
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
#import glob
import numpy as np
from calibration import calibrator as cl
from numpy import any as anny
from scipy.stats import chi2
from scipy.linalg import inv
import numdifftools as ndf

from multiprocess import Process, Queue

modelos = ['poly', 'rational', 'fisheye', 'stereographic']

# https://github.com/uqfoundation/multiprocess/tree/master/py3.6/examples

# %% funcion error
# MAP TO HOMOGENOUS PLANE TO GET RADIUS

def int2flat(cameraMatrix, distCoeffs, model):
    '''
    parametros intrinsecos concatenados como un solo vector
    '''
    
    if model==modelos[3]:
        # stereographic case is special
        kFlat = cameraMatrix[[0,1],2]
        dFlat = np.reshape(distCoeffs, -1)
        
    else:
        kFlat = cameraMatrix[[0,1,0,1],[0,1,2,2]]
        dFlat = np.reshape(distCoeffs, -1)
    
    X = np.concatenate((kFlat, dFlat))
    Ns = np.array([len(kFlat), len(dFlat)])
    Ns = np.cumsum(Ns)

    return X, Ns


def flat2CamMatrix(kFlat, model):
    cameraMatrix = np.eye(3, dtype=float)
    if model==modelos[3]:
        cameraMatrix[[0,1],2] = kFlat
    else:
        cameraMatrix[[0,1,0,1],[0,1,2,2]] = kFlat
    return cameraMatrix


def flat2int(X, Ns, model):
    '''
    hace lo inverso de int2flat
    '''
    kFlat = X[0:Ns[0]]
    dFlat = X[Ns[0]:Ns[1]]

    cameraMatrix = flat2CamMatrix(kFlat, model)
    distCoeffs = dFlat

    return cameraMatrix, distCoeffs


def ext2flat(rVec, tVec):
    '''
    toma un par rvec, tvec y devuelve uno solo concatenado
    '''
    rFlat = np.reshape(rVec, -1)
    tFlat = np.reshape(tVec, -1)

    X = np.concatenate((rFlat, tFlat))

    return X
def flat2ext(X):
    '''
    hace lo inverso de ext2flat
    '''
    rFlat = X[0:3]
    tFlat = X[3:]

    rVecs = np.reshape(rFlat, 3)
    tVecs = np.reshape(tFlat, 3)

    return rVecs, tVecs

# %%

def errorCuadraticoImagen(Xext, Xint, Ns, params, j, mahDist=False):
    '''
    el error asociado a una sola imagen, es para un par rvec, tvec
    necesita tambien los paramatros intrinsecos

    if mahDist=True it returns the squared mahalanobis distance for the
    proyection points
    '''
    # saco los parametros de flat para que los use la func de projection
    cameraMatrix, distCoeffs = flat2int(Xint, Ns, params['model'])
    rvec, tvec = flat2ext(Xext)
    # saco los parametros auxiliares
    #n = params["n"]
    # m = params["m"]
    imagePoints = params["imagePoints"]
    model = params["model"]
    chessboardModel = params["chessboardModel"]
    Cccd = params["Cccd"]
    Cf = params["Cf"]
    Ck = params["Ck"]
    Crt = params["Crt"]
    
    try: # check if there is covariance for this image
        Cov = Cccd[j]
    except:
        Cov = None

    # hago la proyeccion
    xm, ym, Cm = cl.inverse(imagePoints[j,0], rvec, tvec, cameraMatrix,
                            distCoeffs, model, Cov, Cf, Ck, Crt)

    # error
    er = ([xm, ym] - chessboardModel[0,:,:2].T).T
    
    Cmbool = anny(Cm)
    if Cmbool:
        # devuelvo error cuadratico pesado por las covarianzas
        S = np.linalg.inv(Cm)  # inverse of covariance matrix
        # distancia mahalanobis
        Er = np.sum(er.reshape((-1, 2, 1))
                    * S
                    * er.reshape((-1, 1, 2)),
                    axis=(1, 2))

        if not mahDist:
            # add covariance normalisation error
            Er += np.linalg.det(Cm)
    else:
        # error cuadratico sin pesos ni nada
        Er = np.sum(er**2, axis=1)

    return Er


def errorCuadraticoInt(Xint, Ns, XextList, params, mahDist=False):
    '''
    el error asociado a todas la imagenes, es para optimizar respecto a los
    parametros intrinsecos
    '''
    # error
    Er = list()

    for j in range(len(XextList)):
        # print(j)
        Er.append(errorCuadraticoImagen(XextList[j], Xint,Ns, params,
                                        j, mahDist=mahDist))

    return np.concatenate(Er)


def etotalInt(Xint, Ns, XextList, params):
    '''
    calcula el error total como la suma de los errore de cada punto en cada
    imagen
    '''
    return errorCuadraticoInt(Xint, Ns, XextList, params).sum()



# %% metropolis hastings
class metHas:
    
    def __init__(self, Ns, XextList, params, sampleador):
        self.Ns = Ns
        self.XextList = XextList
        self.params = params
        
        self.generados = 0
        self.gradPos = 0
        self.gradNeg = 0
        self.mismo = 0
        
        self.sampleador = sampleador
    
    def nuevo(self, old, oldE):
        # genero nuevo
        new = self.sampleador.rvs() # rn(8) * intervalo + cotas[:,0]
        self.generados += 1
    
        # cambio de error
        newE = etotalInt(new, self.Ns, self.XextList, self.params)
        deltaE = newE - oldE
    
        if deltaE < 0:
            self.gradPos += 1
            print("Gradiente Positivo")
            print(self.generados, self.gradPos, self.gradNeg, self.mismo)
            return new, newE # tiene menor error, listo
        else:
            # nueva opoertunidad, sampleo contra rand
            pb = np.exp(- deltaE / 2)
    
            if pb > rn():
                self.gradNeg += 1
                print("Gradiente Negativo, pb=", pb)
                print(self.generados, self.gradPos, self.gradNeg, self.mismo)
                return new, newE # aceptado a la segunda oportunidad
            else:
    #            # vuelvo recursivamente al paso2 hasta aceptar
    #            print('rechazado, pb=', pb)
    #            new, newE = nuevo(old, oldE)
                self.mismo +=1
                print("Mismo punto,        pb=", pb)
                print(self.generados, self.gradPos, self.gradNeg, self.mismo)
                return old, oldE
    
        return new, newE


# %% funciones para calcular jacobiano y hessiano in y externo
Jint = ndf.Jacobian(errorCuadraticoInt)  # (Ns,)
Hint = ndf.Hessian(errorCuadraticoInt)  #  (Ns, Ns)
Jext = ndf.Jacobian(errorCuadraticoImagen)  # (6,)
Hext = ndf.Hessian(errorCuadraticoImagen)  # (6,6)

# una funcion para cada hilo
def procJint(Xint, Ns, XextList, params, ret):
    ret.put(Jint(Xint, Ns, XextList, params))

def procHint(Xint, Ns, XextList, params, ret):
    ret.put(Hint(Xint, Ns, XextList, params))

def procJext(Xext, Xint, Ns, params, j, ret):
    ret.put(Jext(Xext, Xint, Ns, params, j))

def procHext(Xext, Xint, Ns, params, j, ret):
    ret.put(Hext(Xext, Xint, Ns, params, j))


# %%
def jacobianos(Xint, Ns, XextList, params, hessianos=True):
    '''
    funcion que calcula los jacobianos y hessianos de las variables intrinsecas
    y extrinsecas. hace un hilo para cada cuenta
    '''
    # donde guardar resultado de derivadas de params internos
    jInt = Queue()
    if hessianos:
        hInt = Queue()

    # creo e inicializo los threads
    if hessianos:
        # print('cuentas intrinsecas, 2 processos')
        pHInt = Process(target=procHint, args=(Xint, Ns, XextList,
                                               params, hInt))
        pHInt.start()
    #else:
        # print('cuentas intrinsecas, 1 processo')

    pJInt = Process(target=procJint, args=(Xint, Ns, XextList, params, jInt))

    pJInt.start()  # inicio procesos

    # donde guardar resultados de jaco y hess externos
    n = len(XextList)
    jExt = np.zeros((n, 1, 6), dtype=float)
    qJext = [Queue() for nn in range(n)]

    if hessianos:
        hExt = np.zeros((n, 6, 6), dtype=float)
        qHext = [Queue() for nn in range(n)]

    # lista de threads
    proJ = list()
    if hessianos:
        proH = list()

    # creo e inicializo los threads
    for i in range(n):
        # print('starting par de processos ', i + 3)
        pJ = Process(target=procJext, args=(XextList[i], Xint, Ns,
                                            params, i, qJext[i]))
        proJ.append(pJ)

        if hessianos:
            pH = Process(target=procHext, args=(XextList[i], Xint, Ns,
                                                params, i, qHext[i]))
            proH.append(pH)

        pJ.start()  # inicio procesos
        if hessianos:
            pH.start()

    jInt = jInt.get()  # saco los resultados
    if hessianos:
        hInt = hInt.get()

    for i in range(n):
        jExt[i] = qJext[i].get()  # guardo resultados
        if hessianos:
            hExt[i] = qHext[i].get()


    pJInt.join()  # espero a que todos terminen

    if hessianos:
        pHInt.join()

    [p.join() for p in proJ]

    if hessianos:
        [p.join() for p in proH]

    if hessianos:
        return jInt, hInt, jExt, hExt
    else:
        return jInt, jExt

# %% calculo de varianza de una matriz de covarianza de MC
#PvecA = np.array([[1,0,0,0],
#                  [0,0,1,0],
#                  [0,1,0,0],
#                  [0,0,0,1]])
#
#const = (PvecA + np.eye(4))
# I matrix + transposition permutation matrix
const2 = np.array([[2, 0, 0, 0],
                  [0, 1, 1, 0],
                  [0, 1, 1, 0],
                  [0, 0, 0, 2]])

def varVar2(c, N):
    '''
    para c de 2x2. calculates variance matrix 4x4 asuming wishart distribution
    https://www.statlect.com/probability-distributions/wishart-distribution
    '''
    cKron = np.kron(c,c)

    return const2.dot(cKron) / N




def varMahal(c1, n, c2, rank=False):
    '''
    calculate mahalanobis distance between two matrices, taking the first one
    as reference (order is important)
    if rank enabled, also returns the accumulated probability up to that
    mahalanobis distance taking into account 3 degrees of freedom
    '''
    # se elimina la fina y columna redundante porque no aportan nada
    c1Var = varVar2(c1, n)[[0,1,3]].T[[0,1,3]].T
    c1Pres = inv(c1Var) # precision matrix

    c1flat = c1[[0,0,1],[0,1,1]]
    c2flat = c2[[0,0,1],[0,1,1]]

    cFlatDif = c1flat - c2flat

    mahDist = cFlatDif.dot(c1Pres).dot(cFlatDif)

    if rank:
        ranking = chi2.cdf(mahDist, 3)
        return mahDist, ranking
    else:
        return mahDist


def trasPerMAt(k,l):
    '''
    returns the trasposition permutation matrix of matrix A of size (k, l)
    https://www.statlect.com/probability-distributions/wishart-distribution#refMuirhead
    '''
    n = k*l 
    
    vecA = np.arange(n, dtype=int)
    A = vecA.reshape((k,l))
    vecAT = A.T.reshape(-1)
    
    mat = np.zeros((n,n), dtype=int)
    
    mat[[vecA],[vecAT]] = 1
    
    return mat



def varVarN(c, Nsamples):
    '''
    returns the variance of a covariance matriz of size NxN calculated with 
    Nsamples samples
    '''
    cKron = np.kron(c,c)
    n = c.shape[0]
    
    const = np.eye(n**2) + trasPerMAt(n, n)
    
    return const.dot(cKron) / Nsamples












