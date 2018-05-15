#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 11:14:37 2018

aca tiro todo el codigo de intrinsicFullPyMC3.py que ya casi que ahi no sirve

@author: sebalander
"""


# %%


pathFiles = "/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/"
pathFiles += "extraDataSebaPhD/traces" + str(28)

trace = dict()

trace['xIn'] = np.load(pathFiles + "Int.npy")
trace['xEx'] = np.load(pathFiles + "Ext.npy")

# %% concateno y calculo la diferencia

## %% calculo los estadísticos para analizar ocnvergencia
## medias de cada cadena
#traceIn = trace['xIn'].reshape((-1, nDraws, NintrParams)).transpose((2,0,1))
#traceEx = trace['xEx'].reshape((-1, nDraws, n, 6)).transpose((2,3,0,1))
#
#inMeanChain = np.mean(traceIn, axis=-1)
#exMeanChain = np.mean(traceEx, axis=-1)
#
#inDifChain = (traceIn.T - inMeanChain.T).T
#exDifChain = (traceEx.T - exMeanChain.T).T
#
#inCovChain = np.mean(inDifChain**2, axis=-1)
#exCovChain = np.mean(exDifChain**2, axis=-1)
#
#nDrawsSqrt = np.sqrt(nDraws)
#
## calculo la diferencia mutua de la media
#inMeanChainDiff = (inMeanChain.reshape((NintrParams, -1, 1)) -
#                   inMeanChain.reshape((NintrParams, 1, -1)))
#inMeanChainDiff *= nDrawsSqrt / np.sqrt(inCovChain).reshape((NintrParams, -1,1))



# %%

corner.corner(trace['xIn'])
corner.corner(trace['xEx'][:,2])


# %%
plt.figure()
plt.plot(trace['xIn'][:,2].reshape((nChains, nDraws)).T)
plt.plot(concDat[:,2].reshape((nChains, nDraws)).T + 1)

# %%

# indices para elegir que traza extrínseca
a = 2
b = 1

trazaSelect = trace['xEx'][:,a,b].reshape((nChains, -1)).T
trazaMean = np.mean(trazaSelect, axis=1)
trazaStd = np.sqrt(np.mean((trazaSelect.T - trazaMean)**2, axis=0))
trazaMeanAll = np.mean(trazaMean)
trazaStdAll = np.sqrt(np.mean((trazaSelect.T - trazaMeanAll)**2))
trazaStdMax = trazaMeanAll + trazaStdAll
trazaStdMin = trazaMeanAll - trazaStdAll


plt.figure()
plt.plot(trazaSelect, linewidth=2)
plt.plot(trazaMean, 'k-', linewidth=3)
plt.fill_between(range(nDraws), trazaMean - trazaStd, trazaMean + trazaStd,
                 facecolor='black', alpha=0.3)
plt.fill_between([0, nDraws - 1], [trazaStdMax, trazaStdMax],
                 [trazaStdMin, trazaStdMin], facecolor='green', alpha=0.2)


# %% select to prune burn in
leChain = trace['xEx'].shape[0] / nChains
indexCut = 50000
noBurnInIndexes = np.arange(indexCut, leChain, dtype=int)
noBurnInIndexes = np.array(noBurnInIndexes.reshape((1,-1)) +
                           (np.arange(nChains) * leChain).reshape((-1,1)),
                           dtype=int)
noBurnInIndexes = np.concatenate(noBurnInIndexes)

trace['xIn'] = trace['xIn'][noBurnInIndexes]
trace['xEx'] = trace['xEx'][noBurnInIndexes]

plt.figure()
plt.plot(trace['xEx'][:,1,0].reshape((-1, int(leChain - indexCut))).T)

corner.corner(trace['xIn'])
corner.corner(trace['xEx'][:,4])


inTraceMean = np.mean(trace['xIn'], axis=0)
inTraceCov = np.cov(trace['xIn'].T)

exTraceMean = np.mean(trace['xEx'], axis=0)
exTraceDif = trace['xEx'] - exTraceMean
exTraceCov = np.mean(exTraceDif.reshape((-1, nIm, 6, 1)) *
                     exTraceDif.reshape((-1, nIm, 1, 6)),
                     axis=0)

# saco las desviaciones estandar
SinTrace = np.sqrt(np.diag(inTraceCov))
SexTrace = np.sqrt(exTraceCov[:, range(6), range(6)])



























# %%
plt.figure()
Smin = np.min(trace['xIn'], axis=0)
plt.plot(np.abs(trace['xIn'] - Smin), 'x-', markersize=1)

plt.matshow(inCorr, cmap='coolwarm', vmin=-1, vmax=1)



# %%
plt.figure()
#ax = fig.gca()
ticksCovMatrix = np.arange(NintrParams, NfreeParams, 6) - 0.5
plt.grid('on')
plt.xticks(ticksCovMatrix)
plt.yticks(ticksCovMatrix)
plt.imshow(exCorr, cmap='coolwarm', vmin=-1, vmax=1)


# %%
a, b = [4, 5]
tira = 2*nDraws
plt.figure()
for i in range(nChains):
    plt.plot(trace['xEx'][i*tira:(i+1)*tira, 0, a],
             trace['xEx'][i*tira:(i+1)*tira, 0, b])
for i in range(nChains):
    plt.plot(trace['xEx'][i*tira, 0, a],
             trace['xEx'][i*tira, 0, b], 'o', markersize=10)



# %%
corner.corner(trace['xEx'][:, 4])

# par graficar histogramas de los parametros extrisnecos
difEx = trace['xEx'] - exMean
normedDifEx = difEx / Sex.reshape((n, 6))

paramInd = 0
plt.figure()
for i in range(n):
    plt.hist(normedDifEx[:,i,paramInd], 50, normed=True)

plt.figure()
plt.hist(normedDifEx[:,:,paramInd].reshape(-1), 50, normed=True)






# %% saco un histograma de la distancia desplazada en cada paso
stepsIn = np.linalg.norm(np.diff(trace['xIn'], axis=0) / Sin, axis=1)

# saco los ceros
stepsIn = stepsIn[stepsIn != 0]
stepsIn = stepsIn[stepsIn < 1]

plt.figure()
plt.hist(stepsIn,50)
print(radiusStepsNdim(NintrParams))
print(np.mean(stepsIn), np.std(stepsIn))

# %%
stepsEx = np.linalg.norm(np.diff(trace['xEx'], axis=0) / Sex.reshape((n,-1)), axis=2)

stepsEx = stepsEx[stepsEx.all(axis=1)]

plt.figure()
plt.hist(stepsEx.reshape(-1),100)

plt.figure()
plt.plot(stepsEx[:,1])

print(radiusStepsNdim(6))
print(np.mean(stepsIn), np.std(stepsIn))

# %% pruebo la estadistica de paso promedio en muchas dimensiones
ndimshit = 100
randomshit = np.random.randn(ndimshit,10000)
rads = np.linalg.norm(randomshit, axis=0)

plt.figure()
plt.hist(rads,100)
radiusStepsNdim(ndimshit)



# %% saco media y varianza
inMean = np.mean(trace['xIn'], axis=0)
inCov = np.cov(trace['xIn'].T)
Sin = np.sqrt(np.diag(inCov))
inCorr = inCov / (Sin.reshape((-1, 1)) * Sin.reshape((1, -1)))

exMean = np.mean(trace['xEx'], axis=0)
exCov = np.cov(trace['xEx'].reshape(-1, n*6).T)
Sex = np.sqrt(np.diag(exCov)).reshape((n,6))
exCorr = exCov / (Sex.reshape((-1, 1)) * Sex.reshape((1, -1)))


# %% testeo sobre la imagen

cameraMatrix, distCoeffs = bl.flat2int(inMean, Ns, model)

Cint = intrCalibResults['inCov']
Cf = np.zeros((4,4))
Cf[2:,2:] = inCov[:2,:2]
Ck = inCov[2, 2]
Cfk = inCov[:2, 2]

xy = np.zeros((n, m, 2))
cm = np.zeros((n, m, 2, 2))

Xext = exMean.reshape((n,6))

for j in range(n):
    rVec, tVec = bl.flat2ext(Xext[j])
    xy[j, :, 0], xy[j, :, 1], cm[j] = cl.inverse(
            imagePoints.reshape((n, m, 2))[j], rVec, tVec,
            cameraMatrix, distCoeffs, model, Ci[j], Cf, Ck, Crt=False, Cfk=Cfk)


plt.figure()
plt.scatter(xy[:,:,0], xy[:,:,1])

fig = plt.figure()
ax = fig.gca()
C = cm.reshape(-1,2,2)
mux = xy[:, :, 0].reshape(-1)
muy = xy[:, :, 1].reshape(-1)
col = 'k'
cl.plotPointsUncert(ax, C, mux, muy, col)

# %% guardo los resultdos de la calibracion intrinseca
intrCalibResults = dict()

intrCalibResults['model'] = model
intrCalibResults['inMean'] = inMean
intrCalibResults['exMean'] = exMean
intrCalibResults['inCov'] = inCov
intrCalibResults['exCov'] = exCov
intrCalibResults['nChains'] = nChains
intrCalibResults['chainLength'] = np.int(trace['xEx'].shape[0] / nChains)
intrCalibResults['filenameStem'] = pathFiles
intrCalibResults['help'] = '''
este diccionario tiene los resultados de la calibracion intrinseca
inMean : promedio de los puntos sampleados con Metropolis
exMean : promedio de los puntos sampleados, parametros extrinsecos
inCov : matriz de covarianza de los parametros intrinsecos
exCov : covarianza de los parametros extrinsecos
nChains : cantidad decadenas de sampleo Metropolis
chainLength : longitud de las cadenas de markov
pathFiles : nombre del archivo con las muestras, agregar Int.py o Ext.py
'''

np.save(pathFiles + "IntrCalibResults", intrCalibResults)


# %%



















































# %%
#
#
#plt.figure()
#plt.plot(Sex, 'r')
#plt.plot(Sex / np.abs(exMean.reshape(-1)), 'b')
#
## trunco para que no sea demasiado chico que no puede ser
#Sin = np.max([Sin, np.abs(inMean.reshape(-1)) * 1e-6], axis=0)
#Sex = np.max([Sex, np.abs(exMean.reshape(-1)) * 1e-6], axis=0)
#
#plt.figure()
#plt.plot(np.abs(inMean), Sin, '+')
#plt.plot(np.abs(exMean), Sex.reshape((n,6)), '*')
#
#
#stepsDist = np.linalg.norm(np.diff(trace['xEx'][1:]),axis=(1,2))
#plt.figure()
#
#
#plt.figure()
#for i in range(nChains):
#    plt.hist(stepsDist[i*2*nDraws:(i+1)*2*nDraws],30)

# %%
# indexes to read diagona 2x2 blocks of a matrix
nTot = 6 * n
irange = np.arange(0, nTot, 6, dtype=int)
jrange = np.array([range(6)] * 6, dtype=int)

xInxs = irange.reshape((-1,1,1)) + jrange.reshape((1,6,6))
yInxs = np.transpose(xInxs, (0,2,1))

#xInxs = [[[i, i], [i+1, i+1]] for i in range(0, nTot, 6)]
#yInxs = [[[i, i+1], [i, i+1]] for i in range(0, nTot, 6)]

# extraigo los bloques de 6x6 de cada pose
exCovBloks = exCov[xInxs, yInxs]

## chequeo que la lectura en bloques da bien
#i = 0
#j0 = 6*i
#j1 = j0+6
#print(exCovBloks[i] == exCov[j0:j1, j0:j1])

corner.corner(trace['xEx'][:,1])
exCovBloks[1]

u, s, v = np.linalg.svd(exCovBloks[1])

#l, v2 = np.linalg.eig(exCovBloks[1])

# tomo el sing val mas chico
s[-1] * u[:,-1]
# multiplico por el autovector correspondiente

# %% pruebo grficar las elipses para ciertas correlaciones, para pensar mejor
# como proponer la escala caracteristica de cada parametro
sabc = np.array([3, 5, 7], dtype=float)
rabc = np.array([-0.81, 0.1, 0.5])

fakeCov = sabc.reshape((-1,1)) * sabc.reshape((1,-1))

fakeCov[np.tril_indices_from(fakeCov,k=-1) ] *= rabc
fakeCov[np.triu_indices_from(fakeCov,k=1) ] *= rabc

fakeDist = pm.MvNormal.dist(mu=[0,0,0], cov=fakeCov)
fakeX = fakeDist.random(size=100000)

corner.corner(fakeX)

fakeCovInv = np.linalg.inv(fakeCov)
fakeCovInvdiagsqrt = np.sqrt(np.diag(fakeCovInv))

fakeCov
np.sqrt(np.abs(fakeCovInv))
fakeCovInvdiagsqrt

fakeCovEstim = np.cov(fakeX.T)
Sfake = np.sqrt(np.diag(fakeCovEstim))
1 / Sfake

# %%



























































# %%
#Smean = np.mean(trace['xAll'], axis=0)
#Scov = np.cov(trace['xAll'].T)

Smean = np.load(pathFiles + "Smean.npy")
Scov = np.load(pathFiles + "Scov.npy")



#Smean = np.mean(traceConc, axis=0)
#Scov = np.cov(traceConc.T)

sigmas = np.sqrt(np.diag(Scov))
Scorr = Scov / ( sigmas.reshape((-1,1)) * sigmas.reshape((1,-1)))
plt.matshow(Scorr, cmap='coolwarm')


# saco el promedio de las matrices de covarianza de los bloques 6x6 extrinsecos
xBlk = [[0, 1, 2, 3, 4, 5]] * 6
xBlkAll = np.arange(NintrParams,NfreeParams,6).reshape((-1,1,1)) + [xBlk]*n
yBlkAll = xBlkAll.transpose((0,2,1))
extrS = np.mean(Scov[xBlkAll, yBlkAll], axis=0)

plt.matshow(extrS, cmap='coolwarm')



'''
Smean = np.array(
      [ 3.98213840e+02,  4.11224396e+02,  8.08170703e+02,  4.67121459e+02,
        9.58412207e-02, -1.79782432e-02,  1.71556081e-02, -4.14991879e-03,
       -2.28322462e-01,  9.20027735e-02, -8.14613082e-02, -3.17308934e+00,
       -5.38585657e+00,  6.26164383e+00,  9.45077936e-01,  8.08304282e-02,
       -7.51600297e-02, -3.03507465e+00, -4.06491057e+00,  4.00096601e+00,
        9.50916102e-01, -3.38361474e-01,  9.60425885e-01,  9.60555438e+00,
       -5.72674887e+00,  3.88570300e+00,  1.46053354e-01, -5.57198700e-01,
        6.43733030e-01, -6.43676977e+00, -4.00413471e+00,  3.72567547e+00,
       -3.27605373e-02,  8.62886207e-01, -6.78201432e-04, -1.27359412e-01,
       -2.44355042e+00,  7.63205909e+00,  8.88361698e-01,  5.26028732e-01,
        8.64146244e-01,  7.80177035e+00, -6.70277531e+00,  5.20695030e+00,
        7.30336876e-01,  8.99827967e-01,  1.78243857e+00,  1.86150936e+01,
        5.98526378e+00,  7.29298765e+00, -8.08738428e-01,  1.02303559e-01,
        1.09629125e+00, -4.95598455e+00, -3.51410147e+00,  1.14528430e+01,
        1.75068619e-02,  1.48805221e-02, -1.58488521e+00, -2.35995566e+00,
        4.91849779e+00,  5.37940157e+00, -6.50590283e-02, -3.22829044e-02,
       -3.09914039e+00,  4.02494559e+00,  4.89104703e+00,  5.56949282e+00,
        1.97581406e-01,  6.88105109e-02,  2.27910698e-02, -3.79923073e+00,
        1.99756603e+00,  5.16839394e+00,  5.68235709e-01,  1.93106014e-01,
        2.22484490e-01, -4.74288722e-01, -1.97951364e+01,  1.48902312e+01,
       -5.40230272e-02, -2.11437124e-02, -3.46421590e-02, -4.59416797e+00,
       -5.82225412e-01,  1.97710238e+01, -1.41653099e+00,  3.38020448e-02,
        2.78678950e+00, -1.75695022e+01, -2.96368188e+00,  1.23532714e+01,
       -1.18460309e-01, -1.05137374e+00,  1.69773008e+00, -1.69412489e+00,
       -5.06532137e+00,  7.90455051e+00, -4.14649903e-01, -4.94426169e-01,
        1.38043596e+00, -5.03135277e+00, -5.43547007e+00,  1.26991966e+01,
       -2.50042201e-01,  8.02786932e-01,  1.72645413e+00,  1.47947989e+00,
        1.60842477e+01,  2.09283726e+01, -2.68805247e-01,  1.03164360e+00,
        6.40659057e-01,  1.68264894e+01,  6.68865788e+00,  1.73046827e+01,
       -3.58352735e-02,  8.85232394e-02, -1.41836937e-01, -2.20225564e+00,
       -1.01215599e+01,  1.87058555e+01, -7.46207061e-03, -2.65485197e-01,
       -1.61748892e-01, -9.54394554e+00, -7.97594735e-01,  2.41651826e+00,
        3.29768724e-01,  1.04935133e+00,  1.43090193e+00,  1.22238773e+01,
       -4.11740103e+00,  7.54633038e+00,  6.08319706e-01, -3.85403982e-03,
       -2.99504577e+00,  3.68058002e+00,  5.48597576e-01,  1.45748882e+01,
       -4.18170179e-02, -5.97130394e-01,  3.06790855e-01, -1.12050748e+01,
        1.66399217e+00,  3.24935875e+00,  1.63730986e-02, -1.23804169e+00,
        6.87577835e-03, -5.26068417e+00, -2.40616638e+00,  1.44066156e+00,
       -1.21411378e+00,  2.17619172e-01,  2.55174591e+00, -1.73766332e+01,
        7.20953855e+00,  1.22812383e+01, -9.95886791e-01,  6.40191151e-01,
        2.05514621e+00, -1.08960679e+01,  1.41197214e+01,  1.53498848e+01,
       -1.24198097e-01, -8.39867265e-01, -2.94574793e+00,  1.48699765e+00,
        1.61442015e+01,  1.23622436e+01,  7.38455019e-01,  4.61661711e-01,
        1.33651323e+00,  9.35040071e+00, -6.55144173e+00,  4.39687080e+00,
       -6.33742277e-01, -1.13585852e-01, -3.01844512e+00,  1.92689848e+00,
        6.99976307e+00,  1.01504128e+01,  1.81275476e-02, -2.97138175e-02,
       -6.27459196e-02, -3.89081176e+00, -2.70593243e+00,  4.32909703e+00,
        3.42741379e-01,  6.03378276e-01, -3.02368589e-01,  1.38806225e+01,
       -9.07390614e+00,  1.55793088e+01,  3.82664071e-02,  8.75231097e-01,
        9.78528144e-04,  1.74233332e+01, -2.79635047e+00,  1.07892644e+01,
       -3.29153925e-02,  1.08510732e+00,  5.22249671e-01,  1.81473165e+01,
        1.98678517e+00,  1.03291327e+01])

Scov = np.zeros((Smean.shape[0], Smean.shape[0]))

Scov[:NintrParams,:NintrParams] = np.array(
      [[ 4.50624463e-06, -4.45948406e-07,  2.59841173e-07,
         1.54850295e-06, -1.49079806e-22, -7.24359240e-28,
        -2.01264916e-27, -9.97816956e-28],
       [-4.45948406e-07,  1.47318063e-05, -2.32025200e-06,
        -1.09655759e-06, -8.29869100e-21,  9.17866270e-28,
         2.55030500e-27,  1.26437266e-27],
       [ 2.59841173e-07, -2.32025200e-06,  4.58277279e-06,
        -5.42906035e-08, -1.29573103e-21,  1.66673555e-28,
         4.63106387e-28,  2.29595598e-28],
       [ 1.54850295e-06, -1.09655759e-06, -5.42906035e-08,
         2.32529916e-06,  5.86264191e-22, -1.00901083e-27,
        -2.80356099e-27, -1.38992936e-27],
       [-1.49079806e-22, -8.29869100e-21, -1.29573103e-21,
         5.86264191e-22,  1.59799658e-28, -6.53517210e-30,
        -1.81581292e-29, -9.00230922e-30],
       [-7.24359240e-28,  9.17866270e-28,  1.66673555e-28,
        -1.00901083e-27, -6.53517210e-30,  2.67262657e-31,
         7.42595570e-31,  3.68158794e-31],
       [-2.01264916e-27,  2.55030500e-27,  4.63106387e-28,
        -2.80356099e-27, -1.81581292e-29,  7.42595570e-31,
         2.06331923e-30,  1.02293786e-30],
       [-9.97816956e-28,  1.26437266e-27,  2.29595598e-28,
        -1.38992936e-27, -9.00230922e-30,  3.68158794e-31,
         1.02293786e-30,  5.07144916e-31]])


extrS = np.array(
      [[ 4.74447530e-27,  4.53038827e-28, -9.40778298e-29,
         6.21366640e-26,  2.17814537e-25, -9.40559558e-25],
       [ 4.53038827e-28,  6.30658437e-27,  6.74397365e-28,
         8.33772783e-26,  1.41196774e-24, -3.16477259e-25],
       [-9.40778298e-29,  6.74397365e-28,  3.79095768e-26,
         8.74665288e-25,  1.75611480e-25,  9.40061577e-25],
       [ 6.21366640e-26,  8.33772783e-26,  8.74665288e-25,
         5.45773843e-19, -2.24294151e-20,  1.31456205e-19],
       [ 2.17814537e-25,  1.41196774e-24,  1.75611480e-25,
        -2.24294151e-20,  7.34643697e-19,  3.96008448e-20],
       [-9.40559558e-25, -3.16477259e-25,  9.40061577e-25,
         1.31456205e-19,  3.96008448e-20,  4.35689213e-19]])

'''


# %%

ScovNew = np.zeros_like(Scov)
ScovNew[:NintrParams,:NintrParams] = Scov[:NintrParams,:NintrParams]
ScovNew[xBlkAll, yBlkAll] = extrS

diagmin = (Smean * 1e-6) **2  # minimo relativo para la covarianza

diagmin < np.diag(ScovNew)

plt.figure()
plt.plot(np.diag(ScovNew))
plt.plot(diagmin)


ScovNew[np.diag_indices_from(Scov)] = np.max([np.diag(ScovNew), diagmin], axis=0)

plt.matshow(np.log(np.abs(ScovNew)), cmap='coolwarm')

sigmasNew = np.sqrt(np.diag(ScovNew))
ScorrNew = ScovNew / ( sigmasNew.reshape((-1,1)) * sigmasNew.reshape((1,-1)))

plt.matshow(ScorrNew, cmap='coolwarm', vmin=-1, vmax=1)


# %% metropolis
'''
http://docs.pymc.io/api/inference.html
'''

start = {'xAll' : Smean}
nDraws = 10
nChains = 8


with projectionModel:
    step = pm.Metropolis(S=Scov, scaling=1e-3, tune_interval=50)

    trace = pm.sample(draws=nDraws, step=step, start=start, njobs=nChains,
                      tune=0, chains=nChains, progressbar=True,
                      discard_tuned_samples=False)


plt.figure()
plt.plot(np.abs(trace['xAll'] - trace['xAll'][-1]), '-x', markersize=1.5)


# probabilidad de repeticion

nRepet = np.sum(trace['xAll'][1:,0] == trace['xAll'][:-1,0])
pRep = nRepet / (trace['xAll'].shape[0] - 1)

print('tasa de repeticion es', pRep)

os.system("espeak 'simulation finished'")

# %%
corner.corner(trace['xAll'][:,:NintrParams])
i=19
corner.corner(trace['xAll'][:,NintrParams+6*i:NintrParams+6*(i+1)])

repeats = np.zeros_like(trace['xIn'])
repeats[1:] = trace['xIn'][1:] == trace['xIn'][:-1]

plt.matshow(repeats, cmap='binary')
plt.matshow(repeats.reshape((200,-1)), cmap='binary')

plt.figure()
plt.plot(trace['xAll'][:,7])

plt.figure()
plt.plot(trace['xAll'][:,0] + trace['xAll_interval__'][:,0])
plt.plot(trace['xAll'][:,0])
plt.plot(trace['xAll'][:,0] - trace['xAll_interval__'][:,0])





# %% load all data simulated

iis = [2, 3, 4, 5]

traces = list()

for i in iis:
    traces.append(np.load(pathFiles + "intrinsicTracesAllNoReps"
                          + str(i) + ".npy"))

tracesCon = np.concatenate(traces,axis=0)

Smean = np.mean(tracesCon, axis=0)

Scov = np.cov(tracesCon.T)

np.save(pathFiles + "Smean", Smean)
np.save(pathFiles + "Scov", Scov)






# %%
for trac in traces:
    corner.corner(trac[:,4:6])

corner.corner(tracesCon[:,4:6])

[print(tra.shape) for tra in traces]




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


