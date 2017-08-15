#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 20:35:03 2017

test functions that propagate uncertanty

@author: sebalander
"""
# %%
import time
import timeit

from importlib import reload
reload(cl)
# %%
j=0
ep = 0.05  # standar deviation in parameters

# matriz de incerteza de deteccion en pixeles
Cccd = np.repeat([np.eye(2,2)],imagePoints[j,0].shape[0], axis=0)
Cf = np.diag([2,2,3,3])
Ck = np.diag((distCoeffs.reshape(-1) * ep )**2)
Cr = np.diag((rVecs[j].reshape(-1) * ep )**2)
Ct = np.diag((tVecs[j].reshape(-1) * ep )**2)


Crt = [Cr, Ct]


# %%
xpp, ypp, Cpp = cl.ccd2hom(imagePoints[j,0], cameraMatrix, Cccd, Cf)


# %% undistort
xp, yp, Cp = cl.homDist2homUndist(xpp, ypp, distCoeffs, model, Cpp, Ck)


# %% project to plane z=0 from homogenous
xm, ym, Cm = cl.xypToZplane(xp, yp, rVecs[j], tVecs[j], Cp, Crt)

Caux = Cm
# %%
xm, ym, Cm = cl.inverse(imagePoints[j,0], rVecs[j], tVecs[j],                                        
                        cameraMatrix, distCoeffs, model,
                        Cccd, Cf, Ck, Crt)

er = [xm, ym] - chessboardModel[0,:,:2].T

# %%
statement1 = '''
p1 = np.tensordot(er, Cm, axes=(0,1))[range(54),range(54)]
p2 = p1.dot(er)[range(54),range(54)]
'''

# %%
statement2 = '''
Er = np.empty_like(xp)
for i in range(len(xp)):
    Er[i] = er[:,i].dot(Cm[i]).dot(er[:,i])
'''

# %%
statement3 = '''
q1 = [np.sum(Cm[:,:,0]*er.T,1), np.sum(Cm[:,:,1]*er.T,1)];
q2 = np.sum(q1*er,0)
'''
# %%

t1 = timeit.timeit(statement1, globals=globals(), number=10000) / 1e4

t2 = timeit.timeit(statement2, globals=globals(), number=10000) / 1e4

t3 = timeit.timeit(statement3, globals=globals(), number=10000) / 1e4

print(t1/t3, t2/t3)



# %% testeo poniendo ruido en la imagen
# nro de realizaciones
Nrel = 1000

# ruido de desv est unitaria
noiseI = np.random.multivariate_normal([0,0], Cccd[0], size=(Nrel, m))
noiseF = np.random.multivariate_normal([0,0,0,0], Cf, size=Nrel)
noiseK = np.random.multivariate_normal([0,0,0,0], Ck, size=Nrel)
noiseR = np.random.multivariate_normal([0,0,0], Cr, size=Nrel)
noiseT = np.random.multivariate_normal([0,0,0], Ct, size=Nrel)

imagePointsNoised = imagePoints[j,0] + noise
kFlat = cameraMatrix[[0,1,0,1],[0,1,2,2]]
camMatNoised = kFlat + noiseF
camMatNoised = [fat2CamMAtrix(v)  for v in camMatNoised]
distCoeNoised = distCoeffs.T + noiseK
rVnoised = rVecs[j].T + noiseR
tVnoised = tVecs[j].T + noiseT


objPoitsNoised = np.empty_like(imagePointsNoised)

for i in range(Nrel):
     objPoitsNoised[i,:,0], objPoitsNoised[i,:,1], Cm = cl.inverse(imagePointsNoised[i], rVnoised[i], tVnoised[i],                                        
                        camMatNoised[i], distCoeNoised[i], model)

# %%
iptNois = imagePointsNoised.reshape(-1,2).T

plt.figure()
plt.plot(iptNois[0], iptNois[1], '*')
plt.plot(imagePoints[j,0,:,0], imagePoints[j,0,:,1], '+k', markersize=5)

# %% plot
objNois = objPoitsNoised.reshape(-1,2).T


plt.figure()
plt.plot(objNois[0], objNois[1], '*')
plt.plot(chessboardModel.T[0], chessboardModel.T[1], '+k')
'''
algo da mal hay que revisar cada paso del mapeo es ocmo si me estuvier aolvidando el ultimo
'''
