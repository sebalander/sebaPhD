# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 13:34:51 2016

explore the optimization of pose problem.

toy example. the spript calculates and graphs the error landscape.

tha idea is to check for pathological minima in that lanscape

@author: sebalander
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import time


# %% 
def directProj(x, pose):
    '''
    pose es (x_cam, z_cam, alpha_cam)
    calcula los angulos dados las coord en la escena y la pose
    '''
    return np.arctan2(x-pose[0], -pose[1]) - pose[2]

def inverseProj(theta, pose):
    '''
    devuelve las posiciones en la escena a partir de los angulos
    '''
    return - pose[1] * np.tan(theta + pose[2]) + pose[0]

def errores(x0, theta0, pose):
    Ex = np.mean((x0 - inverseProj(theta0, pose))**2)
    Et = np.mean((theta0 - directProj(x0, pose))**2)
    return Ex, Et

n=1e3
alphas = np.linspace(-np.pi,np.pi,n)

def searchAlpha(xc, zc, x0, theta0):
    '''
    bruteforce searching optimal alpha, keeps position constant
    '''
    # global alphas
    # alphas = np.linspace(-np.pi,np.pi,n)
    
    errVsAlpha = [errores(x0, theta0, [xc, zc, al]) for al in alphas]
    errVsAlpha = np.array(errVsAlpha).T
    
    # plt.plot(alphas, errVsAlpha[0])
    # plt.plot(alphas, errVsAlpha[1])
    # plt.plot(theta0)
    
    # minimum error
    i = np.argmin(errVsAlpha[1]) # minimo global del error en theta
    alphaBestEtheta = [i, alphas[i], errVsAlpha[1,i]]
    
    inter = np.int(n/4)
    indices = [alphaBestEtheta[0]-inter,
               alphaBestEtheta[0]+inter]
    i = alphaBestEtheta[0] - inter + \
        np.argmin(errVsAlpha[0,indices[0]:indices[1]])
    alphaBestEx = [i, alphas[i], errVsAlpha[0,i]]
    print(xc,zc)
    return [alphaBestEx, alphaBestEtheta]



# %%
N = 1000
pose0 = [0.5,-0.5,0]
x0 = np.random.rand(10)
theta0 = directProj(x0, pose0)


# %%
pose = [0.7, -0.4, 0.1]
tic = time.clock()
searchAlpha(pose[0], pose[1], x0, theta0)
toc = time.clock()
toc - tic

# %% calculate error surface
Nsurf = 200
xc = np.linspace(0,1,Nsurf-10)
zc = np.linspace(-1,-0.10,Nsurf)

err = [[searchAlpha(x, z, x0, theta0) for z in zc] for x in xc]

err = np.array(err)

np.save(,err)

# %%
err = np.load('./resources/extToyCalibError.npy')
errXSurf = err[:,:,0,2]
errThetaSurf = err[:,:,1,2]


# %% plot as image
#import matplotlib.cm as cm

fig, axes = plt.subplots(ncols=2, nrows=2)
ax1, ax2, ax3, ax4 = axes.ravel()

#plt.figure()
Z = errThetaSurf
ax1.imshow(Z)
ax1.set_title('MSE in Theta (linearscale)')

#plt.figure()
Z = np.log(errThetaSurf)
ax2.imshow(Z)
ax2.set_title('MSE in Theta (logscale)')

#plt.figure()
Z = errXSurf
ax3.imshow(Z) #, interpolation='bilinear', cmap=cm.inferno,
#                origin='lower', extent=[-3, 3, -3, 3],
#                vmax=abs(Z).max(), vmin=-abs(Z).max())
ax3.set_title('MSE in x (linearscale)')

#plt.figure()
Z = np.log(errXSurf)
ax4.imshow(Z)
ax4.set_title('MSE in x (logscale)')

plt.title('n=1e3, N=1e3, Nsurf=200')



# %%

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection='3d')
#X, Y, Z = axes3d.get_test_data(0.05)

Xc,Zc = np.meshgrid(xc,zc)
ax.plot_surface(Xc, Zc, errXSurf, rstride=8, cstride=8, alpha=0.3)
ax.plot_surface(Xc, Zc, errThetaSurf, rstride=8, cstride=8, alpha=0.3)

cset = ax.contour(Xc, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
cset = ax.contour(Xc, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
cset = ax.contour(Xc, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

ax.set_xlabel('X')
#ax.set_xlim(-40, 40)
ax.set_ylabel('Z')
#ax.set_ylim(-40, 40)
ax.set_zlabel('MSE')
#ax.set_zlim(-100, 100)

plt.show()
