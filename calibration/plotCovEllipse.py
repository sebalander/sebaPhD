'''
para arreglar porque no me da bien la grafica de las elipses a partir de las
matrices de covarianza
'''
# %% imports
import numpy as np
import scipy.linalg as ln
from scipy.special import chdtri, chdtrc
import matplotlib.pyplot as plt


# %%
fi = np.linspace(0, 2*np.pi, 100)
r = np.sqrt(chdtri(2, 0.1))  # radio para que 90% caigan adentro
# r = 1
Xcirc = np.array([np.cos(fi), np.sin(fi)]) * r


def unit2CovTransf(c):
    '''
    returns the matrix that transforms points from unit normal pdf to a normal
    pdf of covariance C. so that
    Xnorm = np.random.randn(2,n)  # generate random points in 2D
    T = unit2CovTransf(C)  # calculate transform matriz
    X = np.dot(T, Xnorm)  # points that follow normal pdf of cov C
    '''
    l, v = ln.eig(C)

    # matrix such that A.dot(A.T)==C
    T =  np.sqrt(l.real) * v

    return T


def plotEllipse(ax, C, mux, muy, col):
    '''
    se grafica una elipse asociada a la covarianza c, centrada en mux, muy
    '''
    
    T = unit2CovTransf(C)
    # roto reescaleo para lleve del circulo a la elipse
    xeli, yeli = np.dot(T, Xcirc)

    ax.plot(xeli+mux, yeli+muy, c=col, lw=0.5)
    v1, v2 = T.T
    ax.plot([mux, mux + v1[0]], [muy, muy + v1[1]], c=col, lw=0.5)
    ax.plot([mux, mux + v2[0]], [muy, muy + v2[1]], c=col, lw=0.5)

# %%
# defino una matriz de covarianza
a = 3
b = 5
r = -0.5*a*b

C = np.array([[a**2, r],[r, b**2]])
mux = muy = 0
col = 'b'

fig = plt.figure()
ax = fig.gca()
ax.set_aspect('equal')
plotEllipse(ax, C, mux, muy, col)


# %% pruebo muchos puntos
n = 10000

Xnor = np.random.randn(2,n)


T = unit2CovTransf(C)
# roto reescaleo para lleve del circulo a la elipse
x, y = X = np.dot(T, Xnor)


X2 = ln.inv(T).dot(X)
Ninside = np.sum(ln.norm(X2, axis=0) <= r)


fig = plt.figure()
ax = fig.gca()
ax.set_aspect('equal')
ax.scatter(x,y)
plotEllipse(ax, C, mux, muy, col)

print(Ninside)

# count how many inside ellipse


