#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 01:12:17 2017

@author: sebalander
"""


# %%
#import glob
import numpy as np
import scipy.linalg as ln
import matplotlib.pyplot as plt
from importlib import reload
from copy import deepcopy as dc
import pyproj as pj

from dev import bayesLib as bl

from multiprocess import Process, Queue
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

# model data files
intrinsicParamsOutFile = imagesFolder + camera + model + "intrinsicParamsML"
intrinsicParamsOutFile = intrinsicParamsOutFile + "0.5.npy"

# calibration points
calibptsFile = "/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/2016-11-13 medicion/calibrExtr/puntosCalibracion.txt"

# load model specific data
distCoeffs = np.load(distCoeffsFile)
cameraMatrix = np.load(linearCoeffsFile)

# load intrinsic calib uncertainty
resultsML = np.load(intrinsicParamsOutFile).all()  # load all objects
Nmuestras = resultsML["Nsamples"]
muDist = resultsML['paramsMU']
covarDist = resultsML['paramsVAR']
muCovar = resultsML['paramsMUvar']
covarCovar = resultsML['paramsVARvar']
Ns = resultsML['Ns']
cameraMatrixOut, distCoeffsOut = bl.flat2int(muDist, Ns)

# Calibration points
calibPts = np.loadtxt(calibptsFile)
lats = calibPts[:,2]
lons = calibPts[:,3]

# according to google earth, a good a priori position is
# -34.629344, -58.370350
y0, x0 = -34.629344, -58.370350
z0 = 15.7 # metros, as measured by oliva
elev = 7 # altura msnm en parque lezama segun google earth
# EGM96 sobre WGS84: https://geographiclib.sourceforge.io/cgi-bin/GeoidEval?input=-34.629344%2C+-58.370350&option=Submit
geoideval = 16.1066
h = elev + geoideval # altura dle terreno sobre WGS84



# %%
a = 6378137
f = 1 /298.257223563
b = a * (1 - f) # 6356752.31424518
e = np.sqrt(a** - b**2) / a
ep = e * a / b

# LLA 2 ECEF
# phi es latitude 
# lambda es longitud
# h heigth above ellipsoid (meters)
# N/rad ardius of curvatiure meters

def lla2ecef(lat, lon, alt):
    '''
    convierte lat lon y elevacion sobre WGS a ECEF
    
    fuente: 
        
    '''
    la = np.deg2rad(lat)
    lo = np.deg2rad(lon)
    cla = np.cos(la)
    sla = np.sin(la)
    clo = np.cos(lo)
    slo = np.sin(lo)
    
    rad = a / np.sqrt(1 - e**2 * sla**2)
    
    r = rad + alt
    
    X = r * cla * clo
    Y = r * cla * slo
    Z = (b**2 * rad / a**2 + alt) * sla
    
    return np.array([X, Y, Z]).T

# convierto a ECEF en metros
XYZ = lla2ecef(lats, lons, h)
xyzOrig = lla2ecef(y0, x0, h) # el origen de coords del mapa
xyz0 = lla2ecef(y0, x0, h + z0)

# %% Ahora saco los vectores para proyectar esto a un plano tanghente local
theta = np.deg2rad(x0) # longitud desde el eje x horizontal
fi = np.pi / 2 - np.deg2rad(y0) # decliancion desde el eje z

ct = np.cos(theta)
st = np.sin(theta)
cf = np.cos(fi)
sf = np.sin(fi)

'''
defiuno los versores este oeste y altura
http://mathworld.wolfram.com/SphericalCoordinates.html
'''
# versor en la direccion radial
rVersor = xyzOrig / ln.norm(xyzOrig)
# e direccion este, versor theta
eastVersor = np.array([-st, ct, 0])
# versor norte, opueto el versor fi
northVersor = - np.array([ct * cf, st * cf, -sf])

# resto el origen
deltaXYZ = XYZ - xyzOrig
delta0 = xyz0 - xyzOrig

# proyecto sobre la rotacion que interesa
# matriz de rotacion para post multiplicar
R = np.array([eastVersor, northVersor, rVersor]).T

xyzPts = deltaXYZ.dot(R)
xyzCam =  delta0.dot(R)


plt.plot(xyzPts[:,0], xyzPts[:,1], '+')
plt.plot(xyzCam[0], xyzCam[1], 'x')


