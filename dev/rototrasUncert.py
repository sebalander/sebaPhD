#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 20:05:06 2017

@author: sebalander
"""

# %%
import sympy as sy
import numpy as np
from numpy import sqrt, cos, sin
from numpy import array

# %% defino variables
# vector de rodriquez en esfericas y traslacion
# despues lo paso a cartesianas, que es como lo tengo en realidad
# pero asi me ahorro la cinematica inversa que es innecesaria
rx, ry, rz, tx, ty, tz = sy.symbols('rx, ry, rz, tx, ty, tz')

rV = sy.Matrix([rx, ry, rz])
tV = sy.Matrix([tx, ty, tz])
rtV = sy.Matrix([rx, ry, rz, tx, ty, tz])

r = sy.sqrt(rV.dot(rV))
ct = sy.cos(r)
st = sy.sin(r)
rhat = rV / r

# matriz de rotacion a partir del vector de rodriguez
rrMat = sy.Matrix([[rx**2, rx*ry, rx*rz],
                   [ry*rx, ry**2, ry*rz],
                   [rz*rx, rz*ry, rz**2]]) / r**2

rMat = sy.Matrix([[0, -rz, ry],
                  [rz, 0, -rx],
                  [-ry, rx, 0]]) / r

R = sy.eye(3) * ct + (1 - ct) * rrMat + st * rMat

# %%
# vector a rotar
xp, yp = sy.symbols('xp, yp')

a = R[0, 0] - R[2, 0] * xp
b = R[0, 1] - R[2, 1] * xp
c = tx - tz * xp
d = R[1, 0] - R[2, 0] * yp
e = R[1, 1] - R[2, 1] * yp
f = ty - tz * yp
q = a * e - d * b

xm = (f * b - c * e) / q
ym = (c * d - f * a) / q

Xp = sy.Matrix([xp, yp])
Xm = sy.Matrix([xm, ym])

# %%
# tengo que saacar el jacobiano respecto a xp, yp, rV y tV
# para el jacobiano respecto a rV primero lo hago respecto a rVsph y uso
# chain rule

JXm_Xp = Xm.jacobian(Xp)
#JXm_rV = Xm.jacobian(rV)
#JXm_tV = Xm.jacobian(tV)
JXm_rtV = Xm.jacobian(rtV)

'''
http://docs.sympy.org/dev/modules/rewriting.html
'''

simps, exps = sy.cse((JXm_Xp, JXm_rtV))

# %% calculo de jacobianos


def jacobianosHom2Map(rx, ry, rz, tx, ty, tz, xp, yp):
    '''
    returns the jacobians needed to calculate the propagation of uncertainty
    
    '''
    x0 = ty - tz*yp
    x1 = rx**2
    x2 = ry**2
    x3 = rz**2
    x4 = x1 + x2 + x3
    x5 = sqrt(x4)
    x6 = sin(x5)
    x7 = x6/x5
    x8 = rx*x7
    x9 = -x8
    x10 = ry*rz
    x11 = 1/x4
    x12 = cos(x5)
    x13 = -x12
    x14 = x13 + 1
    x15 = x11*x14
    x16 = x10*x15
    x17 = -x16 + x9
    x18 = x15*x2
    x19 = x16 + x8
    x20 = x19*yp
    x21 = x12 + x18 - x20
    x22 = x1*x15
    x23 = ry*x7
    x24 = -x23
    x25 = rx*rz
    x26 = x15*x25
    x27 = x24 + x26
    x28 = x27*xp
    x29 = x12 + x22 - x28
    x30 = rz*x7
    x31 = -x30
    x32 = rx*ry
    x33 = x15*x32
    x34 = -x19*xp + x31 + x33
    x35 = -x27*yp + x30 + x33
    x36 = x21*x29 - x34*x35
    x37 = 1/x36
    x38 = x23 - x26
    x39 = x17*x35 - x21*x38
    x40 = tx - tz*xp
    x41 = x36**(-2)
    x42 = x41*(x0*x34 - x21*x40)
    x43 = -x17*x29 + x34*x38
    x44 = x41*(-x0*x29 + x35*x40)
    x45 = x6/x4**(3/2)
    x46 = x2*x45
    x47 = x4**(-2)
    x48 = 2*rx*x14*x47
    x49 = rx*x46 - x2*x48
    x50 = x11*x12
    x51 = x1*x45
    x52 = x32*x45
    x53 = rz*x52
    x54 = 2*rz*x14*x47
    x55 = -x32*x54
    x56 = x53 + x55 + x7
    x57 = x1*x50 - x51 + x56
    x58 = x49 - x57*yp + x9
    x59 = 2*ry*x14*x47
    x60 = ry*x51 - x1*x59
    x61 = x25*x45
    x62 = x25*x50
    x63 = ry*x15
    x64 = -x57*xp + x60 + x61 - x62 + x63
    x65 = rx**3
    x66 = rx*x15
    x67 = 2*x14*x47
    x68 = rz*x51 - x1*x54
    x69 = x32*x50
    x70 = rz*x15
    x71 = x52 + x68 - x69 + x70
    x72 = x45*x65 - x65*x67 + 2*x66 - x71*xp + x9
    x73 = -x61
    x74 = x60 + x62 + x63 - x71*yp + x73
    x75 = -x21*x72 - x29*x58 + x34*x74 + x35*x64
    x76 = ry**3
    x77 = rz*x46 - x2*x54
    x78 = -x52 + x69 + x70 + x77
    x79 = x24 + x45*x76 + 2*x63 - x67*x76 - x78*yp
    x80 = x10*x45
    x81 = x10*x50
    x82 = -x81
    x83 = x49 + x66 - x78*xp + x80 + x82
    x84 = x53 + x55 - x7
    x85 = -x2*x50 + x46 + x84
    x86 = x24 + x60 - x85*xp
    x87 = x49 + x66 - x80 + x81 - x85*yp
    x88 = -x21*x86 - x29*x79 + x34*x87 + x35*x83
    x89 = x3*x45
    x90 = ry*x89 - x3*x59 + x62 + x63 + x73
    x91 = x31 + x77 - x90*yp
    x92 = x3*x50
    x93 = x84 + x89 - x90*xp - x92
    x94 = rx*x89 - x3*x48 + x66 + x80 + x82
    x95 = x31 + x68 - x94*xp
    x96 = x56 - x89 + x92 - x94*yp
    x97 = -x21*x95 - x29*x91 + x34*x96 + x35*x93

    # jacobiano de la pos en el mapa con respecto a las posiciones homogeneas
    JXm_Xp = array([[x37*(tz*x21 + x0*x17) + x39*x42,
                     x37*(-tz*x34 - x17*x40) + x42*x43],
                    [x37*(-tz*x35 - x0*x38) + x39*x44,
                     x37*(tz*x29 + x38*x40) + x43*x44]])
    
    # jacobiano respecto a la rototraslacion, tres primeras columnas son wrt
    # rotacion y las ultimas tres son wrt traslacion
    JXm_rtV = array([[x37*(x0*x64 - x40*x58) + x42*x75,     # x wrt r1
                      x37*(x0*x83 - x40*x79) + x42*x88,     # x wrt r2
                      x37*(x0*x93 - x40*x91) + x42*x97,     # x wrt r3
                      x37*(x13 - x18 + x20),                # x wrt t1
                      x34*x37, x37*(x21*xp - x34*yp)],      # x wrt t2
                     [x37*(-x0*x72 + x40*x74) + x44*x75,    # x wrt t3
                      x37*(-x0*x86 + x40*x87) + x44*x88,    # y wrt r1
                      x37*(-x0*x95 + x40*x96) + x44*x97,    # y wrt r2
                      x35*x37,                              # y wrt t1
                      x37*(x13 - x22 + x28),                # y wrt t2
                      x37*(x29*yp - x35*xp)]])              # y wrt t3


    return JXm_Xp, JXm_rtV

rx, ry, rz = rV = array([ 0.02019567,  0.00262967, -1.58390316])
tx, ty, tz = tV = array([-2.44949087,  4.85340383,  5.27767529])
xp, yp = [1,1]

Jx, Jrt = jacobianosHom2Map(rx, ry, rz, tx, ty, tz, xp, yp)

Jxrt = np.concatenate((Jx, Jrt), axis=1)

# %% testeo contra libreria numerica
from calibration import calibrator as cl
import numdifftools as ndf


def func(X):
    xp, yp, rV, tV = X[0], X[1], X[2:5], X[5:]
    y1, y2, _ = cl.xypToZplane(xp, yp, rV, tV)
    return array([y1,y2])

X = np.concatenate([[xp], [yp], rV, tV], axis=0)

func(X)

jac = ndf.Jacobian(func)
jac.step.base_step=0.1
Jnum = jac(X)


dif = Jxrt-Jnum
# imprimo error relativo
dif.T / Jxrt.T


