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

# %% defino variables
# vector de rodriquez en esfericas y traslacion
# despues lo paso a cartesianas, que es como lo tengo en realidad
# pero asi me ahorro la cinematica inversa que es innecesaria
rx, ry, rz, tx, ty, tz = sy.symbols('rx, ry, rz, tx, ty, tz')

rV = sy.Matrix([rx, ry, rz])
tV = sy.Matrix([tx, ty, tz])

r = sy.sqrt(rV.dot(rV))
ct = sy.cos(r)
st = sy.sin(r)
rhat = rV / r

# matriz de rotacion a partir del vector de rodriguez
rrMat = sy.Matrix([[rx**2, rx*ry, rx*rz],
                   [ry*rx, ry**2, ry*rz],
                   [rz*rz, rz*ry, rz**2]]) / r**2
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
JXm_rV = Xm.jacobian(rV)
JXm_tV = Xm.jacobian(tV)

'''
http://docs.sympy.org/dev/modules/rewriting.html
'''

simps, exps = sy.cse((JXm_Xp, JXm_rV, JXm_tV))

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
    x25 = x15*x3
    x26 = x24 + x25
    x27 = x26*xp
    x28 = x12 + x22 - x27
    x29 = rz*x7
    x30 = -x29
    x31 = rx*ry
    x32 = x15*x31
    x33 = -x19*xp + x30 + x32
    x34 = -x26*yp + x29 + x32
    x35 = x21*x28 - x33*x34
    x36 = 1/x35
    x37 = x23 - x25
    x38 = x17*x34 - x21*x37
    x39 = tx - tz*xp
    x40 = x35**(-2)
    x41 = x40*(x0*x33 - x21*x39)
    x42 = -x17*x28 + x33*x37
    x43 = x40*(-x0*x28 + x34*x39)
    x44 = x6/x4**(3/2)
    x45 = x2*x44
    x46 = rx*x45
    x47 = x4**(-2)
    x48 = 2*rx*x14*x47
    x49 = -x2*x48
    x50 = x11*x12
    x51 = x1*x44
    x52 = x31*x44
    x53 = rz*x52
    x54 = 2*rz*x14*x47
    x55 = -x31*x54
    x56 = x53 + x55 + x7
    x57 = x1*x50 - x51 + x56
    x58 = x46 + x49 - x57*yp + x9
    x59 = 2*ry*x14*x47
    x60 = ry*x51 - x1*x59
    x61 = rx*rz
    x62 = x44*x61
    x63 = x50*x61
    x64 = ry*x15
    x65 = -x57*xp + x60 + x62 - x63 + x64
    x66 = rx**3
    x67 = rx*x15
    x68 = 2*x14*x47
    x69 = x31*x50
    x70 = x3*x44
    x71 = rx*x70 - x3*x48 + x52 - x69
    x72 = x44*x66 - x66*x68 + 2*x67 - x71*xp + x9
    x73 = -x62
    x74 = x60 + x63 + x64 - x71*yp + x73
    x75 = -x21*x72 - x28*x58 + x33*x74 + x34*x65
    x76 = ry**3
    x77 = rz*x45 - x2*x54
    x78 = rz*x15
    x79 = -x52 + x69 + x77 + x78
    x80 = x24 + x44*x76 + 2*x64 - x68*x76 - x79*yp
    x81 = x46 + x49 + x67
    x82 = x10*x44
    x83 = x10*x50
    x84 = -x83
    x85 = -x79*xp + x81 + x82 + x84
    x86 = ry*x70 - x3*x59
    x87 = -x7
    x88 = -x2*x50 + x45 + x86 + x87
    x89 = x24 + x60 - x88*xp
    x90 = x81 - x82 + x83 - x88*yp
    x91 = -x21*x89 - x28*x80 + x33*x90 + x34*x85
    x92 = x63 + x64 + x73 + x86
    x93 = x30 + x77 - x92*yp
    x94 = x3*x50
    x95 = x53 + x55 + x70 + x87 - x92*xp - x94
    x96 = rz**3
    x97 = x44*x96 - x68*x96 + 2*x78 + x82 + x84
    x98 = rz*x51 - x1*x54 + x30 - x97*xp
    x99 = x56 - x70 + x94 - x97*yp
    x100 = -x21*x98 - x28*x93 + x33*x99 + x34*x95
    
    # jacobiano de la pos en el mapa con respecto a las posiciones homogeneas
    JXm_Xp = np.array([[x36*(tz*x21 + x0*x17) + x38*x41,
                        x36*(-tz*x33 - x17*x39) + x41*x42],
                       [x36*(-tz*x34 - x0*x37) + x38*x43,
                        x36*(tz*x28 + x37*x39) + x42*x43]])
    
    # jacobiano respecto al vector de rodriguez
    JXm_rV = np.array([[x36*(x0*x65 - x39*x58) + x41*x75,
                        x36*(x0*x85 - x39*x80) + x41*x91,
                        x100*x41 + x36*(x0*x95 - x39*x93)],
                       [x36*(-x0*x72 + x39*x74) + x43*x75,
                       x36*(-x0*x89 + x39*x90) + x43*x91,
                       x100*x43 + x36*(-x0*x98 + x39*x99)]])
    
    # jacobiano respecto a la traslacion
    JXm_tV = np.array([[x36*(x13 - x18 + x20),
                        x33*x36,
                        x36*(x21*xp - x33*yp)],
                       [x34*x36,
                        x36*(x13 - x22 + x27),
                        x36*(x28*yp - x34*xp)]])

    return JXm_Xp, JXm_rV, JXm_tV

#

rx, ry, rz = [ 0.02019567,  0.00262967, -1.58390316]
tx, ty, tz = [-2.44949087,  4.85340383,  5.27767529]
xp, yp = [1,1]

jacobianosHom2Map(rx, ry, rz, tx, ty, tz, xp, yp)

# %% pruebo con algun valor:
valuesDict = {r:4, a:8, b:0.2, vx:1, vy:2, vz:3, tx:7, ty:8, tz:9}

jacobianos(4, 8, 0.2, 1, 2, 3, 10, 11)


