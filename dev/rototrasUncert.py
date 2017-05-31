#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 20:05:06 2017

@author: sebalander
"""

# %%
import sympy as sy
import numpy as np

# %% defino variables
# vector de rodriquez en esfericas y traslacion
# despues lo paso a cartesianas, que es como lo tengo en realidad
# pero asi me ahorro la cinematica inversa que es innecesaria
r, a, b, tx, ty, tz = sy.symbols('r, a, b, tx, ty, tz')
# vector a rotar
vx, vy, vz = sy.symbols('vx, vy, vz')

rVsph = sy.Matrix([r, a, b])
v = sy.Matrix([vx, vy, vz])
tV = sy.Matrix([tx, ty, tz])

ct = sy.cos(r)
st = sy.sin(r)

# vector de rodrigues en esfericas y cartesianas
rx = rVsph[0] * sy.cos(rVsph[1]) * sy.sin(rVsph[2])
ry = rVsph[0] * sy.sin(rVsph[1]) * sy.sin(rVsph[2])
rz = rVsph[0] * sy.cos(rVsph[2])

rV = sy.Matrix([rx, ry, rz])
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
xp, yp = sy.symbols('xp, yp')

a = R[0, 0] - R[2, 0] * xp
b = R[0, 1] - R[2, 1] * xp
c = tx - tz * xp
d = R[1, 0] - R[2, 0] * yp
e = R[1, 1] - R[2, 1] * yp
f = ty - tz * yp
q = a * e - d * b

x = (f * b - c * e) / q
y = (c * d - f * a) / q

X = sy.Matrix([x, y])
Xp = sy.Matrix([xp, yp])

# tengo que saacar el jacobiano respecto a xp, yp, rV y tV
# para el jacobiano respecto a rV primero lo hago respecto a rVsph y uso
# cain rule

JX_Xp = X.jacobian(Xp)
JX_tV = X.jacobian(tV)
JX_rVsph = X.jacobian(rVsph)
JrV_rVsph = rV.jacobian(rVsph)

'''
http://docs.sympy.org/dev/modules/rewriting.html
'''

simps, exps = sy.cse((JX_Xp, JX_tV, JX_rVsph, JrV_rVsph))

# %% calculo de jacobianos


def jacobianos(r, a, b, tx, ty, tz, xp, yp):
    x0 = ty - tz*yp
    x3, x6, x2 = np.sin([r, a, b])
    x9, x1, x8 = np.cos([r, a, b])
    x4 = x2*x3
    x5 = x1*x4
    x7 = x2*x6
    x10 = -x9
    x11 = x10 + 1
    x12 = x11*x8
    x13 = x12*x7
    x14 = -x13 - x5
    x15, x16, x22, x26 = np.square([x6, x2, x1, x8])
    x17 = x11*x16
    x18 = x15*x17
    x19 = x13 + x5
    x20 = x19*yp
    x21 = x18 - x20 + x9
    x23 = x17*x22
    x24 = x3*x7
    x25 = -x24
    x27 = x11*x26
    x28 = x25 + x27
    x29 = x28*xp
    x30 = x23 - x29 + x9
    x31 = x3*x8
    x32 = x1*x6
    x33 = x17*x32
    x34 = -x28*yp + x31 + x33
    x35 = -x19*xp - x31 + x33
    x36 = x21*x30 - x34*x35
    x37 = 1/x36
    x38 = x24 - x27
    x39 = x14*x34 - x21*x38
    x40 = tx - tz*xp
    x41 = x36**(-2)
    x42 = x41*(x0*x35 - x21*x40)
    x43 = -x14*x30 + x35*x38
    x44 = x41*(-x0*x30 + x34*x40)
    x45 = -x18
    x46 = -x3
    x47 = x16*x3
    x48 = x1*x2
    x49 = x31*x7 + x48*x9
    x50 = x15*x47 + x46 - x49*yp
    x51 = x8*x9
    x52 = x32*x47
    x53 = -x49*xp - x51 + x52
    x54 = x26*x3 - x7*x9
    x55 = x22*x47 + x46 - x54*xp
    x56 = x51 + x52 - x54*yp
    x57 = -x21*x55 - x30*x50 + x34*x53 + x35*x56
    x58 = 2*x33
    x59 = x12*x48 + x25
    x60 = x58 - x59*yp
    x61 = x23 + x45
    x62 = -x59*xp + x61
    x63 = x5*xp - x58
    x64 = x5*yp + x61
    x65 = -x21*x63 - x30*x60 + x34*x62 + x35*x64
    x66 = 2*x12*x2
    x67 = x1*x31 - x17*x6 + x27*x6
    x68 = x15*x66 - x67*yp
    x69 = 2*x1*x13
    x70 = x4 - x67*xp + x69
    x71 = -x31*x6 - x66
    x72 = x22*x66 - x71*xp
    x73 = -x4 + x69 - x71*yp
    x74 = -x21*x72 - x30*x68 + x34*x70 + x35*x73
    x75 = r*x2
    x76 = r*x8
    
    # jacobiano de la pos en el mapa con respecto a las posiciones homogeneas
    JX_Xp = np.array([[x37*(tz*x21 + x0*x14) + x39*x42,
                       x37*(-tz*x35 - x14*x40) + x42*x43],
                      [x37*(-tz*x34 - x0*x38) + x39*x44,
                       x37*(tz*x30 + x38*x40) + x43*x44]], dtype=float)
    
    # jacobiano respecto a la traslacion
    JX_tV = np.array([[x37*(x10 + x20 + x45), x35*x37, x37*(x21*xp - x35*yp)],
                      [x34*x37, x37*(x10 - x23 + x29), x37*(x30*yp - x34*xp)]],
                     dtype=float)
    
    # jacobiano de  posiciones en mapa wrt vector de rodrigues en esfericas
    JX_rVsph = np.array([[x37*(x0*x53 - x40*x50) + x42*x57,
                          x37*(x0*x62 - x40*x60) + x42*x65,
                          x37*(x0*x70 - x40*x68) + x42*x74],
                         [x37*(-x0*x55 + x40*x56) + x44*x57,
                          x37*(-x0*x63 + x40*x64) + x44*x65,
                          x37*(-x0*x72 + x40*x73) + x44*x74]], dtype=float)
    
    # jacobiano de vector de rodrigues cartesiano wrt esfericas
    JrV_rVsph = np.array([[x48, -x6*x75, x1*x76],
                          [x7,   x1*x75, x6*x76],
                          [x8,        0,   -x75]], dtype=float)

    return JX_Xp, JX_tV, JX_rVsph, JrV_rVsph

#
# %% pruebo con algun valor:
valuesDict = {r:4, a:8, b:0.2, vx:1, vy:2, vz:3, tx:7, ty:8, tz:9}

jacobianos(4, 8, 0.2, 1, 2, 3, 10, 11)
