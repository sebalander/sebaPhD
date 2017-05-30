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

# %% esfericas y cartesianas
rx = rVsph[0] * sy.cos(rVsph[1]) * sy.sin(rVsph[2])
ry = rVsph[0] * sy.sin(rVsph[1]) * sy.sin(rVsph[2])
rz = rVsph[0] * sy.cos(rVsph[2])

rV = sy.Matrix([rx, ry, rz])
rhat = rV / r

# jacobiano de cartesianas derivadas respecto a esfericas
JrV_rVsph = rV.jacobian(rVsph)

# rotacion rodrigues
v2 = ct * v + sy.sin(r) * rhat.cross(v) + rhat * (rhat.dot(v)) * (1 - ct)

Jv2_rVsph = v2.jacobian(rVsph)

# %% pruebo con valores especificos
valuesDict = {r:4, a:8, b:0.2, vx:1, vy:2, vz:3, tx:7, ty:8, tz:9}

JrV_rVsph_eval = JrV_rVsph.subs(valuesDict).evalf()
Jv2_rVsph_eval = Jv2_rVsph.subs(valuesDict).evalf()

JrV_rVsph_eval = np.array(sy.matrix2numpy(JrV_rVsph_eval), dtype=float)
Jv2_rVsph_eval = np.array(sy.matrix2numpy(Jv2_rVsph_eval), dtype=float)

# busco el jacobiano del vector rotado con respecto al vector de rodrigues
# en cartesianas
Jv2_rVcar = Jv2_rVsph_eval.dot(np.linalg.inv(JrV_rVsph_eval))

# %% comparar con funcion donde se implementa
def jacobianCartWrtSpher(r, a, b):
    '''
    jacobiano de las coordenadas cartesianas de un vector respecto a sus
    respectivas coordenadas esfericas. es decir, el jacobiano de la funcion
    que transforma las coordenadas esfericas en cartesianas
    '''
    
    ca, cb = np.cos([a, b])
    sa, sb = np.sin([a, b])
    
    sbca = sb*ca
    sacb = sa*sb
    
    J = np.array([[sbca, -r*sacb, r*ca*cb],
                  [sa*sb,  r*sbca, r*sacb],
                  [   cb,        0,   -r*sb]])
    
    return J

jacobianCartWrtSpher(4, 8, 0.2)
JrV_rVsph_eval

# %%

def jacobianRotatedWrtRvecSph(r, a, b, vx, vy, vz):
    '''
    jacobian of the rotated vector with respect to the rotation vector
    expressed in its spherical coordinates
    '''
    cr, ca, cb = np.cos([r, a, b])
    sr, sa, sb = np.sin([r, a, b])
    
    sasb = sa*sb
    sbca = sb*ca
    sacb = sa*cb
    cacb = ca*cb
    cr1 = 1 - cr
    
    aux1 = (vx*sbca + vy*sasb + vz*cb)
    aux2 = (vx*cacb + vy*sacb - vz*sb)
    aux3 = (-vx*sasb + vy*sbca)
    aux3cr1 = aux3*cr1
    aux4 = (vz*sr + aux3cr1)
    cr1aux1 = cr1*aux1
    
    J = np.array([
                    [(vz*sasb - vy*cb)*cr + sr*(aux1*sbca - vx),
                    sbca*aux4 - cr1aux1*sasb,
                    (vy*sb + vz*sacb)*sr + cr1*(aux1*cacb + aux2*sbca)],
                    
                    [ (vx*cb - vz*sbca)*cr + sr*(aux1*sasb - vy),
                    sasb*aux4 + cr1aux1*sbca,
                    + cr1*(aux1*sacb + aux2*sasb) - (vx*sb + vz*cacb)*sr],
                    
                    [aux3*cr + sr*(aux1*cb - vz),
                    aux3cr1*cb - (vx*sbca + vy*sasb)*sr,
                    (vy*cacb - vx*sacb)*sr - cr1*(aux1*sb + aux2*cb)]
                    ])
    
    return J

#
jacobianRotatedWrtRvecSph(4, 8, 0.2, 1, 2, 3)
Jv2_rVsph_eval


# %% matriz de rotacion a partir del vector de rodriguez
rrMat = sy.Matrix([[rx**2, rx*ry, rx*rz],[ry*rx, ry**2, ry*rz],[rz*rz, rz*ry, rz**2]]) / r**2
rMat = sy.Matrix([[0, -rz, ry],[rz, 0, -rx],[-ry, rx, 0]]) / r

R = sy.eye(3) * ct + (1-ct) * rrMat + st *rMat

# %%
xp, yp = sy.symbols('xp, yp')

a = R[0,0] - R[2,0] * xp
b = R[0,1] - R[2,1] * xp
c = tx - tz * xp
d = R[1,0] - R[2,0] * yp
e = R[1,1] - R[2,1] * yp
f = ty - tz * yp
q = a * e - d * b

x = (f * b - c * e) / q
y = (c * d - f * a) / q

X = sy.Matrix([x,y])
Xp = sy.Matrix([xp,yp])

# tengo que saacar el jacobiano respecto a xp, yp, rV y tV
# para el jacobiano respecto a rV primero lo hago respecto a rVsph y uso
# cain rule