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

rV = sy.Matrix([r, a, b])
v = sy.Matrix([vx, vy, vz])
tV = sy.Matrix([tx, ty, tz])

ct = sy.cos(r)

# %% esfericas y cartesianas
rx = rV[0] * sy.cos(rV[1]) * sy.sin(rV[2])
ry = rV[0] * sy.sin(rV[1]) * sy.sin(rV[2])
rz = rV[0] * sy.cos(rV[2])

rVcar = sy.Matrix([rx, ry, rz])
rhat = rVcar / r

# jacobiano de cartesianas derivadas respecto a esfericas
JrVcar_rV = rVcar.jacobian(rV)

# rotacion rodrigues
v2 = ct * v + sy.sin(r) * rhat.cross(v) + rhat * (rhat.dot(v)) * (1 - ct)

Jv2_rV = v2.jacobian(rV)

# %% pruebo con valores especificos
valuesDict = {r:4, a:8, b:0.2, vx:1, vy:2, vz:3, tx:7, ty:8, tz:9}

JrVcar_rV_eval = JrVcar_rV.subs(valuesDict).evalf()
Jv2_rV_eval = Jv2_rV.subs(valuesDict).evalf()

JrVcar_rV_eval = np.array(sy.matrix2numpy(JrVcar_rV_eval), dtype=float)
Jv2_rV_eval = np.array(sy.matrix2numpy(Jv2_rV_eval), dtype=float)

# busco el jacobiano del vector rotado con respecto al vector de rodrigues
# en cartesianas
Jv2_rVcar = Jv2_rV_eval.dot(np.linalg.inv(JrVcar_rV_eval))

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
JrVcar_rV_eval

# %%

def jacobianRotatedWrtRvecSph(r,a,b,vx,vy,vz):
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
jacobianRotatedWrtRvecSph(4, 8, 0.2,1,2,3)
Jv2_rV_eval


