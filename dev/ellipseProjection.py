#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 18:02:20 2017

@author: sebalander
"""

# %%
import sympy as sy
import numpy as np

# %% defino variables
x, y, z, xm, ym, mx, my, s, a, b, c = sy.symbols('x, y, z, xm, ym, mx, my, s, a, b, c')
r11, r12, r21, r22, r31, r32, tx, ty, tz = sy.symbols('r11, r12, r21, r22, r31, r32, tx, ty, tz')



# %% expresion de la rototraslacion
R = np.array([[r11, r12],[r21, r22], [r31, r32]])
T = np.array([tx, ty, tz])
xym = np.array([xm, ym])

# coords en camara ref frame from map coords
x, y, z = R.dot(xym) + T


# %% expresion del conoide
C = np.array([[a,b],[b,c]])
X3d = np.array([[x-mx*z],[y-my*z]])

# esta expresion debe ser cero
conoide = (X3d.T.dot(C).dot(X3d) - s*z).reshape(-1)[0]
conexp = sy.expand(conoide)

#sy.collect(conexp, xm**2)
#sy.collect(conexp, ym**2)
#sy.collect(conexp, ym)
#sy.collect(conexp, xm*ym)

cuadXmYm = conexp.as_poly(xm, ym)


# %% tieneq que dar igual a esta expresion:
X, Y, MX, MY, A, B, C = sy.symbols('X, Y, MX, MY, A, B, C')

XX = np.array([[X - MX],[Y - MY]])
equiv = XX.T.dot(np.array([[A,B],[B,C]])).dot(XX).reshape(-1)[0]
equiv = sy.expand(equiv)
equiv = equiv.as_poly(X,Y)

'''
# los tres elementos de la martiz de covarianza:
A*X**2
+ 2*B*X*Y
+ C*Y**2
# ahora el centro, dos ecs con 2 incognitas
+ (-2*A*MX - 2*B*MY)*X
+ (-2*B*MX - 2*C*MY)*Y
# y esto parece que no sirve
+ A*MX**2
+ 2*B*MX*MY
+ C*MY**2
'''
# %%
# de aca sale C11
C11 = (a*mx**2*r31**2 - 2*a*mx*r11*r31 + a*r11**2 + 2*b*mx*my*r31**2 - 2*b*mx*r21*r31 - 2*b*my*r11*r31 + 2*b*r11*r21 + c*my**2*r31**2 - 2*c*my*r21*r31 + c*r21**2)

# de aca sale C12, C21
C12 = + (2*a*mx**2*r31*r32 - 2*a*mx*r11*r32 - 2*a*mx*r12*r31 + 2*a*r11*r12 + 4*b*mx*my*r31*r32 - 2*b*mx*r21*r32 - 2*b*mx*r22*r31 - 2*b*my*r11*r32 - 2*b*my*r12*r31 + 2*b*r11*r22 + 2*b*r12*r21 + 2*c*my**2*r31*r32 - 2*c*my*r21*r32 - 2*c*my*r22*r31 + 2*c*r21*r22) / 2

# de aca sale C22
#  + (a*mx**2*r32**2 - 2*a*mx*r12*r32 + a*r12**2 + 2*b*mx*my*r32**2 - 2*b*mx*r22*r32 - 2*b*my*r12*r32 + 2*b*r12*r22 + c*my**2*r32**2 - 2*c*my*r22*r32 + c*r22**2)*ym**2
C22 = + (a*mx**2*r32**2 - 2*a*mx*r12*r32 + a*r12**2 + 2*b*mx*my*r32**2 - 2*b*mx*r22*r32 - 2*b*my*r12*r32 + 2*b*r12*r22 + c*my**2*r32**2 - 2*c*my*r22*r32 + c*r22**2)

'''
# de estas dos que siguen saco mux, muy pero es mas facil hacer la proyeccion
+ (2*a*mx**2*r31*tz - 2*a*mx*r11*tz - 2*a*mx*r31*tx + 2*a*r11*tx + 4*b*mx*my*r31*tz - 2*b*mx*r21*tz - 2*b*mx*r31*ty - 2*b*my*r11*tz - 2*b*my*r31*tx + 2*b*r11*ty + 2*b*r21*tx + 2*c*my**2*r31*tz - 2*c*my*r21*tz - 2*c*my*r31*ty + 2*c*r21*ty - r31*s)*xm

+ (2*a*mx**2*r32*tz - 2*a*mx*r12*tz - 2*a*mx*r32*tx + 2*a*r12*tx + 4*b*mx*my*r32*tz - 2*b*mx*r22*tz - 2*b*mx*r32*ty - 2*b*my*r12*tz - 2*b*my*r32*tx + 2*b*r12*ty + 2*b*r22*tx + 2*c*my**2*r32*tz - 2*c*my*r22*tz - 2*c*my*r32*ty + 2*c*r22*ty - r32*s)*ym

# esta es redundante
+ a*mx**2*tz**2 - 2*a*mx*tx*tz + a*tx**2 + 2*b*mx*my*tz**2 - 2*b*mx*ty*tz - 2*b*my*tx*tz + 2*b*tx*ty + c*my**2*tz**2 - 2*c*my*ty*tz + c*ty**2 - s*tz
'''






