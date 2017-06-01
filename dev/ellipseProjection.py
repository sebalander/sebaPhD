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


# %%

def rototrasCovariance(Cp, xp, yp, rV, tV):
    '''
    propaga la elipse proyectada por el conoide soble el plano del mapa
    solo toma en cuenta la incerteza en xp, yp
    '''
    a, b, _, c = Cp.flatten()
    mx, my = (xp, yp)
    r11, r12, r21, r22, r31,r32 = cv2.Rodrigues(rV)[0][:,:2].flatten()
    tx, ty, tz = tV.flatten()
    
    C11 = (a*(mx**2*r31**2 + r11**2)
           + c*(r21**2 + my**2*r31**2)
           - 2*a*mx*r11*r31
           + 2*b*mx*my*r31**2
           - 2*b*mx*r21*r31
           - 2*b*my*r11*r31
           + 2*b*r11*r21
          
           - 2*c*my*r21*r31)
    
    C12 = (a*mx**2*r31*r32 - a*mx*r11*r32 - a*mx*r12*r31 + a*r11*r12 + 2*b*mx*my*r31*r32 - b*mx*r21*r32 - b*mx*r22*r31 - b*my*r11*r32 - b*my*r12*r31 + b*r11*r22 + b*r12*r21 + c*my**2*r31*r32 - c*my*r21*r32 - c*my*r22*r31 + c*r21*r22)
    
    C22 = (a*mx**2*r32**2 - 2*a*mx*r12*r32 + a*r12**2 + 2*b*mx*my*r32**2 - 2*b*mx*r22*r32 - 2*b*my*r12*r32 + 2*b*r12*r22 + c*my**2*r32**2 - 2*c*my*r22*r32 + c*r22**2)
    
    C = np.array([[C11, C12], [C12, C22]])
    
    #s=1
    #alfa = -(a*mx**2*r31*tz - a*mx*r11*tz - a*mx*r31*tx + a*r11*tx + 2*b*mx*my*r31*tz - b*mx*r21*tz - b*mx*r31*ty - b*my*r11*tz - b*my*r31*tx + b*r11*ty + b*r21*tx + c*my**2*r31*tz - c*my*r21*tz - c*my*r31*ty + c*r21*ty - r31*s)
    #
    #beta = -(a*mx**2*r32*tz - a*mx*r12*tz - a*mx*r32*tx + a*r12*tx + 2*b*mx*my*r32*tz - b*mx*r22*tz - b*mx*r32*ty - b*my*r12*tz - b*my*r32*tx + b*r12*ty + b*r22*tx + c*my**2*r32*tz - c*my*r22*tz - c*my*r32*ty + c*r22*ty - r32*s)
    #
    #MUx, MUy = ln.inv(C).dot([alfa, beta])

    return C


# %%
C11 = cuadXmYm.terms()[0][1]
C12 = cuadXmYm.terms()[1][1]
C22 = cuadXmYm.terms()[3][1]

sims, exp = sy.cse((C11, C12, C22))

def rototrasCovariance(Cp, xp, yp, rV, tV):
    '''
    propaga la elipse proyectada por el conoide soble el plano del mapa
    solo toma en cuenta la incerteza en xp, yp
    '''
    a, b, _, c = Cp.flatten()
    mx, my = (xp, yp)
    r11, r12, r21, r22, r31,r32 = cv2.Rodrigues(rV)[0][:,:2].flatten()
    tx, ty, tz = tV.flatten()
    br11 = b*r11
    bmx = b*mx
    amx = a*mx
    cmy = c*my
    x0, x1, x2, x3, x4, x8, x9, x10, x11, x12, x13, x14 = 2*np.array([br11, amx*r11, bmx*r21, br11*my, cmy*r21, bmx*my, b*r12, amx*r12, bmx*r22, b*my*r12, cmy*r22, r31*r32])
    x5, mx2, my2, x15 = np.array(r31, mx, my, r32)**2
    x6 = a*mx2
    x7 = c*my2
    
    C11 = a*r11**2 + c*r21**2 + r21*x0 - r31*x1 - r31*x2 - r31*x3 - r31*x4 + x5*x6 + x5*x7 + x5*x8,
    
    C12 = r12*(2*a*r11) + r21*x9 + r22*x0 + r22*(2*c*r21) - r31*x10 - r31*x11 - r31*x12 - r31*x13 - r32*x1 - r32*x2 - r32*x3 - r32*x4 + x14*x6 + x14*x7 + (4*r31*r32)*(bmx*my),
    
    C22 = a*r12**2 + c*r22**2 + r22*x9 - r32*x10 - r32*x11 - r32*x12 - r32*x13 + x15*x6 + x15*x7 + x15*x8]
    Cm = np.array([[C11, C12], [C12, C22]])
    
    return Cm