# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Mon Oct 23 18:00:37 2017



@author: sebalander
"""
# %% imports
from numpy import deg2rad, cos, sin, sqrt, array, pi
from numpy.linalg import norm

# %% funciton definitions

# constantes
a = 6378137 # este
f = 1 / 298.257223563
#b = a * (1 - f)
e2 = f * (2 - f)
#e = np.sqrt(e2)
#ep = e * a / b
a2 = a**2
b2 = a2 * (1 - f)**2


def lla2ecef(lla):
    '''
    convierte lat lon y elevacion sobre WGS a coords ECEF
    
    N/rad ardius of curvatiure meters
    phi es latitude 
    lambda es longitud
    h heigth above ellipsoid (meters)
    
    fuente:
        Datum Transformations of GPS Positions, Application Note, 5th July 1999
        https://microem.ru/files/2012/08/GPS.G1-X-00006.pdf
    '''
    llaT = lla.T # easier to manipulate vectors
    
    la = deg2rad(llaT[0]) # to radians
    lo = deg2rad(llaT[1])
    
    cla = cos(la) # precalculate sine cosine
    sla = sin(la)
    clo = cos(lo)
    slo = sin(lo)
    
    rad = a / sqrt(1 - e2 * sla**2) # radius of curvature
    
    r = rad + llaT[2] # add topoligical elevation wrt to ellipsoid
    
    X = r * cla * clo
    Y = r * cla * slo
    Z = (b2 * rad / a2 + llaT[2]) * sla
    
    return array([X, Y, Z]).T


def ltpR(lla0):
    '''
    returns the ORIGIN and ROTATION matrix to postmultiply ecef coordinates
    so thatraise RuntimeError("Cannot activate multiple GUI eventloops")
    they are in the local tangent plane, east being x, north being y, height z.

    given a X ecef 3D coord use
    > (X - T).dot(R)
    to convert to ltp frame of reference
    '''
    T = lla2ecef(lla0)
    
    fi = pi / 2 - deg2rad(lla0[0]) # decliancion desde el eje z
    theta = deg2rad(lla0[1]) # longitud desde el eje x horizontal
    
    ct = cos(theta)
    st = sin(theta)
    cf = cos(fi)
    sf = sin(fi)
    
    '''
    defino los versores este oeste y altura
    http://mathworld.wolfram.com/SphericalCoordinates.html
    tambien coincide en como se convierte velocidades en 
    Datum Transformations of GPS Positions, Application Note, 5th July 1999
    '''
    # versor en la direccion radial
    rVersor = T / norm(T)
    # e direccion este, versor theta
    eastVersor = array([-st, ct, 0])
    # versor norte, opueto el versor fi
    northVersor = - array([ct * cf, st * cf, -sf])
    
    # proyecto sobre la rotacion que interesa
    # matriz de rotacion para post multiplicar
    R = array([eastVersor, northVersor, rVersor]).T
    return R, T


def lla2ltp(lla, lla0):
    '''
    converts lla to local tangent plane coords centering in another given lla
    X is a (N,3) or (3,) vector containing lat, lon, alt in columns.
    X0 is a (3,) vector. altitude is wrt WGS84 reference ellipsoid.
    '''
    
    X = lla2ecef(lla)
    
    R, T = ltpR(lla0)
    
    return (X - T).dot(R)
