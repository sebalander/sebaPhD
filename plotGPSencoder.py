# -*- coding: utf-8 -*-
'''
grafiacar las posiciones de gps para ver como dieron, si se puede estimar la
trayectoria masomenos bien y si de ahi se puede calibrar un poco mejor el
encoder

@author: sebalander
'''
# %%
import numpy as np
import matplotlib.pyplot as pt
import datetime
import time

# %%

ar = "resources/20161113192738.txt"

'''
# descomentar esta parte para cargar datos dela otra forma
data = np.loadtxt(ar,
                  dtype=str,
                  delimiter=',',
                  skiprows=3).T

# time, lat, lon, elevation, accuracy, bearing, speed, satellites, provider, hdop, vdop, pdop, geoidheight, ageofdgpsdata, dgpsid

times = [datetime.datetime.strptime(t,"%Y-%m-%dT%H:%M:%SZ") for t in data[0] ]

times = np.array([time.mktime(t.timetuple()) for t in times])



lat = np.array(map(float, data[1]))
lon = np.array(map(float, data[2]))
alt = np.array(map(float, data[3]))
acc = np.array(map(float, data[4]))
spe = np.array(map(float, data[6]))
sts = np.array(map(int,   data[7]))
hdp = np.array(map(float, data[9]))
#vdp = map(float, data[10])
#pdp = map(float, data[11])

'''

def datestr2num(st):
    dttme = datetime.datetime.strptime(str(st),"b'%Y-%m-%dT%H:%M:%SZ'")
    return time.mktime(dttme.timetuple())

converters = {0 : datestr2num}

times = np.loadtxt(ar,
                  dtype=int,
                  delimiter=',',
                  converters=converters,
                  skiprows=1,
                  usecols=(0,)).T


def str2num(st):
    return float(str(st)[2:-1])


converters = {1 : str2num,
              2 : str2num,
              3 : str2num,
              4 : str2num,
              6 : str2num}

data = np.loadtxt(ar,
                  dtype=float,
                  delimiter=',',
                  converters=converters,
                  skiprows=1,
                  usecols=(1,2,3,4,6)).T
#                  usecols=(1,2,3,4,5,6,7,9,10,11)).T

lat = data[0]
lon = data[1]
ele = data[2]
acc = data[3]
spe = data[4]


# %% posiciones

pt.plot(lon, lat, '-+')
pt.show()

# %% ploteo los datos en gral
fig, axes = pt.subplots(3,1,sharex=True)
axes[0].plot(times,lat)
axes[1].plot(times,lon)
axes[2].plot(times,spe*3.6,) # en km/h

pt.show()

# %% altura
pt.figure()
pt.plot(times,alt)
pt.show()

# %% convertir angulo a metros necesito el radio de la tierra en ese punto y la
# altura sobre wgs84
# usar wgs84 https://www.nga.mil/ProductsServices/GeodesyandGeophysics/Pages/WorldGeodeticSystem.aspx
#http://earth-info.nga.mil/GandG/wgs84/gravitymod/wgs84_180/intptW.html

# segun wikipedia los semiejes del elipsoide:
# https://en.wikipedia.org/wiki/World_Geodetic_System
a = 6378137.0 # Semi-major axis a [m]
b = 6356752.314245 # Semi-minor axis b [m]

x = a * np.cos(lat) * np.cos(lon)
y = a * np.cos(lat) * np.sin(lon)
z = b * np.sin(lat)

r = np.linalg.norm([x,y,z],axis=0)

# corrijo por las diferentes alturas:
k = (r + alt) / r
x *= k
y *= k
z *= k

# %%
from mpl_toolkits.mplot3d import Axes3D

fig = pt.figure()
ax = fig.gca(projection='3d')
ax.plot(x,y,z)

# %% proyecto sobre plano perpendicular a punto medio (referencia arbitraria)
latM = np.mean(lat)
lonM = np.mean(lon)

clat = np.cos(latM)
slat = np.sin(latM)
clon = np.cos(lonM)
slon = np.sin(lonM)


Rx = np.array([[1,0,0],
               [0,clon,-slon],
               [0,slon,clon]])

Ry = np.array([[clat,0,-slat],
               [0,1,0],
               [slat,0,clat]])

R = np.dot(Rx,Ry)

xyz = np.array([x,y,z])
X, Y, Z = np.dot(R,xyz)

xM = np.mean(x)
yM = np.mean(y)
zM = np.mean(z)

xyzM = np.array([xM, yM, zM])/np.linalg.norm([xM, yM, zM])

pt.errorbar(X,Y,acc,acc)




