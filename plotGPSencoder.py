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

<<<<<<< HEAD
ar = "resources/20161113192738.txt"
=======
ar = "resources/encoderGPS/20161113192738.txt"
indMax = 1900  # indice hasta donde tomar los datos de gps
>>>>>>> c4a14a75d27e3ec968a671f1971e993dac7f5ef8

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
<<<<<<< HEAD
                  usecols=(0,)).T

=======
                  usecols=(0,)).T[:indMax]

# restar 3 horas porque esta en UTC para que este en hora de argentina
times -= 3*60*60
inter = times[1:] - times[:-1] # [secs]
>>>>>>> c4a14a75d27e3ec968a671f1971e993dac7f5ef8

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
<<<<<<< HEAD
                  usecols=(1,2,3,4,6)).T
=======
                  usecols=(1,2,3,4,6))[:indMax].T
>>>>>>> c4a14a75d27e3ec968a671f1971e993dac7f5ef8
#                  usecols=(1,2,3,4,5,6,7,9,10,11)).T

lat = data[0]
lon = data[1]
ele = data[2]
acc = data[3]
spe = data[4]


<<<<<<< HEAD
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




=======

# %% cargar datos de encoder
# tiempos en microsegundos dt1 y dt2 son incertezas y Tau es el intervalo
dt1, dt2, Tau = np.loadtxt("resources/encoderGPS/m1_seba.txt").T
timesEnc0 = np.loadtxt("resources/encoderGPS/mtime_seba.txt")
vtime = np.loadtxt("resources/encoderGPS/vtime_seba.txt")

timesEnc = [datetime.datetime(np.int(t[0]),
                              np.int(t[1]),
                              np.int(t[2]),
                              np.int(t[3]),
                              np.int(t[4]),
                              np.int(t[5]),
                              np.int((t[5] - np.int(t[5]))*1e6))
            for t in timesEnc0]

timesEnc = np.array([time.mktime(t.timetuple()) + t.microsecond / 1.0e6
                     for t in timesEnc], dtype=np.float64)

# pt.plot(timesEnc,'-+')

# la vuelta es 
L = 3.78/2  # la distancia recorrida en una vuelta de rueda
sL = 0.03/2  # un sigma de incerteza
speEnc = 1e6 * L / Tau
speEncMax = 1e6 * (L + sL) / (Tau - dt2)
speEncMin = 1e6 * (L - sL) / (Tau + dt1)

# %% World Geodetic System 1984 (WGS 84)
# según wikipedia https://en.wikipedia.org/wiki/Geodetic_datum
# WGS 84 Defining Parameters Parameter
a = 6378137.0  # semi mayor axis [m]
f_1 = 298.257223563  # Reciprocal of flattening

# WGS 84 derived geometric constants
b = 6356752.3142  # Semi-minor axis m
e2 = 6.69437999014e-3  # First eccentricity squared
e22 = 6.73949674228e-3  # Second eccentricity squared

# to rectangular coordinates, calculate velocity
# https://en.wikipedia.org/wiki/Reference_ellipsoid
a2 = a**2
b2 = b**2

'''
voy a tomar siempre longitud antes de latitud para darle un orden
'''


def lonLat2Rect(long, lati, elev, a2, b2, inRadians=True):
    if not inRadians:
        lati, long = (np.deg2rad(lati), np.deg2rad(long))

    cLat = np.cos(lati)
    sLat = np.sin(lati)
    # radius of curvature in the prime vertical
    N = a2 / np.sqrt(a2 * cLat**2 + b2 * sLat**2)
    Nele = N + elev

    X = Nele * cLat * np.cos(long)
    Y = Nele * cLat * np.sin(long)
    Z = ((b2/a2) * N + elev) * sLat
    return X, Y, Z


# %% posiciones, saco las posiciones de redferencia tambien


f, a = pt.subplots()
pt.plot(lon, lat, '-+')
pos = []


def onclick(event):
    pos.append(np.array([event.xdata, event.ydata]))


f.canvas.mpl_connect('button_press_event', onclick)
f.show()

# %% tomo las ultimas dos posiciones clickeadas como referencia de cero y de
# direccion x
orig = np.empty(3)
dirX = np.empty(3)

try:
    orig[:2] = pos[-2]
    dirX[:2] = pos[-1]
except:
    orig[:2] = np.array([np.mean(lon), np.mean(lat)])
    dirX[:2] = np.array(orig + [1, 0])

# le elijo una elevacion en funcion de los datos mas cercanos
distOrig = (lon-orig[0])**2 + (lat-orig[1])**2
distOrigMin = np.min(distOrig)
# pongo como critero 5 distancias minimas
orig[2] = np.mean(ele[distOrig < distOrigMin*5])

# le elijo una elevacion en funcion de los datos mas cercanos
distDirX = (lon-dirX[0])**2 + (lat-dirX[1])**2
distDirXMin = np.min(distDirX)
# pongo como critero 5 distancias minimas
dirX[2] = np.mean(ele[distDirX < distDirXMin*5])


# %%
orig = lonLat2Rect(orig[0], orig[1], orig[2], a2, b2)
dirX = lonLat2Rect(dirX[0], dirX[1], dirX[2], a2, b2)

# calculo los versores de direccion del marco de referencia deseado respecto
# al marco de referencia tierra
xv = dirX / np.linalg.norm(dirX)
zv = orig / np.linalg.norm(orig)
yv = np.cross(zv, xv)

X, Y, Z = lonLat2Rect(lon, lat, ele, a2, b2)  # todas las posiciones
XYZearth = np.array([X, Y, Z]).T - orig

'''
chequear bien que este calculando correctamente la matriz de rotacion!!!!
'''

Rot = np.linalg.inv(np.array([xv, yv, zv]).T) # rotation from earth to calle

XYZ = np.dot(Rot, XYZearth.T)

pt.plot(XYZ[1],XYZ[2])

pt.plot(XYZ[0],XYZ[1])

# %% ploteo los datos en gral; RESTANDO 1 SEG A TIEMPO DE GPS 

#tlim = [times[0], times[1900]]

fig, axes = pt.subplots(2, 1, sharex=True)
axes[0].plot(times-1, X, '-+r')
axes[0].set_ylabel('X')
#axes[0].set_xlim(tlim)

ax2 = axes[0].twinx()
ax2.plot(times-1, Y, '-*b')
ax2.set_ylabel('Y')
#ax2.set_xlim(tlim)

axes[1].plot(timesEnc, speEnc, 'ro-', markersize=3)
axes[1].plot(timesEnc, speEncMax, 'r')
axes[1].plot(timesEnc, speEncMin, 'r')
axes[1].plot(times-1, spe, 'ko-', markersize=3)
axes[1].set_ylim([0, 20])
#axes[1].legend()
axes[1].set_xlabel('tiempo segundos desde 1979')
axes[1].set_ylabel('velocidad en m/s')

pt.show()
>>>>>>> c4a14a75d27e3ec968a671f1971e993dac7f5ef8
