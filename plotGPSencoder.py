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



