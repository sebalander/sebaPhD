# -*- coding: utf-8 -*-
"""
Created on Mon May 16 19:00:12 2016

reading arduino uno output

@author: sebalander
"""

# %% IMPORTS
import serial
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time


# %% DECLARATIONS
ser = serial.Serial('/dev/ttyUSB0', 9600)
N = 50  # number of data points to read
path = "/home/alumno/Code/sebaPhD/resources/arduinoSpeed/"

# %% LOOP
while True:
    data = list()
    arch = datetime.datetime.fromtimestamp(time.time())
    nombre = path+arch.strftime("%Y %m %d %H %M %S")+'txt'
    arch = open(nombre,'w')
    
    while len(data) < N:
        try:
            intervals = ser.readline()
            hora = datetime.datetime.fromtimestamp(time.time())
            # año mes dia hora minuto segundo microsegundo
            hora = hora.strftime("%Y %m %d %H %M %S %f  ")
            linea = hora + intervals
            #print(linea)
            data.append(linea)
            #print(len(data))
            print>>arch, linea
        except:
            pass
    
    #with open(arch,'w') as f:
    #    f.writelines('%s' % s for s in data)
    arch.close()
    print("file completed", nombre)
    
    del data
    del arch
    del intervals
    del hora
    del nombre

