# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:18:54 2017

testear la deteccion surg y BS

@author: sebalander
"""

import detectSURF

# %% llamar con stos parametros de entrada
# crear clase con esto
path = '/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/2016-11-13 medicion/'
imFile = path + "vcaSnapShot.png"
roiFile = path + 'roi.png'
videosFile = path + 'Videos/vca_2016*.avi'
vidOutputBSSURF = path + 'Videos/vca_BS_SURF.avi'

# %% preparar para leer videos


