# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:18:54 2017

testear la deteccion surg y BS

@author: sebalander
"""
# %%
import detectSURF
import importlib

# %% llamar con stos parametros de entrada
# crear clase con esto
path = '/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/2016-11-13 medicion/'
roiFile='roi.png'
videosFile='Videos/vca_2016*.avi'
vidOutputBSSURF='Videos/vca_BS_SURF.avi'
surfThres=500
bsHist=5000
bsThres=16
blurSigmaX=2
equalDiskSize=500
nVideosMax=3
showOnscreen = True
writeToDIsk = True

# %% preparar para leer videos
importlib.reload(detectSURF)
dtc = detectSURF.detector(path,
                          roiFile,
                          videosFile,
                          vidOutputBSSURF,
                          surfThres,
                          bsHist,
                          bsThres,
                          blurSigmaX,
                          equalDiskSize,
                          nVideosMax,
                          showOnscreen,
                          writeToDIsk)
# vars(dtc)

# %%
dtc.detectar()

