#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:19:55 2018

simular las posese extrinsicas y los datos de calibracion

@author: sebalander
"""
# %% imports
import numpy as np
import matplotlib.pyplot as plt

# %%
s = (1600, 900)
fov = np.deg2rad(183 / 2)  # angulo hasta donde ve la camara
print((s[0] / 2) / np.tan(fov/2)) # radio maximo en la imagen y angulo maximo
print( 952*1600/1920)  # estiamndo del k de la camara en modo full
k = 800  # a proposed k for the camera, en realidad estimamos que va a ser #815

hs = np.array([7.5, 10])
angs = np.deg2rad([0, 30, 60])
Npts = np.array([10, 20])
rad = 100  # radio en metros

