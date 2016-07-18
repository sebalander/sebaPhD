# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:32:04 2016

we take the intrinsic parameters of the camera and calibrate it's pose using
fiducial points in the floor

@author: sebalander
"""

# %%
import cv2
import numpy as np
import glob

# %% LOAD DATA
imgShapeFile = "./resources/fishWideShape.npy"
distCoeffsFile = "./resources/fishWideDistCoeffs.npy"
linearCoeffsFile = "./resources/fishWideLinearCoeffs.npy"



imgSize = tuple(np.load(imgShapeFile))

distCoeffs = np.load(distCoeffsFile)
cameraMatrix = np.load(linearCoeffsFile)

