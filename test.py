# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 12:30:28 2016

test stuff

test inverse rational function

@author: sebalander
"""
# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

from inverseRational import inverseRational

# %% DATA FILES
imageFile = "/home/sebalander/code/sebaPhD/resources/fishGrid/FEsheetBoard1920pix.png"
cornersFile = "/home/sebalander/code/sebaPhD/resources/fishGrid/FEsheetCorners.npy"
patternFile = "./resources/fishGrid/FEsheetPattern.npy"

distCoeffsFile = "./resources/fishDistCoeffs.npy"
linearCoeffsFile = "./resources/fishLinearCoeffs.npy"

rvecOptimFile = "./resources/fishGrid/FEsheetRvecOptim.npy"
tvecOptimFile = "./resources/fishGrid/FEsheetTvecOptim.npy"

# %% LOAD DATA
img = cv2.imread(imageFile)
corners = np.load(cornersFile)
objectPoints = np.load(patternFile)

distCoeffs = np.load(distCoeffsFile)
cameraMatrix = np.load(linearCoeffsFile)

rVec = np.load(rvecOptimFile)
tVec = np.load(tvecOptimFile)

# %%
u,v = corners[0,0]

inverseRational(u, v, cameraMatrix, distCoeffs)


# %% test distortion
r = np.linspace(0,10,100)

def distort(r, k):
    '''
    returns distorted radious
    '''
    r2 = r**2
    r4 = r2**2
    r6 = r2*r4
    
    rd = r * (1 + k[0,0]*r2 + k[1,0]*r4 + k[2,0]*r6) / \
        (1 + k[3,0]*r2 + k[4,0]*r4 + k[5,0]*r6)
    return rd

rd = distort(r, distCoeffs)

plt.plot(r,rd)