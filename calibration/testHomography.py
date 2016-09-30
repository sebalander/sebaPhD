# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:06:58 2016

@author: sebalander
"""


# %%
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy import linalg
import poseCalibration as pc

# %% LOAD DATA
#imagesFolder = "./resources/fishChessboardImg/"
#cornersFile = "/home/sebalander/code/sebaPhD/resources/fishCorners.npy"
#patternFile = "/home/sebalander/code/sebaPhD/resources/chessPattern.npy"
#imgShapeFile = "/home/sebalander/code/sebaPhD/resources/fishShape.npy"

imagesFolder = "./resources/PTZchessboard/zoom 0.0/"
cornersFile = "./resources/PTZchessboard/zoom 0.0/ptzCorners.npy"
patternFile = "./resources/chessPattern.npy"
imgShapeFile = "./resources/ptzImgShape.npy"

corners = np.load(cornersFile).transpose((0,2,1,3))
fiducialPoints = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.jpg')

# output files
distCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzDistCoeffs.npy"
linearCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzLinearCoeffs.npy"
rvecsFile = "./resources/PTZchessboard/zoom 0.0/ptzRvecs.npy"
tvecsFile = "./resources/PTZchessboard/zoom 0.0/ptzTvecs.npy"

# %% custom homography

#xA = np.array([1,4,6])
#yA = np.array([-1,0,2])


Hcust = 2*np.array([[0,1,0],[0,0,1],[1,0,0]])

src = fiducialPoints[0]+[0,0,1]
src2 = np.array([np.dot(Hcust, sr) for sr in src])
dst = np.array([src2[:,0]/src2[:,2], src2[:,1]/src2[:,2]]).T

# %% plot 

plt.plot(src[:,0], src[:,1],'+k')
plt.plot(dst[:,0], dst[:,1],'xr')

# %% fit with function

H = cv2.findHomography(src[:,:2], dst, method=8)[0]

Hcust
H

# %%

rot = np.array([[1,0,0],
                [0,-1,0],
                [0,0,-1]], dtype='float32')
rVec = cv2.Rodrigues(rot)[0]

rot2 = cv2.Rodrigues(rVec)[0]

rot
rot2

