# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:06:58 2016

@author: sebalander
"""


# %%
import cv2
import numpy as np
import scipy as sp
import glob
import matplotlib.pyplot as plt
import poseCalibration as pc
np.random.seed(0)

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
imgSize = np.load(imgShapeFile)
images = glob.glob(imagesFolder+'*.jpg')

# output files
distCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzDistCoeffs.npy"
linearCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzLinearCoeffs.npy"
rvecsFile = "./resources/PTZchessboard/zoom 0.0/ptzRvecs.npy"
tvecsFile = "./resources/PTZchessboard/zoom 0.0/ptzTvecs.npy"


# %%
reload(pc)


# %% use real data
f = 5e2 # proposal of f, can't be estimated from homography

rVecs, tVecs, Hs = pc.estimateInitialPose(fiducialPoints, corners, f, imgSize)

pc.plotHomographyToMatch(fiducialPoints, corners[1:3], f, imgSize, images[1:3])

pc.plotForwardHomography(fiducialPoints, corners[1:3], f, imgSize, Hs[1:3], images[1:3])

pc.plotBackwardHomography(fiducialPoints, corners[1:3], f, imgSize, Hs[1:3])


# %% custom sinthetic homography

# estos valores se ven lindos, podrían ser random tambien
rVec = np.array([[-1.17365947],
                 [ 1.71987668],
                 [-0.48076979]])
tVec = np.array([[ 2.53529204],
                 [ 1.53850073],
                 [ 1.362088  ]])

pc.fiducialComparison3D(rVec, tVec, fiducialPoints)

H = pc.pose2homogr(rVec, tVec)


# %% produce sinthetic corners (no image to compare though)
f = 1e2
imgSize = np.array([800,600])

src = fiducialPoints[0]+[0,0,1]
dst = np.array([np.dot(H, sr) for sr in src])
dst = np.array([dst[:,0]/dst[:,2],
                dst[:,1]/dst[:,2]]).T
dst = f * dst + imgSize/2

# sinthetic corners. always have shape (Nimg,Npts,1,2)
corners = np.reshape(dst,(1,len(dst),1,2))


# %% test on sinthetic data
rVecs, tVecs, Hs = pc.estimateInitialPose(fiducialPoints, corners, f, imgSize)

pc.plotHomographyToMatch(fiducialPoints, corners, f, imgSize)

pc.plotForwardHomography(fiducialPoints, corners, f, imgSize, Hs)

pc.plotBackwardHomography(fiducialPoints, corners, f, imgSize, Hs)
