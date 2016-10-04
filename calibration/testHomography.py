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
imgSize = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.jpg')

# output files
distCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzDistCoeffs.npy"
linearCoeffsFile = "./resources/PTZchessboard/zoom 0.0/ptzLinearCoeffs.npy"
rvecsFile = "./resources/PTZchessboard/zoom 0.0/ptzRvecs.npy"
tvecsFile = "./resources/PTZchessboard/zoom 0.0/ptzTvecs.npy"


# %%
reload(pc)

# %% random homography
#rVec = cv2.Rodrigues(pc.euler(np.random.rand()*np.pi*2,
#                              np.random.rand()*np.pi*2,
#                              np.random.rand()*np.pi*2))[0]
#tVec = np.random.rand(3)*2 + 1
#rVec = rVec.reshape(3,1)
#tVec = tVec.reshape(3,1)

rVec = np.array([[-1.17365947],
                 [ 1.71987668],
                 [-0.48076979]])
tVec = np.array([[ 2.53529204],
                 [ 1.53850073],
                 [ 1.362088  ]])
f = 1e2

pc.fiducialComparison3D(rVec, tVec, fiducialPoints)

H = pc.pose2homogr(rVec, tVec)


# %% produce sinthetic data

src = fiducialPoints[0]+[0,0,1]
src2 = np.array([np.dot(H, sr) for sr in src])
dst = np.array([f*src2[:,0]/src2[:,2],
                f*src2[:,1]/src2[:,2],
                np.ones(src.shape[0])]).T

# plot data
plt.figure()
plt.plot(src[:,0], src[:,1],'+k')
plt.plot(dst[:,0], dst[:,1],'xr')



# %% fit homography from sinthetic data

# fit homography
H2 = cv2.findHomography(src[:,:2], dst[:,:2], method=0)[0]
rVec, tVec = pc.homogr2pose(H2)

cv2.Rodrigues(rVec)[0]
tVec

# test it forward
src2F = np.array([np.dot(H2, sr) for sr in src])
dst2F = np.array([src2F[:,0]/src2F[:,2], src2F[:,1]/src2F[:,2]]).T

# plot data
plt.figure()
plt.plot(dst[:,0], dst[:,1],'+k')
plt.plot(dst2F[:,0], dst2F[:,1],'xr')
plt.title("En plano imagen")


# test it backward
Hi = sp.linalg.inv(H2)
dst2B = np.array([np.dot(Hi, ds) for ds in dst])
src2B = np.array([dst2B[:,0]/dst2B[:,2],
                  dst2B[:,1]/dst2B[:,2],
                  np.ones(src.shape[0])]).T


fiducialProjected = (src2B-[0,0,1]).reshape(fiducialPoints.shape)
pc.fiducialComparison3D(rVec, tVec,
                        fiducialPoints, fiducialProjected,
                        label1="fiducial points",
                        label2="ajuste")


# %%
def joinPoints(pts1, pts2):
    plt.figure()
    plt.plot(pts1[:,0], pts1[:,1],'+k')
    plt.plot(pts2[:,0], pts2[:,1],'xr')
    
    # unir con puntos
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i,0], pts2[i,0]],
                 [pts1[i,1], pts2[i,1]],'-k')



# %% load from real data

src = fiducialPoints[0]+[0,0,1]
f = 1e2 # proposal of f, can't be estimated from homography?
i=5
for i in [0,3,6,9]:
    i
    dst = (corners[i,:,0] - np.array(imgSize)/2) / f
    
    # cambio el signo de Y y parece que todo bien
    dst = np.concatenate((dst*[1,-1],
                          np.ones((dst.shape[0],1))),
                         axis=1)
    
    # plot data
    # plt.figure()
    # plt.plot(src[:,0], src[:,1],'+k')
    # plt.plot(dst[:,0], dst[:,1],'xr')
    # plt.title("%d src y dst en el mismo plano"%i)
    # joinPoints(src, dst)
    
    # fit homography
    H = cv2.findHomography(src[:,:2], dst[:,:2], method=0)[0]
    rVec, tVec = pc.homogr2pose(H)
    
    cv2.Rodrigues(rVec)[0]
    tVec
    
    # test it forward
    src2F = np.array([np.dot(H, sr) for sr in src])
    dst2F = np.array([src2F[:,0]/src2F[:,2], src2F[:,1]/src2F[:,2]]).T
    
    # plot data
    # plt.figure()
    # plt.plot(dst[:,0], dst[:,1],'+k')
    # plt.plot(dst2F[:,0], dst2F[:,1],'xr')
    # plt.title("%d en plano imagen"%i)
    # joinPoints(dst, dst2F)
    
    # test it backward
    Hi = sp.linalg.inv(H)
    dst2B = np.array([np.dot(Hi, ds) for ds in dst])
    src2B = np.array([dst2B[:,0]/dst2B[:,2],
                      dst2B[:,1]/dst2B[:,2],
                      np.ones(src.shape[0])]).T
    
    
    fiducialProjected = (src2B-[0,0,1]).reshape(fiducialPoints.shape)
    pc.fiducialComparison3D(rVec, tVec,
                            fiducialPoints, fiducialProjected,
                            label1="fiducial points",
                            label2="%d ajuste"%i)

