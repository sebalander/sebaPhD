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

def pose2homogr(rVec,tVec,f):
    
    # calcular las puntas de los versores
    if rVec.shape == (3,3):
        [x,y,z] = rVec
    else:
        [x,y,z], _ = cv2.Rodrigues(rVec)
    
    
    H = np.array([x,y,tVec[:,0]]).T
    
    return f*H



def homogr2pose(H):
    r1B = H[:,0]
    r2B = H[:,1]
    r3B = np.cross(r1B,r2B)
    
    # convert to versors of B described in A ref frame
    r1 = np.array([r1B[0],r2B[0],r3B[0]])
    r2 = np.array([r1B[1],r2B[1],r3B[1]])
    r3 = np.array([r1B[2],r2B[2],r3B[2]])
    
    rot = np.array([r1, r2, r3]).T
    # make sure is orthonormal
    rotNorm = rot.dot(linalg.inv(linalg.sqrtm(rot.T.dot(rot))))
    rVec, _ = cv2.Rodrigues(rotNorm) # make into rot vector
    
    # rotate to get displ redcribed in A
    # tVec = np.dot(rot, -homography[2]).reshape((3,1))
    tVec = H[:,2].reshape((3,1))
    
    # rescale
    k = np.sqrt(np.linalg.norm(r1)*np.linalg.norm(r2))
    tVec = tVec / k
    
    return [rVec, tVec]



# %% custom homography

rVec = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
tVec = np.array([[4],[5],[8]])
f = 4.215687215

H = pose2homogr(rVec, tVec, f)

# %%
reload(pc)

# %% random homography
rVec = cv2.Rodrigues(pc.euler(np.random.rand()*np.pi*2,
                              np.random.rand()*np.pi*2,
                              np.random.rand()*np.pi*2))[0]
tVec = np.random.rand(3)*2 + 1
f = np.random.rand()

rVec = rVec.reshape(3,1)
tVec = tVec.reshape(3,1)

rVec
cv2.Rodrigues(rVec)[0]
#sarasa1(rVec)
#sarasa2(rVec)

tVec

pc.fiducialComparison3D(rVec, tVec, fiducialPoints)

H = pose2homogr(rVec, tVec, f)



# %%

src = fiducialPoints[0]+[0,0,1]
src2 = np.array([np.dot(H, sr) for sr in src])
dst = np.array([src2[:,0]/src2[:,2], src2[:,1]/src2[:,2]]).T

# %% plot 

plt.figure()
plt.plot(src[:,0], src[:,1],'+k')
plt.plot(dst[:,0], dst[:,1],'xr')



# %% fit with function

H2 = cv2.findHomography(src[:,:2], dst, method=8)[0]

H
H2
# %%

rVec2, tVec2 = homogr2pose(H2)

tVec
tVec2
# %%

pc.fiducialComparison3D(rVec2,tVec2,fiducialPoints)


## %%
#import cv2
#from cv2 import Rodrigues
#
#rVec = np.array([[ 1.26001985],
#                 [ 5.55466931],
#                 [ 2.65262444]])
#
#def sarasa1(rVec):
#    ret = Rodrigues(rVec)[0]
#    return ret
#
#
#def sarasa2(rVec):
#    ret = cv2.Rodrigues(rVec)[0]
#    return ret
#
## estos dos dan bien
#Rodrigues(rVec)[0]
#cv2.Rodrigues(rVec)[0]
#
## estos dos dan mal
#sarasa1(rVec)
#sarasa2(rVec)
#
