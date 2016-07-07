# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:23:21 2016

adjusts intrinsic and extrinsica calibration parameteres using fiducial points
must provide initial conditions

@author: sebalander
"""



# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2.fisheye as fe
import glob

# %% FUNCTION DECLARATION
# proyect points to image
def proyectRationalManualy(scene3Dpoints,
                           rvec,
                           tvec,
                           cameraMatrix,
                           distCoeffs,
                           imagePointsProjected):
    '''
    scene3Dpoints (1,n,3)
    rvec (3,1)
    tvec (3,1)
    cameraMatrix (3,3)
    distCoeffs (14,1)
    imagePointsProjected (n,1,2)  
    
    '''
    # roto translation of points
    R, _ = cv2.Rodrigues(rvec)
    rotated = np.array([XYZ.dot(R) for XYZ in  chessboardModel[0]])
    xyz = np.add(rotated, tvec.transpose())
   
   # divide by z
    x, y = [xyz[:,0]/xyz[:,2], xyz[:,1]/xyz[:,2] ]
    
    # rational distortion
    # auxiliar calculations    
    x2 = x**2
    y2 = y**2
    xy = 2*x*y
    r2 = x2 + y2
    r4 = r2**2
    r6 = r2*r4
    
    # key: distCoeffs is (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
    k = np.concatenate((distCoeffs[0:2,0],distCoeffs[4:8,0]))
    p = distCoeffs[2:4,0]
    s = distCoeffs[8:12,0]
    ratFraction = (1 + k[0] * r2 + k[1] * r4 + k[2] * r6) / \
                   (1 + k[3] * r2 + k[4] * r4 + k[5] * r6)
    
    x_d = x*ratFraction + p[0]*xy + p[1]*(r2 + 2*x2) + s[0]*r2 + s[1]*r4
    y_d = y*ratFraction + p[0]*(r2 + 2*y2) + p[1]*xy + s[2]*r2 + s[3]*r4
    # proyect to image
    ## terminar de ver en
    # http://docs.opencv.org/ref/master/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d&gsc.tab=0
    # return
    
    



# %% LOAD DATA
imagesFolder = "./resources/fishWideChessboardImg/"
cornersFile = "./resources/fishWideCorners.npy"
patternFile = "./resources/fishWidePattern.npy"
imgShapeFile = "./resources/fishWideShape.npy"
distCoeffsFile = "./resources/fishWideDistCoeffs.npy"
linearCoeffsFile = "./resources/fishWideLinearCoeffs.npy"

imgpoints = np.load(cornersFile)
chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.png')

# %%
x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)
np.add(x1, x2)
