# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:30:53 2016

calibrates using fisheye distortion model (polynomial in theta)

help in
http://docs.opencv.org/ref/master/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d&gsc.tab=0

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

# %% LINEAR APPROACH TO CAMERA CALIBRATION
n = corners.shape[0] # number of images
m = corners.shape[1] # points per image

#i = 3
#j = 31
#x = corners[i,0,j]
#p = fiducialPoints[0,j]
cer = np.zeros(4)


def lineas(x,y,xp,yp):
    
    lin = [[x, 0, -x*xp, y, 0, -y*xp, 1, 0, -xp],
     [0, x, -x*yp, 0, y, -y*yp, 0, 1, -yp]]
    
    return lin

i = 0 # indice de imagen, hasta n
j = 0 # indice de punto, hasta m

xp = corners[i,j,0,0] - imgSize[0]/2
yp = corners[i,j,0,1] - imgSize[1]/2
x = fiducialPoints[0,j,0]
y = fiducialPoints[0,j,1]

lineas(x,y,xp,yp)

def H(fiducialPoints, corner, imgSize, f):
    m = fiducialPoints.shape[1]
    h = [ lineas(fiducialPoints[0,j,0],
                 fiducialPoints[0,j,1],
                 (corner[j,0,0] - imgSize[0]/2)/f,
                 (corner[j,0,1] - imgSize[1]/2)/f)
         for j in range(m)]
    
    h = np.array(h)
    h = np.reshape(h,(m*2,9))
    return h

H(fiducialPoints, corners[0], imgSize, f)

def rotTras1(fiducialPoints, corner, imgSize, f):
    
    h = H(fiducialPoints, corner, imgSize, f)
    
    # uso svd
    U, S, V = linalg.svd(h)
    # elijo el menor
    v = V[8]
    # chequear que no este repetido?
    r1 = v[0:3]
    r2 = v[3:6]
    tVec = v[6:9].reshape((3,1))
    
    # np.dot(r1, r2) # no dan perpendiculares
    r3 = np.cross(r1,r2)
    
    # componer matriz de rotacion cruda
    # ortonormalizarla
    rot = np.array([r1, r2, r3])
    
    rot = rot.dot(linalg.inv(linalg.sqrtm(rot.T.dot(rot))))
    
    rVec, _ = cv2.Rodrigues(rot)
    
    return rVec, tVec

def initialPoses1(fiducialPoints, corners, imgSize, f):
    initialPoses = [rotTras1(fiducialPoints, corner, imgSize, f) for corner in corners]
    
    rVecsIni = [pose[0] for pose in initialPoses]
    tVecsIni = [pose[1] for pose in initialPoses]
    
    return rVecsIni, tVecsIni

rVecsIni, tVecsIni = initialPoses1(fiducialPoints, corners, imgSize, f)

# %% 

def initialPoses3(fiducialPoints, corners, imgSize, f):
    
    t = np.mean(fiducialPoints[0],axis=0)+[0,0,1]
    tVec = t.reshape((3,1))
    
    rot = np.array([[1,0,0],
                    [0,-1,0],
                    [0,0,-1]], dtype='float32')
    rVec = cv2.Rodrigues(rot)[0]
    
    rVecsIni = [rVec]*corners.shape[0]
    tVecsIni = [tVec]*corners.shape[0]
    
    return rVecsIni, tVecsIni

rVecsIni, tVecsIni = initialPoses3(fiducialPoints, corners, imgSize, f)

# %% 

def rotTras2(fiducialPoints, corner, imgSize, f):
    # i = 8
    # corner = corners[i]
    srcPoints = fiducialPoints[0,:,:2]
    dstPoints = (corner[:,0] - np.array(imgSize)/2) / f
    method = 0 #cv2.RANSAC
    
    homography = cv2.findHomography(srcPoints, dstPoints, method=method)[0]
    
    r1 = homography[:,0]
    r2 = homography[:,1]
    r3 = np.cross(r1,r2)
    
    # get versors of A described in B ref frame
    #r1B = homography[:,0]
    #r2B = homography[:,1]
    #r3B = np.cross(r1B,r2B)
    
    # convert to versors of B described in A ref frame
    #r1 = np.array([r1B[0],r2B[0],r3B[0]])
    #r2 = np.array([r1B[1],r2B[1],r3B[1]])
    #r3 = np.array([r1B[2],r2B[2],r3B[2]])
    
    rot = np.array([r1, r2, r3]).T
    # make sure is orthonormal
    rot = rot.dot(linalg.inv(linalg.sqrtm(rot.T.dot(rot))))
    rVec, _ = cv2.Rodrigues(rot) # make into rot vector
    
    # rotate to get displ redcribed in A
    # tVec = np.dot(rot, -homography[2]).reshape((3,1))
    
    tVec = homography[:,2].reshape((3,1))
    
    return [rVec, tVec]



def initialPoses2(fiducialPoints, corners, imgSize, f):
    
    initialPoses = [rotTras2(fiducialPoints, corner, imgSize, f) for corner in corners]
    
    rVecsIni = [pose[0] for pose in initialPoses]
    tVecsIni = [pose[1] for pose in initialPoses]
    
    return rVecsIni, tVecsIni



# %%

model= 'rational'
i = 3

# %% 
# hacer optimizacion para terminar de ajustar el pinhole y asi sacar la pose para cada imagen. usar las funciones de las librerias extrinsecas
# es para cada imagen por separado
f = 1e3
linearCoeffs = np.array([[f, 0, imgSize[0]/2],
                         [0, f, imgSize[1]/2],
                         [0, 0, 1]], dtype='float32')

distCoeffs = np.zeros((14,1)) # no hay distorsion

# %% test initial calibration INVERSE

rVecsIni, tVecsIni = initialPoses2(fiducialPoints, corners, imgSize, f)


fiducialProjectedIni = pc.inverse(corners[i],
                                  rVecsIni[i], tVecsIni[i],
                                  linearCoeffs, distCoeffs, model)

# plot
# pc.fiducialComparison(fiducialPoints, fiducialProjectedIni)
pc.fiducialComparison3D(rVecsIni[i], tVecsIni[i], fiducialPoints, fiducialProjectedIni, label1 = 'Fiducial points', label2 = 'Projected points')


# %% optimise rVec, tVec
rVecOpt, tVecOpt, optParams = pc.calibrateInverse(fiducialPoints, corners[i], rVecsIni[i], tVecsIni[i], linearCoeffs, distCoeffs,model)

fiducialProjectedOpt = pc.inverse(corners[i], rVecOpt, tVecOpt, linearCoeffs, distCoeffs,model)
# plot
pc.fiducialComparison(fiducialPoints, fiducialProjectedOpt)
pc.fiducialComparison3D(rVecOpt, tVecOpt, fiducialPoints, fiducialProjectedOpt, label1 = 'Fiducial points', label2 = 'Optimised points')


# %% test initial calibration DIRECT

cornersProjectedIni = pc.direct(fiducialPoints,
                                rVecsIni[i], tVecsIni[i],
                                linearCoeffs, distCoeffs, model)

# plot corners in image
img = cv2.imread(images[i])
pc.cornerComparison(img, corners[i], cornersProjectedIni)

# %% optimise rVec, tVec
rVecOpt, tVecOpt, optParams = pc.calibrateDirect(fiducialPoints, corners[i], rVecsIni[i], tVecsIni[i], linearCoeffs, distCoeffs, model)

# test mapping with optimised conditions
cornersProjectedOpt = pc.direct(fiducialPoints, rVecOpt, tVecOpt, linearCoeffs, distCoeffs, model)
pc.cornerComparison(img, corners[i], cornersProjectedOpt)


# %%
# para todas las imagenes juntas dejar la pose fija y optimizar los parametros de distorsion 

# %% optimizar todo junto a la vez tomando lo anterior como condiciones iniciales.