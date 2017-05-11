# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:30:53 2016

calibrates intrinsic with diff distortion model

@author: sebalander
"""

# %%
import glob
import numpy as np
from calibration import calibrator as cl
import matplotlib.pyplot as plt


# %% LOAD DATA
# input
# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'
# puede ser ['rational', 'fisheye', 'poly']
modelos = ['poly', 'rational', 'fisheye']
model = modelos[2]
plotCorners = False


imagesFolder = "./resources/intrinsicCalib/" + camera + "/"
cornersFile =      imagesFolder + camera + "Corners.npy"
patternFile =      imagesFolder + camera + "ChessPattern.npy"
imgShapeFile =     imagesFolder + camera + "Shape.npy"

# output
distCoeffsFile =   imagesFolder + camera + model + "DistCoeffs.npy"
linearCoeffsFile = imagesFolder + camera + model + "LinearCoeffs.npy"
tVecsFile =        imagesFolder + camera + model + "Tvecs.npy"
rVecsFile =        imagesFolder + camera + model + "Rvecs.npy"

# load data
imgpoints = np.load(cornersFile)
chessboardModel = np.load(patternFile)
imgSize = tuple(np.load(imgShapeFile))
images = glob.glob(imagesFolder+'*.png')

n = len(imgpoints)  # cantidad de imagenes
# Parametros de entrada/salida de la calibracion
objpoints = np.array([chessboardModel]*n)

## %% para que calibrar ocmo fisheye no de error
##    flags=flags, criteria=criteria)
##cv2.error: /build/opencv/src/opencv-3.2.0/modules/calib3d/src/fisheye.cpp:1427: error: (-215) svd.w.at<double>(0) / svd.w.at<double>((int)svd.w.total() - 1) < thresh_cond in function CalibrateExtrinsics
##http://answers.opencv.org/question/102750/fisheye-calibration-assertion-error/
## saco algunas captiras de calibracion
#indSelect = np.arange(n)
#np.random.shuffle(indSelect)
#indSelect = indSelect<10

# %% OPTIMIZAR
#rms, K, D, rVecs, tVecs = cl.calibrateIntrinsic(objpoints[indSelect], imgpoints[indSelect], imgSize, model)

#reload(cl)

rms, K, D, rVecs, tVecs = cl.calibrateIntrinsic(objpoints, imgpoints, imgSize, model)



# %% plot fiducial 
#points and corners to ensure the calibration data is ok
if plotCorners:
    for i in range(n): # [9,15]:
        rVec = rVecs[i]
        tVec = tVecs[i]
        fiducial1 = chessboardModel
        
        cl.fiducialComparison3D(rVec, tVec, fiducial1)
#

# %% TEST MAPPING (DISTORTION MODEL)
# pruebo con la imagen j-esima

if plotCorners:
    for j in range(n):  # range(len(imgpoints)):
        imagePntsX = imgpoints[j, 0, :, 0]
        imagePntsY = imgpoints[j, 0, :, 1]
    
        rvec = rVecs[j]
        tvec = tVecs[j]
    
        imagePointsProjected = cl.direct(chessboardModel, rvec, tvec, K, D, model)
        imagePointsProjected = imagePointsProjected.reshape((-1,2))
    
        xPos = imagePointsProjected[:, 0]
        yPos = imagePointsProjected[:, 1]
    
        plt.figure()
        im = plt.imread(images[j])
        plt.imshow(im)
        plt.plot(imagePntsX, imagePntsY, 'xr', markersize=10)
        plt.plot(xPos, yPos, '+b', markersize=10)
        #fig.savefig("distortedPoints3.png")
#

# %% SAVE CALIBRATION
np.save(distCoeffsFile, D)
np.save(linearCoeffsFile, K)
np.save(tVecsFile, tVecs)
np.save(rVecsFile, rVecs)