#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 19:26:11 2018

@author: sebalander
"""
# %%
import cv2
import glob
import numpy as np
from cv2 import fisheye as fe
import matplotlib.pyplot as plt

images = glob.glob("/home/sebalander/Code/sebaPhD/resources/intrinsicCalib/vcaWide/vlcsnap*.png")

nIm = len(images)

# %%
# cantidad esquinas internas del tablero:
# los cuadraditos del chessboard-1
patternSize = (9, 6)

# criterio de finalizacion de cornerSubPix
subpixCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, # termination criteria type
            30, # max number of iterations
            0.001) # min accuracy

#Arrays para pts del objeto y pts de imagen para from all the images.
objectPoints = [] #3d points in real world
imagePoints = [] #2d points in image plane

# Se arma un vector con la identificacion de cada cuadrito
# La dimensión de este vector es diferente al que se usa en la calibracion
# de la PTZ, es necesaria una dimension mas para que corra el calibrate
chessboardModel = np.zeros((1,6*9,3), np.float32)
chessboardModel[0, :, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) #rellena las columnas 1 y 2


# %%
for picture in images:
    img = cv2.imread(picture, cv2.IMREAD_GRAYSCALE)

    found, corners = cv2.findChessboardCorners(img, patternSize)
    if found:
        cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), subpixCriteria)
        imagePoints.append(corners.reshape(1, -1, 2))


#        cv2.drawChessboardCorners(img, patternSize, corners, found)
#        cv2.imshow('Puntos detectados', img)
#        cv2.waitKey(0)
    else:
        print('No se encontraron esquinas en ' + picture)

imageSize = img.shape # [::-1]


# %% calibracion rational Extended, con incertezas

objectPoints = [chessboardModel]*nIm

# Parametros de entrada/salida de la calibracion
cameraMatrix = np.eye(4)
distCoeffs = np.zeros((14,1))

flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_ZERO_TANGENT_DIST

retAll = cv2.calibrateCameraExtended(objectPoints, imagePoints, imageSize,
                                     cameraMatrix, distCoeffs, flags=flags)

#[, rvecs[, tvecs[, stdDeviationsIntrinsics[, stdDeviationsExtrinsics[, perViewErrors[, flags[, criteria]]]]]]])

(retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics,
                             stdDeviationsExtrinsics, perViewErrors) = retAll
# stdDeviationsIntrinsics
# (f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3, s_4, \tau_x, \tau_y)

rvecs = np.array(rvecs)
tvecs = np.array(tvecs)

intrinsics = np.zeros((18,1))
intrinsics[:4, 0] = cameraMatrix[[0, 1, 0, 1],[0, 1, 2, 2]]
intrinsics[4:] = distCoeffs

stdDeviationsRots, stdDeviationsTras = stdDeviationsExtrinsics.reshape((nIm, 2, 3, 1)).transpose((1, 0, 2, 3))


#extrinsics = np.concatenate((rvecs, tvecs), axis=1).reshape((-1, 1))
rVecList = np.abs(rvecs).reshape(-1)
tVecList = np.abs(tvecs).reshape(-1)
intrList = np.abs(intrinsics).reshape(-1)

plt.figure()
plt.scatter(rVecList, stdDeviationsRots.reshape(-1), label='rvec')
plt.scatter(tVecList, stdDeviationsTras.reshape(-1), label='tvecs')
plt.scatter(intrList, stdDeviationsIntrinsics, label='intr')
for i in range(len(intrList)):
    plt.text(intrList[i], stdDeviationsIntrinsics[i], i)
plt.legend(loc=4)

# errores estimados:
eAbsR = np.mean(stdDeviationsRots.reshape(-1))
retPoly = np.polyfit(tVecList, stdDeviationsTras.reshape(-1), 1, full=True)
ab, residuals, rank, singular_values, rcond = retPoly
resSTD = np.sqrt(residuals / 3 / nIm)

print("\nerror relativo de f_x, f_y:")
print(stdDeviationsIntrinsics[:2].flatten() / intrList[:2])
print("\nerror de c_x, c_y:")
print(stdDeviationsIntrinsics[2:4].flatten(), 'pixeles')
print("\nerrores relativos de  k_1, k_2, k_3, k_4, k_5, k_6")
print(stdDeviationsIntrinsics[[4, 5, 8, 9, 10, 11]].flatten())
print('\najuste de error abs para Rvecs [rads, degs]')
print(eAbsR, np.deg2rad(eAbsR))
print("\najuste de recta para Tvecs y sigma")
print(ab[0], ab[1] + resSTD)

 # %%stdDeviationsIntrinsics[2:4].flatten()
# Parametros de entrada/salida de la calibracion
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(nIm)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(nIm)]
K = np.zeros((3, 3))
D = np.zeros((4, 1))
objectPoints = [chessboardModel] * nIm
#calibrationFlags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
calibrationFlags=cv2.fisheye.CALIB_FIX_SKEW
#calibrationFlags = 0 # se obtiene el mismo resultado que con el de arriba
calibrationCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, # termination criteria type
            100, # max number of iterations
            0.001) # min accuracy

rms, _, _, _, _ = fe.calibrate(objpoints,
                               imgpoints,
                               img.shape[::-1],
                               K,
                               D,
                               rvecs,
                               tvecs,
                               calibrationFlags,
                               calibrationCriteria)

# Estos son los valores de la PTZ -  Averiguar fe
apertureWidth = 4.8 #[mm]
apertureHeight = 3.6

# fovx: Output field of view in degrees along the horizontal sensor axis
# fovy: Output field of view in degrees along the vertical sensor axis
# focalLength: Focal length of the lens in mm
# principalPoint: Principal point in mm
# aspectRatio: fy/fx

fovx,fovy,focalLength,principalPoint,aspectRatio = \
    cv2.calibrationMatrixValues(K,
                                img.shape[::-1], # imageSize in pixels
                                apertureWidth,   #  Physical width in mm of the sensor
                                apertureHeight  # Physical height in mm of the sensor
                                )

print fovx
print fovy
print focalLength
print principalPoint
print aspectRatio



