# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:34:13 2016

@author: sebalander
"""




# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %%
# input
# 6x9 chessboard
#imageFile = "./resources/fishChessboard/Screenshot from fishSeba.mp4 - 12.png"

# 8x11 A4 shetts chessboard
imageFile = "./resources/PTZgrid/ptz_(0.850278, -0.014444, 0.0).jpg"
cornersIniFile = "./resources/PTZgrid/PTZgridImageInitialConditions.txt"

# output
cornersFile = "./resources/PTZgrid/ptzCorners.npy"
patternFile = "./resources/PTZgrid/ptzChessPattern.npy"
imgShapeFile = "./resources/PTZgrid/ptzImgShape.npy"

# load
# corners set by hand, read as (n,1,2) size
cornersIni = np.array([[crnr] for crnr in np.loadtxt(cornersIniFile)])
img = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)


# %% BINARIZE IMAGE 
# see http://docs.opencv.org/3.0.0/d7/d4d/tutorial_py_thresholding.html
th = cv2.adaptiveThreshold(img,
                            255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY,
                            501,
                            0)

plt.imshow(th)
plt.scatter(cornersIni[:,0,0],cornersIni[:,0,1])

# %% FIND CORNERS (NOT WORKING)
# cantidad esquinas internas del tablero:
# los cuadraditos del chessboard-1
pattW = 8  # width 
pattH = 11  # height
patternSize = (pattW, pattH)

found, corners = cv2.findChessboardCorners(th, patternSize);

# %% refine corners

# criterio de finalizacion de cornerSubPix
subpixCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, # termination criteria type
            30, # max number of iterations
            0.1) # min accuracy

corners = cv2.cornerSubPix(th,
                           cornersIni,
                           (41, 41),
                           (9, 9),
                           subpixCriteria);

# %%
#Arrays para pts del objeto y pts de imagen para from all the images.
objpoints = [] #3d points in real world
imgpoints = [] #2d points in image plane





# Se arma un vector con la identificacion de cada cuadrito
# La dimensión de este vector es diferente al que se usa en la calibracion
# de la PTZ, es necesaria una dimension mas para que corra el calibrate
chessboardModel = np.zeros((1,pattH*pattW,3), np.float32)
chessboardModel[0, :, :2] = np.mgrid[0:pattW, 0:pattH].T.reshape(-1, 2) #rellena las columnas 1 y 2


# %% SAVE DATA POINTS
np.save(cornersFile, imgpoints)
np.save(patternFile, chessboardModel)
np.save(imgShapeFile, img.shape)
