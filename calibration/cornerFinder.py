# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 21:50:26 2016

save chessboard pattern and image conrners to a file for later calibration (so
as not to do it again for every calibration model)

@author: lilian+sebalander
"""
# %%
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

# %%
hayquegraficar = True

# cam puede ser ['vca', 'vcaWide', 'ptz'] son los datos que se tienen
camera = 'vcaWide'

# input
imagesFolder = "./resources/intrinsicCalib/" + camera + "/"

# output
cornersFile =      imagesFolder + camera + "Corners.npy"
patternFile =      imagesFolder + camera + "ChessPattern.npy"
imgShapeFile =     imagesFolder + camera + "Shape.npy"

images = glob.glob(imagesFolder+'*.png')


# %%
# cantidad esquinas internas del tablero:
# los cuadraditos del chessboard-1
pattW = 9  # width 
pattH = 6  # height
patternSize = (pattW, pattH) 

# criterio de finalizacion de cornerSubPix
subpixCriteria = (cv2.TERM_CRITERIA_EPS, # + cv2.TERM_CRITERIA_MAX_ITER, # termination criteria type
            1000000, # max number of iterations
            0.001) # min accuracy


# %%
#Arrays para pts del objeto y pts de imagen para from all the images.
objpoints = [] #3d points in real world
imgpoints = [] #2d points in image plane

# Se arma un vector con la identificacion de cada cuadrito
# La dimensión de este vector es diferente al que se usa en la calibracion
# de la PTZ, es necesaria una dimension mas para que corra el calibrate
chessboardModel = np.zeros((1,pattH*pattW,3), np.float32)
chessboardModel[0, :, :2] = np.mgrid[0:pattW, 0:pattH].T.reshape(-1, 2) #rellena las columnas 1 y 2

# %%
noencuentra = []
for picture in images:
    img = cv2.imread(picture, cv2.IMREAD_GRAYSCALE);
    
    found, corners = cv2.findChessboardCorners(img, patternSize);
    
    if found:
        corners = cv2.cornerSubPix(img, corners, (11, 11), (1, 1), subpixCriteria);
        imgpoints.append(corners.reshape(1, -1, 2));
        
        if hayquegraficar:
            plt.figure()
            plt.imshow(cv2.imread(picture)[:,:,::-1])
            plt.scatter(corners[:,0,0], corners[:,0,1])
    
    else:
        print('No se encontraron esquinas en ' + picture)
        noencuentra.append(picture);

0 #

# %% SAVE DATA POINTS
np.save(cornersFile, imgpoints)
np.save(patternFile, chessboardModel)
# hay que pasar img shape al reves porque requiere width height en ese orden
np.save(imgShapeFile, img.shape[::-1])
