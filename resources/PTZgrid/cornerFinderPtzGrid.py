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
imageFile = "ptz_(0.850278, -0.014444, 0.0).jpg"
cornersIniFile = "PTZgridImageInitialConditions.txt"

# output
cornersFile = "ptzCorners.npy"
patternFile = "ptzGridPattern.npy"
imgShapeFile = "ptzImgShape.npy"

# load
# corners set by hand, read as (n,1,2) size
# must format as float32
cornersIni = np.array([[crnr] for crnr in np.loadtxt(cornersIniFile)],
                       dtype='float32')
img = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)
imgCol = cv2.imread(imageFile)

# %% BINARIZE IMAGE 
# see http://docs.opencv.org/3.0.0/d7/d4d/tutorial_py_thresholding.html
th = cv2.adaptiveThreshold(img,
                            255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY,
                            501,
                            0)

# haceomos un close para sacar manchas
kernel = np.ones((5,5),np.uint8)
closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

plt.imshow(th)
plt.imshow(closed)
plt.imshow(imgCol)
plt.plot(cornersIni[:,0,0],cornersIni[:,0,1],'ow')

# %% refine corners

# criterio de finalizacion de cornerSubPix
subpixCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, # termination criteria type
            300, # max number of iterations
            0.01) # min accuracy

corners = np.copy(cornersIni)

cv2.cornerSubPix(closed,
                 corners,
                 (15, 15),
                 (5, 5),
                 subpixCriteria);


plt.imshow(imgCol[:,:,[2,1,0]])
plt.plot(cornersIni[:,0,0],cornersIni[:,0,1],'+r', label="Initial")
plt.plot(corners[:,0,0],corners[:,0,1],'xb', label="Optimized")
plt.legend()

# %% DEFINE FIDUCIAL POINTS IN 3D SCENE, by hand
# shape must be (1,n,3), float32
nx = 8
ny = 12
xx = range(nx)
y0 = 12
yy = range(y0,y0-ny,-1)

grid = np.array([[[[x, y, 0] for x in xx] for y in yy]], dtype='float32')
grid = grid.reshape((1,nx*ny,3))
toDelete = np.logical_and(grid[0,:,0] < 2, grid[0,:,1] < 2)
grid = grid[:,np.logical_not(toDelete),:]

# scale to the size of A4 sheet
grid[0,:,0] *= 0.21
grid[0,:,1] *= 0.297
# %% PLOT FIDUCIAL POINTS
fig = plt.figure()
from mpl_toolkits.mplot3d import Axes3D
ax = fig.gca(projection='3d')
ax.scatter(grid[0,:,0], grid[0,:,1], grid[0,:,2])
plt.show()

# %% SAVE DATA POINTS
np.save(cornersFile, corners)
np.save(patternFile, grid)
np.save(imgShapeFile, img.shape)
