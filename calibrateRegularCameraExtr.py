# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:23:21 2016

adjusts extrinsic calibration parameteres using fiducial points
must provide initial conditions

@author: sebalander
"""

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2.fisheye as fe
import glob

# %% LOAD DATA
# input files
imageFile = "/home/sebalander/code/sebaPhD/resources/fishGrid/FEsheetBoard1920pix.png"
cornersFile = "/home/sebalander/code/sebaPhD/resources/fishGrid/FEsheetCorners.npy"
patternFile = "./resources/fishGrid/FEsheetPattern.npy"
rvecInitialFile = "./resources/fishGrid/FErvecInitial.npy"
tvecInitialFile = "./resources/fishGrid/FEtvecInitial.npy"

# intrinsic parameters (input)
distCoeffsFile = "./resources/fishDistCoeffs.npy"
linearCoeffsFile = "./resources/fishLinearCoeffs.npy"

corners = np.load(cornersFile)
chessboardModel = np.load(patternFile)
distCoeffs = np.load(distCoeffsFile)
linearCoeffs = np.load(linearCoeffsFile)

# output files
poseFile = "./resources/fishGrid/FEpose.npy"


