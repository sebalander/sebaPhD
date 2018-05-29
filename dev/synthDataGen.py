#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:19:55 2018

simular las posese extrinsicas y los datos de calibracion

@author: sebalander
"""
# %% imports
import numpy as np
import matplotlib.pyplot as plt
import glob

class dataPaper():
    '''
    Fields with default values:

        model = 'stereographic' string indicating camera intrinsic model
                ['poly', 'rational', 'fisheye', 'stereographic']

        camera = 'vcaWide' string camera model

        s = np.array([1600, 900]) is the image size

        k = 800 is the sintehtic stereographic parameter

        uv= np.array([800, 450]) is the stereographic optical center

        cornersIn = None numpy array that holds the detected corners in
                    chessboard images

        chessModel = None numpy array of the chessboard model, 3D

        cornersEx = None numpy array of the real world corners in urban setting

        objExtr = None numpy array of corresponding lat-lon (or whatever unit)
                  of world calibration points.
    '''

    def __init__(self, model='stereographic', camera='vcaWide', s=[1600, 900],
                 k=800, uv=[800, 450], cornersIn=None,
                 chessModel=None, cornersEx=None, objExtr=None, imagesFiles=None):
        self.model = model
        self.camera = camera
        self.s = np.array(s)
        self.k = k
        self.uv = np.array(uv)
        self.cornersIn = cornersIn
        self.chessModel = chessModel
        self.cornersEx = cornersEx
        self.objExtr = objExtr
        self.imagesFiles = imagesFiles
        self.ocvRvecs = None
        self.ocvTvecs = None
        self.nIm = 0
        self.nPt = 0




# %% intrínsecos
camera = 'vcaWide'
model='stereographic'
s = np.array((1600, 900))
uv = s / 2.0
k = 800.0  # a proposed k for the camera, en realidad estimamos que va a ser #815

allData = dataPaper(model, camera, s, k, uv)

allData


# %% poses de calibratecamera y corners

imagesFolder = "./resources/intrinsicCalib/" + camera + "/"
cornersFile = imagesFolder + camera + "Corners.npy"
patternFile = imagesFolder + camera + "ChessPattern.npy"
imgShapeFile = imagesFolder + camera + "Shape.npy"

# model data files opencv
# distCoeffsFileOCV = imagesFolder + camera + 'fisheye' + "DistCoeffs.npy"
# linearCoeffsFileOCV = imagesFolder + camera + 'fisheye' + "LinearCoeffs.npy"
tVecsFileOCV = imagesFolder + camera + 'fisheye' + "Tvecs.npy"
rVecsFileOCV = imagesFolder + camera + 'fisheye' + "Rvecs.npy"


# ## load data
allData.cornersIn = np.load(cornersFile)
allData.chessModel = np.load(patternFile)
allData.s = np.load(imgShapeFile)
allData.imagesFiles = glob.glob(imagesFolder + '*.png')

allData.nIm = dataPaper.cornersIn.shape[0]
allData.nPt = dataPaper.cornersIn.shape[2]  # cantidad de puntos por imagen

# load model specific data from opencv Calibration
# distCoeffsOCV = np.load(distCoeffsFileOCV)
# cameraMatrixOCV = np.load(linearCoeffsFileOCV)
allData.ocvRvecs = np.load(rVecsFileOCV)
allData.ocvTvecs = np.load(tVecsFileOCV)

# %% table of poses sintéticas
fov = np.deg2rad(183 / 2)  # angulo hasta donde ve la camara
print((s[0] / 2) / np.tan(fov/2)) # radio maximo en la imagen y angulo maximo
print( 952*1600/1920)  # estiamndo del k de la camara en modo full

hs = np.array([7.5, 10])
angs = np.deg2rad([0, 30, 60])
Npts = np.array([10, 20])
rad = 100  # radio en metros


