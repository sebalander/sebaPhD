# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:47:18 2016

defines the class poseCalibrator that recieves extrinsic calibration data:
- fiducial points
- corners found on image that correspond to the fiducial points
- camera intrinsic parameters
- initial pose guess for pose
and calculates:
- optimum pose minimizing error in image
- optimum pose minimizing error in scene

also perhaps plots things and in the future calculates error.
long term: help user define initial pose guess, help find corners in image

following tutorial in http://www.tutorialspoint.com/python/python_classes_objects.htm

@author: sebalander
"""
# %% IMPORTS
import poseStereographicCalibration as stereographic

import poseRationalCalibration as rational

import poseUnifiedCalibration as unified

import poseFisheyeCalibration as fisheye

# %% ========== ========== DIRECT FISHEYE ========== ==========
def directFisheye():
    '''TODO'''
# %% ========== ========== INVERSE FISHEYE ========== ==========
def inverseFisheye():
    '''TODO'''

