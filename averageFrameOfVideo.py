# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:39:51 2016

takes a video and calculates the average of all frames (not BG, just average).
saves said average frame

@author: sebalander
"""
# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %%
videoForAveraging = "/home/sebalander/code/VisionUNQextra/robotMovil_Ulises/puntosCalibracion.mp4"
averageFrameFile = "/home/sebalander/code/VisionUNQextra/robotMovil_Ulises/puntosCalibracionImage.png"

# %% INITIALISE VIDEO
cap = cv2.VideoCapture(videoForAveraging)

# %% LOOP

ret, frame = cap.read()
frameAvrg = frame.copy()
ret, frame = cap.read()

while ret:
    N = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print(N)
    alfa = (N - 1) / N
    
    frameAvrg = cv2.addWeighted(frameAvrg, alfa, frame, 1-alfa, 0)
    
    ret, frame = cap.read()

# %%

cv2.imwrite(averageFrameFile, frameAvrg)

cap.release()