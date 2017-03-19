# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 17:08:24 2016

@author: sebalander
"""
# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# %%
# video = "/home/sebalander/code/VisionUNQextra/2 hrs/01_20140820_100717.avi"
video = "/home/sebalander/code/VisionUNQextra/balkon/balkon02_athirdsize0115-0630.avi"
cap = cv2.VideoCapture(video)

ret, frame = cap.read()

xy = [[394],[367]]

# %%
rgb = list()
while ret:
    print(cap.get(cv2.CAP_PROP_POS_FRAMES),
          ' out of ',
          cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rgb.append(frame[xy][0])
    ret, frame = cap.read()

# %%
cap.release()
# %%
rgb = np.array(rgb)
s = rgb.shape
rgb = np.reshape(rgb, (s[0],s[1]*s[2])).T

# %%
plt.plot(rgb[0],'-+r',rgb[1],'-+g',rgb[2],'-+b')

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.plot(rgb[0],rgb[1],rgb[2],'+k')

# %% plot while showing color coordinate
cap = cv2.VideoCapture(video)
ret, frame = cap.read()
n = 20

# chech un LumCbCr color space or something like that
traza = np.array([frame[xy[::-1]][0] for i in range(n) ]).T


plt.ion()
figFrame = plt.figure(1)
axFr = figFrame.gca()
frame = cv2.circle(frame,(xy[0][0], xy[1][0]),5,0)
foto = axFr.imshow(frame[:,:,::-1])


figRGB = plt.figure(2)
axRGB = figRGB.gca(projection='3d')
# axRGB = figRGB.gca()
plotRGB = axRGB.plot(traza[0], traza[1],traza[2],'-+')[0]
axRGB.set_xbound(0,255)
axRGB.set_ybound(0,255)
axRGB.set_zbound(0,255)

while ret:
    print(cap.get(cv2.CAP_PROP_POS_FRAMES),
          ' out of ',
          cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    traza[:,:-1] = traza[:,1:]
    traza[:,-1] = np.array(frame[xy[::-1]][0])
    plotRGB.set_xdata(traza[0])
    plotRGB.set_ydata(traza[1])
    plotRGB.set_3d_properties(zs=traza[2])
    
    frame = cv2.circle(frame,(xy[0][0], xy[1][0]),5,0)
    foto.set_data(frame[:,:,::-1])
    # figFrame.canvas.draw_idle()
    ret, frame = cap.read()
    plt.pause(0.1)



