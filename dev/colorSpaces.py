# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 20:07:16 2016

takes  video and shows it in another color space to try to reduce
sadow inference

@author: sebalander
"""
# %%
import cv2

# %%
archivoVideo = '/home/sebalander/code/VisionUNQextra/2 hrs/01_20140926_140530.avi'
cap = cv2.VideoCapture(archivoVideo)
ret, frame = cap.read()
# %%
while ret:
    print(ret)
    cv2.imshow("frame", frame)
    ret, frame = cap.read()

# %%
cap.release()
cv2.destroyAllWindows()

