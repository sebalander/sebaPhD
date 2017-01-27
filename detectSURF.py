'''
test surf detection and FLANN matching

jan 2017

@author: sebalander
'''
# %% imports
import numpy as np
import cv2
from skimage import io as Iio
from  skvideo import io as Vio
from copy import deepcopy as dc
import matplotlib.pyplot as plt

# %% data

# sample screenshot
imFile = "/home/sebalander/Code/VisionUNQextra/2016-11-13 medicion/vcaSnapShot.png"
roiFile = '/home/sebalander/Code/VisionUNQextra/2016-11-13 medicion/roi.png'

im = cv2.imread(imFile)
showKp = dc(im)
roi = cv2.imread(roiFile,cv2.IMREAD_GRAYSCALE)

# %% surf on single image
surf = cv2.xfeatures2d.SURF_create(1000)

keypoints, descriptors = surf.detectAndCompute(im,roi)
showKp = cv2.drawKeypoints(im,keypoints,showKp)
io.imshow(showKp[:,:,::-1])


# %% surf on video
imFile = "/home/sebalander/Code/VisionUNQextra/2016-11-13 medicion/vcaSnapShot.png"
roiFile = '/home/sebalander/Code/VisionUNQextra/2016-11-13 medicion/roi.png'
vidFile = '/home/sebalander/Code/VisionUNQextra/2016-11-13 medicion/Videos/vca_2016-11-12 17:44:31.487787.avi'

roi = cv2.imread(roiFile,cv2.IMREAD_GRAYSCALE)[:904] # lo recorto xq es mas chico?
im = cv2.imread(imFile)
vidGen = Vio.vreader(vidFile)

surf = cv2.xfeatures2d.SURF_create(500)

# %%
fig = plt.imshow(im)
for frame in vidGen:
    
    frame.shape
    roi.shape
    
    keypoints, descriptors = surf.detectAndCompute(frame,mask=roi)
    
    im = cv2.drawKeypoints(frame,keypoints,im)
    
    fig.set_data(im)
    fig.draw()
     # no puedo mostrar la imagen dinamicamente

