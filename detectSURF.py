'''
test surf detection and FLANN matching

jan 2017

@author: sebalander
'''
# %% imports
import numpy as np
import cv2
from copy import deepcopy as dc
import matplotlib.pyplot as plt

# %% data

# sample screenshot
imFile = "/home/sebalander/Code/VisionUNQextra/2016-11-13 medicion/vcaSnapShot.png"
roiFile = '/home/sebalander/Code/VisionUNQextra/2016-11-13 medicion/roi.png'

im = cv2.imread(imFile)
showKp = dc(im)
roi = cv2.imread(roiFile,cv2.IMREAD_GRAYSCALE)
cv2.namedWindow('frame',cv2.WINDOW_KEEPRATIO+cv2.WINDOW_AUTOSIZE)


# %% surf on single image
surf = cv2.xfeatures2d.SURF_create(1000)

keypoints, descriptors = surf.detectAndCompute(im,roi)
showKp = cv2.drawKeypoints(im,keypoints,showKp)
cv2.imshow('frame',cv2.pyrDown(showKp[:,:,::-1]))


# %% surf on video
imFile = "/home/sebalander/Code/VisionUNQextra/2016-11-13 medicion/vcaSnapShot.png"
roiFile = '/home/sebalander/Code/VisionUNQextra/2016-11-13 medicion/roi.png'
vidFile = '/home/sebalander/Code/VisionUNQextra/2016-11-13 medicion/Videos/vca_2016-11-12 17:44:31.487787.avi'

roi = cv2.imread(roiFile,cv2.IMREAD_GRAYSCALE)[:904] # lo recorto xq es mas chico?

vid = cv2.VideoCapture(vidFile) # open video
frame = vid.read() # get one frame
sh = np.shape(frame)
imProc = dc(frame) # image to be processed

surf = cv2.xfeatures2d.SURF_create(500)
bs = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=16, detectShadows=False)

coloredIm = np.array([[[255,0,0] 
                            for j in range(sh[1])] 
                                 for i in range(sh[0])],dtype=np.uint8)

# %%
i=0
cv2.namedWindow('frame',cv2.WINDOW_KEEPRATIO+cv2.WINDOW_AUTOSIZE)
vid = cv2.VideoCapture(vidFile) # open video

blurKernel = [3,3] # np.array([[0,1,0],[1,1,1],[0,1,0]])
erodeKernel = np.ones([7,7])
dilateKernel = np.ones([5,5])


while vid.isOpened(): #  and i<500:
    print(i)
    frame = vid.read()[1]
    frame2 = cv2.blur(frame,blurKernel)
    frameRoi = cv2.bitwise_and(frame2,frame2,mask=roi)
    
    fgmask = bs.apply(frameRoi)
    #fgmask = cv2.dilate(fgmask,dilateKernel)
    #fgmask = cv2.erode(fgmask,erodeKernel)
    #fgmask = cv2.dilate(fgmask,dilateKernel)
    
    colored2 = cv2.bitwise_and(coloredIm,coloredIm,mask=fgmask)
    im = cv2.addWeighted(frame,0.5,colored2,0.5,0)
    #im = cv2.bitwise_and(frame,frame,mask=roi)
    
    # keypoints_now, descriptors_now = surf.detectAndCompute(frame,mask=roi)
    #im = cv2.drawKeypoints(frame,keypoints,im)
    
    cv2.imshow('frame',im)
    cv2.waitKey(10)
    i+=1

# %%
vid.release()
