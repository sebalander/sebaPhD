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
from glob import glob

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
roiFile = '/home/sebalander/Code/VisionUNQextra/2016-11-13 medicion/roi.png'
videosFile = '/home/sebalander/Code/VisionUNQextra/2016-11-13 medicion/Videos/vca_2016*.avi'
vidOutput = '/home/sebalander/Code/VisionUNQextra/2016-11-13 medicion/Videos/vca_BS.avi'


vidList = glob(videosFile)
vidList.sort()
vidN = len(vidList)
roi = cv2.imread(roiFile,cv2.IMREAD_GRAYSCALE)[:904] # lo recorto xq es mas chico?

# open first video to get one frame sample
vid = cv2.VideoCapture(vidList[0])
vid.isOpened()
frame = vid.read()[1] # get one frame
sh = np.shape(frame)
vid.release()

# create surf object
surf = cv2.xfeatures2d.SURF_create(500)
bs = cv2.createBackgroundSubtractorMOG2(history=5000,
                                        varThreshold=16,
                                        detectShadows=False)

# an image of plain color
coloredIm = np.array([[[255,255,0] 
                            for j in range(sh[1])] 
                                 for i in range(sh[0])],dtype=np.uint8)

# open ocv window
cv2.namedWindow('frame',cv2.WINDOW_KEEPRATIO+cv2.WINDOW_AUTOSIZE)

# filtering parameters
blurKsize = (7,7)
blurSigmaX = 2
erodeKernel = np.ones([2,2])
dilateKernel = np.ones([5,5])

# %% open video for writing
fcc = 
out = cv2.VideoWriter(vidOutput,['XVID'],10,sh[:1])


# %%
for i, vidFile in enumerate(vidList):
    vid = cv2.VideoCapture(vidFile)
    framesN = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    
    while vid.get(cv2.CAP_PROP_POS_FRAMES) < framesN: #vid.isOpened:
        
        frame = vid.read()[1]
        frame2 = cv2.GaussianBlur(frame,blurKsize,blurSigmaX) # reduce noise
        #frameRoi = cv2.bitwise_and(frame2,frame2,mask=roi) # apply roi mask
        
        fgmask = bs.apply(frame)
        #fgmask = cv2.dilate(fgmask,dilateKernel)
        #fgmask = cv2.erode(fgmask,erodeKernel)
        #fgmask = cv2.dilate(fgmask,dilateKernel)
        
        colored2 = cv2.bitwise_and(coloredIm,coloredIm,mask=fgmask)
        im = cv2.addWeighted(frame,0.5,colored2,0.5,0)
        #im = cv2.bitwise_and(frame,frame,mask=roi)
        
        # keypoints_now, descriptors_now = surf.detectAndCompute(frame,mask=roi)
        #im = cv2.drawKeypoints(frame,keypoints,im)
        textIm = "Frame %d de %d. "%(vid.get(cv2.CAP_PROP_POS_FRAMES),framesN)
        textIm = textIm + "Video  " + vidFile[68:87]
        
        im = cv2.putText(im,textIm,(50,850),
                         cv2.FONT_HERSHEY_COMPLEX ,1,[0,0,0])
        
        cv2.imshow('frame',cv2.pyrDown(im))
        cv2.waitKey(1)
    vid.release()

