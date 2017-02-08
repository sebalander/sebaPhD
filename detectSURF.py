'''
test surf detection and FLANN matching

jan 2017

@author: sebalander
'''
# %% imports
import numpy as np
import cv2
from copy import deepcopy as dc
# import matplotlib.pyplot as plt
from glob import glob
from skimage.morphology import disk
from skimage.filters import rank
from skimage.color.adapt_rgb import adapt_rgb, each_channel


@adapt_rgb(each_channel)
def localEqualize(image, selem):
    return rank.equalize(image, selem=selem)

# %% data de entrada
path = '/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/2016-11-13 medicion/'
roiFile = path + 'roi.png'
videosFile = path + 'Videos/vca_2016*.avi'
vidOutputBSSURF = path + 'Videos/vca_BS_SURF.avi'

showOnscreen = True
writeToDIsk = True

surfTrhes = 500
surfHist = 5000
blurSigmaX = 2
equalDiskSize = 500
nVideosMax = 5  # number of videos to process, default 0

# %% preparacion o inicializacion

vidList = glob(videosFile)
vidList.sort()
vidN = len(vidList)

# filtering parameters
blurKsize = (blurSigmaX*4+1, blurSigmaX*4+1)
selem = disk(equalDiskSize)

# open first video to get one frame sample
vid = cv2.VideoCapture(vidList[0])
vid.isOpened()
frame = vid.read()[1]  # get one frame
sh = np.shape(frame)  # get it's shape
vid.release()

roi = cv2.imread(roiFile, cv2.IMREAD_GRAYSCALE)[:904]  # recorto xq es mas chico?
im = dc(frame)

# create surf object
surf = cv2.xfeatures2d.SURF_create(surfTrhes)
bs = cv2.createBackgroundSubtractorMOG2(history=surfHist,
                                        varThreshold=16,
                                        detectShadows=False)

# an image of plain color
coloredIm = np.array([[[255, 255, 0] for j in range(sh[1])]
                      for i in range(sh[0])], dtype=np.uint8)


# %% ejecucion

# open ocv window
if showOnscreen:
    cv2.namedWindow('frame',
                    cv2.WINDOW_KEEPRATIO+cv2.WINDOW_AUTOSIZE)

# open video for writing
if writeToDIsk:
    fcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter(vidOutputBSSURF, fcc, 10, sh[:2])

for i, vidFile in enumerate(vidList):
    vid = cv2.VideoCapture(vidFile)
    framesN = vid.get(cv2.CAP_PROP_FRAME_COUNT)

    while vid.get(cv2.CAP_PROP_POS_FRAMES) < framesN:

        frame = vid.read()[1]

        frame2 = cv2.GaussianBlur(frame, blurKsize, blurSigmaX)  # reduce noise
        frame2 = localEqualize(frame2, selem)

        fgmask = bs.apply(frame2) #  esta linea da error:
                                  #  cv2.error: /build/opencv/src/opencv-3.1.0/modules/python/src2/cv2.cpp:163: error: (-215) The data should normally be NULL! in function allocate

        im = cv2.bitwise_and(coloredIm, coloredIm, mask=fgmask)
        im = cv2.addWeighted(frame, 0.5, im, 0.5, 0)

        keypoints_now, descriptors_now = surf.detectAndCompute(frame, mask=roi)
        im = cv2.drawKeypoints(im, keypoints_now, im)
        textIm = "Frame %d/%d. " % (vid.get(cv2.CAP_PROP_POS_FRAMES), framesN)
        textIm = textIm + "Video  " + vidFile[84:]

        print(textIm)

        im = cv2.putText(im, textIm, (50, 850),
                         cv2.FONT_HERSHEY_COMPLEX, 1, [0, 0, 0])
        if writeToDIsk:
            out.write(im)

        if showOnscreen:
            cv2.imshow('frame', cv2.pyrDown(im))
            keyPressed = cv2.waitKey(-1)


    vid.release()

    # salir despues de hacer 5 videos
    if nVideosMax!=0 and i > nVideosMax:
        break

# close
if writeToDIsk:
    out.release()

if showOnscreen:
    cv2.destroyAllWindows()

