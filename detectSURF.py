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


#@adapt_rgb(each_channel)
#def localEqualize(image, selem):
#    return rank.equalize(image, selem=selem)

## %% data de entrada
#path = '/home/sebalander/Code/VisionUNQextra/Videos y Mediciones/2016-11-13 medicion/'
#roiFile = path + 'roi.png'
#videosFile = path + 'Videos/vca_2016*.avi'
#vidOutputBSSURF = path + 'Videos/vca_BS_SURF.avi'
#
#showOnscreen = True
#writeToDIsk = True
#
#surfTrhes = 500
#surfHist = 5000
#blurSigmaX = 2
#equalDiskSize = 500
#nVideosMax = 5  # number of videos to process, default 0

# %% preparacion o inicializacion
class detector():
    '''
    para detectar BS y SURF
    '''
    def __init__(self, path,
                 roiFile='roi.png',
                 videosFile='Videos/vca_2016*.avi',
                 vidOutputBSSURF='Videos/vca_BS_SURF.avi',
                 surfThres=500,
                 bsHist=5000,
                 bsThres=16,
                 blurSigmaX=2,
                 equalDiskSize=500,
                 nVideosMax=3,
                 showOnscreen=True,
                 writeToDIsk=True):

        self.showOnscreen = showOnscreen
        self.writeToDIsk = writeToDIsk

        # paths
        self.path = path
        self.roiFile = path + roiFile
        self.videosFile = path + videosFile
        self.vidOutputBSSURF = path + vidOutputBSSURF

        # generate list of videos
        self.vidList = glob(self.videosFile)
        self.vidList.sort()
        self.vidN = len(self.vidList)

        # filtering parameters
        self.blurSigmaX = blurSigmaX
        self.blurKsize = (blurSigmaX*4+1, blurSigmaX*4+1)
        self.selem = disk(equalDiskSize)

        # open first video to get one frame sample
        vid = cv2.VideoCapture(self.vidList[0])
        vid.isOpened()
        frame = vid.read()[1]  # get one frame
        self.sh = np.shape(frame)  # get it's shape
        vid.release()

        # recorto xq es mas chico?
        self.roi = cv2.imread(self.roiFile, cv2.IMREAD_GRAYSCALE)[:904]
        self.im = dc(frame)

        # create surf object
        self.surf = cv2.xfeatures2d.SURF_create(surfThres)
        self.bs = cv2.createBackgroundSubtractorMOG2(history=bsHist,
                                                     varThreshold=bsThres,
                                                     detectShadows=False)

        # an image of plain color
        self.coloredIm = np.array([[[255, 255, 0] for j in range(self.sh[1])]
                              for i in range(self.sh[0])], dtype=np.uint8)

    # %% ejecucion
    def detectar(self):
        # open ocv window
        if self.showOnscreen:
            cv2.namedWindow('frame',
                            cv2.WINDOW_KEEPRATIO+cv2.WINDOW_AUTOSIZE)

        # open video for writing
        if self.writeToDIsk:
            fcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
            out = cv2.VideoWriter(self.vidOutputBSSURF,
                                  fcc, 10, self.sh[:2])

        for i, vidFile in enumerate(self.vidList):
            vid = cv2.VideoCapture(vidFile)
            framesN = vid.get(cv2.CAP_PROP_FRAME_COUNT)

            while vid.get(cv2.CAP_PROP_POS_FRAMES) < framesN:
                frame = vid.read()[1]

                # find surf keypoints
                keypoints_now, descriptors_now = (
                            self.surf.detectAndCompute(frame, mask=self.roi))

                # aplly to BS algorith MOG2
                frame2 = cv2.GaussianBlur(frame, self.blurKsize,
                                          self.blurSigmaX)  # reduce noise
                # equlize too slow
                # frame2 = localEqualize(frame2, self.selem)
                fgmask = self.bs.apply(frame2)
                # esta linea da error:
                # cv2.error: /build/opencv/src/opencv-3.1.0/modules/python/src2/cv2.cpp:163: error: (-215) The data should normally be NULL! in function allocate
                # ver https://github.com/opencv/opencv/issues/5667
                # parece que se arreglo en abril de 2016 pero la release
                # sigueinte es 3.2 asi que habra que instalar al nueva?


                # compose images
                im = cv2.bitwise_and(self.coloredIm, self.coloredIm,
                                     mask=fgmask)
                im = cv2.addWeighted(frame, 0.5, im, 0.5, 0)
                im = cv2.drawKeypoints(im, keypoints_now, im)

                textIm = "Frame %d/%d. " % (vid.get(cv2.CAP_PROP_POS_FRAMES),
                                            framesN)
                textIm = textIm + "Video  " + vidFile[len(self.path):]
        
                print(textIm)
        
                im = cv2.putText(im, textIm, (50, 850),
                                 cv2.FONT_HERSHEY_COMPLEX, 1, [0, 0, 0])
                if self.writeToDIsk:
                    out.write(im)
        
                if self.showOnscreen:
                    cv2.imshow('frame', cv2.pyrDown(im))
                    cv2.waitKey(1)
        
        
            vid.release()
        
            # salir despues de hacer 5 videos
            if self.nVideosMax!=0 and i > self.nVideosMax:
                break
        
        # close
        if self.writeToDIsk:
            out.release()
        
        if self.showOnscreen:
            cv2.destroyAllWindows()

