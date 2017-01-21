# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 20:24:54 2016

@author: lilian

mueve la ptz y saca foto
"""
# %%
import cv2
from PTZCamera import PTZCamera
import numpy as np
from dotsphere import dotsphere1, dotsphere2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import sleep

def fovVsZoom(z):
    '''
    returns aprox field of viev for ptz in radians
    '''
    return 2 * np.arctan(0.0248305/(z + 0.0485368))

# %%
ip = '192.168.1.49'
portHTTP = 80
portRTSP = '554'
usr = 'admin'
psw = '12345'

cam = PTZCamera(ip, portHTTP, usr, psw)
cam.getStatus()

#url = 'rtsp://'+ip+':'+portRTSP+'/Streaming/Channels/1?transportmode=unicast&profile=Profile_1'
url = 'rtsp://' + ip + '/live.sdp'
cap = cv2.VideoCapture(url)
cap.isOpened()
domoPath = "./resources/PTZdomo/"



#%%
header = "encoderPan,  encoderTil,  fi,  theta,   points"

Zoom = [0.0]

# saca aprox 24 fotos por minuto

for z in Zoom:
    solidAngle = fovVsZoom(z)**2
    density = 4 * 4 * np.pi / solidAngle  # more than the "correct" density
    points = dotsphere2(density).T
    # remove points in upper half 
    points = points[:,points[2]>=0]
    nPics = points.shape[1]
    print(np.shape(points))
    
    # calculate corresponding angles
    fi = np.arctan2(points[1],points[0])   # pan angle?
    r_xy = np.sqrt(points[0]**2, points[1]**2)
    the = np.arctan2(r_xy, points[2]) # tilt angle?
    
    # sort to take less time
    indSorted = np.lexsort((the,fi))
    the, fi = the[indSorted], fi[indSorted]
    
    points = points[:,indSorted]
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[0],points[1],points[2],'-o')
    plt.plot([0],[0],[0])
    plt.show()
    
    # convert to encoder, asuming home position
    ePan = fi / np.pi
    eTil = the * 4 / np.pi -1
    
    data = np.concatenate(([ePan],[eTil],[fi],[the],points)).T
    
    np.savetxt("%sangulos_%1.1f.txt"%(domoPath,z),data,header=header)
    
    #plt.figure()
    for i in range(nPics):
        ep, et = ePan[i], eTil[i]
        cam.moveAbsolute(ep, et, z)
        sleep(0.1)
        print(i, 'de', nPics, ep, et)
        # muevo el indice a ultio frame que vio la camara
        cap.set(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO,1)
        ret = False
        while not ret:
            ret, frame = cap.read()

        # Guardar imagen
        cv2.imwrite('%sdomo_%1.1f_%d.jpg'%(domoPath,z,i), frame)


