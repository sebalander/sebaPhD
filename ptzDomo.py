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
def angles2point(pan,tilt):
    return [np.cos(pan)*np.cos(tilt),
            np.sin(pan)*np.cos(tilt),
            np.sin(tilt)]


def point2angles(points):
    pan = np.arctan2(points[1],points[0])   # pan angle?
    r_xy = np.sqrt(points[0]**2, points[1]**2)
    tilt = np.arctan2(r_xy, points[2]) # tilt angle?
    return pan, tilt

# %%
def anglesMeridian(z, densityFactor):
    '''
    distributes pans and tilts according to custum made method that evenly
    distributes pictures along meridians
    '''
    
    a = fovVsZoom(z)
    Ntilts = np.int(np.ceil(densityFactor * np.pi / 2 / a))
    tilts = np.linspace(0,np.pi/2,Ntilts)
    
    # circunferencias calculated half the correct value so that the pans are 
    # half of the expected number
    # apparently the FOV in pan is twice the FOV in tilt
    circunferencias = np.cos(tilts) * np.pi
    Npans = np.array(np.ceil(densityFactor * circunferencias / a),dtype=np.int)
    pans = [np.linspace(-np.pi,np.pi,n) for n in Npans]
    
    t = 0
    angles = np.concatenate([[pans[t]], [np.ones(pans[t].shape)*tilts[t]]])
    
    for t in range(Ntilts)[1:]:
        aux = np.concatenate([[pans[t]], [np.ones(Npans[t])*tilts[t]]])
        angles = np.concatenate((angles,aux),axis=1)
    
    return angles[0], angles[1]

# %%
def anglesIco(z,densityFactor):
    '''
    distributes angles avenly using icosaedron method, via dotsphere
    
    '''
    solidAngle = fovVsZoom(z)**2
    density = densityFactor * 4 * np.pi / solidAngle  # more than the "correct" density
    points = dotsphere2(density).T
    # remove points in upper half 
    points = points[:,points[2]>=0]
    nPics = points.shape[1]
    print(np.shape(points))
    
    # calculate corresponding angles
    pan, tilt = point2angles(points)
    
    # sort to take less time
    indSorted = np.lexsort((tilt,pan))
    tilt, pan = tilt[indSorted], pan[indSorted]

    return pan, tilt

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
    densityFactor = 4 # increases image density wrpt the "correct" density
    
    pan, tilt = anglesMeridian(z, densityFactor)
    nPics = len(pan)
    # convert to encoder, asuming home position
    ePan = pan / np.pi
    eTil = tilt * 4 / np.pi -1
    
    data = np.concatenate(([ePan],[eTil],[pan],[tilt])).T
    
    np.savetxt("%sangulos_%1.1f.txt"%(domoPath,z),data,header=header)
    
    
    #plt.figure()
    for i in range(nPics):
        ep, et = ePan[i], eTil[i]
        cam.moveAbsolute(ep, et, z)
        sleep(1)
        print(i, 'de', nPics, ep, et)
        # muevo el indice a ultio frame que vio la camara
        cap.set(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO,1)
        ret = False
        while not ret:
            ret, frame = cap.read()

        # Guardar imagen
        cv2.imwrite('%seomo_%1.1f_%d.jpg'%(domoPath,z,i), frame)


