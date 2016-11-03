# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 21:29:10 2016

@author: lilian

graba en continuo sacando timestamp de cada frame con la vca
para opencv '2.4.9.1'
"""
# %%
import cv2     
from numpy import savetxt
import datetime
import psutil
import time
print("Libraries imported")

# %%
def captureTStamp(files, duration, cod,  fps=0, verbose=True):
    '''
    files = [ur, saveVideoFile, saveDateFile, saveMillisecondFile]
    duration = time in mintes
    cod = codec
    fps = frames per second for video to be saved
    verbose = print messages to screen
    
    si fpscam =0 trata de llerlo de la captura. para fe hay que especificarla
    
    
    Examples
    --------
    
    # para la FE
    duration = 1 # in minutes
    files = ['rtsp://192.168.1.48/live.sdp',
             "/home/alumno/Documentos/sebaPhDdatos/vca_test_video.avi",
             "/home/alumno/Documentos/sebaPhDdatos/vca_test_tsFrame.txt",
             "/home/alumno/Documentos/sebaPhDdatos/vca_test_msFrame.txt"]
    fpsCam = 12
    cod = 'X264'
    
    captureTStamp(files, duration, cod, fps=fpsCam)
    
    # para la PTZ
    
    '''
    
    fourcc = cv2.cv.CV_FOURCC(cod[0],cod[1],cod[2],cod[3]) # Códec de video
    
    if verbose:
        print(files)
        print("Duration",duration,"minutes")
        print("fps",fps)
        print("codec",cod)
    
    # Inicializacion
    tFin = datetime.datetime.now() + datetime.timedelta(minutes=duration)
    
    ts = []  # timestamp de la captura
    ms = []  # ts de la captura en ms
    
    
    # abrir captura
    cap = cv2.VideoCapture(files[0])
    if not cap.isOpened():
        print("capture not opened")
        return
    # configurar writer
    w = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    if not fps:
        fps = cap.get(cv2.CAP_PROP_FPS)
    #para fe especificar los fps pq toma cualquier cosa con la propiedad
    
    out = cv2.VideoWriter(files[1], fourcc, fps,( w, h), True)
    
    if verbose:
        print("capture open",cap.isOpened())
        print("frame size",w,h)
        print("output opened",out.isOpened())
    
    
    # Primera captura
    ret, frame = cap.read()
    if ret:
        t = datetime.datetime.now()
        ts.append(t)
        ms.append(int(round(time.time() * 1000))) # actual time in miliseconds
        out.write(frame)
        if verbose:
            print("first frame captured")
        
    # loop
    while (t <= tFin):
        ret, frame = cap.read()
        
        if ret:
            t = datetime.datetime.now()
            ts.append(t)
            ms.append(int(round(time.time() * 1000)))
            out.write(frame)
            if verbose:
                print("seconds elapsed and date",cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)/1000,t)
            
        else:
            break
    # end of loop
    
    # release and save
    out.release()
    cap.release()
    
    if verbose:
        print('loop exited, cap, out released, times saved to files')
        
    savetxt(files[2],ts, fmt= ["%s"])
    savetxt(files[3],ms, fmt= ["%f"])

# %% para la FE
#url = 'rtsp://192.168.1.48/Streaming/Channels/1?transportmode=unicast&profile=Profile_1'
duration = 0.2 # in minutes
files = ['rtsp://192.168.1.48/live.sdp',
         "/home/alumno/Documentos/sebaPhDdatos/vca_test_video.avi",
         "/home/alumno/Documentos/sebaPhDdatos/vca_test_tsFrame.txt",
         "/home/alumno/Documentos/sebaPhDdatos/vca_test_msFrame.txt"]
fpsCam = 20
cod = 'XVID'
    
# %%
captureTStamp(files, duration, cod, fpsCam)

# %% para la PTZ
duration = 0.2 # in minutes
files = ["rtsp://192.168.1.49/live.sdp",
         "/home/alumno/Documentos/sebaPhDdatos/ptz_test_video.avi",
         "/home/alumno/Documentos/sebaPhDdatos/ptz_test_tsFrame.txt",
         "/home/alumno/Documentos/sebaPhDdatos/ptz_test_msFrame.txt"]  

fpsCam = 20
cod = 'XVID'

captureTStamp(files, duration, cod, fpsCam)