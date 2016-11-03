# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 16:47:28 2016

@author: lilian, sebalander
"""
    
from onvif import ONVIFCamera
import PTZCamera

class IPCamera(ONVIFCamera):
    
    def __init__(self, host, port ,user, passwd):
        ONVIFCamera.__init__(self, host, port, user, passwd)
        self.mediaService = self.create_media_service()
        self.deviceService = self.create_devicemgmt_service()
        
        self.capture = None
        
    def getTimestamp(self):
        request = self.devicemgmt.create_type('GetSystemDateAndTime')
        response = self.devicemgmt.GetSystemDateAndTime(request)

        year = response.LocalDateTime.Date.Year
        month = response.LocalDateTime.Date.Month
        day = response.LocalDateTime.Date.Day
        
        hour = response.LocalDateTime.Time.Hour
        minute = response.LocalDateTime.Time.Minute
        second = response.LocalDateTime.Time.Second
        
        print str(year) + "/" + self.formatTimePart(month) + "/" + self.formatTimePart(day)
        print self.formatTimePart(hour) + ":" + self.formatTimePart(minute) + ":" + self.formatTimePart(second)
        
    def setTimestamp(self):
        request = self.devicemgmt.create_type('SetSystemDateAndTime')
#        request.DateTimeType = 'Manual'
#        request.UTCDateTime.Time.Hour = 11
        
        response = self.devicemgmt.SetSystemDateAndTime(request)
        
        print response

      
        
        
    def formatTimePart(self, param):
        sParam = str(param)
        if len(sParam) == 1:
            return "0" + sParam
        else:
            return sParam



# %%
from cv2 import VideoCapture, VideoWriter
from cv2.cv import CV_CAP_PROP_FPS as prop_fps
from cv2.cv import CV_FOURCC as fourcc
from cv2.cv import CV_CAP_PROP_FRAME_WIDTH as frame_width
from cv2.cv import CV_CAP_PROP_FRAME_HEIGHT as frame_height
from cv2.cv import CV_CAP_PROP_POS_MSEC as pos_msec
from numpy import savetxt
import datetime
from time import time

# %%
def captureTStamp(files, duration, cod,  fps=0, verbose=True):
    '''
    guarda por un tiempo en minutos (duration) el video levantado desde la
    direccion indicada en el archvo indicado. tambíen archivos con los time
    stamps de cada frame.
    
    files = [ur, saveVideoFile, saveDateFile, saveMillisecondFile]
    duration = time in mintes
    cod = codec
    fps = frames per second for video to be saved
    verbose = print messages to screen
    
    si fpscam =0 trata de llerlo de la captura. para fe hay que especificarla
    
    para opencv '2.4.9.1'
    
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
    
    # %% para la PTZ
    duration = 0.2 # in minutes
    files = ["rtsp://192.168.1.49/live.sdp",
             "/home/alumno/Documentos/sebaPhDdatos/ptz_test_video.avi",
             "/home/alumno/Documentos/sebaPhDdatos/ptz_test_tsFrame.txt",
             "/home/alumno/Documentos/sebaPhDdatos/ptz_test_msFrame.txt"]  
    
    fpsCam = 20
    cod = 'XVID'
    
    captureTStamp(files, duration, cod, fpsCam)
    
    '''
    
    fcc = fourcc(cod[0],cod[1],cod[2],cod[3]) # Códec de video
    
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
    cap = VideoCapture(files[0])
    if not cap.isOpened():
        print("capture not opened")
        return
    # configurar writer
    w = int(cap.get(frame_width))
    h = int(cap.get(frame_height))
    if not fps:
        fps = cap.get(prop_fps)
    #para fe especificar los fps pq toma cualquier cosa con la propiedad
    
    out = VideoWriter(files[1], fcc, fps,( w, h), True)
    
    if verbose:
        print("capture open",cap.isOpened())
        print("frame size",w,h)
        print("output opened",out.isOpened())
    
    
    # Primera captura
    ret, frame = cap.read()
    if ret:
        t = datetime.datetime.now()
        ts.append(t)
        ms.append(int(round(time() * 1000))) # actual time in miliseconds
        out.write(frame)
        if verbose:
            print("first frame captured")
        
    # loop
    while (t <= tFin):
        ret, frame = cap.read()
        
        if ret:
            t = datetime.datetime.now()
            ts.append(t)
            ms.append(int(round(time() * 1000)))
            out.write(frame)
            if verbose:
                print("seconds elapsed and date",cap.get(pos_msec)/1000,t)
            
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
