# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 16:47:28 2016

@author: lilian, sebalander
"""
    
from onvif import ONVIFCamera
#import PTZCamera

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
from cv2 import VideoCapture
from cv2.cv import CV_CAP_PROP_FPS as prop_fps
from cv2.cv import CV_FOURCC as fourcc
from cv2.cv import CV_CAP_PROP_FRAME_WIDTH as frame_width
from cv2.cv import CV_CAP_PROP_FRAME_HEIGHT as frame_height
from cv2.cv import CV_CAP_PROP_POS_MSEC as pos_msec

from skvideo.io import VideoWriter
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
    
    si fpscam=0 trata de llerlo de la captura. para fe hay que especificarla
    
    para opencv '2.4.9.1'
    
    Examples
    --------
    
    from cameraUtils import captureTStamp
    
    # para la FE
    duration = 1 # in minutes
    files = ['rtsp://192.168.1.48/live.sdp',
             "/home/alumno/Documentos/sebaPhDdatos/vca_test_video.avi",
             "/home/alumno/Documentos/sebaPhDdatos/vca_test_tsFrame.txt"]
    fpsCam = 12
    cod = 'XVID'
    
    captureTStamp(files, duration, cod, fps=fpsCam)
    
    # %% para la PTZ
    duration = 0.2 # in minutes
    files = ["rtsp://192.168.1.49/live.sdp",
             "/home/alumno/Documentos/sebaPhDdatos/ptz_test_video.avi",
             "/home/alumno/Documentos/sebaPhDdatos/ptz_test_tsFrame.txt"]  
    
    fpsCam = 20
    cod = 'XVID'
    
    captureTStamp(files, duration, cod, fpsCam)
    
    '''
    
    # fcc = fourcc(cod[0],cod[1],cod[2],cod[3]) # Códec de video
    
    if verbose:
        print(files)
        print("Duration",duration,"minutes")
        print("fps",fps)
        print("codec",cod)
    
    # Inicializacion
    tFin = datetime.datetime.now() + datetime.timedelta(minutes=duration)
    
    ts = list()  # timestamp de la captura
    
    # abrir captura
    cap = VideoCapture(files[0])
    while not cap.isOpened():
        cap = VideoCapture(files[0])
    
    print("capture opened")
	# configurar writer
    w = int(cap.get(frame_width))
    h = int(cap.get(frame_height))
    if not fps:
        fps = cap.get(prop_fps)
    #para fe especificar los fps pq toma cualquier cosa con la propiedad
    
    #out = VideoWriter(files[1], fcc, fps,( w, h), True)
    out = VideoWriter(files[1], frameSize=(w,h), fourcc='XVID')
    out.open()
    
    if verbose:
        print("capture open",cap.isOpened())
        print("frame size",w,h)
        print("output opened",out.isOpened())
    
    if not out.isOpened() or not cap.isOpened():
        out.release()
        cap.release()
        # exit function if unable to open cap or out
        return
    
    # Primera captura
    ret, frame = cap.read()
    if ret:
        t = datetime.datetime.now()
        ts.append(t)
        out.write(frame)
        if verbose:
            print("first frame captured")
        
    # loop
    while (t <= tFin):
        ret, frame = cap.read()
        
        if ret:
            t = datetime.datetime.now()
            ts.append(t)
            out.write(frame)
            if verbose:
                print("seconds elapsed",cap.get(pos_msec)/1000)
                print(frame.size)

    # end of loop
    
    # release and save
    out.release()
    cap.release()
    
    del out
    del cap
    
    if verbose:
        print('loop exited, cap, out released, times saved to files')
        
    savetxt(files[2],ts, fmt= ["%s"])


# %% functions originaly from PTZCamera
#from onvif import ONVIFCamera
#from cameraUtils import IPCamera
import urllib
import logging
import time

class PTZCamera(IPCamera):

    def __init__(self, host, port ,user, passwd):
        IPCamera.__init__(self, host, port, user, passwd)
        
        self.ptzService = self.create_ptz_service()
        self.profile = self.mediaService.GetProfiles()[0]
        
        self.initializePanTiltBoundaries()
        
    def initializePanTiltBoundaries(self):
        # Get PTZ configuration options for getting continuous move range
        request = self.ptzService.create_type('GetConfigurationOptions')
        request.ConfigurationToken = self.profile.PTZConfiguration._token
        ptz_configuration_options = self.ptzService.GetConfigurationOptions(request)        
        
        self.XMAX = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].XRange.Max
        self.XMIN = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].XRange.Min
        self.YMAX = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].YRange.Max
        self.YMIN = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].YRange.Min
        self.ZMAX = ptz_configuration_options.Spaces.ContinuousZoomVelocitySpace[0].XRange.Max
        self.ZMIN = ptz_configuration_options.Spaces.ContinuousZoomVelocitySpace[0].XRange.Min
    
        
    def getStreamUri(self):
#        return self.mediaService.GetStreamUri()[0]
        return 'rtsp://192.168.1.49:554/Streaming/Channels/1?transportmode=unicast&profile=Profile_1'
        
        
    def getStatus(self):
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('GetStatus')
        request.ProfileToken = media_profile._token
        
        ptzStatus = self.ptzService.GetStatus(request)
        pan = ptzStatus.Position.PanTilt._x
        tilt = ptzStatus.Position.PanTilt._y
        zoom = ptzStatus.Position.Zoom._x

        return (pan, tilt, zoom)
        
    def continuousToRight(self):
        panVelocityFactor = self.XMAX
        tiltVelocityFactor = 0
        zoomVelocityFactor = 0
        self.continuousMove(panVelocityFactor, tiltVelocityFactor, zoomVelocityFactor)
        
    def continuousToLeft(self):
        panVelocityFactor = self.XMIN
        tiltVelocityFactor = 0
        zoomVelocityFactor = 0
        self.continuousMove(panVelocityFactor, tiltVelocityFactor, zoomVelocityFactor)
        
    def continuousToUp(self):
        panVelocityFactor = 0
        tiltVelocityFactor = self.YMAX
        zoomVelocityFactor = 0
        self.continuousMove(panVelocityFactor, tiltVelocityFactor, zoomVelocityFactor)
        
    def continuousToDown(self):
        panVelocityFactor = 0
        tiltVelocityFactor = self.YMIN
        zoomVelocityFactor = 0
        self.continuousMove(panVelocityFactor, tiltVelocityFactor, zoomVelocityFactor)
        
    def continuousZoomIn(self):
        panVelocityFactor = 0
        tiltVelocityFactor = 0
        zoomVelocityFactor = self.ZMAX
        self.continuousMove(panVelocityFactor, tiltVelocityFactor, zoomVelocityFactor)
        
    def continuousZoomOut(self):
        panVelocityFactor = 0
        tiltVelocityFactor = 0
        zoomVelocityFactor = self.ZMIN
        self.continuousMove(panVelocityFactor, tiltVelocityFactor, zoomVelocityFactor)
        
    def continuousMove(self, panFactor, tiltFactor, zoomFactor):
        request = self.ptzService.create_type('ContinuousMove')
        request.ProfileToken = self.profile._token
        request.Velocity.PanTilt._x = panFactor
        request.Velocity.PanTilt._y = tiltFactor
        request.Velocity.Zoom._x = zoomFactor
      
        self.ptzService.ContinuousMove(request)
      # Wait a certain time
        timeout = 1
        time.sleep(timeout)
      # Stop continuous move
        self.ptzService.Stop({'ProfileToken': request.ProfileToken})


    def oneStepRight(self):
        status = self.getStatus()
        logging.info("Movimiento hacia derecha desde " + str(status))
        actualPan = status[0]
        actualTilt = status[1]
        
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('AbsoluteMove')
        request.ProfileToken = media_profile._token
        pan = actualPan - float(2)/360
        if pan <= -1:
            pan = 1

        request.Position.PanTilt._x = pan
        request.Position.PanTilt._y = actualTilt
        absoluteMoveResponse = self.ptzService.AbsoluteMove(request)
        
    def oneStepLeft(self):
        status = self.getStatus()
        print "Movimiento hacia izquierda desde " + str(status)
        actualPan = status[0]
        actualTilt = status[1]
        
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('AbsoluteMove')
        request.ProfileToken = media_profile._token
        pan =  round(actualPan + float(2)/360 , 6)
        if pan >= 1:
            pan = -1
        print pan
        request.Position.PanTilt._x = pan
        request.Position.PanTilt._y = actualTilt
        absoluteMoveResponse = self.ptzService.AbsoluteMove(request)
        
        
    def oneStepUp(self):
        status = self.getStatus()
        print "Movimiento hacia arriba desde " + str(status)
        actualPan = status[0]
        actualTilt = status[1]
        
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('AbsoluteMove')
        request.ProfileToken = media_profile._token
        tilt =  round(actualTilt - float(2)/90, 6)
        pan = actualPan
        if tilt <= -1:
            tilt = -1
            pan = actualPan
        elif tilt >= 1:
                tilt = 1
                pan = actualPan + 180*float(2)/360
                
        request.Position.PanTilt._x = pan
        request.Position.PanTilt._y = tilt
        absoluteMoveResponse = self.ptzService.AbsoluteMove(request)
        
        
        
    def oneStepDown(self):
        status = self.getStatus()
        print "Movimiento hacia abajo desde " + str(status)
        actualPan = status[0]
        actualTilt = status[1]
        
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('AbsoluteMove')
        request.ProfileToken = media_profile._token
        tilt = round(actualTilt + float(2)/90, 6)
        pan = actualPan
        if tilt <= -1:
            tilt = -1
            pan = actualPan
        elif tilt >= 1:
                tilt = 1
                pan = actualPan + 180*float(2)/360

        request.Position.PanTilt._x = pan
        request.Position.PanTilt._y = tilt
        absoluteMoveResponse = self.ptzService.AbsoluteMove(request)
        
        
    def oneStepZoomIn(self):
        status = self.getStatus()
        print "Zoom in desde " + str(status)
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('AbsoluteMove')
        request.ProfileToken = media_profile._token
        
        if status[2] < 0.05:
            paso = 0.07
        else:
            paso = 0.035
            
        pZoom = status[2] + paso
        if pZoom > 1:
            pZoom = 1
        
        request.Position.Zoom._x = pZoom
        absoluteMoveResponse = self.ptzService.AbsoluteMove(request)
        
    def oneStepZoomOut(self):
        status = self.getStatus()
        print "Zoom out desde " + str(status)
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('AbsoluteMove')
        request.ProfileToken = media_profile._token
        
        pZoom = status[2] - 0.01    # Con este paso anda bien
        if pZoom < 0:
            pZoom = 0

        request.Position.Zoom._x = pZoom
        absoluteMoveResponse = self.ptzService.AbsoluteMove(request)
        
    
    def continuousRight(self):
        logging.info("Movimiento continuo hacia derecha")

        
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('AbsoluteMove')
        request.ProfileToken = media_profile._token
        pan = actualPan - float(2)/360
        if pan <= -1:
            pan = 1

        request.Position.PanTilt._x = pan
        request.Position.PanTilt._y = actualTilt
        absoluteMoveResponse = self.ptzService.AbsoluteMove(request)
    
    
    
    def moveAbsolute(self, pan, tilt, zoom = 0):
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('AbsoluteMove')
        request.ProfileToken = media_profile._token
        
#        pPan = round(1 - float(pan)/180, 6)
#        pTilt = round(1 - float(tilt)/45, 6)
#        pZoom = round(float(zoom/100), 6)
#       
        request.Position.PanTilt._x = pan
        request.Position.PanTilt._y = tilt
        request.Position.Zoom._x = zoom
        absoluteMoveResponse = self.ptzService.AbsoluteMove(request)
        
        
    def setHomePosition(self):
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('SetHomePosition')
        request.ProfileToken = media_profile._token
        self.ptzService.SetHomePosition(request)
        
    def gotoHomePosition(self):
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('GotoHomePosition')
        request.ProfileToken = media_profile._token
        self.ptzService.GotoHomePosition(request)
        
    def getSnapshotUri(self):
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.mediaService.create_type('GetSnapshotUri')
        request.ProfileToken = media_profile._token
        response = self.mediaService.GetSnapshotUri(request)
        
        logging.info(response.Uri)
#        urllib.urlretrieve("http://10.2.1.49/onvif-http/snapshot", "local-filename.jpeg")
           
    ''' Metodo para probar capturas en la PTZ '''
    def testAbsolute(self, pan, tilt, zoom = 0):
        media_profile = self.mediaService.GetProfiles()[0]
        
        request = self.ptzService.create_type('AbsoluteMove')
        request.ProfileToken = media_profile._token
        
               
        request.Position.PanTilt._x = pan
        request.Position.PanTilt._y = tilt
        request.Position.Zoom._x = zoom
        testAbsoluteResponse = self.ptzService.AbsoluteMove(request)            
   
