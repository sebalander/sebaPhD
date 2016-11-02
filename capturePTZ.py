# -*- coding: utf-8 -*-
'''

leer cada frame de la ptz y guardarlo con su timestamp.
'''
import cv2
from numpy import savetxt
import datetime
import psutil
import time


#time.sleep(1)
url = 'rtsp://10.2.1.49:554/Streaming/Channels/1?transportmode=unicast&profile=Profile_1'
#url = 'rtsp://10.2.1.48/live.sdp'

#*** Inicializacion ***
tFin = datetime.datetime.now() + datetime.timedelta(minutes=10)
#filename = 'vca_fe_768_'
filename = 'ptz2_'

#fps = [] #fps calculado con el ts. Se calcula con la captura del segundo frame
#ts = []  #timestamp de la captura
#bw = [] # ancho de banda calculado de la placa cableada. Se calcula dsd el segundo frame
ms = [] # ts de la captura en ms


#*** Configuración de capture y writer ***
cap = cv2.VideoCapture(url)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fpsCam = cap.get(cv2.CAP_PROP_FPS)
#para fe especificar los fps pq toma cualquier cosa con la propiedad
#fpsCam = 12
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Códec de video

out = cv2.VideoWriter(filename+"video.avi", fourcc, fpsCam,(w, h),True)

#*** Primera captura
ret, frame = cap.read()
if ret:
    t0 = datetime.datetime.now()
#    iostat = psutil.net_io_counters(pernic=True)
#    bytesRcv = iostat['enp5s0'][1]
#    ts.append(t0)
    ms.append(int(round(time.time() * 1000))) # actual time in milliseconds
    out.write(frame)
    
#    ts.append(tFrame.strftime('%Y-%m-%d %H:%M:%S %f'))

while (t0 <= tFin):
    ret, frame = cap.read()
    if ret:
        t1 = datetime.datetime.now()
        ms.append(int(round(time.time() * 1000)))
        iostat = psutil.net_io_counters(pernic=True)
        out.write(frame)
#        print 'escribe'
        
#        ts.append(t1)
        
        tsDiff = (t1 - t0).total_seconds()
        
#        bytesRxCalc = iostat['enp5s0'][1] - bytesRcv 
#        bw.append(bytesRxCalc/tsDiff) # bytes/s
        
#        fps.append(1/tsDiff)
        
        t0 = t1
#        bytesRcv = iostat['enp5s0'][1]
      
    else:
        break


out.release()
cap.release()

#savetxt(filename+"tsFrame.txt",ts, fmt= ["%s"])
#savetxt(filename+"bw.txt",bw, fmt= ["%f"])
#savetxt(filename+"fps.txt",fps, fmt= ["%f"])
savetxt(filename+"msFrame.txt",ms, fmt= ["%f"])

''' Error obtenido '''
#[mpeg4 @ 0x49bc880] timebase 1/180000 not supported by MPEG 4 standard,
#the maximum admitted value for the timebase denominator is 65535
