# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
@author: sebalander

para grabar video de a intervalos fijos hasta una fecha predefinida
ademas mueve la PTZ a los lugares predefinidos
"""
# %%
from cameraUtils import captureTStamp
from datetime import timedelta, datetime
from PTZCamera import PTZCamera
from time import sleep
from sys import argv

# %%
# load arguments
arguments = argv
print(arguments)

yea = int(argv[1])
mon = int(argv[2])
day = int(argv[3])
hor = int(argv[4])
mnt = int(argv[5])
fpsCam = int(argv[6])
duration = int(argv[7])


## date
#yea=2016
#mon=11
#day=7
#hor=16
#mnt=0
## set video duration in minutes
#duration=0.5
## framerate in fps
#fpsCam=10

# para la PTZ
url = 'rtsp://192.168.1.49/live.sdp'
cod = 'XVID'

ip = '192.168.1.49'
portHTTP = 80
portRTSP = '554'
usr = 'admin'
psw = '12345'

endDate = datetime(yea,mon,day,hor,mnt,0)
lastStart = endDate - timedelta(minutes=duration)

# pan tilt y zoom para grabar
posiciones = [[0.7, -0.3, 0],
              [0.9, -1, 0]]

cam = PTZCamera(ip, portHTTP, usr, psw)
cam.getStatus()

nPos = len(posiciones) - 1 # one less to make easier counting

i=0

# %%
while(datetime.now() < lastStart):
    eP, eT, z = posiciones[i]
    print(eP, eT, z)
    
    # to avoid connection reset by peer when moving camera
    while True:
        try:
            cam.moveAbsolute(eP, eT, z)
        except:
            # reconnect if connection was reset
            print("problem conecting, retrying in 2 secs")
            sleep(2)
            cam = PTZCamera(ip, portHTTP, usr, psw)
        break
    
    # position index counting
    if i < nPos:
        i=i+1
    else:
        i=0
    sleep(2) # esperar que se mueva y enfoque minimamente
    ahora = datetime.now()
    files = [url,
             "/home/alumno/Documentos/sebaPhDdatos/ptz_%s.avi"%ahora,
             "/home/alumno/Documentos/sebaPhDdatos/ptz_%s_tsFrame.txt"%ahora]
    
    
    videoFailed = captureTStamp(files, duration, cod, fps=fpsCam)
    # video saved correctly <=> videoFailed=0


# %% last video to complete desired endDate
lastDuration = endDate - datetime.now()
lastDuration = lastDuration.total_seconds() / 60

eP, eT, z = posiciones[i]
print(eP, eT, z)
cam.moveAbsolute(eP, eT, z)
sleep(2) # esperar que se mueva y enfoque minimamente
ahora = datetime.now()
files = [url,
         "/home/alumno/Documentos/sebaPhDdatos/ptz_%s.avi"%ahora,
         "/home/alumno/Documentos/sebaPhDdatos/ptz_%s_tsFrame.txt"%ahora]

videoFailed = captureTStamp(files, duration, cod, fps=fpsCam)
# video saved correctly <=> videoFailed=0

while videoFailed:
    # keep trying
    sleep(2)
    videoFailed = captureTStamp(files, duration, cod, fps=fpsCam)


