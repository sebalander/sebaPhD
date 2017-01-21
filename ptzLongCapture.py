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
iteration = int(argv[8])

# para la PTZ
url = 'rtsp://192.168.1.49/live.sdp'
cod = 'XVID'

ip = '192.168.1.49'
portHTTP = 80
portRTSP = '554'
usr = 'admin'
psw = '12345'

ahora = datetime.now()

endDate = datetime(yea,mon,day,hor,mnt,0)
lastStart = endDate - timedelta(minutes=duration)

# calculate duration in minutes
durationTillEnd = endDate - ahora
durationTillEnd = durationTillEnd.total_seconds() / 60

if duration < durationTillEnd:
    ret = 1 # more videos are needed
else:
    duration = durationTillEnd
    ret = 0 # this is the last video


# pan tilt y zoom para grabar
posiciones = [[0.7, -0.3, 0],
              [0.9, -1, 0]]
nPos = 2 # cantidad de posiciones diferentes

cam = PTZCamera(ip, portHTTP, usr, psw)
cam.getStatus()

# %%
eP, eT, z = posiciones[iteration % nPos ]

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

sleep(2) # esperar que se mueva y enfoque minimamente
ahora = datetime.now()
files = [url,
         "/home/alumno/Documentos/sebaPhDdatos/ptz_%s.avi"%ahora,
         "/home/alumno/Documentos/sebaPhDdatos/ptz_%s_tsFrame.txt"%ahora]


videoFailed = captureTStamp(files, duration, cod, fps=fpsCam)
# video saved correctly <=> videoFailed=0

