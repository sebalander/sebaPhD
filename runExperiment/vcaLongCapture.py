# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
@author: sebalander

para grabar video de a intervalos fijos hasta una fecha predefinida
"""
# %%
from cameraUtils import captureTStamp
from datetime import timedelta, datetime
from time import sleep
from sys import argv

# %%
# load arguments
arguments = argv
#print(arguments)

yea = int(argv[1])
mon = int(argv[2])
day = int(argv[3])
hor = int(argv[4])
mnt = int(argv[5])
fpsCam = int(argv[6])
duration = int(argv[7])

# para la FE
url = 'rtsp://192.168.1.48/live.sdp'
cod = 'XVID'

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


files = [url,
         "/home/alumno/Documentos/sebaPhDdatos/vca_%s.avi"%ahora,
         "/home/alumno/Documentos/sebaPhDdatos/vca_%s_tsFrame.txt"%ahora]

print(duration)
videoFailed = captureTStamp(files, duration, cod, fps=fpsCam, verbose=True)
# video saved correctly <=> videoFailed=0

print(ret)

