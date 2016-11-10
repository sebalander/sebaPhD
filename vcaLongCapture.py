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
print(arguments)

yea = int(argv[1])
mon = int(argv[2])
day = int(argv[3])
hor = int(argv[4])
mnt = int(argv[5])
fpsCam = int(argv[6])
duration = int(argv[7])/60.0

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

# para la FE
url = 'rtsp://192.168.1.48/live.sdp'
cod = 'XVID'

endDate = datetime(yea,mon,day,hor,mnt,0)
lastStart = endDate - timedelta(minutes=duration)

# %%
while(datetime.now() < lastStart):
    ahora = datetime.now()
    files = [url,
             "/home/alumno/Documentos/sebaPhDdatos/vca_%s.avi"%ahora,
             "/home/alumno/Documentos/sebaPhDdatos/vca_%s_tsFrame.txt"%ahora]
    
    captureTStamp(files, duration, cod, fps=fpsCam)
    # sleep(60)

# %% last video to complete desired endDate
lastDuration = endDate - datetime.now()
lastDuration = lastDuration.total_seconds() / 60

ahora = datetime.now()
files = [url,
         "/home/alumno/Documentos/sebaPhDdatos/vca_%s.avi"%ahora,
         "/home/alumno/Documentos/sebaPhDdatos/vca_%s_tsFrame.txt"%ahora]

captureTStamp(files, lastDuration, cod, fps=fpsCam)


