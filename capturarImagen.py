# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 20:24:54 2016

@author: lilian

mueve la ptz y saca foto
"""

# -*- coding: utf-8 -*-
import cv2
from PTZCamera import PTZCamera


url = 'rtsp://10.2.1.49:554/Streaming/Channels/1?transportmode=unicast&profile=Profile_1'
cap = cv2.VideoCapture(url)

cam = PTZCamera('10.2.1.49', 80, 'admin', '12345')
(aPan, aTilt, aZoom) = cam.getStatus()





##**********************************************************************
## CAPTURA MOVIENDO TILT
##**********************************************************************

# Datos hardcoded
yIni = -0.901778
yFin = -0.092667
cantFotos = 20
y = abs(yIni - yFin)/ cantFotos #factor de aumento de pan

# Posicion inicial
cam.moveAbsolute(aPan, yIni, aZoom)
print yIni

i = 0
j = 0
r = 0
while(True):
        
    # Captura
    ret, frame = cap.read()

    if (i%25 == 0):
        j+=1
        if (j==3) & (r < cantFotos+1):
            j = 0
            r+=1
            # Guardar imagen
            cv2.imwrite('captura' + str(r) +'.jpg', frame)
            
            # Mover
            tilt = yIni + (y*r)
            print tilt
            cam.moveAbsolute(aPan, tilt, aZoom)
    i+=1
    


##**********************************************************************
## CAPTURA MOVIENDO PAN
##**********************************************************************
#
## Datos hardcoded
#xIni = 0.385556
#xFin = 0.134167
#cantFotos = 20
#x = (xIni - xFin)/ cantFotos #factor de aumento de pan
#
## Posicion inicial
#cam.moveAbsolute(xIni, aTilt, aZoom)    
#print xIni
##time.sleep(5)
#i = 0
#j = 0
#r = 0
#while(True):
#        
#    # Captura
#    ret, frame = cap.read()
#
#    if (i%25 == 0):
#        j+=1
#        if (j==3) & (r < cantFotos+1):
#            j = 0
#            r+=1
#            # Guardar imagen
#            cv2.imwrite('captura' + str(r) +'.jpg', frame)
#            
#            # Mover
#            pan = xIni - (x*r)
#            print pan
#            cam.moveAbsolute(pan, aTilt, aZoom)
#    i+=1
#    
