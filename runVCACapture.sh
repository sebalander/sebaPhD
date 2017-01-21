#!/bin/bash

# to catch user Ctrl+c 
control_c(){
    echo "saliendo"
    kill $PID
    exit
}


# all parameters must be integers
# set finishing date
yea=2016
mon=11
day=14
hor=8
mnt=0
finishDate=$(date -d "$yea-$mon-$day $hor:$mnt" +'%s') # fecha final en segundos desde 1979
# set video duration in minutes
vcaDur=2
# framerate in fps
vcaFPS=10

ahora=$(date +%s) # fecha actual en segundos desde 1979
i=0

trap control_c SIGINT # trap keyboard interrupt control-c

while [ $ahora -le $finishDate ]; do
    python vcaLongCapture.py $yea $mon $day $hor $mnt $vcaFPS $vcaDur
    echo $i
    ((i++))
    ahora=$(date +%s) # fecha actual en segundos desde 1979
done
