#!/bin/bash

# all parameters must be integers
# set finishing date
yea=2016
mon=11
day=7
hor=12
mnt=0

# set video duration in minutes
vcaDur=30
ptzDur=30

# framerate in fps
vcaFPS=10
ptzFPS=10

# call as separate processes
python ptzLongCapture.py $yea $mon $day $hor $mnt $ptzFPS $ptzDur &
python vcaLongCapture.py $yea $mon $day $hor $mnt $vcaFPS $vcaDur
