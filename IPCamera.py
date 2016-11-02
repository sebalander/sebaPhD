# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 16:47:28 2016

@author: lilian
"""
    
from onvif import ONVIFCamera


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
