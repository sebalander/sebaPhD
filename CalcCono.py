# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:19:08 2017

@author: ulises
"""
import numpy as np

def calcCono(model,tam = 1) :
    assert(tam>0),"El tamaño de entrada no puede ser menor que 1"
    
    config = model.get_config()
    cant = len(config["layers"])
    
    for cont in range(cant-1,-1,-1):
        capa = config["layers"][cont]
        if "Pool" in capa["class_name"] and  "Global" not in capa["class_name"]:
            z=capa["config"]["pool_size"][0]
            s=capa["config"]["strides"][0]
            tam = (tam-1)*s+z
        if "Conv" in capa["class_name"]:
            r = capa["config"]["nb_row"]
            ds = capa["config"]["subsample"][0]
#            if capa["config"]["boder_mode"]=="same":     
#            if capa["config"]["boder_mode"]=="same":
#            if capa["config"]["boder_mode"]=="same":
            #no lo uso porque quiero el cono efectivo, no el
            #el ajustado
            tam = np.ceil((tam+r-1)/ds)
            
        
    return tam
        
