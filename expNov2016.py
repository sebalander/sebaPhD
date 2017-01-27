# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 17:24:19 2017

aplying calibration scheme to the data form Nov 2016

@author: sebalander
"""



# %%
import numpy as np


# %% Initial position
# height calculated from picture
h_pix = np.linalg.norm([544-530,145-689]) # balcon

pixs = np.array([np.linalg.norm([541-545,319-299]), # parabolica 1
                 np.linalg.norm([533-552,310-307]), # parabolica 2
                 np.linalg.norm([459-456,691-624]), # persona
                 np.linalg.norm([798-756,652-651]), # 5xancho de cajon cerveza
                 np.linalg.norm([767-766,668-613])]) # 4xalto de cajon cerveza

# %%corresponding values in meters
# usando las medidas de cajon de cerveza de ulises
mets = np.array([0.6, 0.6, 1.8, 5*0.297, 4*0.342]) # 

h_m = h_pix*mets/pixs
h_m

# %%according to google earth, a good a priori position is
# -34.629344, -58.370350
y0, x0 = -34.629344, -58.370350
z0 = 15.7 # metros, as measured by oliva
# and a rough conversion to height is using the radius of the earth
# initial height 
z0 = z0 * 180.0 / np.pi / 6400000.0 # now in degrees

# %% initial pose


