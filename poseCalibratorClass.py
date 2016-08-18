# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:47:18 2016

defines the class poseCalibrator that recieves extrinsic calibration data:
- fiducial points
- corners found on image that correspond to the fiducial points
- camera intrinsic parameters
- initial pose guess for pose
and calculates:
- optimum pose minimizing error in image
- optimum pose minimizing error in scene

also perhaps plots things and in the future calculates error.
long term: help user define initial pose guess, help find corners in image

following tutorial in http://www.tutorialspoint.com/python/python_classes_objects.htm

@author: sebalander
"""
# %% IMPORTS
import numpy as np
import cv2
from lmfit import minimize, Parameters


# %% DECLARATION
class poseCalibrator:
    '''
    declaration(arguuments)
    por ahora todos los metodos modifican directamente las class variables
    para setear y getear las variables no hay seters y geters, se hace directo
    
    
    variables (input and aputput)
    ---------
    
    example
    -------
    
    
    '''
    
    # INIT
    def __init__(self,corners=[],
                      fiducialPoints=[],
                      intrinsicLinear=[],
                      intrinsicDistortion=[],
                      initialRotation=[],
                      initialTraslation=[],
                      model='rational'):
        '''
        sets variables calling the setParams method
        '''
        self.setParams(corners=corners,
                       fiducialPoints=fiducialPoints,
                       intrinsicLinear=intrinsicLinear,
                       intrinsicDistortion=intrinsicDistortion,
                       initialRotation=initialRotation,
                       initialTraslation=initialTraslation,
                       model=model)
    
    # METHOD THAT SETS VARIABLES
    def setParams(self,corners=[],
                      fiducialPoints=[],
                      intrinsicLinear=[],
                      intrinsicDistortion=[],
                      initialRotation=[],
                      initialTraslation=[],
                      model='rational'):
        '''
        set inputs independently of instantiation, usefull for updating
        variables without creating a new instace
        '''
        self.imageCorners = corners
        self.fiducialPoints = fiducialPoints
        self.intrinsicLinear = intrinsicLinear
        self.intrinsicDistortion = intrinsicDistortion
        self.initialRotation = initialRotation
        self.initialTraslation = initialTraslation
        self.model = model
    
    # CHECKER METHODS
    # methods that check that all variables are in the correct format
    
    # ========== ========== PROJECTION METHODS ========== ==========
    def inverseRational(self,optimised):
        '''
        inverseRational(objPoints, rVec/rotMatrix, tVec, cameraMatrix,
                        distCoeffs)-> objPoints
        takes corners in image and returns coordinates in scene
        corners must be of size (n,1,2)
        objPoints has size (1,n,3)
        ignores tangential and tilt distortions
        '''
        if optimised:
            rVec = self.optimisedRotation
            tVec = self.optimisedTraslation
        else:
            rVec = self.initialRotation
            tVec = self.initialTraslation
        
        if rVec.shape != (3,3):
            rVec, _ = cv2.Rodrigues(rVec)
        
        xpp = ((self.imageCorners[:,0,0]-self.intrinsicLinear[0,2]) /
                self.intrinsicLinear[0,0])
        ypp = ((self.imageCorners[:,0,1]-self.intrinsicLinear[1,2]) /
                self.intrinsicLinear[1,1])
        rpp = np.sqrt(xpp**2 + ypp**2)
        
        # polynomial coeffs
        # # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
        poly = [[self.intrinsicDistortion[4,0],
                 -r*self.intrinsicDistortion[7,0],
                 self.intrinsicDistortion[1,0],
                 -r*self.intrinsicDistortion[6,0],
                 self.intrinsicDistortion[0,0],
                 -r*self.intrinsicDistortion[5,0],
                 1,
                 -r] for r in rpp]
        
        roots = [np.roots(p) for p in poly]
        rp_rpp = np.array([roo[np.isreal(roo)].real[0] for roo in roots]) / rpp
        
        xp = xpp * rp_rpp
        yp = ypp * rp_rpp
        
        # auxiliar calculations
        a = rVec[0,0] - rVec[2,0] * xp
        b = rVec[0,1] - rVec[2,1] * xp
        c = tVec[0,0] - tVec[2,0] * xp
        d = rVec[1,0] - rVec[2,0] * yp
        e = rVec[1,1] - rVec[2,1] * yp
        f = tVec[1,0] - tVec[2,0] * yp
        q = a*e-d*b
        
        X = -(c*e - f*b)/q # check why wrong sign, why must put '-' in front?
        Y = -(f*a - c*d)/q
        
        shape = (1,self.imageCorners.shape[0],3)
        XYZ = np.array([X, Y, np.zeros(shape[1])]).T
        XYZ = np.reshape(XYZ, shape)
        
        self.projectedFiducialPoints = XYZ
    
    def directRational(self,optimised):
        '''
        '''
        if optimised:
            rVec = self.optimisedRotation
            tVec = self.optimisedTraslation
        else:
            rVec = self.initialRotation
            tVec = self.initialTraslation
        
        projected, jaco = cv2.projectPoints(self.fiducialPoints,
                                            rVec,
                                            tVec,
                                            self.intrinsicLinear,
                                            self.intrinsicDistortion)
        
        if optimised:
            self.projectedCornersOptimised = projected
            self.projectedJacobianOptimised = jaco
        else:
            self.projectedCornersInitial = projected
            self.projectedJacobianInitial = jaco
    
    
    def inverseFisheye():
        '''TODO'''
    
    def directFisheye():
        '''TODO'''
    
    def direct(self,optimised=True):
        '''
        calls appropiate direct projection function according to model
        using optimised pose or initial pose values
        '''
        if self.model=='rational':
            self.directRational(optimised)
        elif self.model=='fisheye':
            self.directFisheye(optimised)
        else:
            print("model must be 'rational' or 'fisheye'")
        
    def inverse(self,optimised=True):
        '''
        calls appropiate inverse projection function according to model
        using optimised pose or initial pose values
        '''
        if self.model=='rational':
            self.inverseRational(optimised)
        elif self.model=='fisheye':
            self.inverseFisheye(optimised)
        else:
            print("model must be 'rational' or 'fisheye'")
    
    
    
    
    
    
    
    # ========== ========== PARAMETER HANDLING ========== ==========
    def retrieveParameters(self, params):
        '''
        
        '''
        rvec = np.zeros((3,1))
        tvec = np.zeros((3,1))
        for i in range(3):
            rvec[i,0] = params['rvec%d'%i].value
            tvec[i,0] = params['tvec%d'%i].value
        
        cameraMatrix = np.zeros((3,3))  
        cameraMatrix[0,0] = params['cameraMatrix0'].value
        cameraMatrix[1,1] = params['cameraMatrix1'].value
        cameraMatrix[0,2] = params['cameraMatrix2'].value
        cameraMatrix[1,2] = params['cameraMatrix3'].value
        cameraMatrix[2,2] = 1
    
        distCoeffs = np.zeros((14,1))
        for i in range(14):
            distCoeffs[i,0] = params['distCoeffs%d'%i].value 
    
        return rvec, tvec, cameraMatrix, distCoeffs
    
    
    def formatParameters(self):
        '''
        get params into correct format for lmfit optimisation
        '''
        params = Parameters()
        for i in range(3):
            params.add('rvec%d'%i,
                       value=self.initialRotation[i,0], vary=True)
            params.add('tvec%d'%i,
                       value=self.initialTraslation[i,0], vary=True)
        
        
        params.add('cameraMatrix0',
                   value=self.intrinsicLinear[0,0], vary=False)
        params.add('cameraMatrix1',
                   value=self.intrinsicLinear[1,1], vary=False)
        params.add('cameraMatrix2',
                   value=self.intrinsicLinear[0,2], vary=False)
        params.add('cameraMatrix3',
                   value=self.intrinsicLinear[1,2], vary=False)
        
        for i in range(14):
            params.add('distCoeffs%d'%i,
                       value=self.intrinsicDistortion[i,0], vary=False)
        
        self.initialParams = params
    
    
    
    
    
    
    
    
    # ========== ========== DEFINE RESIDUALS ========== ==========
    def residualDirect(self, params, objectPoints, corners):
        '''
        '''
        rvec, tvec, cameraMatrix, distCoeffs = self.retrieveParameters(params)
        
        if self.model=='rational':
            projectedCorners, _ = cv2.projectPoints(objectPoints,
                                                    rvec,
                                                    tvec,
                                                    cameraMatrix,
                                                    distCoeffs)
        elif self.model=='fisheye':
            'TODO'
        else:
            print("model must be 'rational' or 'fisheye'")
        
        return corners[:,0,:] - projectedCorners[:,0,:]
    
    def residualInverse(self, params, objectPoints, corners):
        '''
        '''
        rvec, tvec, cameraMatrix, distCoeffs = self.retrieveParameters(params)
        
        # project points
        projectedPoints = self.inverse(corners,
                                       rvec,
                                       tvec,
                                       cameraMatrix,
                                       distCoeffs)
        
        return objectPoints[0,:,:2] - projectedPoints[0,:,:2]
    
    
    
    
    
    
    
    
    
    
    
    # ========== ========== OPTIMIZATION ========== ==========
    def optimizePoseDirect(self):
        '''
        '''
        self.formatParameters() # generate Parameters obj
        
        out = minimize(self.residualDirect,
                       self.initialParams,
                       args=(self.fiducialPoints,
                             self.imageCorners))
        
        self.directMinimizeOutput = out
        self.optimisedRotationDirect, self.optimisedTraslationDirect, _ , _ = \
                                            self.retrieveParameters(out.params)
    
    def optimizePoseInverse(self):
        '''
        '''
        self.formatParameters() # generate Parameters obj
        
        out = minimize(self.residualInverse,
                       self.initialParams,
                       args=(self.fiducialPoints,
                             self.imageCorners))
        
        self.inverseMinimizeOutput = out
        self.optimisedRotationInverse, self.optimisedTraslationInverse, _ , _ = \
                                            self.retrieveParameters(out.params)
    