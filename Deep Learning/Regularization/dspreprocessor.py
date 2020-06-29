# -*- coding: utf-8 -*-
"""
Created on Sun May 10 18:00:38 2020

@author: Rahul Sapireddy
"""

# import the opencvlibrary
import cv2

class DsPreprocessor:
    def __init__(self, width, height):
        # save the width and height into the attributes of the class
        self.width = width
        self.height = height
        
    def preprocess(self, image):
        # ignore the aspect ratio and resize the image
        return cv2.resize(image, (self.width,self.height), interpolation = cv2.INTER_AREA)
    
"""
INTERPOLATION [optional] 
flag that takes one of the following methods. 
INTER_NEAREST – a nearest-neighbor interpolation 
INTER_LINEAR – a bilinear interpolation (used by default) 
INTER_AREA – resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method. 
INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood 
INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood.
"""
