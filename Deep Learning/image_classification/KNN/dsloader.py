# -*- coding: utf-8 -*-
"""
Created on Sun May 10 18:27:32 2020

@author: Rahul Sapireddy
"""

# importing required packages
import cv2
import numpy as np
import os # for going through the folders

class DsLoader:
    def __init__(self, preprocessor=None):
        # save the preprocessor passed in (if any)
        self.preprocesssor = preprocessor
        
        # initialize an empty list if the list of passed preprocessor is empty
        if self.preprocesssor is None:
            self.preprocesssor = []
            
    def load(self, imagePaths):
        # initialize the list for data and labels
        data = []
        labels = []
        
        # loop through all the image paths passed in
        for(i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            # we take the label name as cats, dogs and pandas here
            
            if self.preprocesssor is not None:
                for p in self.preprocesssor:
                    image = p.preprocess(image)
            
            # displaying the progress in format
            # Preprocessed 500/3000
            if i > 500 and (i+1)%500==0:
                print("Processed {}/{}".format(i+1, len(imagePaths)))
            
            data.append(image)
            labels.append(label)
            
        # return two arrays, data and labels to the code that called the function
        return(np.array(data),np.array(labels))
        
        
        
                