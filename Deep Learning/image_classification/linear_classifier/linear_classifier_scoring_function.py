# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:07:19 2020

@author: Rahul Sapireddy
"""

# import numpy and opencv
import numpy as np
import cv2

# defing class labels
class_labels = ["dog", "cat", "panda"]

# set the seed to use inside random generator
np.random.seed(1)

# generate random weight and bias
w = np.random.randn(3, 3072)
b = np.random.randn(3)

# load the image
input_image = cv2.imread("C:\\Users\\Rahul Sapireddy\\Desktop\\EDUNOIX\\Deep Learning\\\images\\test1.png")

# resize and flatten the image
input_image = cv2.resize(input_image, (32,32)).flatten()

# finding the dot product of weight and feature vector 
# then add bias to it
score_results = w.dot(input_image) + b

# printing the scores
for (label,score_result) in zip(class_labels, score_results):
    print("Score of {}: {:.2f}".format(label,score_result))

