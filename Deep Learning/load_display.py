# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:08:02 2020

@author: Rahul Sapireddy
"""

import cv2

image = cv2.imread("C:\\Users\\Rahul Sapireddy\\Desktop\\EDUNOIX\\Deep Learning\\\images\\cat.jpeg")

print(image.shape) 
"""
gives number of rows - height of the image
columns - width of the image
number of channels in the image
"""

cv2.imshow("Test Image",image)
print(image[30,50]) 
# Accesing a single pixel
# Access the colour of the pixel at this point