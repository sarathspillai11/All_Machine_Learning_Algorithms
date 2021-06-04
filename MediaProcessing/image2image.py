# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 21:09:56 2019

@author: kpmg
"""

# import the necessary packages
import cv2 
import numpy as np 
original = cv2.imread("comparison.jpg") 
duplicate = cv2.imread("comparison1.jpg") 
# 1) Check if 2 images are equals 
if original.shape == duplicate.shape: 
    print("The images have same size and channels") 
difference = cv2.subtract(original, duplicate) 
b, g, r = cv2.split(difference) 
if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
    print("The images are completely Equal") 
