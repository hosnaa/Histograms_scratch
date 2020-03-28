# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:12:29 2020

@author: hosna
"""

import numpy as np
from numpy import asarray
from PIL import Image
import matplotlib.pyplot as plt

def global_threshold (threshold, img):
    img = asarray(img)
    img = img.astype('float32')
    print(img)
    for row in range (img.shape[0]):
        for col in range (img.shape[1]):
            if img[row, col] < threshold:
                img[row, col] = 0
            else:
                img[row, col] = 255
    return img

# load image
image = Image.open('rose.jpg').convert('L')
thresh = 70
new_img = global_threshold(thresh, image)
print(new_img.shape)
img_h = new_img.shape[0]
img_w = new_img.shape[1]

#array is in float type and the image should have every pixel in the format 0-255 (which is uint8)
new_img = new_img.astype(np.uint8) #made it save safely
new_img_plot = Image.fromarray(new_img)

new_img_plot.save('img_global_thresholded.png')  # cannot save it why?
image.show()
new_img_plot.show()