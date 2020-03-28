# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:42:41 2020

@author: hosna
"""

import numpy as np
from numpy import asarray
from PIL import Image
import matplotlib.pyplot as plt



def normalize_color_image(img):
    image = Image.open(img)
    #image.show()
    
    #read image as array to take size of image
    pixels = asarray(image)
    pixels = pixels.astype('float32')
    #print (pixels.shape)
    #print(pixels.shape[1])
    
    #get minimum and maximum intensity values of image(for each channel) and set a range out of them
    old_minR = pixels[..., 0].min()
    old_minG = pixels[..., 1].min()
    old_minB = pixels[..., 2].min()
    
    old_maxR = pixels[..., 0].max()
    old_maxG = pixels[..., 1].max()
    old_maxB = pixels[..., 2].max()
    
    #Or: max_channels = np.amax([np.amax(img[:,:,0]), np.amax(img[:,:,1]), np.amax(img[:,:,2])])
    
    old_rangeR = old_maxR - old_minR
    old_rangeG = old_maxG - old_minG
    old_rangeB = old_maxB - old_minB
    
    #(normalize to 0-255; if it's not)
    #new_min = 0
    #new_max = 255
    #new_range = new_max - new_min
    
    #print((pixels.min(), pixels.max()))      # note that it will change afterwards
    
    #formula for normalization from (0-255): Inew = (Iold-old_min) * (new_range/old_range) + new_min
    #formula for normalization from (0-1): Inew = (Iold-old_min) /old_range
    
    #for each pixel change its intensity using the formula above
    for rows in range (pixels.shape[0]):
        for col in range (pixels.shape[1]):
            pixels[rows, col,0]  = (pixels[rows, col,0] - old_minR) / old_rangeR
            pixels[rows, col,1]  = (pixels[rows, col,1] - old_minG) / old_rangeG
            pixels[rows, col,2]  = (pixels[rows, col,2] - old_minB) / old_rangeB
            #print(pixels[rows, col])
    #print((pixels.min(), pixels.max()))     # it should be changed by now
    
    #from the array/matrix of normalized pixels obtained from the above for loop, create the image
    #pixels = np.concatenate((pixels[...,0], pixels[...,1],pixels[...,2]), axis = 0)
    #print('3.2')
    #img = Image.fromarray(pixels,'RGB')      # get an image from normalized pixels
    #img.save('norm_colored_img.png')       # change the name each time, or we may concatenate it with a random variable
    #img.show()
    plt.imshow(pixels)
    return pixels
    

def normalize_grey_image(img):
    
    image = Image.open(img).convert('L')
    pixels = asarray(image)
    pixels = pixels.astype('float32')
    
    #need only one as its only one channel
    old_min = pixels.min()
    old_max = pixels.max()
    old_range = old_max - old_min
    
    for rows in range (pixels.shape[0]):
        for col in range (pixels.shape[1]):
            pixels[rows, col]  = (pixels[rows, col] - old_min) / old_range

    #img = Image.fromarray(pixels,'L')
    #img.show()
    #img.save('norm_grey_img.png')
    plt.imshow(pixels,cmap = plt.get_cmap('gray'))
    return pixels
    
#trial
my_img = 'rose.jpg'
image = Image.open(my_img)

pixels = asarray(image)
pixels = pixels.astype('float32')
print(pixels.shape)
#check if its RGB or Gray
try:
    if (pixels.shape[2] == 3):
        image_norm = normalize_color_image(my_img)
        print("3")
    
#elif (pixels.shape[2] == 1):
except IndexError:  #(pixels.shape[2] == None):
    image_norm_array = normalize_grey_image(my_img)
    print("1")
    

