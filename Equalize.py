# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:10:15 2020

@author: hosna
"""
import numpy as np
from numpy import asarray
from PIL import Image
import matplotlib.pyplot as plt

#Draw histogram and its equalized form

#another way to show an image
#we may try this for color image

#def equalize_color_image(img)
    #image2 = Image.open('rose.jpg').convert('L')
    
    #image2 = Image.open('rose.jpg')
    #img_float = img_arr.astype('float32')
    #img_h = img_arr.shape[0]
    #img_w = img_arr.shape[1]
    # display the image
    #plt.imshow(image2, cmap='gray')
    #img_arr = np.asarray(image2)
    
    # put pixels in a 1D array by flattening out img array
    #flat = img_arr.flatten()
    
    #hist function is used to plot the histogram of an image.
    #plt.hist(flat.ravel(), bins=256)
    #cdf, bins, patches = plt.hist(flat, bins=256, range=(0,256), normed=True, cumulative=True)
    #new_pixels = np.interp(flat, bins[:-1], cdf*255)
    #plt.hist(new_pixels.ravel(), bins=256) #histo after equaliazation
    #output_image = Image.fromarray((new_pixels.reshape((img_h, img_w))))

#For RGB there is no “correct” way of doing it. However, one of the more widely used approaches 
#is to convert an RGB image into a different format and then equalize on one of those channels. 
#For example, we can convert the image to YCbCr and apply histogram equalization to the luma component (Y)


#img parameter is a flattened (1-dimensional) array containing pixel values for our image. 
#The function outputs an array where each index represents the number of pixels at that grey level 
#e.g. if histogram[5] = 99 it means 99 pixels in the image have a grey level of 5.

def make_histogram(img):
    # Take a flattened greyscale image and create a historgram from it 
    histogram = np.zeros(256, dtype=int)
    for i in range(img.size):
        histogram[img[i]] += 1
    return histogram

#This function returns an array where each index is the sum of everything before it + itself. 
#For example, if the histogram started with [5, 2, 9, 1, ...] the start of cumsum would be [5, 7, 16, 17, ...].

def make_cumsum(histogram):
    # Create an array that represents the cumulative sum of the histogram 
    cumsum = np.zeros(256, dtype=int)
    cumsum[0] = histogram[0]
    for i in range(1, histogram.size):
        cumsum[i] = cumsum[i-1] + histogram[i]
    return cumsum


#The mapping is created from the following formula:
#M(i)=max(0,round((grey_levels∗cumsum(i))/(h∗w))−1)
#where:
#i = 0 … grey_levels
#grey_levels = number of grey levels in our image (usually 256)
#cumsum(i) = i’th element of the array produced by our make_cumsum(histogram) function
#h, w = the height and width of our image
    
#The array returned can be used as a mapping function where the index is the old grey value and the number at that index is the new one. 
#For example, if we want to find a new grey value for 60 we can do mapping[60] and the new value will be returned.
def make_mapping(cumsum, img_h, img_w):
    # Create a mapping s.t. each old colour value is mapped to a new
     #   one between 0 and 255 
    mapping = np.zeros(256, dtype=int)

    grey_levels = 256
    for i in range(grey_levels):
        mapping[i] = max(0, round((grey_levels*cumsum[i])/(img_h*img_w))-1)
    return mapping

#create the mapped image
#new_image[i]=mapping[img[i]]
#The output of this function is an array containing the pixel values of the new, histogram equalized image! 
#All that needs doing now is restructuring and rendering / saving it
def apply_mapping(img, mapping):
    # Apply the mapping to our image 
    new_image = np.zeros(img.size, dtype=int)
    for i in range(img.size):
        new_image[i] = mapping[img[i]]
    return new_image


my_img = 'rose.jpg'
image2 = Image.open(my_img).convert('L')
image2.save('img_gray.png')
image2.show()
#print(image2.size)
img_arr = np.asarray(image2)
img_float = img_arr.astype('float32')
img_h = img_arr.shape[0]
img_w = img_arr.shape[1]
flat = img_arr.flatten()
hist = make_histogram(flat)

#histo = Image.fromarray(hist.reshape((256, 256)))
#histo.save('initial_histogram.png')
plt.plot(hist)
plt.show()

#plt.hist(flat.ravel(), bins=256)

cs = make_cumsum(hist)
#cs.save('cumulative_sum.png')
plt.plot(cs)
plt.show()

new_intensity = make_mapping(cs, img_h, img_w)
#print(new_intensity)
#print(img_float[5][1])
#print(new_intensity[5])

new_img = apply_mapping(flat,new_intensity) #new_img is 1D
#print(new_img)

hist_equ= make_histogram(new_img)
#hist_equ.save('equalize_histogram.png')
plt.plot(hist_equ)
plt.show()

output_image = Image.fromarray(np.uint8(new_img.reshape((img_h, img_w))))  
#output_image = Image.fromarray((new_img.reshape((img_h, img_w)))) #what's the difference?
output_image.save('img_equalize.png')
#output_image.show()
