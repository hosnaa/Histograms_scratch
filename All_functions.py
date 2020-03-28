# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:02:36 2020

@author: hosna
"""
import numpy as np
from numpy import asarray
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

"""
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
    
"""


#Norm(0-1)
"""
# convert from integers to floats; after reading the image as array
pixels = pixels.astype('float32')  #without it; error of "output array is read-only", why?
# normalize to the range 0-1
pixels /= 255.0
print((pixels.min(), pixels.max()))
"""
"""

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
"""

"""
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
"""
"""
def Local_thresholding(image_array, size,ratio):
     #print( len(thelist)- size + 1)
     #print( len(thelist[0]) - size + 1)
     new_array=np.ones(shape=(len(image_array),len(image_array[0])))

     for row  in range( len(image_array)- size + 1 ):
         for col  in range( len(image_array[0]) - size + 1 ):
             #for row1  in range( len(thelist)- size + 1 ):
             window=image_array[row:row+size,col:col+size]
             minm=window.min()
             maxm=window.max()
             #print(minm,maxm)
             threshold =minm+((maxm-minm)*ratio)
             #print(threshold)
             if window[0,0] < threshold:
                 new_array[row,col]=0
                 print('ok1')
                    #new_array.append(0)
                 #print ('t')
                #print('x')      
             else:
                new_array[row,col]=1
                print('ok2')
#                      
     #print(new_array)
     #img = Image.fromarray(new_array,'L')
     #img.show()
     #new_array = new_array.astype(np.uint8)
     #plt.imshow(new_array)
     print(new_array)
     plt.imshow(new_array, cmap = plt.get_cmap('gray'))
                
img = 'rose.jpg'
image = Image.open(img).convert('L')

#image.show()
    
#read image as array to take size of image
pixels = asarray(image)
#pixels = pixels.astype('float32')
pixels = np.array(pixels)
#print(pixels)        
Array=Local_thresholding(pixels,10 ,0.5)
#print(Array)
"""
"""
##density_plot

def make_histogram(img):
    # Take a flattened greyscale image and create a historgram from it 
    histogram = np.zeros(256, dtype=int)
    for i in range(img.size):
        histogram[img[i]] += 1
    return histogram

my_img1 = 'rose.jpg'
image1 = Image.open(my_img1).convert('L')
#image2.save('img_gray.png')
img_arr1 = np.asarray(image1)
img_float1 = img_arr1.astype('float32')
img_h1 = img_arr1.shape[0]
img_w1 = img_arr1.shape[1]
flat1 = img_arr1.flatten()
histogram1 = make_histogram(flat1)
plt.plot(histogram1)
plt.show()
#sns.distplot(histogram1, hist=True, kde=True, 
#             bins=int(180/1), color = 'darkblue', 
#             hist_kws={'edgecolor':'black'},
#             kde_kws={'linewidth': 4})

my_img = 'img_equalize.png'
image2 = Image.open(my_img).convert('L')
#image2.save('img_gray.png')
img_arr = np.asarray(image2)
img_float = img_arr.astype('float32')
#img_h = img_arr.shape[0]
#img_w = img_arr.shape[1]
flat = img_arr.flatten()
#histogram = make_histogram(flat)
#plt.plot(histogram)
#plt.show()


## Density Plot and Histogram of all arrival delays
#sns.distplot(histogram, hist=True, kde=True, 
#             bins=int(180/5), color = 'darkblue', 
#             hist_kws={'edgecolor':'black'},
#             kde_kws={'linewidth': 4})


#Density Plot and Histogram of all arrival delays
sns.distplot(flat, hist=True, kde=True, 
                color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 5})

"""
