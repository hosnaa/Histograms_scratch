# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:13:25 2020

@author: hosna
"""
import numpy as np
from numpy import asarray
from PIL import Image
import matplotlib.pyplot as plt

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