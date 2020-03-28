# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 20:17:33 2020

@author: hosna
"""

import numpy as np
from numpy import asarray
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

def make_histogram(img):
    # Take a flattened greyscale image and create a historgram from it 
    histogram = np.zeros(256, dtype=int)
    for i in range(img.size):
        histogram[img[i]] += 1
    return histogram

my_img = 'img_equalize.png'
image2 = Image.open(my_img).convert('L')
img_arr = np.asarray(image2)
img_float = img_arr.astype('float32')
flat = img_arr.flatten()

sns.distplot(flat, hist=True, kde=True, 
                color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 5})