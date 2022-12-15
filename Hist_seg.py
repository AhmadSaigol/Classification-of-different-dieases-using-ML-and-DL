# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 15:40:00 2022

@author: LaptopJR
"""

from matplotlib import pyplot as plt
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float, img_as_ubyte, io, transform
import cv2
import time
start_time = time.time()

"""
FAILED ATTEMPTS 

def circular_crop(image_data):
    height,width = image_data.shape
    lum_img = Image.new('L', [height,width] , 0)
    
    draw = ImageDraw.Draw(lum_img)
    draw.pieslice([(0,0), (height,width)], 0, 360, 
              fill = 255, outline = "white")
    img_arr =np.array(image_data)
    lum_img_arr =np.array(lum_img)
    cropped_image = np.dstack((img_arr,lum_img_arr))
    return cropped_image


 #   Define physical shape of filter mask
def circular_filter(image_data, radius):
     kernel = np.zeros((2*radius+1, 2*radius+1))
     y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
     mask = x**2 + y**2 <= radius**2
     kernel[mask] = 1                
     filtered_image = gf(image_data, np.median, footprint = kernel)
     return filtered_image
"""

## READING ##
#img=img_as_float(io.imread("samples/3b96f124-bcc3-4dcd-a551-707cc610e3f2.png",as_gray=True))
#img=img_as_float(io.imread("samples/05f46c41-856a-4781-822b-244d82cebd5f.png",as_gray=True))
#img=img_as_float(io.imread("samples/6c12f4cf-0626-48f9-b4c3-0532ca4a4da4.png",as_gray=True))
img=img_as_float(io.imread("/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/data/raw_data/code_testing/train/0a0aa8c9-6b33-445d-9b90-dfba8a1a3572.png",as_gray=True))

## Denoising image ##
sigma_est= np.mean(estimate_sigma(img, multichannel=True))
denoise = denoise_nl_means(img, h=1.15*sigma_est, fast_mode=True, patch_size=5, patch_distance=3)
denoise_ubyte= img_as_ubyte(denoise)

## Lung extraction ##
#mask = denoise_ubyte > 0

##Crop Circle
def check_if_inside(point, center, radius):
    equation=(point[0]-center[0])**2 + (point[1]-center[1])**2 - radius**2
    if equation>0:
        return False
    else:
        return True

center = [139,115] #Temp
radius = 90 #Temp

for rows in range(denoise_ubyte.shape[0]):
    for cols in range(denoise_ubyte.shape[1]):
        if not check_if_inside([rows,cols], center, radius):
           denoise_ubyte[cols][rows]=0


toprocess_img = transform.resize(denoise_ubyte[center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius], (15,15))
#new=mask*denoise_ubyte


##PLOTTING##

#plt.hist(denoise_ubyte.flat, bins=50, range=(0,255))
plt.figure()
#plt.imshow(toprocess_img,cmap='gray', vmin=0, vmax=255)
plt.imshow(toprocess_img,cmap='gray')

plt.savefig("/home/ahmad/Documents/TUHH/Semester 3/Intelligent Systems in Medicine/Project/Classification-of-different-dieases-using-ML-and-DL/experiments/crop_images/test.png")
##Computational time
print("--- %s seconds ---" % (time.time() - start_time))




