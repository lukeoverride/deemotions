'''
    Luca Surace, University of Calabria - Plymouth University
    
    This code creates for each image in the path (first argument), two gaussian noised images:
    the first one with 7-value, the second one with a value of 14.
'''

import numpy as np
import glob,os
import cv2
import sys
def noisy(noise_value,image):
	row,col,ch= image.shape
	mean = 0
	var = noise_value
	gauss = np.random.normal(mean,var,(row,col,ch))
	gauss = gauss.reshape(row,col,ch)
	noisy = image + gauss
	noisy = np.clip(noisy,0,255)
	return noisy.astype(np.uint8)

os.chdir(sys.argv[1])
for file in glob.glob("*.png"):
	fileNoExtension = file[0:len(file)-4];
	img = cv2.imread(file)
	imgNoised7 = noisy(7,img)
	imgNoised14 = noisy(14,img)
	cv2.imwrite(fileNoExtension+"_noised1.png",imgNoised7)
	cv2.imwrite(fileNoExtension+"_noised2.png",imgNoised14)
