import numpy as np
import os
import cv2
import sys
def noisy(noise_typ,image):
	if noise_typ == "gauss":
		row,col= image.shape
		mean = 0
		var = 14
		#sigma = var**0.5
		gauss = np.random.normal(mean,var,(row,col))
		gauss = gauss.reshape(row,col)
		noisy = image + gauss
		noisy = np.clip(noisy,0,255)
		
		return noisy.astype(np.uint8)
#do it for all images in the dir
img = cv2.imread(sys.argv[1]+".png",0)
imgMod = noisy('gauss',img)
cv2.imshow('image',imgMod)
cv2.waitKey(0)
cv2.destroyAllWindows()
