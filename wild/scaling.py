#first @param input the path in which the images to be resized are located

import sys, glob, os
import cv2



def scale_images(input_path):
    os.chdir(input_path)
    NEW_SIZE = 64.0

    for file in glob.glob("*.jpg"):
        fileNoExtension = file[0:len(file)-4]
        img = cv2.imread(file)
        high = img.shape[0]
        width = img.shape[1]
        #computes scaling factor
        scaling_factor_y = float(NEW_SIZE/high)
        scaling_factor_x = float(NEW_SIZE/width)
        #scales the image
        res = cv2.resize(img,None,fx=scaling_factor_x, fy=scaling_factor_y, interpolation = cv2.INTER_CUBIC)
        if not os.path.exists("Scaled/"): os.makedirs("Scaled")
        cv2.imwrite("Scaled/"+file,res)
