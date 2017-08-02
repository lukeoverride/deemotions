#first @param input the path in which the images to be resized are located

import sys, glob, os
import cv2

os.chdir(sys.argv[1])

for file in glob.glob("*"):
    img = cv2.imread(file)
    high = img.shape[0]
    width = img.shape[1]
    #computes scaling factor
    #scales the image
    if width>2500:
        print file
        res = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(file,res)
