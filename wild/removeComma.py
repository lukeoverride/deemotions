import sys, glob, os
import cv2

os.chdir(sys.argv[1])

for file in glob.glob("*.jpg"):
    if ("," in file):
        print file
        #newstr = file.replace(",", "")
        #os.system("mv "+file+" "+newstr)

