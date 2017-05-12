'''
    Luca Surace, University of Calabria - Plymouth University
    
    This code performs image preprocessing, in particular:
    1) scales all faces by imposing a fix distance between pupils. This distance is the average of the pupils distances in the dataset;
    2) makes eye alignment by face rotation, setting the angle of the line passing through the pupils to 0 degrees.
    @argument 1: path of the original images and landmarks (input)
    @argument 2: path of the folder to save transformed images and landmarks (output)
'''



import sys
import numpy as np
import cv2
import cv
import math
import glob, os

os.chdir(sys.argv[1]) #changes current working directory
somma = 0;
#eyes constants margins. Do not modify. LM/RM = Left Margin/Right Margin with respect to the observer of the picture
LEFT_EYE_LM = 36;
LEFT_EYE_RM = 39;
RIGHT_EYE_LM = 42;
RIGHT_EYE_RM = 45;

for file in glob.glob("*.txt"):
	x, y = np.loadtxt(file, unpack=True)
	leftPupil = ((x[LEFT_EYE_LM]+x[LEFT_EYE_RM])/2, (y[LEFT_EYE_LM]+y[LEFT_EYE_RM])/2);
	rightPupil = ((x[RIGHT_EYE_LM]+x[RIGHT_EYE_RM])/2, (y[RIGHT_EYE_LM]+y[RIGHT_EYE_RM])/2);
	res = cv2.norm(leftPupil, rightPupil);
	somma = somma+res;

averageDist = somma/len(glob.glob("*.txt"));
print averageDist;

for file in glob.glob("*.png"):
	fileNoExtension = file[0:len(file)-4];
	img = cv2.imread(file,0)
	x, y = np.loadtxt(fileNoExtension+"_landmarks.txt", unpack=True)
	leftPupil = ((x[LEFT_EYE_LM]+x[LEFT_EYE_RM])/2, (y[LEFT_EYE_LM]+y[LEFT_EYE_RM])/2);
	rightPupil = ((x[RIGHT_EYE_LM]+x[RIGHT_EYE_RM])/2, (y[RIGHT_EYE_LM]+y[RIGHT_EYE_RM])/2);
	#computes scaling factor
	currDist = cv2.norm(leftPupil, rightPupil);
	scaling_factor = averageDist/currDist;
	xsc, ysc = (x*scaling_factor,y*scaling_factor);
	coordinates = (xsc,ysc);
	#computes rotation angle
	angle = math.atan((rightPupil[1]-leftPupil[1])/(rightPupil[0]-leftPupil[0]));
	angleDegrees = angle*180/math.pi;
	#scales the image
	res = cv2.resize(img,None,fx=scaling_factor, fy=scaling_factor, interpolation = cv2.INTER_CUBIC)
	#rotates the images
	rows,cols = res.shape
	M = cv2.getRotationMatrix2D((cols/2,rows/2),angleDegrees,1)
	dst = cv2.warpAffine(res,M,(cols,rows))
	matrice = [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
	result = np.matmul(matrice,coordinates);
	xtransf,ytransf = result;
	xtrasl = xtransf + M[0][2];
	ytrasl = ytransf + M[1][2];
	#write new landmarks on file
	output = np.column_stack((xtrasl.flatten(),ytrasl.flatten()))
	np.savetxt(sys.argv[2]+fileNoExtension+"_landmarks.txt",output,delimiter='   ')
	cv2.imwrite(sys.argv[2]+file,dst)
	
	
	
	
	
	
	
	
	
