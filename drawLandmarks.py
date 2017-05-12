'''
    Luca Surace, University of Calabria - Plymouth University
    
    This code projects the landmark points in the .txt files on the pictures (.png files) present in the
    directory. The path dir is taken by first input argument.
    It also makes an example of translation and rotation and write results on a .txt file, comment if you do not need this
    and use x,y arrays instead of xtrasl, ytrasl.
'''

import sys
import numpy as np
import cv2

img = cv2.imread(sys.argv[1]+".png",0)
x, y = np.loadtxt(sys.argv[1]+"_landmarks.txt", unpack=True)
font = cv2.FONT_HERSHEY_SIMPLEX

scaling_factor = 0.75;
xsc, ysc = (x*scaling_factor,y*scaling_factor);
rows,cols = img.shape
angle = 0.52;
angleDegrees = 30.0;
res = cv2.resize(img,None,fx=scaling_factor, fy=scaling_factor, interpolation = cv2.INTER_CUBIC)
rows,cols = res.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),angleDegrees,1)
dst = cv2.warpAffine(res,M,(cols,rows))
matrice = [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
coordinates = (xsc,ysc);
result = np.matmul(matrice,coordinates);
xtransf,ytransf = result;
xtrasl = xtransf + M[0][2];
ytrasl = ytransf + M[1][2];

output = np.column_stack((xtrasl.flatten(),ytrasl.flatten()))
np.savetxt('output.txt',output,delimiter='   ')
for i in range(0,len(xtransf)):
#for i in range(0,len(x)):
	cv2.circle(dst,(int(xtrasl[i]),int(ytrasl[i])), 7, (255,0,0), 1)
	#cv2.circle(img,(int(x[i]),int(y[i])), 7, (128,0,0), 1)
	#cv2.putText(img,str(i),(int(x[i])-5,int(y[i])+5), font, 0.25,(128,0,0),1)
#cv2.imshow('image',dst)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


