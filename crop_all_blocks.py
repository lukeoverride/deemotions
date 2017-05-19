'''
    Copyright (C) 2017 Luca Surace - University of Calabria, Plymouth University
    
    This file is part of Deemotions. Deemotions is an Emotion Recognition System
    based on Deep Learning method.

    Deemotions is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Deemotions is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Deemotions.  If not, see <http://www.gnu.org/licenses/>.
    
    -----------------------------------------------------------------------
    
    This file crops CK+ images in five blocks (tight face, mouth, left eye, top nose, nose tip)
    using landmarks files given with the CK+ dataset.
    
'''

import cv2
import numpy as np
import sys
import glob,os

def crop_face():
	FACE_BLOCK_WIDTH = 52.0;
	FACE_BLOCK_HEIGHT = 72.0;
	FOREHEAD_PADDING = 50;
	#left and right margins
	WIDTH_LM = 0;
	WIDTH_LM2 = 36;
	WIDTH_RM = 16;
	WIDTH_RM2 = 45;
	HEIGHT_BOTTOM = 8;
	HEIGHT_TOP = 19;
	xa = int((x[WIDTH_LM]+x[WIDTH_LM2])/2);
	xb = int((x[WIDTH_RM]+x[WIDTH_RM2])/2);
	yc = int(y[HEIGHT_BOTTOM]);
	yd = int(y[HEIGHT_TOP]-FOREHEAD_PADDING);
	crop(FACE_BLOCK_WIDTH,FACE_BLOCK_HEIGHT,"face",xa,xb,yc,yd);
	xs = (xa,xb);
	return xs;
	
def crop_mouth(xs):
	MOUTH_BLOCK_WIDTH = 40.0;
	MOUTH_BLOCK_HEIGHT = 24.0;
	WIDTH_LM = 48;
	WIDTH_RM = 54;
	HEIGHT_BOTTOM = 8;
	HEIGHT_BOTTOM2 = 57;
	HEIGHT_TOP = 33;
	xa = int((x[WIDTH_LM]+xs[0])/2);
	xb = int((x[WIDTH_RM]+xs[1])/2);
	yc = int((y[HEIGHT_BOTTOM]+y[HEIGHT_BOTTOM2])/2);
	yd = int(y[HEIGHT_TOP]);
	crop(MOUTH_BLOCK_WIDTH,MOUTH_BLOCK_HEIGHT,"mouth",xa,xb,yc,yd);
	
def crop_eye():
	EYE_BLOCK_WIDTH = 32.0;
	EYE_BLOCK_HEIGHT = 24.0;
	FOREHEAD_PADDING = 20;
	#left and right margins
	WIDTH_LM = 0;
	WIDTH_RM = 29;
	HEIGHT_BOTTOM = 29;
	HEIGHT_TOP = 19;
	xa = int(x[WIDTH_LM]);
	xb = int(x[WIDTH_RM]);
	yc = int(y[HEIGHT_BOTTOM]);
	yd = int(y[HEIGHT_TOP]-FOREHEAD_PADDING);
	crop(EYE_BLOCK_WIDTH,EYE_BLOCK_HEIGHT,"eye",xa,xb,yc,yd);
	
def crop_topnose():
	TOPNOSE_BLOCK_WIDTH = 40.0;
	TOPNOSE_BLOCK_HEIGHT = 36.0;
	FOREHEAD_PADDING = 15;
	#left and right margins
	WIDTH_LM = 38;
	WIDTH_RM = 43;
	HEIGHT_BOTTOM = 28;
	HEIGHT_TOP = 19;
	xa = int(x[WIDTH_LM]);
	xb = int(x[WIDTH_RM]);
	yc = int(y[HEIGHT_BOTTOM]);
	yd = int(y[HEIGHT_TOP]-FOREHEAD_PADDING);
	crop(TOPNOSE_BLOCK_WIDTH,TOPNOSE_BLOCK_HEIGHT,"topnose",xa,xb,yc,yd);
	
def crop_nosetip():
	NOSETIP_BLOCK_WIDTH = 40.0;
	NOSETIP_BLOCK_HEIGHT = 32.0;
	#left and right margins
	WIDTH_LM = 37;
	WIDTH_RM = 44;
	HEIGHT_BOTTOM = 51;
	HEIGHT_TOP = 29;
	xa = int(x[WIDTH_LM]);
	xb = int(x[WIDTH_RM]);
	yc = int(y[HEIGHT_BOTTOM]);
	yd = int(y[HEIGHT_TOP]);
	crop(NOSETIP_BLOCK_WIDTH,NOSETIP_BLOCK_HEIGHT,"nosetip",xa,xb,yc,yd);


def crop(width,height,block_name,xa,xb,yc,yd):
	AB = xb-xa;
	CD = yc-yd;
	crop_img = img[yd:yd+CD, xa:xa+AB]
	scaling_factor_x = width/AB;
	scaling_factor_y = height/CD;
	res = cv2.resize(crop_img,None,fx=scaling_factor_x, fy=scaling_factor_y, interpolation = cv2.INTER_CUBIC)
	cv2.imwrite(sys.argv[2]+block_name+"/"+file,res)
	
os.chdir(sys.argv[1]) #changes current working directory
os.mkdir(sys.argv[2]+"face")
os.mkdir(sys.argv[2]+"mouth")
os.mkdir(sys.argv[2]+"eye")
os.mkdir(sys.argv[2]+"topnose")
os.mkdir(sys.argv[2]+"nosetip")

for file in glob.glob("*.png"):
	fileNoExtension = file[0:len(file)-4];
	img = cv2.imread(file,0)
	x, y = np.loadtxt(fileNoExtension+"_landmarks.txt", unpack=True)
	xs = crop_face()
	crop_mouth(xs);
	crop_eye();
	crop_topnose();
	crop_nosetip();


