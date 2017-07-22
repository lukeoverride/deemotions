#!/usr/bin/env python

import io
import os, sys, glob
import numpy as np

from google.cloud import vision

def detect_labels(path):
    """Detects labels in the file."""
    vision_client = vision.Client()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision_client.image(content=content)

    return image.detect_labels()
           
        
def main(emotionStr):
    if (emotionStr == 'Positive'):
        emoCode = np.array([1,0,0])
    elif (emotionStr == 'Neutral'):
        emoCode = np.array([0,1,0])
    elif (emotionStr == 'Negative'):
        emoCode = np.array([0,0,1])
        
    for file in sorted(glob.glob("*")):
        if (file == "Faces"):
            continue
        labels = detect_labels(file)
        labels_list = list()
        for label in labels:
            print label.description
            labels_list.append(label.description.encode('utf-8'))
        #Write the CSV file
        fd = open('wild_GAF_labels.csv','a')
        
        fd.write(file + "," + str(emoCode) + "," + str(labels_list) + "\n")
        '''
        el_num = 0
        for x in labels_list:
            fd.write(x)
            if (el_num < len(labels_list)-1):
                fd.write(",") 
            el_num += 1
        fd.write("]\n")
        '''
        fd.close()
         

if __name__ == '__main__':
    os.chdir(sys.argv[1])

    main(sys.argv[2])
