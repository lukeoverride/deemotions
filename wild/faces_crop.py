#!/usr/bin/env python

# Copyright 2015 Google, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Draws squares around detected faces in the given image."""

import argparse
import numpy as np
import os,sys,glob

from google.cloud import vision
from PIL import Image, ImageDraw


def detect_face(face_file, max_results=4):
    """Uses the Vision API to detect faces in the given file.

    Args:
        face_file: A file-like object containing an image with faces.

    Returns:
        An array of Face objects with information about the picture.
    """
    content = face_file.read()
    # [START get_vision_service]
    image = vision.Client().image(content=content)
    # [END get_vision_service]

    return image.detect_faces()


def crop_faces(file_name, image, faces):
    """Crop faces, then saves each detected face into a separate file.

    Args:
      image: a file containing the image with the faces.
      faces: a list of faces found in the file. This should be in the format
          returned by the Vision API.
    """
    img = np.asarray(Image.open(image))
    img_length = img.shape[1]
    img_high = img.shape[0]
    count = 0
    
    for face in faces:
        box = [(bound.x_coordinate, bound.y_coordinate)
               for bound in face.bounds.vertices]

        x0 = box[0][0]
        y0 = box[0][1]
        x1 = box[1][0]
        y1 = box[1][1]
        x2 = box[2][0]
        y2 = box[2][1]
        x3 = box[3][0]
        y3 = box[3][1]
        top_left_x = x0
        top_left_y = y0
        length = x1-x0
        high = y3-y0

        if length != high:
            padding = np.abs(high - length)
            padding = float(padding)
            if (length > high):
                y0 = y0 - (padding/2)
                y1 = y1 - (padding/2)
                y2 = y2 + (padding/2)
                y3 = y3 + (padding/2)
                top_left_y = y0
                high = high + padding
            else:
                x0 = x0 - (padding / 2)
                x3 = x3 - (padding / 2)
                x1 = x1 + (padding / 2)
                x2 = x2 + (padding / 2)
                top_left_x = x0
                length = length + padding
            
            #shift if the window is outside the image
            #move left
            if (x1 > img_length and x2 > img_length):
                gap = x1 - img_length
                x0 = x0 - gap
                x3 = x3 - gap
                x1 = x1 - gap
                x2 = x2 - gap
                top_left_x = top_left_x - gap
            #move down
            if (y0 < 0 and y1 < 0):
                print "caso 2"
                top_left_y = top_left_y + np.abs(y0)
            #move right
            if (x0 < 0 and x3 < 0):
                gap = np.abs(x0)
                x0 = x0 + gap
                x3 = x3 + gap
                x1 = x1 + gap
                x2 = x2 + gap
                top_left_x = top_left_x + gap
            #move left
            if (y2 > img_high and y3 > img_high):
                top_left_y = top_left_y - (y2-img_high)
                
        top_left_x = int(top_left_x)
        length = int(length)
        top_left_y = int(top_left_y)
        high = int(high)
        cropped = img[top_left_y:top_left_y+high,top_left_x:top_left_x+length]
        cropped = Image.fromarray(cropped)
        fileNoExtension = file_name[0:len(file_name) - 4];
        cropped.save("Faces/"+fileNoExtension+"_face_"+str(count)+".jpg")
        count += 1

def main(max_results):
    for file in glob.glob("*.jpg"):
        print file
        with open(file, 'rb') as image:
            faces = detect_face(image, max_results)
            print('Found {} face{}'.format(
		        len(faces), '' if len(faces) == 1 else 's'))

            print('Writing for original file {}'.format(file))
            # Reset the file pointer, so we can read the file again
            image.seek(0)
            crop_faces(file, image,faces)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detects faces in the given image.')
    parser.add_argument(
        'input_dir', help='the directory of the images you\'d like to detect faces in.')
    parser.add_argument(
        '--max-results', dest='max_results', default=4,
        help='the max results of face detection.')
    args = parser.parse_args()
    
    os.chdir(args.input_dir)
    os.mkdir("Faces")

    main(args.max_results)
