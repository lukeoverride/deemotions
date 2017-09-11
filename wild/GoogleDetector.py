#!/usr/bin/env python

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

    This code calls Google Vision API for detection of faces and labels
    in the given image.

'''

import io

from google.cloud import vision

class GoogleDetector:


    def detect_face(self, face_file, max_results=4):
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


    def detect_labels(self, path):
        """Detects labels in the file."""
        vision_client = vision.Client()

        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = vision_client.image(content=content)

        labels = image.detect_labels()

        labels_list = list()
        for label in labels:
            labels_list.append(label.description.encode('utf-8'))
        return labels_list
