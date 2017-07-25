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
            print label.description
            labels_list.append(label.description.encode('utf-8'))
        return labels_list