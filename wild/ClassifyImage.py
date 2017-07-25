import cv2
import numpy as np
import tensorflow as tf
import os,sys,glob
from cnn_wildemotion_detection import CnnEmotionDetection
from GoogleDetector import GoogleDetector
from ImagePreprocessing import ImagePreprocessing


def classify_image(test_path, image_path):
    #image = cv2.imread(image_path).astype(np.float32)

    google_detector = GoogleDetector()
    image_preprocessor = ImagePreprocessing()
    print image_path
    labels = google_detector.detect_labels(image_path)

    with open(image_path, 'rb') as image:
        faces = google_detector.detect_face(image)
        # Reset the file pointer, so we can read the file again
        image.seek(0)
        image_preprocessor.crop_faces(image_path, image, faces)

    image_preprocessor.scale_images(test_path+"Faces/",64.0)
    sess = tf.Session()
    emotion_detector = CnnEmotionDetection(sess)
    emotion_detector.load_variables('/home/napster/emotions/wild/checkpoints/emotiW_detection_171851/cnn_emotiW_detection-1499.meta',
                                    '/home/napster/emotions/wild/checkpoints/emotiW_detection_171851/')
    #chiamare le predictions tu tutte le immagini e avere la media
    emotion_map = emotion_detector.getEmotionMap(test_path+"Scaled/")
    print emotion_map
    cnn_predictions = emotion_detector.getMean(emotion_map)

    print cnn_predictions
    print labels



def main(test_path):
    os.chdir(test_path)
    for image_path in sorted(glob.glob("*")):
        classify_image(test_path, image_path)

if __name__ == '__main__':
    main(sys.argv[1])