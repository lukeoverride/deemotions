import cv2
import numpy as np
import tensorflow as tf
import os,sys,glob
from cnn_wildemotion_detection import CnnEmotionDetection
from GoogleDetector import GoogleDetector
from ImagePreprocessing import ImagePreprocessing
from bayesian_network import BayesianNetwork

def classify_image(test_path, image_path,real_label):

    targets = ['Positive','Negative','Neutral']

    google_detector = GoogleDetector()
    image_preprocessor = ImagePreprocessing()
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
    emotion_map = emotion_detector.getEmotionMap(test_path+"Scaled/")
    cnn_predictions = emotion_detector.getMean(emotion_map)

    my_bayes_net = BayesianNetwork()
    is_correct = my_bayes_net.initModel("/home/napster/emotions/wild/wild_GAF_labels_histogram_train_global.csv",False)
    print("Model correct: " + str(is_correct))
    reverse_index_list = [0,2,1]
    posterior = my_bayes_net.inferenceWithCNN(labels,reverse_index_list[np.argmax(cnn_predictions)])
    final_predictions = np.argmax(posterior['emotion_node'].values)
    bayesian_label = targets[final_predictions]


    print image_path
    print "Real label: ",real_label
    if (real_label == bayesian_label):
        print "Bayesian label: ",bayesian_label,
        print "*"
    else:
        print "Bayesian label: ", bayesian_label
    print "CNN label: ",targets[reverse_index_list[np.argmax(cnn_predictions)]]
    print labels

    return final_predictions




def main(test_path,real_label):
    os.chdir(test_path)
    predicted_counter = [0,0,0]
    for image_path in sorted(glob.glob("*")):
        final_predictions = classify_image(test_path, image_path,real_label)
        predicted_counter[final_predictions] += 1
        print predicted_counter
        print ""



if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])