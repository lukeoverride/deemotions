import tensorflow as tf
import cv2
from cnn_wildemotion_detection import CnnEmotionDetection
import numpy as np
import glob
import sys
from six.moves import cPickle as pickle
import csv

'''
Returns a map containing, for each image, the average of all faces emotions value

'''
def getEmotionMap(emotion_detector, csv_path):
    emotion_all_faces = {}
    with open(csv_path, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        first_line = 0  # To jump the header line
        nuovo = 0
        for row in reader:
            if (first_line != 0):
                tokens = row[0].split("face")
                completeFileName = tokens[0].split("/")
                fileName = completeFileName[len(completeFileName) - 1]
                image = cv2.imread(row[0]).astype(np.float32)
                current_pred = emotion_detector.getEmotionsPredictions(image)
                if (fileName in emotion_all_faces):
                    emotion_all_faces[fileName] = np.append(emotion_all_faces[fileName], current_pred, axis=0)
                else:
                    nuovo += 1
                    emotion_all_faces[fileName] = current_pred
            first_line = 1
        print nuovo
        return emotion_all_faces

def computeMean(emotion_map):
    for key in emotion_map:
        emotion_map[key] = np.mean(emotion_map[key], axis=0)
        # Write the CSV file
        fd = open('wild_GAF_cnn_predictions.csv', 'a')
        fd.write(key + "," + str(emotion_map[key]) + "\n")
        fd.close()
    return emotion_map

def compute_accuracy(predictions, labels, verbose=False):
    '''This function return the accuracy

    @param predictions the output of the network for each image passed
    @param labels the correct category (target) for the image passed
    @return it returns the accuracy as number of instances correctly
    classified over the total number of instances
    '''
    #takes the highest value in the predictions and makes the one_hot vector
    predictions_normalized = np.zeros(predictions.shape)
    row = np.arange(predictions.shape[0])
    col = np.argmax(predictions, axis=1)
    predictions_normalized[row,col] = 1
    difference = np.absolute(predictions_normalized - labels)
    result = np.sum(difference,axis=1)
    correct = np.sum(result==0).astype(np.float32)
    predict_positive = np.sum(col==0)
    predict_negative = np.sum(col==2)
    predict_neutral = np.sum(col==1)
    if (verbose == True):
        print correct/predictions.shape[0]
        print [predict_positive,predict_negative,predict_neutral]
    return correct/predictions.shape[0]


def main():

    sess = tf.Session()
    emotion_detector = CnnEmotionDetection(sess)
    emotion_detector.load_variables('./checkpoints/emotiW_detection_171851/cnn_emotiW_detection-1499.meta',
                                 './checkpoints/emotiW_detection_171851/')

    emotion_all_faces_positive = getEmotionMap(emotion_detector,"./wild_GAF_faces_val_positive_all.csv")
    emotion_all_faces_negative = getEmotionMap(emotion_detector, "./wild_GAF_faces_val_negative_all.csv")
    emotion_all_faces_neutral = getEmotionMap(emotion_detector, "./wild_GAF_faces_val_neutral_all.csv")

    emotion_all_faces_positive = computeMean(emotion_all_faces_positive)
    emotion_all_faces_negative = computeMean(emotion_all_faces_negative)
    emotion_all_faces_neutral = computeMean(emotion_all_faces_neutral)

    print len(emotion_all_faces_positive)
    print len(emotion_all_faces_negative)
    print len(emotion_all_faces_neutral)
    total_positive = len(emotion_all_faces_positive)
    total_negative = len(emotion_all_faces_negative)
    total_neutral = len(emotion_all_faces_neutral)
    test_label_positive = np.zeros((total_positive, 3))
    test_label_negative = np.zeros((total_negative, 3))
    test_label_neutral = np.zeros((total_neutral, 3))
    test_label_positive[:,0] = 1
    test_label_negative[:,2] = 1
    test_label_neutral[:,1] = 1
    emotion_all_faces_positive_array = np.asarray(emotion_all_faces_positive.values())
    emotion_all_faces_negative_array = np.asarray(emotion_all_faces_negative.values())
    emotion_all_faces_neutral_array = np.asarray(emotion_all_faces_neutral.values())
    emotion_all_faces_positive_array = emotion_all_faces_positive_array.reshape((total_positive,3))
    emotion_all_faces_negative_array = emotion_all_faces_negative_array.reshape((total_negative, 3))
    emotion_all_faces_neutral_array = emotion_all_faces_neutral_array.reshape((total_neutral, 3))
    compute_accuracy(emotion_all_faces_positive_array, test_label_positive, True)
    compute_accuracy(emotion_all_faces_negative_array, test_label_negative, True)
    compute_accuracy(emotion_all_faces_neutral_array, test_label_neutral, True)
    



if __name__ == "__main__":
    main()
