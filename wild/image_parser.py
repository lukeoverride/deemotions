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
    
    This code generates pickle files from GAF pictures. These files are used
    later in the process to load images in the CNN.

'''


import cv2
import os.path
import numpy as np
import csv
import glob
import sys
from six.moves import cPickle as pickle

##
# Given the input directory containing the image folders (Person01, Person02, Person03, etc)
# it generates a CSV (comma separated value) files containing the image address and the 
# noised-emotion values. The images are cropped and the face is saved in the output folder.
# @param input_path the folder containing the database folders (images). It is also the output directory to use for saving the CSV files.
# It must be in the form ../Positive|Negative|Neutral/*/*/
def create_csv(input_path):

    #Image counter
    counter = 0
    roll = 0.0

    #Write the header
    fd = open('/home/napster/verywild_GAF.csv','w')
    fd.write("path, id, emotion" + "\n")
    fd.close()

    splitPath = input_path.split('/')
    emotionStr = splitPath[len(splitPath)-3]
    
    if (emotionStr == 'Positive'):
        emoCode = np.array([1,0,0])
    elif (emotionStr == 'Neutral'):
        emoCode = np.array([0,1,0])
    elif (emotionStr == 'Negative'):
        emoCode = np.array([0,0,1])
    

    id = 0

    #Iterate through all the folder specified in the input path
    for image_path in sorted(glob.glob(input_path+"*")):
        splitted = image_path.split('/')
        image_name = splitted[len(splitted)-1]
        image_no_extension = image_name[0:len(image_name)-4];
        #Write the CSV file
        fd = open('/home/napster/verywild_GAF.csv','a')
        fd.write(image_path + "," + str(int(id)) + "," + str(emoCode) + "\n")
        fd.close()
        id += 1

##
# Generate a pickle file containing Numpy arrays ready to use for
# the Leave-One-Out (loo) coross-validation test. There are 15 pickle files.
# In each pickle file there is a test matrix containing the images of a 
# single subject and a training matrix containing the images of all 
# the other subjects.
# @param csv_path the path to the CSV file generated with create_csv function
# @param output_path the path where saving the 118 pickle files
# @param shuffle if True it randomises the position of the images in the training dataset
def create_loo_pickle(csv_path, output_path):

    if not os.path.exists(output_path): os.makedirs(output_path)

    #Saving the TEST file names in a list
    image_list = list()
    with open(csv_path, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        first_line = 0 #To jump the header line
        for row in reader:
            if(first_line != 0): image_list.append(row[0]) #prende la prima colonna come stringa
            first_line = 1

    #Loading the labels
    id_vector = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(1), dtype=np.int32) #prende la seconda colonna (indice 1) come vettore
    emotion_vector = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(2),dtype=np.ndarray)

    #Printing shape
    print("Tot Images: " + str(len(image_list)))
    print("Person ID: " + str(id_vector.shape))
    print("Emotion: " + str(emotion_vector.shape))
    
    #local variable clered at ech cycle
    training_list = list()
    training_emotion_list = list()

    #for each element
    for i in id_vector:

        #Check if the image exists
        if os.path.isfile(image_list[i]):
            image = cv2.imread(str(image_list[i])) #color
            img_h, img_w, img_d = image.shape
            #img_d = 3
        else:
            print("The image do not exist: " + image_list[i])
            raise ValueError('Error: the image file do not exist.')

        training_list.append(image)
        training_emotion_list.append(emotion_vector[i])

    #Create arrays
    training_array = np.asarray(training_list)
    training_emotion_array = np.asarray(training_emotion_list)

    training_array = np.reshape(training_array, (-1, img_h*img_w*img_d))
    training_emotion_array = np.reshape(training_emotion_array, (-1, 1))

    print("Training dataset: ", training_array.shape)
    print("Training emotion label: ", training_emotion_array.shape)

    #saving the dataset in a pickle file
    pickle_file = output_path + "/GAF_p" + str(i) + ".pickle"
    print("Saving the dataset in: " + pickle_file)
    print("... ")
    try:
         print("Opening the file...")
         f = open(pickle_file, 'wb')
         save = {
           'training_dataset': training_array,
           'training_emotion_label': training_emotion_array,
               }

         print("Training dataset: ", training_array.shape)
         print("Training emotion label: ", training_emotion_array.shape)

         print("Saving the file...")
         pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
         print("Closing the file...")
         f.close()

         print("The dataset has been saved and it is ready for the training! \n")
    except Exception as e:
         print('Unable to save data to', pickle_file, ':', e)
         raise

##
# Given a pickle file name and an element number it show the element
# and the associated noised-emotion labels.
# @param pickle_file path to the pickle file
# @param element an integer that specifies which element to return
# @param element_type the dataset to acces (training or test)
# @param img_size the size of the image (default 64x64 pixels)
def show_pickle_element(pickle_file, element, element_type="training", img_size=32):

    #Check if the file exists
    if os.path.isfile(pickle_file) == False:
        print("The pickle file do not exist: " + pickle_file)
        raise ValueError('Error: the pickle file do not exist.')

    with open(pickle_file, 'rb') as f:
    #Open the specified dataset and return the element
        if(element_type == "training"):

                handle = pickle.load(f)
                training_dataset = handle['training_dataset']
                training_emotion_label = handle['training_emotion_label']
                del handle  # hint to help gc free up memory
                print("Selected element: " + str(element))
                print("emotion: " + str(training_emotion_label[element]))
                print("")
                img = training_dataset[element]
                img = np.reshape(img, (72,52,3))
                cv2.imwrite( "./image.jpg", img );
                #cv2.imshow('image',img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

        else:
            raise ValueError('Error: element_type must be training or test.')


##
# To use the function reported below you have to and add the right folder paths.
#
#
def main():

    #1- It creates the CSV file and cropped/resized faces
    # First of all you have to specify where the uncompressed folder with the dataset is located
    # Specify an output folder and the image size (be careful to choose this size, it must be less
    # than the dimension of the original faces). You can choose if save the image in grayscale or colours.

    create_csv(input_path=sys.argv[1])


    #2- It creates 118 pickle files containing numpy arrays with images and labels.
    # You have to specify the CSV file path created in step 1.

    #create_loo_pickle(csv_path=sys.argv[1]+"wild_GAF.csv", output_path=sys.argv[1])


    #3- You can check that everything is fine using this function.
    # In this example it takes a random element and save it in the current folder.
    # It prints the emotion labels of the element.
    #element = np.random.randint(2600)

    #show_pickle_element(pickle_file="/home/napster/GAF_2/Train/Negative/Faces/Pickles/GAF_p4469.pickle", element=2, element_type="training", img_size=32)


if __name__ == "__main__":
    main()

