#!/usr/bin/env python

##
# Massimiliano Patacchiola, Plymouth University 2016
# massimiliano.patacchiola@plymouth.ac.uk
#
# Python example for the manipulation of the Prima Head Pose Image Database:
# http://www-prima.inrialpes.fr/perso/Gourier/Faces/HPDatabase.html
#
# It contains three functions that allow creating a CSV file and to crop/resize
# the faces. It can generate 15 pickle files for the leave-one-out (Jack Knife)
# cross-validation test on unknown subjects. 
# To use the file you have to insert the right paths in the main function.
#
# Requirements: 
# OpenCV (sudo apt-get install libopencv-dev python-opencv) 
# Numpy (sudo pip install numpy)
# Six (sudo pip install six)

import cv2
import os.path
import random
import numpy as np
import csv
import glob
import re
from six.moves import cPickle as pickle

##
# Given the input directory containing the image folders (Person01, Person02, Person03, etc)
# it generates a CSV (comma separated value) files containing the image address and the 
# noised-emotion values. The images are cropped and the face is saved in the output folder.
# @param input_path the folder containing the database folders (images)
# @param label_path the folder containing the emotion labels files
# @param the output directory to use for saving the CSV files
def create_csv(input_path, label_path, output_path):

    #Image counter
    counter = 0
    roll = 0.0

    #Write the header
    fd = open(output_path + '/prima_label.csv','w')
    fd.write("path, id, serie, emotion, noised" + "\n")
    fd.close()

    #Iterate through all the folder specified in the input path
    #per ogni immagine, splitta il nome e prenditi i valori di persona e seq number, vai nella cartella appropriata, apri il file e prendi il valore dell'emozione
        for image_path in glob.glob(input_path+"*.png"):

            splitted = image_path.split('/')
            image_name = splitted[len(splitted)-1]
            image_no_extension = image_name[0:len(image_name)-4];
            splitted = image_name.split('_')
            if (len(splitted) == 4):
				image_no_extension = image_name[0:len(image_name)-12];
            person_id = splitted[0][1:len(splitted[0])];
            seq_n = splitted[1]
			noised = 0;
            if (len(splitted) == 4):
				noised = 1;
			#RETRIEVE HERE THE EMOTION FILE ASSOCIATED
            emotion = int(np.loadtxt(label_path+"S"+person_id+"/"+seq_n+"/"+image_no_extension+"_emotion.txt", unpack=True))
            
            #Write the CSV file
            #e' meglio se la output dir sia assoluta e non relativa, quindi partendo dalla root
            fd = open(input_path + '/prima_label.csv','a')
            fd.write(image_path + "," + str(int(person_id)) + "," + str(int(seq_n)) + "," + str(emotion) + "," + str(int(noised)) + "\n")
            fd.close()

            counter += 1

##
# Generate a pickle file containing Numpy arrays ready to use for
# the Leave-One-Out (loo) coross-validation test. There are 15 pickle files.
# In each pickle file there is a test matrix containing the images of a 
# single subject and a training matrix containing the images of all 
# the other subjects.
# @param csv_path the path to the CSV file generated with create_csv function
# @param output_path the path where saving the 15 pickle files
# @param shuffle if True it randomises the position of the images in the training dataset
def create_loo_pickle(csv_path, output_path, shuffle=False):

    #Saving the TEST file names in a list
    image_list = list()
    with open(csv_path, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        first_line = 0 #To jump the header line
        for row in reader:
            if(first_line != 0): image_list.append(row[0]) #prende la prima colonna come stringa
            first_line = 1

    #Loading the labels
    person_id_vector = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(1), dtype=np.float32) #prende la seconda colonna (indice 1) come vettore
    seq_n_vector = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(2), dtype=np.float32)
    emotion_vector = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(3), dtype=np.float32)
    noised_vector = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(4), dtype=np.float32)

    #Printing shape
    print("Tot Images: " + str(len(image_list)))
    print("Person ID: " + str(person_id_vector.shape))
    print("Sequence n: " + str(seq_n_vector.shape))
    print("Emotion: " + str(emotion_vector.shape))
    
    #per ogni persona
    for i in range(1,118):
    	
    		
    		#local variable clered at ech cycle
		    training_list = list()
		    training_emotion_list = list()
		    training_noised_list = list()
		    
		    valid_list = list()
		    valid_emotion_list = list()
		    valid_noised_list = list()

		    test_list = list()
		    test_emotion_list = list()
		    test_noised_list = list()
		    
		    row_counter = 0
		    insertedInTraining = 0
		    
		    for person_id in person_id_vector:
		        #Check if the image exists
		        if os.path.isfile(image_list[row_counter]):
		            image = cv2.imread(str(image_list[row_counter])) #colour
		            img_h, img_w, img_d = image.shape
		        else:
		            print("The image do not exist: " + image_list[row_counter])
		            raise ValueError('Error: the image file do not exist.')
	 
		        #Separate test and training sets     
		        #FARE IN MODO CHE PRENDO SOLO LE IMMAGINI ORIGINALI E NON QUELLE RUMOROSE PER IL TEST SET     
		        if(int(person_id) == i && noised_vector[row_counter] == 0): #if it is not noised 
		             test_list.append(image)
		             test_emotion_list.append(emotion_vector[row_counter])
		             test_noised_list.append(noised_vector[row_counter])
		        else:
		        #900 training set, 78 validation set, selected ramdomly
		        	 r = random.uniform(0,1)
		        	 if (r < 0.9 && insertedInTraining < 900):
				     	training_list.append(image)         
				     	training_emotion_list.append(emotion_vector[row_counter])
				     	training_noised_list.append(noised_vector[row_counter])
				     	insertedInTraining += 1;
				     else
				     	valid_list.append(image)
				     	valid_emotion_list.append(emotion_vector[row_counter])
				     	valid_noised_list.append(noised_vector[row_counter])
				     #mi serve anche tenermi l'informazione sul numero di seq e noised value?
		        row_counter += 1
		    

        #Create arrays
        training_array = np.asarray(training_list)
        training_emotion_array = np.asarray(training_emotion_list) 
        training_noised_array = np.asarray(training_noised_list)
    
        test_array = np.asarray(test_list)
        test_emotion_array = np.asarray(test_emotion_list) 
        test_noised_array = np.asarray(test_noised_list) 

        training_array = np.reshape(training_array, (-1, img_h*img_w*img_d))
        training_emotion_array = np.reshape(training_emotion_array, (-1, 1)) 
        training_noised_array = np.reshape(training_noised_array, (-1, 1))
     
        test_array = np.reshape(test_array, (-1, img_h*img_w*img_d)) 
        test_emotion_array = np.reshape(test_emotion_array, (-1, 1)) 
        test_noised_array = np.reshape(test_noised_array, (-1, 1)) 

        print("Training dataset: ", training_array.shape)
        print("Training emotion label: ", training_emotion_array.shape)
        print("Training noised label: ", training_emotion_array.shape)
        print("Test dataset: ", test_array.shape)
        print("Test emotion label: ", test_emotion_array.shape)
        print("Test noised label: ", test_noised_array.shape)

        #saving the dataset in a pickle file
        pickle_file = output_path + "/prima_p" + str(i) + "_out.pickle"
        print("Saving the dataset in: " + pickle_file)
        print("... ")
        try:
             print("Opening the file...")
             f = open(pickle_file, 'wb')
             save = {
               'training_dataset': training_array,
               'training_emotion_label': training_emotion_array,
               'training_noised_label': training_noised_array,    
               'test_dataset': test_array,
               'test_emotion_label': test_emotion_array,
               'test_noised_label': test_noised_array   
               # aggiungere salvataggio valid set 
                   }

             print("Training dataset: ", training_array.shape)
             print("Training emotion label: ", training_emotion_array.shape)
             print("Training noised label: ", training_noised_array.shape)
             print("Test dataset: ", test_array.shape)
             print("Test emotion label: ", test_emotion_array.shape)
             print("Test noised label: ", test_noised_array.shape)

             print("Saving the file...")
             pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
             print("Closing the file...")
             f.close()

             print("")
             print("The dataset has been saved and it is ready for the training! \n")
             print("")

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
def show_pickle_element(pickle_file, element, element_type="training", img_size=64):

    #Check if the file exists
    if os.path.isfile(pickle_file) == False:
        print("The pickle file do not exist: " + pickle_file)
        raise ValueError('Error: the pickle file do not exist.')

    #Open the specified dataset and return the element
    if(element_type == "training"):
        with open(pickle_file, 'rb') as f:
            handle = pickle.load(f)
            training_dataset = handle['training_dataset']
            training_emotion_label = handle['training_emotion_label']
            training_noised_label = handle['training_noised_label']
            del handle  # hint to help gc free up memory
            print("Selected element: " + str(element))
            print("emotion: " + str(training_emotion_label[element]))
            print("noised: " + str(training_noised_label[element]))
            print("")
            img = training_dataset[element]
            img_h, img_w, img_d = img.shape
            img = np.reshape(img, (img_size,img_size,img_w))
            cv2.imwrite( "./image.jpg", img );
            #cv2.imshow('image',img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

    elif(element_type == "test"):
            handle = pickle.load(f)
            test_dataset = handle['test_dataset']
            test_emotion_label = handle['test_emotion_label']
            test_noised_label = handle['test_noised_label']
            del handle  # hint to help gc free up memory
            print("Selected element: " + str(element))
            print("emotion: " + str(test_emotion_label[element]))
            print("noised: " + str(test_noised_label[element]))
            print("")
            img = test_dataset[element]
            img_h, img_w, img_d = image.shape
            img = np.reshape(img, (img_size,img_size,img_w))
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

    create_csv(input_path="./", label_path="./output", output_path="./")


    #2- It creates 15 pickle files containing numpy arrays with images and labels.
    # You have to specify the CSV file path created in step 1.

    #create_loo_pickle(csv_path="./prima_label.csv", output_path="./output", shuffle=False)


    #3- You can check that everything is fine using this function.
    # In this example it takes a random element and save it in the current folder.
    # It prints the noised and emotion labels of the element.
    #element = np.random.randint(2600)

    #show_pickle_element(pickle_file="./output/prima_p1_out.pickle", element=element, element_type="training", img_size=64)


if __name__ == "__main__":
    main()

