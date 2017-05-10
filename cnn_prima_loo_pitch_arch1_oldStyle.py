#!/usr/bin/env python


# Massimiliano Patacchiola, Plymouth University 2016
#
#In this example I load a pickle file containing head pose images.
#The model take as input an image (or a batch) and return a single value (or a vector)
#representing the pitch value of the head in the image.
#
#DATASET: It requires the file 'prima_pitch_p1_out.pickle' which must be in the same folder of this script.
#This pickle file is based on the "prima dataset" which is available for free.
#
#TENSORBOARD: this code works on my system and it shows correctly the value of the learning rate
#and the loss at each epoch. It saves a log file in the foder '/tmp/log/pitch_logs_p1_161944'
#You should notice that the name of the file is based on the current time. You should check this
#name in the folder and use it in Tensorboard.
#You can run tensorboard with this command: tensorboard --logdir="file:///tmp/log/pitch_logs_p1_161944"
#(where the name of the file can change based on the current time). In the code below I used 
#the tag 'Tensorboard' in the comments, every time I declared a Tensorboard-related variable.
#Important: to visualise the variable you have to wait a couple of minutes after the simulation started.
#Tensorboard is slow and it can take a while in order to load the first results.
#
#COMPATIBILITY: I hope that the instance declared are compatible with your version of Tensorflow.
#I noticed that my call for the Tensorboard stuff are quite different from the one you used. I hope
#you can run it without problems. Moreover some function like 'datetime' that I used for saving the log
#file based on the current time, could not work on windows. Try to replace it with something else or remove
#it from the code. The log file can have always the same name. 


import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import datetime
#Here OpenCV is used in order to visualise images.
#However I commented the cv2 parts in the code below.
#You can use antother library if you want.
#import cv2 

#VEDERE DOPO
def accuracy(predictions, labels, verbose=False): #visto che non e' continuo il mio output ma discreto, devo usare un altro metodo statistico? confusion matrix, accuracy classica
    '''This function return the accuracy for the current batch.

    Because the network has only one neuron as output, and the output
    is considered continous, the accuracy is measured through RMSE
    @param predictions the output of the network for each image passed
    @param labels the correct category (target) for the immage passed
    @param verbose if True prints information on terminal
    @return it returns the RMSE (Root Mean Squared Error)
    '''
    if (verbose == True):
        # Convert back to degree
        predictions_degree = predictions * 180
        predictions_degree -= 90
        labels_degree = labels * 180
        labels_degree -= 90
        RMSE_pitch = np.sum(np.square(predictions_degree - labels_degree), dtype=np.float32) * 1 / predictions.shape[0]
        RMSE_pitch = np.sqrt(RMSE_pitch)
        RMSE_std = np.std(np.sqrt(np.square(predictions_degree - labels_degree)), dtype=np.float32)
        # MAE = Mean Absolute Error
        MAE_pitch = np.sum(np.absolute(predictions_degree - labels_degree), dtype=np.float32) * 1 / predictions.shape[0]
        MAE_std = np.std(np.absolute(predictions_degree - labels_degree), dtype=np.float32)

        print("=== RMSE and MAE (DEGREE) ===")

        for counter in range(0, 10):
            random = np.random.randint(0, predictions.shape[0])
            print("pitch[" + str(random) + "]: " + str(labels[random]) + " / " + str(predictions[random]))
        for counter in range(0, 10):
            random = np.random.randint(0, predictions.shape[0])
            temp_predicted = int((predictions[random] * 180) - 90)
            temp_label = int((labels[random] * 180) - 90)            
            print("pitch[" + str(random) + "]: " + str(temp_label) + " / " + str(temp_predicted))
           
        print("==============================")            
        print("RMSE mean: " + str(RMSE_pitch) + " degree")
        print("RMSE std: " + str(RMSE_std) + " degree")
        print("MAE mean: " + str(MAE_pitch) + " degree")
        print("MAE std: " + str(MAE_std) + " degree")
        print("==============================")
        # Convert back to degree
        predictions_degree = predictions * 180
        predictions_degree -= 90
        labels_degree = labels * 180
        labels_degree -= 90
    # It returns the RMSE
    return np.sqrt(np.sum(np.square(predictions_degree - labels_degree), dtype=np.float32) * 1 / predictions.shape[0])

def create_batch(train_dataset, train_labels):
    addedElements = []
    batch_data = np.zeros((0,train_dataset.shape[1],train_dataset.shape[2],train_dataset.shape[3]))
    batch_labels = np.zeros((0,train_labels.shape[1]))
    for i in range(0,7):
        codeForI = np.zeros(7,int)
        codeForI[i] = 1
        for n in range(0,3):
            k = int(np.random.uniform(0,train_labels.shape[0]))
            while ((train_labels[k] != codeForI).any() or k in addedElements):
                k = int(np.random.uniform(0,train_labels.shape[0]))
            batch_data = np.append(batch_data,[train_dataset[k]],axis=0)
            batch_labels = np.append(batch_labels,[train_labels[k]],axis=0)
            addedElements = np.append(addedElements,k)

    emotionUsed = []
    for person in range(0,4):
        emot = int(np.random.uniform(0,7))
        k = int(np.random.uniform(0, train_labels.shape[0]))
        emotCode = np.zeros(7,int)
        emotCode[emot] = 1
        while ((train_labels[k] != emotCode).any() or k in addedElements or emot in emotionUsed):
            emot = int(np.random.uniform(0, 7))
            emotCode = np.zeros(7, int)
            emotCode[emot] = 1
            k = int(np.random.uniform(0, train_labels.shape[0]))
        batch_data = np.append(batch_data, [train_dataset[k]], axis=0)
        batch_labels = np.append(batch_labels, [train_labels[k]], axis=0)
        addedElements = np.append(addedElements, k)
        emotionUsed = np.append(emotionUsed,emot)
    batch = (batch_data,batch_labels)
    return batch

def extractArraysRemoveBrackets(labels):
    labels_new = np.zeros((labels.size, 7), int)
    for i in range(0, labels.size):
        for t in range(1, 15, 2):
            labels_new[i][t / 2] = labels[i][t]
    return labels_new


def main(block_name):
              
        pickle_file = "./output/"+block_name+"/prima_p5.0_out.pickle"
        batch_size = 25 #we pass a batch of 25 images as input
        patch_size = 5 # filter size

        if (block_name == "face"):
            image_size_h = 72
            image_size_w = 52
        elif (block_name == "mouth"):
            image_size_h = 24
            image_size_w = 40
        elif (block_name == "left_eye"):
            image_size_h = 24
            image_size_w = 32
        elif (block_name == "top_nose"):
            image_size_h = 36
            image_size_w = 40
        elif (block_name == "nose_tip"):
            image_size_h = 32
            image_size_w = 40

        num_labels = 7 #the output of the network (7 neuron)
        #num_channels = 3  # colour images have 3 channels
        num_channels = 1 # grayscale images have 1 channel

        # Load the pickle file containing the dataset 
        #The pickle file contains some Numpy arrays and matrices.
        #e.g. the name 'training_dataset' is the name of the numpy
        #array as it was stored in the pickle file. Here it can be
        #used again.
        with open(pickle_file, 'rb') as f:
            save = pickle.load(f)
            train_dataset = save['training_dataset']
            train_labels = save['training_emotion_label']
            valid_dataset = save['validation_dataset']
            valid_labels = save['validation_emotion_label']
            test_dataset = save['test_dataset']
            test_labels = save['test_emotion_label']
            del save  # hint to help gc free up memory
            # Here I print the dimension of the three datasets
            print('Training set', train_dataset.shape, train_labels.shape)
            print('Validation set', valid_dataset.shape, valid_labels.shape)
            print('Test set', test_dataset.shape, test_labels.shape)

        # Reshape the datasets
        #The format used in my model is: (-1, image_size, image_size, channels)
        #Here I reshape the datasets to follow this convention
        train_dataset = train_dataset.reshape((-1, image_size_w, image_size_h, num_channels)).astype(np.float32)
        train_labels = train_labels.reshape((-1)).astype(np.ndarray)
        valid_dataset = valid_dataset.reshape((-1, image_size_w, image_size_h, num_channels)).astype(np.float32)
        valid_labels = valid_labels.reshape((-1)).astype(np.ndarray)
        test_dataset = test_dataset.reshape((-1, image_size_w, image_size_h, num_channels)).astype(np.float32)
        test_labels = test_labels.reshape((-1)).astype(np.ndarray)

        # create the arrays from string, removing brackets as well
        train_labels_new = extractArraysRemoveBrackets(train_labels)
        valid_labels_new = extractArraysRemoveBrackets(valid_labels)
        test_labels_new = extractArraysRemoveBrackets(test_labels)

        #Un altro metodo e' normalizzare tra 0 e 1 e shiftare tra -0.5 e 0.5
        #train_dataset -= 127
        #valid_dataset -= 127
        #test_dataset -= 127

        #Printing the new shape of the datasets
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels_new.shape)
        print('Test set', test_dataset.shape, test_labels_new.shape)

        #Declaring the graph object necessary to build the model
        graph = tf.Graph()
        with graph.as_default():

            print("Init Tensorflow variables...")
            # Input data. They are Tensorflow placeholders, meaning empty containers
            #that we can fill with data during the session. They must have the same 
            #shape used in our convention.
            tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size_w, image_size_h, num_channels))
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
            tf_valid_dataset = tf.constant(valid_dataset)
            tf_test_dataset = tf.constant(test_dataset)

            # Variables used by the model. You should change them if you change the model.
            # Conv layer
            # [patch_size, patch_size, num_channels, depth]
            conv1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, 6], stddev=0.1), name="conv1y_w")
            conv1_biases = tf.Variable(tf.zeros([6]), name="conv1y_b")
            # Conv layer
            # [patch_size, patch_size, depth, depth]
            conv2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, 6, 12], stddev=0.1), name="conv2y_w")
            conv2_biases = tf.Variable(tf.zeros([12]), name="conv2y_b")


            # Output layer
            conv1_size_w = (image_size_w - patch_size + 1)/2
            conv2_size_w = (conv1_size_w - patch_size + 1)/2
            conv1_size_h = (image_size_h - patch_size + 1)/2
            conv2_size_h = (conv1_size_h - patch_size + 1)/2
            layer_out_weights = tf.Variable(tf.truncated_normal([conv2_size_w * conv2_size_h * 12, num_labels], stddev=0.1), name="outy_w")
            layer_out_biases = tf.Variable(tf.zeros([num_labels], name="outy_b"))

            # dropout (keep probability)
            keep_prob = tf.placeholder(tf.float32) 													

            def model(data, _dropout=1.0): #uso ReLu invece della sigmoid visto che e' discreto il mio output?
                '''The model of the network.

                This function takes as input the batch, which is a matrix where each row is 
                an image, and it returns the output of the network. The output can be a single real value
                (if the input is a single image) or a vector of real values (if the input is a batch).
                This model uses the sigmoid function instead of ReLu because it produces a continous 
                output in the range [-1,+1].
                @param data it is an image or a batch (matrix) containing the images to process
                @param _dropout it is the dropout probability, leave to 1.0 if not used
                @return the output of the network
                '''
                X = tf.reshape(data, shape=[-1, image_size_w, image_size_h, num_channels])
                print("SHAPE X: " + str(X.get_shape()))  # Convolution Layer 1
                conv1 = tf.sigmoid(tf.nn.bias_add(tf.nn.conv2d(X, conv1_weights, strides=[1, 1, 1, 1], padding='VALID'), conv1_biases))
                print("SHAPE conv1: " + str(conv1.get_shape()))
                # Max Pooling (down-sampling)
                pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                print("SHAPE pool1: " + str(pool1.get_shape()))

                # Convolution Layer 2
                conv2 = tf.sigmoid(tf.nn.bias_add(tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='VALID'), conv2_biases))
                print("SHAPE conv2: " + str(conv2.get_shape())) 
                # Max Pooling (down-sampling)
                pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                print("SHAPE pool2: " + str(pool2.get_shape()))

                # Output layer
                out = tf.reshape(pool2, [-1, layer_out_weights.get_shape().as_list()[0]])
                out = tf.sigmoid(tf.matmul(out, layer_out_weights) + layer_out_biases)
                print("SHAPE out: " + str(out.get_shape()))

                return out


            #Here the model function created above is called
            #In Tensorflow this line is called during the session,
            #because when we want to estimate the loss, it is
            #necessary to have the model_ouput.
            model_output = model(tf_train_dataset, keep_prob)
            #The loss or cost is estimated based on the
            #difference between the output of the network and 
            #the target values. The loss requires the model output,
            #if the loss estimation is called from a Tensorflow session
            #the graph will automatically estimates the model_output.
            loss = tf.nn.l2_loss(model_output - tf_train_labels)
            #LA LOSS E' CALCOLATA SULLA DIFFERENZA DI VALORI DEL TRAINING SET, NON SUL TEST. Ovvio, devo migliorare il modello passo passo, leggere di piu' sulle conv!
            
            #Here I use L2 regularization. Which is a term		#la regolarizzazione aggiunge informazioni per prevenire overfitting (Wikipedia)
            #necessary to keep the weight values low.
            #Adding the regularization terms to the loss
            beta = 5e-4
            loss += (beta * tf.nn.l2_loss(conv1_weights)) 
            loss += (beta * tf.nn.l2_loss(conv2_weights)) 
            loss += (beta * tf.nn.l2_loss(layer_out_weights))

            #This summary allows printing the loss in Tensorboard          
            loss_summ = tf.summary.scalar("loss", loss)

            # Optimizer. Here the optimizer object is declared.
            #Uncomment/Comment to use a different one. The value of
            #the learning rate can be decreased during the simulation.
            global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
            learning_rate = tf.train.exponential_decay(0.0125, global_step, 70, 0.1, staircase=True)
            lrate_summ = tf.summary.scalar("learning rate", learning_rate) #save in a summary for Tensorboard
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

            #Predictions for the training, validation, and test data.
            train_prediction = model_output
            valid_prediction = model(tf_valid_dataset)													
            test_prediction = model(tf_test_dataset)

            #Declaring a saver object in order to save all the variables.
            saver = tf.train.Saver()

            #Number of epoch
            total_epochs = 100

            #Here the session starts. All the stuff declared above can be called (if they are Tensorflow objects)
            #because they are stored in memory and are available.
            with tf.Session(graph=graph) as session:
                #Summary definition for Tensorboard
                #I do this operations only once when I start the session.
                merged_summaries = tf.summary.merge_all()
                now = datetime.datetime.now()
                log_path = "/tmp/log/pitch_logs_p1_" + str(now.hour) + str(now.minute) + str(now.second)
                writer_summaries = tf.summary.FileWriter(log_path, session.graph_def)
                tf.initialize_all_variables().run()

                #Here the simulation starts.
                #You should notice that in the session.run call I ask for:
                #[optimizer, loss, train_prediction, merged_summaries]
                #All this object are estimated following the graph we created before.
                #e.g. if I ask for 'loss' then Tensorflow will estimate all the variables
                #necessary in order to get that value.
                for epoch in range(total_epochs):
                    batch = create_batch(train_dataset, train_labels_new)
                    batch_data = batch[0]
                    batch_labels = batch[1]
                    #If you do not want to use batch training, you can feed the network
                    #with the entire dataset. This can be expensive if the dataset is huge.
                    feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 1.0}
                    _, l, predictions, my_summary = session.run([optimizer, loss, model_output, merged_summaries],
                                                    feed_dict=feed_dict)
                    writer_summaries.add_summary(my_summary, epoch) #Here write the summary (current variables) for Tensorboard

                    if (epoch % 10 == 0):
                        print("")
                        print("Loss at epoch: ", epoch, " is " , l)
                        print("Global Step: " + str(global_step.eval()) + " of " + str(total_epochs))
                        print("Learning Rate: " + str(learning_rate.eval()))
                        print("Minibatch size: " + str(batch_labels.shape))
                        #accuracy(predictions, batch_labels, True)
                        #print("")                        
                        #print("Validation size: " + str(valid_labels_new.shape))
                        #print("Validation RMSE: %.2f%%" % accuracy(valid_prediction.eval(), valid_labels_new, True))
                        print("")
                feed_dict_test = {tf_test_dataset: test_dataset}
                output_predictions = session.run([model_output], feed_dict=feed_dict_test)
                #print output_predictions
                #TODO accuracy tra predictions e test_labels_new
                #At the end of the simulation I save the network.
                saver.save(session, "./sessions/tensorflow/cnn_arch1_pitch_p1" , global_step=epoch)  # save the session
                #print("# Test RMSE: %.2f" % accuracy(valid_prediction.eval(), valid_labels_new, True))
                print("# Test size: " + str(test_labels_new.shape))


if __name__ == "__main__":
    main("mouth")
