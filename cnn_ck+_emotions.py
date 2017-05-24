'''
    Copyright (C) 2017 Luca Surace - University of Calabria, Plymouth University
                  2016 Massimiliano Patacchiola, Plymouth University
    
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

    This file contains the CNN structure to classify emotional pictures.
    It is training on the "CK+ dataset" and also computes
    loss and accuracy, which are written in a .txt file.
    
    The emotional images are loaded from a pickle file.
    The model take as input an image (or a batch) and return a vector
    representing the emotion target value of the face given as input.
    
    DATASET: It requires a pickle file which must be in the same folder of this script.
    This file is based on the "CK+ dataset" which is available for free.
    
    TENSORBOARD: this code works on my system and it shows correctly the value of the learning rate
    and the loss at each epoch. It saves a log file in the foder '/tmp/log/pitch_logs_p1_161944'
    You should notice that the name of the file is based on the current time. You should check this
    name in the folder and use it in Tensorboard.
    You can run tensorboard with this command: tensorboard --logdir="file:///tmp/log/pitch_logs_p1_161944"
    (where the name of the file can change based on the current time). In the code below I used 
    the tag 'Tensorboard' in the comments, every time I declared a Tensorboard-related variable.
    Important: to visualise the variable you have to wait a couple of minutes after the simulation started.
    Tensorboard is slow and it can take a while in order to load the first results.

'''


import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import datetime
import os,sys,glob

def accuracy(predictions, labels, verbose=False):
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
    if (verbose == True):
        print correct/predictions.shape[0]
    return correct/predictions.shape[0]

def create_batch(train_dataset, train_labels):
    '''
    This function creates a batch sized 25 from the input. In my example there are 7 emotions classes, so
    we randomly chose 3 emotional faces out of every class, and we assigned the remaining 4 faces (randomly selected) 
    to 4 different classes, randomly selected as well.
    :param train_dataset: image data
    :param train_labels:  label data
    :return: batch as tuple(batch_data, batch_labels)
    '''
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
    labels_new = np.zeros((labels.size, 7))
    for i in range(0, labels.size):
        for t in range(1, 15, 2):
            labels_new[i][t / 2] = labels[i][t]
    return labels_new

def minmax_normalization(data):
    return ((data - np.min(data))/np.max(data))

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.float32(np.interp(image.flatten(), bins[:-1], cdf))
    return image_equalized.reshape(image.shape)


def model(data, image_size_w, image_size_h, num_channels, conv1_weights, conv1_biases, conv2_weights, conv2_biases,
          dense1_weights, dense1_biases, layer_out_weights, layer_out_biases, _dropout=1.0):
    '''The model of the network.

    This function takes as input the batch, which is a matrix where each row is 
    an image, and it returns the output of the network. The output can be a single real value
    (if the input is a single image) or a vector of real values (if the input is a batch).
    @param data it is an image or a batch (matrix) containing the images to process
    @param _dropout it is the dropout probability, leave to 1.0 if not used
    @return the output of the network
    '''
    X = tf.reshape(data, shape=[-1, image_size_w, image_size_h, num_channels])
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, conv1_weights, strides=[1, 1, 1, 1], padding='VALID'), conv1_biases))

    # Max Pooling (down-sampling)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Convolution Layer 2
    conv2 = tf.nn.bias_add(tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='VALID'), conv2_biases)
    # Max Pooling (down-sampling)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer 1
    dense1 = tf.reshape(pool2, [-1, dense1_weights.get_shape().as_list()[0]])  # Reshape conv3
    dense1 = tf.matmul(dense1, dense1_weights) + dense1_biases

    # Output layer
    out = tf.matmul(dense1, layer_out_weights) + layer_out_biases

    # Output layer
    #out = tf.reshape(pool2, [-1, layer_out_weights.get_shape().as_list()[0]])
    #out = tf.matmul(out, layer_out_weights) + layer_out_biases
    #print("SHAPE out: " + str(out.get_shape()))

    return out


def main(block_name):

    for pickle_file in glob.glob(sys.argv[1]+block_name+"/*.pickle"):
        subject = pickle_file[len(pickle_file) - 12:len(pickle_file) - 7];
        #pickle_file = "./output/"+block_name+"/prima_p10.0_out.pickle"
        batch_size = 25
        patch_size = 5 # filter size
        myInitializer = None


        if (block_name == "face"):
            image_size_h = 72
            image_size_w = 52
        elif (block_name == "mouth"):
            image_size_h = 24
            image_size_w = 40
        elif (block_name == "eye"):
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

        train_dataset = image_histogram_equalization(train_dataset)
        valid_dataset = image_histogram_equalization(valid_dataset)
        test_dataset = image_histogram_equalization(test_dataset)

        train_dataset = minmax_normalization(train_dataset)
        valid_dataset = minmax_normalization(valid_dataset)
        test_dataset = minmax_normalization(test_dataset)


        #Printing the new shape of the datasets
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels_new.shape)
        print('Test set', test_dataset.shape, test_labels_new.shape)

        #Declaring the graph object necessary to build the model
        graph = tf.Graph()
        with graph.as_default():

            print("Init Tensorflow variables...")
            tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size_w, image_size_h, num_channels))
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
            tf_valid_dataset = tf.constant(valid_dataset)
            tf_test_dataset = tf.constant(test_dataset)

            # Conv layer
            # [patch_size, patch_size, num_channels, depth]
            #conv1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, 6], stddev=0.1), name="conv1y_w")
            conv1_weights = tf.get_variable(name="conv1y_w",shape=[patch_size,patch_size,num_channels,6],initializer=myInitializer)
            conv1_biases = tf.Variable(tf.zeros([6]), name="conv1y_b")
            # Conv layer
            # [patch_size, patch_size, depth, depth]
            conv2_weights = tf.get_variable(name="conv2y_w",shape=[patch_size, patch_size, 6, 12], initializer=myInitializer)
            conv2_biases = tf.Variable(tf.zeros([12]), name="conv2y_b")


            # Output layer
            conv1_size_w = (image_size_w - patch_size + 1)/2
            conv2_size_w = (conv1_size_w - patch_size + 1)/2
            conv1_size_h = (image_size_h - patch_size + 1)/2
            conv2_size_h = (conv1_size_h - patch_size + 1)/2
            #layer_out_weights = tf.Variable(tf.truncated_normal([conv2_size_w * conv2_size_h * 12, num_labels], stddev=0.1), name="outy_w")
            #layer_out_biases = tf.Variable(tf.zeros([num_labels], name="outy_b"))

            dense1_weights = tf.get_variable(name="dense1y_w",shape=[conv2_size_w * conv2_size_h * 12, 256], initializer=myInitializer)
            dense1_biases = tf.Variable(tf.zeros([256], name="dense1y_b"))

            # Output layer
            layer_out_weights = tf.get_variable(name="outy_w",shape=[256, num_labels], initializer=myInitializer)
            layer_out_biases = tf.Variable(tf.zeros(shape=[num_labels]), name="outy_b")

            # dropout (keep probability) - not used really up to now
            keep_prob = tf.placeholder(tf.float32) 													

            model_output = model(tf_train_dataset, image_size_w, image_size_h, num_channels, conv1_weights, conv1_biases, conv2_weights, conv2_biases,
                                 dense1_weights, dense1_biases, layer_out_weights, layer_out_biases, keep_prob)
            loss = tf.nn.l2_loss(model_output - tf_train_labels)

            #La regolarizzazione aggiunge informazioni per prevenire overfitting (Wikipedia)
            beta = 5e-4
            loss += (beta * tf.nn.l2_loss(conv1_weights)) 
            loss += (beta * tf.nn.l2_loss(conv2_weights))
            loss += (beta * tf.nn.l2_loss(dense1_weights))
            loss += (beta * tf.nn.l2_loss(layer_out_weights))

            loss_summ = tf.summary.scalar("loss", loss)

            global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
            learning_rate = tf.train.exponential_decay(0.00125, global_step, 300, 0.5, staircase=True)
            lrate_summ = tf.summary.scalar("learning rate", learning_rate) #save in a summary for Tensorboard
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

            train_prediction = model_output
            valid_prediction = model(tf_valid_dataset, image_size_w, image_size_h, num_channels, conv1_weights, conv1_biases, conv2_weights, conv2_biases,
                                     dense1_weights, dense1_biases, layer_out_weights, layer_out_biases)
            test_prediction = model(tf_test_dataset, image_size_w, image_size_h, num_channels, conv1_weights, conv1_biases, conv2_weights, conv2_biases,
                                    dense1_weights, dense1_biases, layer_out_weights, layer_out_biases)

            saver = tf.train.Saver()

            total_epochs = 500

            with tf.Session(graph=graph) as session:
                merged_summaries = tf.summary.merge_all()
                now = datetime.datetime.now()
                log_path = "./sessions/summary_log/summaries_logs_p"+subject+ str(now.hour) + str(now.minute) + str(now.second)
                writer_summaries = tf.summary.FileWriter(log_path, session.graph)
                tf.global_variables_initializer().run()

                epochs = np.ndarray(0,int)
                losses = np.ndarray(0,np.float32)
                accuracy_batch = np.ndarray(0,np.float32)
                accuracy_valid = np.ndarray(0,np.float32)

                for epoch in range(total_epochs):
                    batch = create_batch(train_dataset, train_labels_new)
                    batch_data = batch[0]
                    batch_labels = batch[1]
                    feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 1.0}
                    _, l, predictions, my_summary = session.run([optimizer, loss, model_output, merged_summaries],
                                                    feed_dict=feed_dict)
                    writer_summaries.add_summary(my_summary, epoch)

                    epochs = np.append(epochs, int(epoch+1))
                    losses = np.append(losses, l)
                    accuracy_batch = np.append(accuracy_batch, accuracy(predictions, batch_labels, False))
                    accuracy_valid = np.append(accuracy_valid, accuracy(valid_prediction.eval(), valid_labels_new, False))

                    if (epoch % 50 == 0):
                        print("")
                        print("Loss at epoch: ", epoch, " is " , l)
                        print("Global Step: " + str(global_step.eval()) + " of " + str(total_epochs))
                        print("Learning Rate: " + str(learning_rate.eval()))
                        print("Minibatch size: " + str(batch_labels.shape))
                        print("Validation size: " + str(valid_labels_new.shape))
                        accuracy(predictions, batch_labels, True)
                        print("")


                saver.save(session, "./sessions/tensorflow/cnn_arch1_pitch_p"+subject , global_step=epoch)  # save the session
                accuracy_test = accuracy(test_prediction.eval(),test_labels_new, True)
                output = np.column_stack((epochs.flatten(), losses.flatten(), accuracy_batch.flatten(), accuracy_valid.flatten()))
                np.savetxt("./sessions/epochs_log/subject_"+subject+".txt", output, header="epoch    loss    accuracy_batch    accuracy_valid", footer="accuracy_test:\n"+str(accuracy_test), delimiter='   ')
                print("# Test size: " + str(test_labels_new.shape))



if __name__ == "__main__":
    main("mouth")
