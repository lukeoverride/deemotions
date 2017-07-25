#!/usr/bin/env python

# The MIT License (MIT)
# Copyright (c) 2017 Massimiliano Patacchiola
# https://mpatacchiola.github.io
# https://mpatacchiola.github.io/blog/
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import cv2  # to visualize a preview
import csv
import datetime
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


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


def extractArraysRemoveBrackets(labels):
    labels_new = np.zeros((labels.size, 3))
    for i in range(0, labels.size):
        for t in range(1, 7, 2):
            labels_new[i][t / 2] = labels[i][t]
    return labels_new

def main():            
        # Load the standard file
        image_size = 64
        batch_size = 63
        patch_size = 5
        num_labels = 3
        num_channels = 3  # colour
        tot_epochs = 1500 # Epochs
        # Change this path based on your datasets location
        pickle_file_positive_train = "./pickles_GAF/GAF_train_positive.pickle"
        pickle_file_negative_train = "./pickles_GAF/GAF_train_negative.pickle"
        pickle_file_neutral_train = "./pickles_GAF/GAF_train_neutral.pickle"
        pickle_file_positive_val = "./pickles_GAF/GAF_val_positive.pickle"
        pickle_file_negative_val = "./pickles_GAF/GAF_val_negative.pickle"
        pickle_file_neutral_val = "./pickles_GAF/GAF_val_neutral.pickle"

        with open(pickle_file_positive_train, 'rb') as f:
            save = pickle.load(f)
            train_dataset_positive = save['training_dataset']
            train_label_positive = save['training_emotion_label']
            del save  # hint to help gc free up memory
            # Here we take only part of the train and test set
            print('Training set positive', train_dataset_positive.shape, train_label_positive.shape)

        with open(pickle_file_negative_train, 'rb') as f:
            save = pickle.load(f)
            train_dataset_negative = save['training_dataset']
            train_label_negative = save['training_emotion_label']
            del save  # hint to help gc free up memory
            # Here we take only part of the train and test set
            print('Training set negative', train_dataset_negative.shape, train_label_negative.shape)

        with open(pickle_file_neutral_train, 'rb') as f:
            save = pickle.load(f)
            train_dataset_neutral = save['training_dataset']
            train_label_neutral = save['training_emotion_label']
            del save  # hint to help gc free up memory
            # Here we take only part of the train and test set
            print('Training set neutral', train_dataset_neutral.shape, train_label_neutral.shape)

        with open(pickle_file_positive_val, 'rb') as f:
            save = pickle.load(f)
            valid_dataset_positive = save['training_dataset']
            valid_label_positive = save['training_emotion_label']
            del save  # hint to help gc free up memory
            # Here we take only part of the train and test set
            print('Validation set', valid_dataset_positive.shape, valid_label_positive.shape)

        with open(pickle_file_negative_val, 'rb') as f:
            save = pickle.load(f)
            valid_dataset_negative = save['training_dataset']
            valid_label_negative = save['training_emotion_label']
            del save  # hint to help gc free up memory
            # Here we take only part of the train and test set
            print('Validation set', valid_dataset_negative.shape, valid_label_negative.shape)

        with open(pickle_file_neutral_val, 'rb') as f:
            save = pickle.load(f)
            valid_dataset_neutral = save['training_dataset']
            valid_label_neutral = save['training_emotion_label']
            del save  # hint to help gc free up memory
            # Here we take only part of the train and test set
            print('Validation set', valid_dataset_neutral.shape, valid_label_neutral.shape)
            
        # Creating the test set taking the first 100 images
        # In our competition the validation set is considered as test set for computing classification accuracy
        train_dataset_positive = train_dataset_positive.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
        train_dataset_negative = train_dataset_negative.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
        train_dataset_neutral = train_dataset_neutral.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)

        total_validation_pos = valid_dataset_positive.shape[0]
        total_validation_neg = valid_dataset_negative.shape[0]
        total_validation_neu = valid_dataset_neutral.shape[0]

        indices_positive_test = np.random.randint(total_validation_pos, size=100)
        indices_negative_test = np.random.randint(total_validation_neg, size=100)
        indices_neutral_test = np.random.randint(total_validation_neu, size=100)

        indices_positive_valid = np.random.randint(total_validation_pos, size=50)
        indices_negative_valid = np.random.randint(total_validation_neg, size=50)
        indices_neutral_valid = np.random.randint(total_validation_neu, size=50)

        #fare il reshape di tutti i valid e poi concatenarli per creare il test
        #test_dataset = np.concatenate((valid_dataset_positive[0:200], valid_dataset_negative[0:200], valid_dataset_neutral[0:200]), axis=0)
        #test_label = np.concatenate((valid_label_positive[0:200], valid_label_negative[0:200], valid_label_neutral[0:200]), axis=0)

        #valid_dataset = np.concatenate((valid_dataset_positive[0:200], valid_dataset_negative[0:200], valid_dataset_neutral[0:200]), axis=0)
        #valid_label = np.concatenate((valid_label_positive[0:200], valid_label_negative[0:200], valid_label_neutral[0:200]), axis=0)

        test_dataset = np.concatenate((np.take(valid_dataset_positive, indices_positive_test, axis=0),
                                     np.take(valid_dataset_negative, indices_negative_test, axis=0),
                                     np.take(valid_dataset_neutral, indices_neutral_test, axis=0)))
        test_label = np.concatenate((np.take(valid_label_positive, indices_positive_test, axis=0),
                                       np.take(valid_label_negative, indices_negative_test, axis=0),
                                       np.take(valid_label_neutral, indices_neutral_test, axis=0)))
        valid_dataset = np.concatenate((np.take(valid_dataset_positive, indices_positive_valid, axis=0),
                                       np.take(valid_dataset_negative, indices_negative_valid, axis=0),
                                       np.take(valid_dataset_neutral, indices_neutral_valid, axis=0)))
        valid_label = np.concatenate((np.take(valid_label_positive, indices_positive_valid, axis=0),
                                     np.take(valid_label_negative, indices_negative_valid, axis=0),
                                     np.take(valid_label_neutral, indices_neutral_valid, axis=0)))


        #reshape
        train_label_positive = train_label_positive.reshape((-1)).astype(np.ndarray)
        train_label_negative = train_label_negative.reshape((-1)).astype(np.ndarray)
        train_label_neutral = train_label_neutral.reshape((-1)).astype(np.ndarray)
        test_dataset = test_dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
        test_label = test_label.reshape((-1)).astype(np.ndarray)
        valid_dataset = valid_dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
        valid_label = valid_label.reshape((-1)).astype(np.ndarray)

        # create the arrays from string, removing brackets as well
        train_label_positive = extractArraysRemoveBrackets(train_label_positive)
        train_label_negative = extractArraysRemoveBrackets(train_label_negative)
        train_label_neutral = extractArraysRemoveBrackets(train_label_neutral)

        test_label = extractArraysRemoveBrackets(test_label)
        valid_label = extractArraysRemoveBrackets(valid_label)


        #Estimating the number of elements in both datasets
        total_positive = train_dataset_positive.shape[0]
        total_negative = train_dataset_negative.shape[0]
        total_neutral = train_dataset_neutral.shape[0]

        # Normalisation
        train_dataset_positive -= 127
        train_dataset_negative -= 127
        train_dataset_neutral -= 127
        valid_dataset -= 127
        test_dataset -= 127
        train_dataset_positive /= 255
        train_dataset_negative /= 255
        train_dataset_neutral /= 255
        valid_dataset /= 255
        test_dataset /= 255

        graph = tf.Graph()
        with graph.as_default():
            tf_initializer = None #tf.random_normal_initializer()
            # Input data.
            tf_input_vector = tf.placeholder(tf.float32, shape=(image_size, image_size, num_channels))
            tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
            tf_valid_dataset = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
            tf_valid_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
            tf_test_dataset = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
            # Variables.
            # Conv layer
            # [patch_size, patch_size, num_channels, depth]
            conv1_weights = tf.get_variable("conv1_48d_w", [11, 11, num_channels, 64], initializer=tf_initializer)
            conv1_biases = tf.Variable(tf.zeros([64]), name="conv1_48d_b")
            # Conv layer
            # [patch_size, patch_size, num_channels, depth]
            conv2_weights = tf.get_variable("conv2_48d_w", [5, 5, 64, 128], initializer=tf_initializer)
            conv2_biases = tf.Variable(tf.zeros([128]), name="conv2_48d_b")
            # Conv layer
            # [patch_size, patch_size, num_channels, depth]
            conv3_weights = tf.get_variable("conv3_48d_w", [3, 3, 128, 256], initializer=tf_initializer)
            conv3_biases = tf.Variable(tf.zeros([256]), name="conv3_48d_b")
            # Dense layer
            # [ 5*5 * previous_layer_out , num_hidden] wd1
            # after 2 poolings the 64x64 image is reduced to size 16x16
            dense1_weights = tf.get_variable("dense1_48d_w", [16 * 16 * 256, 512], initializer=tf_initializer)
            dense1_biases = tf.Variable(tf.random_normal(shape=[512]), name="dense1_48d_b")
            # Output layer
            layer_out_weights = tf.get_variable("out_48d_w", [512, num_labels], initializer=tf_initializer)
            layer_out_biases = tf.Variable(tf.random_normal(shape=[num_labels]), name="out_48d_b")
            # dropout (keep probability)
            keep_prob = tf.placeholder(tf.float32)

            # Model.
            def model(data, _dropout=1.0):
                X = tf.reshape(data, shape=[-1, image_size, image_size, num_channels])
                print("SHAPE X: " + str(X.get_shape()))  # Convolution Layer 1
                conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, conv1_weights, strides=[1, 1, 1, 1], padding='SAME'), conv1_biases))
                print("SHAPE conv1: " + str(conv1.get_shape()))
                # Max Pooling (down-sampling)
                pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
                print("SHAPE pool1: " + str(pool1.get_shape()))
                # Apply Normalization
                norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
                # Apply Dropout
                norm1 = tf.nn.dropout(norm1, _dropout)
                # Second Convolution
                conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(norm1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME'), conv2_biases))
                print("SHAPE conv2: " + str(conv2.get_shape()))
                # Max Pooling (down-sampling)
                pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
                print("SHAPE pool2: " + str(pool2.get_shape()))
                # Apply Normalization
                norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
                # Apply Dropout
                norm2 = tf.nn.dropout(norm2, _dropout)
                # Third convolution
                conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(norm2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME'),conv3_biases))
                print("SHAPE conv2: " + str(conv3.get_shape()))
                # Fully connected layer
                dense1 = tf.reshape(conv3, [-1, dense1_weights.get_shape().as_list()[0]])  # Reshape conv3
                print("SHAPE dense1: " + str(dense1.get_shape()))
                dense1 = tf.nn.relu(tf.matmul(dense1, dense1_weights) + dense1_biases)  # Relu
                dense1 = tf.nn.dropout(dense1, _dropout)
                # Output layer
                out = tf.matmul(dense1, layer_out_weights) + layer_out_biases
                print("SHAPE out: " + str(out.get_shape()))
                # Return the output with logits
                return out

            # Training computation.
            logits = model(tf_train_dataset, keep_prob)
            loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))      
            #- Adding the regularization terms to the loss
            #beta =  5e-4 #it was: 5e-4 = 0.0005
            #loss += (beta * tf.nn.l2_loss(conv1_weights)) 
            #loss += (beta * tf.nn.l2_loss(dense1_weights))
            #loss += (beta * tf.nn.l2_loss(layer_out_weights))   
            loss_summ = tf.summary.scalar("loss", loss)

            # Find the batch accuracy and save it in summary
            accuracy = tf.equal(tf.argmax(tf_train_labels, 1), tf.argmax(logits, 1))
            accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
            accuracy_summary = tf.summary.scalar("accuracy", accuracy)

            valid_prediction = model(tf_valid_dataset)
            accuracy_valid = tf.equal(tf.argmax(tf_valid_labels, 1), tf.argmax(valid_prediction, 1))
            accuracy_valid = tf.reduce_mean(tf.cast(accuracy_valid, tf.float32))
            accuracy_valid_summary = tf.summary.scalar("accuracy_valid", accuracy_valid)
            
            cnn_emot_output = model(tf_input_vector)

            # Optimizer.
            learning_rate = 0.0001
            global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
            #learning_rate = tf.train.exponential_decay(0.0001, global_step, 700, 0.1, staircase=True)
            #lrate_summ = tf.scalar_summary("learning rate", learning_rate)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
            #optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, global_step=global_step)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp').minimize(loss, global_step=global_step)
            #optimizer = tf.train.AdagradOptimizer(learning_rate=0.00625).minimize(loss, global_step=global_step)
            #optimizer = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.95).minimize(loss, global_step=global_step)
            #optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001, rho=0.95, epsilon=1e-08, use_locking=False, name='Adadelta').minimize(loss, global_step=global_step)

            # Predictions for the training, validation, and test data.
            train_prediction = logits
            #valid_prediction = model(tf_validation_dataset)
            # Call test_prediction and pass the test inputs to have test accuracy
            test_prediction = model(tf_test_dataset)
            _, test_accuracy = tf.metrics.accuracy(labels=tf.argmax(test_label, 1), predictions=tf.argmax(test_prediction, 1))
            _, test_recall = tf.metrics.recall(labels=tf.argmax(test_label, 1), predictions=tf.argmax(test_prediction, 1))
            _, test_precision = tf.metrics.precision(labels=tf.argmax(test_label, 1), predictions=tf.argmax(test_prediction, 1))
            _, test_false_positives = tf.metrics.false_positives(labels=tf.argmax(test_label, 1), predictions=tf.argmax(test_prediction, 1))
            _, test_false_negatives = tf.metrics.false_negatives(labels=tf.argmax(test_label, 1), predictions=tf.argmax(test_prediction, 1))

            # Save all the variables
            saver = tf.train.Saver({'conv1_48d_w': conv1_weights, 'conv2_48d_w': conv2_weights, 'conv3_48d_w': conv3_weights,
                                     'conv1_48d_b': conv1_biases, 'conv2_48d_b': conv2_biases, 'conv3_48d_b': conv3_biases,
                                    'dense1_48d_w': dense1_weights, 'dense1_48d_b': dense1_biases, 'out_48d_w':layer_out_weights, 'out_48d_b': layer_out_biases})
            with tf.Session(graph=graph) as session:
                # Summary definition
                merged_summaries = tf.summary.merge_all()
                now = datetime.datetime.now()
                log_path = "./logs/log_emotiW_detection_" + str(now.hour) + str(now.minute) + str(now.second)
                writer_summaries = tf.summary.FileWriter(log_path, session.graph)
                tf.global_variables_initializer().run()
                tf.local_variables_initializer().run()
                # tf.initialize_all_variables().run()
                print('Initialized')

                for step in range(tot_epochs):
                    # Pick random images in equal number from positive, negative and neutral dataset
                    quantity_positive = int(batch_size/3)
                    quantity_negative = quantity_positive
                    quantity_neutral = quantity_negative
                    indices_positive = np.random.randint(total_positive, size=quantity_positive)
                    indices_negative = np.random.randint(total_negative, size=quantity_negative)
                    indices_neutral = np.random.randint(total_neutral, size=quantity_neutral)
                    batch_data = np.concatenate((np.take(train_dataset_positive, indices_positive, axis=0),
                                                 np.take(train_dataset_negative, indices_negative, axis=0),
                                                 np.take(train_dataset_neutral, indices_neutral, axis=0)))
                    batch_labels = np.concatenate((np.take(train_label_positive, indices_positive, axis=0),
                                                   np.take(train_label_negative, indices_negative, axis=0),
                                                   np.take(train_label_neutral, indices_neutral, axis=0)))


                    feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, tf_valid_dataset: valid_dataset, tf_valid_labels: valid_label, keep_prob: 0.5}
                    _, acc, l, predictions, acc_valid, pred_valid, my_summary = session.run([optimizer, accuracy, loss, train_prediction, accuracy_valid, valid_prediction, merged_summaries],
                                                    feed_dict=feed_dict)
                    writer_summaries.add_summary(my_summary, step)
                    if (step % 25 == 0):
                        print("")
                        print("Loss at step: ", step, " is " , l)
                        print("Global Step: " + str(global_step.eval()) + " of " + str(tot_epochs))
                        #print("Learning Rate: " + str(learning_rate.eval()))
                        print("Minibatch size: " + str(batch_labels.shape))
                        print("Accuracy_train: " + str(acc))
                        print ("Accuracy_valid: " +str(acc_valid))
                        print ("predictions positive: "+str(predictions[0]))
                        print ("predictions positive: "+str(predictions[1]))
                        print ("predictions positive: "+str(predictions[4]))
                        print ("predictions positive: "+str(predictions[10]))
                        print ("predictions negative: "+str(predictions[30]))
                        print ("predictions negative: "+str(predictions[34]))
                        print ("predictions negative: "+str(predictions[40]))
                        print ("predictions negative: "+str(predictions[42]))
                        print ("predictions neutral: "+str(predictions[60]))
                        print ("predictions neutral: "+str(predictions[52]))
                        print ("predictions neutral: "+str(predictions[50]))
                        print ("predictions neutral: "+str(predictions[47]))
                        print("")

                # Save and test the network      
                checkpoint_path = "./checkpoints/emotiW_detection_" + str(now.hour) + str(now.minute) + str(now.second)
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                saver.save(session, checkpoint_path + "/cnn_emotiW_detection" , global_step=step)  # save the session
                feed_dict = {tf_test_dataset: test_dataset, keep_prob: 1.0}
                test_pred, test_acc, test_rec, test_prec, test_fp, test_fn = session.run([test_prediction, test_accuracy, test_recall, test_precision, test_false_positives, test_false_negatives], feed_dict=feed_dict)
                
                #Saving the TEST file names in a list
                results = {}
                with open("./wild_GAF_faces_val_positive.csv", 'rb') as csvfile:
                    reader = csv.reader(csvfile)
                    first_line = 0 #To jump the header line
                    for row in reader:
                        if(first_line != 0):
                            tokens = row[0].split("face")
                            completeFileName = tokens[0].split("/")
                            fileName = completeFileName[len(completeFileName)-1]
                            image = cv2.imread(row[0]).astype(np.float32)
                            image -= 127.0
                            image /= 255.0
                            feed_dict = {tf_input_vector: image}
                            current_pred = session.run([cnn_emot_output], feed_dict=feed_dict)
                            print current_pred
                            if (fileName in results):
                                results[fileName] = np.append(results[fileName],current_pred,axis=0)
                            else:
                                results[fileName] = current_pred
                        first_line = 1
                        
                images_positive = len(results)

                with open("./wild_GAF_faces_val_negative.csv", 'rb') as csvfile:
                    reader = csv.reader(csvfile)
                    first_line = 0 #To jump the header line
                    for row in reader:
                        if(first_line != 0):
                            tokens = row[0].split("face")
                            completeFileName = tokens[0].split("/")
                            fileName = completeFileName[len(completeFileName)-1]
                            image = cv2.imread(row[0]).astype(np.float32)
                            image -= 127.0
                            image /= 255.0
                            feed_dict = {tf_input_vector: image}
                            current_pred = session.run([cnn_emot_output], feed_dict=feed_dict)
                            print current_pred
                            if (fileName in results):
                                results[fileName] = np.append(results[fileName],current_pred,axis=0)
                            else:
                                results[fileName] = current_pred
                        first_line = 1
                        
                images_negative = len(results)-images_positive

                with open("./wild_GAF_faces_val_neutral.csv", 'rb') as csvfile:
                    reader = csv.reader(csvfile)
                    first_line = 0 #To jump the header line
                    for row in reader:
                        if(first_line != 0):
                            tokens = row[0].split("face")
                            completeFileName = tokens[0].split("/")
                            fileName = completeFileName[len(completeFileName)-1]
                            image = cv2.imread(row[0]).astype(np.float32)
                            image -= 127.0
                            image /= 255.0
                            feed_dict = {tf_input_vector: image}
                            current_pred = session.run([cnn_emot_output], feed_dict=feed_dict)
                            print current_pred
                            if (fileName in results):
                                results[fileName] = np.append(results[fileName],current_pred,axis=0)
                            else:
                                results[fileName] = current_pred
                        first_line = 1
                        
                images_neutral = len(results)-(images_positive+images_negative)
                print images_positive
                print images_negative
                print images_neutral

                for key in results:
                    results[key] = np.mean(results[key], axis=0)
                    # Write the CSV file
                    fd = open('wild_GAF_cnn_predictions.csv', 'a')
                    fd.write(key + "," + str(results[key]) + "\n")
                    fd.close()

                global_test_label = np.zeros((len(results),num_labels))
                global_test_label[0:images_positive,0]=1
                global_test_label[images_positive:images_positive+images_negative, 2] = 1
                global_test_label[images_positive+images_negative:images_positive+images_negative+images_neutral, 1] = 1

                results_array = np.asarray(results.values())
                results_array = results_array.reshape((len(results), 3))
                compute_accuracy(results_array[0:images_positive],global_test_label[0:images_positive],True)
                compute_accuracy(results_array[images_positive:images_positive+images_negative],global_test_label[images_positive:images_positive+images_negative],True)
                compute_accuracy(results_array[images_positive+images_negative:images_positive+images_negative+images_neutral],global_test_label[images_positive+images_negative:images_positive+images_negative+images_neutral],True)
                
                
                print("# Tot. images tested: " + str(test_dataset.shape[0]))
                print("# Test accuracy: " + str(test_acc))
                print("# Test recall: " + str(test_rec))
                print("# Test precision: " + str(test_prec))
                print("# Test false positives: " + str(test_fp))
                print("# Test false negatives: " + str(test_fn))
                print("")


if __name__ == "__main__":
    main()

