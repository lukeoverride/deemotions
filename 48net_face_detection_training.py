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


from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import cv2  # to visualize a preview
import datetime
import os

def main():            
        # Load the standard file
        image_size = 48
        batch_size = 64
        patch_size = 5
        num_labels = 2
        num_channels = 3  # colour
        tot_epochs = 50 # Epochs
        # Change this path based on your datasets location
        pickle_file_positive = "./positive_dataset_48net_28986.pickle"
        pickle_file_negative = "./negative_dataset_48net_198081.pickle"

        with open(pickle_file_positive, 'rb') as f:
            save = pickle.load(f)
            train_dataset_positive = save['training_dataset']
            train_label_positive = save['training_label']
            del save  # hint to help gc free up memory
            # Here we take only part of the train and test set
            print('Training set', train_dataset_positive.shape, train_label_positive.shape)

        with open(pickle_file_negative, 'rb') as f:
            save = pickle.load(f)
            train_dataset_negative = save['training_dataset']
            train_label_negative = save['training_label']
            del save  # hint to help gc free up memory
            # Here we take only part of the train and test set
            print('Training set', train_dataset_negative.shape, train_label_negative.shape)

        # Creating the test set taking the first 100 images
        test_dataset = np.concatenate((train_dataset_positive[0:100, :, :], train_dataset_negative[0:100, :, :]), axis=0)
        test_label = np.concatenate((train_label_positive[0:100, :], train_label_negative[0:100, :]), axis=0)
        train_dataset_positive = train_dataset_positive[100:, :, :]
        train_dataset_negative = train_dataset_negative[100:, :, :]
        train_label_positive = train_label_positive[100:, :]
        train_label_negative = train_label_negative[100:, :]

        #Estimating the number of elements in both datasets
        total_positive = train_dataset_positive.shape[0]
        total_negative = train_dataset_negative.shape[0]

        # Normalisation
        #train_dataset -= 127
        #validation_dataset -= 127
        #test_dataset -= 127
        #train_dataset /= 255
        #validation_dataset /= 255
        #test_dataset /= 255

        graph = tf.Graph()
        with graph.as_default():
            tf_initializer = None #tf.random_normal_initializer()
            # Input data.
            tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
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
            # after 2 poolings the 48x48 image is reduced to size 12x12
            dense1_weights = tf.get_variable("dense1_48d_w", [12 * 12 * 256, 512], initializer=tf_initializer)
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

            # Optimizer.
            # learning_rate = 0.001 #it was: 0.001
            global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
            #learning_rate = tf.train.exponential_decay(0.000098, global_step, 15000, 0.1, staircase=True)
            #lrate_summ = tf.scalar_summary("learning rate", learning_rate)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
            #optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, global_step=global_step)
            #optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp').minimize(loss, global_step=global_step)
            #optimizer = tf.train.AdagradOptimizer(learning_rate=0.00625).minimize(loss, global_step=global_step)
            #optimizer = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.95).minimize(loss, global_step=global_step)
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001, rho=0.95, epsilon=1e-08, use_locking=False, name='Adadelta').minimize(loss, global_step=global_step)

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
                log_path = "./logs/log_48net_detection_" + str(now.hour) + str(now.minute) + str(now.second)
                writer_summaries = tf.summary.FileWriter(log_path, session.graph)
                tf.global_variables_initializer().run()
                tf.local_variables_initializer().run()
                # tf.initialize_all_variables().run()
                print('Initialized')

                for step in range(tot_epochs):
                    # Pick random images in euqal number from positive and negative dataset
                    quantity_positive = int(batch_size/2)
                    quantity_negative = batch_size - quantity_positive
                    indices_positive = np.random.randint(total_positive, size=quantity_positive)
                    indices_negative = np.random.randint(total_negative, size=quantity_negative)
                    batch_data = np.concatenate((np.take(train_dataset_positive, indices_positive, axis=0),
                                                 np.take(train_dataset_negative, indices_negative, axis=0)))
                    batch_labels = np.concatenate((np.take(train_label_positive, indices_positive, axis=0),
                                                   np.take(train_label_negative, indices_negative, axis=0)))


                    feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 1.0}
                    _, acc, l, predictions, my_summary = session.run([optimizer, accuracy, loss, train_prediction, merged_summaries],
                                                    feed_dict=feed_dict)
                    writer_summaries.add_summary(my_summary, step)
                    if (step % 50 == 0):
                        print("")
                        print("Loss at step: ", step, " is " , l)
                        print("Global Step: " + str(global_step.eval()) + " of " + str(tot_epochs))
                        #print("Learning Rate: " + str(learning_rate.eval()))
                        print("Minibatch size: " + str(batch_labels.shape))
                        print("Accuracy: " + str(acc))                
                        print("")                        

                # Save and test the network      
                checkpoint_path = "./checkpoints/48net_detection_" + str(now.hour) + str(now.minute) + str(now.second) 
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path) 
                saver.save(session, checkpoint_path + "/cnn_48net_detection" , global_step=step)  # save the session
                feed_dict = {tf_test_dataset: test_dataset, keep_prob: 1.0}
                test_acc, test_rec, test_prec, test_fp, test_fn = session.run([test_accuracy, test_recall, test_precision, test_false_positives, test_false_negatives], feed_dict=feed_dict)
                print("# Tot. images tested: " + str(test_dataset.shape[0])) 
                print("# Test accuracy: " + str(test_acc))
                print("# Test recall: " + str(test_rec))
                print("# Test precision: " + str(test_prec))
                print("# Test false positives: " + str(test_fp))
                print("# Test false negatives: " + str(test_fn))
                print("")

if __name__ == "__main__":
    main()

