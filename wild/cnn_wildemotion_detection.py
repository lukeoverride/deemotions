#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import tensorflow as tf
import cv2
import os.path
import imp #to check for missing modules
import math
import datetime

#Check if dlib is installed
try:
    imp.find_module('dlib')
    IS_DLIB_INSTALLED = True
    import dlib
    print('[DEEPGAZE] face_detection_slindow.py: the dlib library is installed.')
except ImportError:
    IS_DLIB_INSTALLED = False
    print('[DEEPGAZE] face_detection_slindow.py: the dlib library is not installed.')

#Enbale if you need printing utilities
DEBUG = False


class CnnEmotionDetection:
    """ Emotion detection class which uses convolutional neural network

        It manages input (colour) picture larger than 64x64 pixels. The CNN are robust
        to variance in the input features and can handle occlusions and bad
        lighting conditions. The output values are in the ranges (degrees): 
    """

    def __init__(self, tf_session):
        """ Init the class

        @param tf_session An external tensorflow session
        """
        self._sess = tf_session


    def print_allocated_variables(self):
        """ Print all the Tensorflow allocated variables

        """
        all_vars = tf.all_variables()

        print("[DEEPGAZE] Printing all the Allocated Tensorflow Variables:")
        for k in all_vars:
            print(k.name)     

    def _allocate_variables(self):
        """ Allocate variables in memory (for internal use)
            
        The variables must be allocated in memory before loading
        the pretrained weights. In this phase empty placeholders
        are defined and later fill with the real values.
        """
        self._image_size = 64
        self._num_labels = 3
        self._num_channels = 3
        # Input data [batch_size, image_size, image_size, channels]
        self.tf_input_vector = tf.placeholder(tf.float32, shape=(self._image_size, self._image_size, self._num_channels))
        
        # Variables.
        #Conv layer
        #[patch_size, patch_size, num_channels, depth]
        self.conv1_weights = tf.Variable(tf.truncated_normal([11, 11, self._num_channels, 64], stddev=0.1),name="conv1_weights")
        self.conv1_biases = tf.Variable(tf.zeros([64]),name="conv1_biases")
        #Conv layer
        #[patch_size, patch_size, depth, depth]
        self.conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=0.1),name="conv2_weights")
        self.conv2_biases = tf.Variable(tf.random_normal(shape=[128]),name="conv2_biases")
        #Conv layer
        #[patch_size, patch_size, depth, depth]
        self.conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1),name="conv3_weights") #was[3, 3, 128, 256]
        self.conv3_biases = tf.Variable(tf.random_normal(shape=[256]),name="conv3_biases")

        #Dense layer
        #[ 5*5 * previous_layer_out , num_hidden] wd1
        #here 5*5 is the size of the image after pool reduction (divide by half 3 times)
        self.dense1_weights = tf.Variable(tf.truncated_normal([16 * 16 * 256, 512], stddev=0.1),name="dense1_weights") #was [5*5*256, 1024]
        self.dense1_biases = tf.Variable(tf.random_normal(shape=[512]),name="dense1_biases")
        #Dense layer
        #[ , num_hidden] wd2
        #self.hy_dense2_weights = tf.Variable(tf.truncated_normal([256, 256], stddev=0.01))
        #self.hy_dense2_biases = tf.Variable(tf.random_normal(shape=[256]))
        #Output layer
        self.out_weights = tf.Variable(tf.truncated_normal([512, self._num_labels], stddev=0.1),name="out_weights")
        self.out_biases = tf.Variable(tf.random_normal(shape=[self._num_labels]),name="out_biases")

        # dropout (keep probability)
        self.keep_prob = tf.placeholder(tf.float32)


        # Model.
        def model(data, _dropout=1.0):

            X = tf.reshape(data, shape=[-1, self._image_size, self._image_size, self._num_channels])
            if(DEBUG == True): print("SHAPE X: " + str(X.get_shape()))

            # Convolution Layer 1
            conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, self.conv1_weights, strides=[1, 1, 1, 1], padding='SAME'), self.conv1_biases))
            if(DEBUG == True): print("SHAPE conv1: " + str(conv1.get_shape()))
            # Max Pooling (down-sampling)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            if (DEBUG == True): print("SHAPE pool1: " + str(pool1.get_shape()))
            # Apply Normalization
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            norm1 = tf.nn.dropout(norm1, _dropout)
            # Second Convolution
            conv2 = tf.nn.relu(
                tf.nn.bias_add(tf.nn.conv2d(norm1, self.conv2_weights, strides=[1, 1, 1, 1], padding='SAME'), self.conv2_biases))
            if (DEBUG == True): print("SHAPE conv2: " + str(conv2.get_shape()))
            # Max Pooling (down-sampling)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            if (DEBUG == True): print("SHAPE pool2: " + str(pool2.get_shape()))
            # Apply Normalization
            norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            norm2 = tf.nn.dropout(norm2, _dropout)
            # Third convolution
            conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(norm2, self.conv3_weights, strides=[1, 1, 1, 1], padding='SAME'), self.conv3_biases))
            #if (DEBUG == True): print("SHAPE conv2: " + str(conv3.get_shape()))
            # Fully connected layer
            dense1 = tf.reshape(conv3, [-1, self.dense1_weights.get_shape().as_list()[0]])  # Reshape conv3
            if (DEBUG == True): print("SHAPE dense1: " + str(dense1.get_shape()))
            dense1 = tf.nn.relu(tf.matmul(dense1, self.dense1_weights) + self.dense1_biases)  # Relu
            dense1 = tf.nn.dropout(dense1, _dropout)
            # Output layer
            out = tf.matmul(dense1, self.out_weights) + self.out_biases
            if (DEBUG == True): print("SHAPE out: " + str(out.get_shape()))
            # Return the output with logits
            return out #add softmax if needed

        # Get the result from the model

        self.cnn_yaw_output = model(self.tf_input_vector)

    def load_variables(self, MetaFilePath, DirectoryPath):
        """ Load varibles from a tensorflow file

        It must be called after the variable allocation.
        This function take the variables stored in a local file
        and assign them to pre-allocated variables.      
        @param MetaFilePath path to a valid checkpoint metafile
        @param DirectoryPath path to checkpoint directory
        """

        #Allocate the variables in memory
        self._allocate_variables()

        saver = tf.train.import_meta_graph(MetaFilePath)

        tf.train.Saver({"conv1_48d_w": self.conv1_weights, "conv1_48d_b": self.conv1_biases,
                        "conv2_48d_w": self.conv2_weights, "conv2_48d_b": self.conv2_biases,
                        "conv3_48d_w": self.conv3_weights, "conv3_48d_b": self.conv3_biases,
                        "dense1_48d_w": self.dense1_weights, "dense1_48d_b": self.dense1_biases,
                        "out_48d_w": self.out_weights, "out_48d_b": self.out_biases
                        }).restore(self._sess, tf.train.latest_checkpoint(DirectoryPath))


    def getEmotionsPredictions(self, image):
        """ Return the emotions predictions according to the CNN estimate.

        @param image It is a colour image.
        """
        image = image - 127
        image = image / 255
        feed_dict = {self.tf_input_vector : image}
        emotion_predictions = self._sess.run([self.cnn_yaw_output], feed_dict=feed_dict)
        return emotion_predictions






