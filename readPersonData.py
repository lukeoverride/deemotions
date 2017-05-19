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
    
    This code reads the files written by cnn_ck+_emotions code and plots the data in a 2D graph (epochs, data)
    The figure 1 shows the loss and the various accuracy values with respect to epochs. Each loss/accuracy value is computed as
    the average of all subjects, for a given epoch.
    The figure 2 shows the accuracy values on test set (single value) for each subject.
    @:argument sys.argv[1] is the path where the data files are located
'''


import numpy as np
import os,glob,sys
import subprocess
import matplotlib.pyplot as plt

os.chdir(sys.argv[1])
epochs = 500
npeople = len(glob.glob("*.txt"))
overall_loss = np.ndarray((0,epochs),np.float32)
overall_accuracy_batch = np.ndarray((0,epochs),np.float32)
overall_accuracy_valid = np.ndarray((0,epochs),np.float32)
overall_accuracy_test = np.ndarray(0,np.float32)

#row=person, column=epoch
accuracy_test_sum = 0

for file in glob.glob("*.txt"):
    pipe = subprocess.Popen('tail '+file+' --lines=1', shell=True, stdout=subprocess.PIPE).stdout
    lastLine = pipe.read()
    tokens = lastLine.split()
    accuracy_test = tokens[1]
    overall_accuracy_test = np.append(overall_accuracy_test,np.float32(accuracy_test))
    epochs, losses, accuracy_batch, accuracy_valid = np.loadtxt(file, unpack=True,skiprows=1,delimiter='   ')
    overall_loss = np.append(overall_loss,[losses],axis=0)
    overall_accuracy_batch = np.append(overall_accuracy_batch,[accuracy_batch],axis=0)
    overall_accuracy_valid = np.append(overall_accuracy_valid,[accuracy_valid],axis=0)

average_loss_per_epoch = np.average(overall_loss,axis=0)
average_accuracyB_per_epoch = np.average(overall_accuracy_batch,axis=0)
average_accuracyV_per_epoch = np.average(overall_accuracy_valid,axis=0)
people = np.arange(0,npeople,1)
print np.average(overall_accuracy_test)

plt.subplot(3,1,1)
plt.plot(epochs,average_loss_per_epoch,'r-')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.subplot(3,1,2)
plt.plot(epochs,average_accuracyB_per_epoch,'g-')
plt.xlabel("epochs")
plt.ylabel("accuracy_batch")
plt.subplot(3,1,3)
plt.plot(epochs,average_accuracyV_per_epoch,'b-')
plt.xlabel("epochs")
plt.ylabel("accuracy_validation")
plt.figure(2)
plt.subplot(1,1,1)
plt.plot(people,overall_accuracy_test,'c-')
plt.xlabel("subject")
plt.ylabel("accuracy_test")
plt.show()
