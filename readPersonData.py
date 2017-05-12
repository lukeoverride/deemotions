'''
    Luca Surace, University of Calabria - Plymouth University
    
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
epochs = 100
npeople = len(glob.glob("*.txt"))
overall_loss = np.ndarray((0,epochs),np.float32)
overall_accuracy_batch = np.ndarray((0,epochs),np.float32)
overall_accuracy_valid = np.ndarray((0,epochs),np.float32)
overall_accuracy_test = np.ndarray(0,np.float32)

#row=person, column=epoch

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
