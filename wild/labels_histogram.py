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

    This code generates an histogram from CSV file containing the image labels.

'''

import operator
import csv
import numpy as np
import sys

def main():
    labels_positive = collectAllLabels(sys.argv[1])
    labels_negative = collectAllLabels(sys.argv[2])
    labels_neutral = collectAllLabels(sys.argv[3])
    histogramPositive = createLabelsHistogram(labels_positive)
    histogramNegative = createLabelsHistogram(labels_negative)
    histogramNeutral = createLabelsHistogram(labels_neutral)
    global_list = labels_positive + labels_negative + labels_neutral
    global_unique_list = list(set(global_list))
    createLabelsGlobal(global_unique_list,histogramPositive,histogramNegative,histogramNeutral)
    
def collectAllLabels(csv_label_path):
    label_list = list()
    with open(csv_label_path, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            el_num = 0
            for element in row:
                if (el_num > 1):
                    splitters = row[el_num].split("'")
                    if (len(splitters) > 1):
                        label_list.append(splitters[1])
                el_num += 1
    return label_list


def createLabelsHistogram(label_list):    
    
    frequent_labels = {}
    for label in label_list:
        if label in frequent_labels:
            frequent_labels[label] += 1
        else:
            frequent_labels[label] = 1
            
    return frequent_labels
            
def writeSortedLabelsHistogram(frequent_labels):
    
    sorted_labels = sorted(frequent_labels.items(), key=operator.itemgetter(1), reverse=True)
    for s_label in sorted_labels:
        # Write the CSV file
        fd = open('wild_GAF_labels_histogram.csv', 'a')
        fd.write(str(s_label) + "\n")
        fd.close()
        
def createLabelsGlobal(unique_list,histogramPositive,histogramNegative,histogramNeutral):
    for label in unique_list:
        occurrences_pos = 0
        occurrences_neg = 0
        occurrences_neu = 0
        if (label in histogramPositive):
            occurrences_pos = histogramPositive[label]
        if (label in histogramNegative):
            occurrences_neg = histogramNegative[label]
        if (label in histogramNeutral):
            occurrences_neu = histogramNeutral[label]
        # Write the CSV file
        fd = open('wild_GAF_labels_train_global.csv', 'a')
        fd.write(label + "," + str(occurrences_pos) + "," + str(occurrences_neg) + "," + str(occurrences_neu) + "\n")
        fd.close()
    
        

            
if __name__ == '__main__':
    main()
    
    
