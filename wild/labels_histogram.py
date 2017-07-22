import operator
import csv
import numpy as np
import sys

def main(csv_label_path):    
    label_list = list()
    #csv_path = "./resources/wild_GAF_labels.csv" 
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
    
    frequent_labels = {}
    for label in label_list:
        if label in frequent_labels:
            frequent_labels[label] += 1
        else:
            frequent_labels[label] = 1
    
    sorted_labels = sorted(frequent_labels.items(), key=operator.itemgetter(1), reverse=True)
    print sorted_labels
        
    for s_label in sorted_labels:
        # Write the CSV file
        fd = open('wild_GAF_labels_histogram_'+sys.argv[2]+'.csv', 'a')
        fd.write(str(s_label) + "\n")
        fd.close()
            
if __name__ == '__main__':
    main(sys.argv[1])
