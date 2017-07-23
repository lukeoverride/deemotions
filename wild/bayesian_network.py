from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import sys
import csv
import numpy as np


class BayesianNetwork:
    """ This class implement the bayesian network for emotion recognition (positive, negative, neutral)
        based on labels returned by a scene descriptor.
    """

    def __init__(self):
        """Init the object
        """
        pass

    def initModel(self, csv_path, verbose=True):
        """Init the model from a CSV file containing the label and the counting for: positive, negative, neutral

        @param: csv_path the path to the CSV file
        @return True if the model has been correctly initialised, otherwise False.
        """
        self.labels_list = list()
        with open(csv_path, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            is_first_line = True #To jump the header line
            for row in reader:
                if is_first_line:
                    total_positive = float(row[1])
                    total_negative = float(row[2])
                    total_neutral = float(row[3])
                    is_first_line = False
                else:
                    self.labels_list.append(row[0])

        if verbose: print self.labels_list
        # Loading the labels
        positive_vector = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(1), dtype=np.int32)
        negative_vector = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(2), dtype=np.int32)
        neutral_vector = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(3), dtype=np.int32)

        # Generate the connections between the father and the children
        cpd_emotion_node = TabularCPD(variable='emotion_node', variable_card=3, values=[[0.33,0.33,0.33]])

        nodes_list = list()
        for label in self.labels_list:
            nodes_list.append(('emotion_node',label))

        self.model = BayesianModel(nodes_list)
        self.model.add_cpds(cpd_emotion_node)

        cpds_list = list()
        for i in range(len(self.labels_list)):
            p_true_positive = float(positive_vector[i]/total_positive)
            p_true_negative = float(negative_vector[i]/total_negative)
            p_true_neutral = float(neutral_vector[i]/total_neutral)
            p_false_positive = 1.0 - p_true_positive
            p_false_negative = 1.0 - p_true_negative
            p_false_neutral = 1.0 - p_true_neutral 
                      
            cpd_label = TabularCPD(variable=self.labels_list[i], variable_card=2, values=[[p_false_positive, p_false_negative, p_false_neutral], 
                                                                                          [p_true_positive, p_true_negative, p_true_neutral]], 
                                                                                           evidence=['emotion_node'], evidence_card=[3])
            cpds_list.append(cpd_label)
            self.model.add_cpds(cpd_label)
            # print(cpd_label.get_values())
            # print("")

        # Associating the CPDs with the network
        #print("Total CPD: " + str(len(cpds_list)))
        #self.model.add_cpds(cpds_list)
        return self.model.check_model()  # return True if the model is correct


    def inference(self, labels_list):
        """It does the inference using the dictionary passed as input

        @param: labels_list a list of entries: ['carnival', 'festival', 'professional']
            where 1=True and 0=False
        """
        labels_dictionary = {}
        for label in labels_list:
            if label in self.labels_list:
                labels_dictionary[label] = 1
            else:
                labels_dictionary[label] = 0
        infer = VariableElimination(self.model)
        print(infer.query(['emotion_node'], labels_dictionary) ['emotion_node'])
        
        
    def returnLabelsListFromNewImage(self, csv_file_row):
        el_num = 0
        toReturn = list()
        for element in csv_file_row:
            if (el_num > 1):
                splitters = csv_file_row[el_num].split("'")
                if (len(splitters) > 1):
                    toReturn = splitters[1]
            el_num += 1
        return toReturn


