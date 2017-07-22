from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import sys
import csv
import numpy as np
# Defining the model structure. We can define the network by just passing a list of edges.



csv_path = sys.argv[1]
#Saving the TEST file names in a list
labels_list = list()
with open(csv_path, 'rb') as csvfile:
    reader = csv.reader(csvfile)
    first_line = 0 #To jump the header line
    for row in reader:
        if(first_line != 0): labels_list.append(row[0])
        else:
            total_positive = row[1]
            total_negative = row[2]
            total_neutral = row[3]
            first_line = 1
#Loading the labels
positive_vector = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(1), dtype=np.int32)
negative_vector = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(2),dtype=np.int32)
neutral_vector = np.genfromtxt(csv_path, delimiter=',', skip_header=1, usecols=(3),dtype=np.int32)

#per ogni label incontrata aggiungere nodo ('emotion', label)
nodes_list = list()
for label in labels_list:
    nodes_list.append(('emotion_node',label))
    
#model = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])
model = BayesianModel(nodes_list)

# Defining individual CPDs.

#definire tabella per emotion: positiva, negativa, neutra
cpd_emotion_node = TabularCPD(variable='emotion_node', variable_card=3, values=[[0.33,0.33,0.33]])
#cpd_d = TabularCPD(variable='D', variable_card=2, values=[[0.6, 0.4]])
#cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.7, 0.3]])

# The representation of CPD in pgmpy is a bit different than the CPD shown in the above picture. In pgmpy the colums
# are the evidences and rows are the states of the variable. So the grade CPD is represented like this:
#
#    +---------+---------+---------+---------+---------+
#    | diff    | intel_0 | intel_0 | intel_1 | intel_1 |
#    +---------+---------+---------+---------+---------+
#    | intel   | diff_0  | diff_1  | diff_0  | diff_1  |
#    +---------+---------+---------+---------+---------+
#    | grade_0 | 0.3     | 0.05    | 0.9     | 0.5     |
#    +---------+---------+---------+---------+---------+
#    | grade_1 | 0.4     | 0.25    | 0.08    | 0.3     |
#    +---------+---------+---------+---------+---------+
#    | grade_2 | 0.3     | 0.7     | 0.02    | 0.2     |
#    +---------+---------+---------+---------+---------+
'''
cpd_g = TabularCPD(variable='G', variable_card=3, 
                   values=[[0.3, 0.05, 0.9,  0.5],
                           [0.4, 0.25, 0.08, 0.3],
                           [0.3, 0.7,  0.02, 0.2]],
                  evidence=['I', 'D'],
                  evidence_card=[2, 2])
'''
cpds_list = list()

for i in range(len(labels_list)):
    p_true_positive = float(positive_vector[i]/total_positive)
    p_true_negative = float(negative_vector[i]/total_negative)
    p_true_neutral = float(neutral_vector[i]/total_neutral)
    p_false_positive = 1.0-p_true_positive
    p_false_negative = 1.0-p_true_negative
    p_false_neutral = 1.0-p_true_neutral 
                      
    cpd_label = TabularCPD(variable=label_list[i], variable_card=2,
                           values=[[p_false_positive,p_false_negative,p_false_neutral],
                                  [p_true_positive,p_true_negative,p_true_neutral]])
    cpds_list.append(cpd_label)
                               
                 
'''
cpd_l = TabularCPD(variable='L', variable_card=2, 
                   values=[[0.1, 0.4, 0.99],
                           [0.9, 0.6, 0.01]],
                   evidence=['G'],
                   evidence_card=[3])

cpd_s = TabularCPD(variable='S', variable_card=2,
                   values=[[0.95, 0.2],
                           [0.05, 0.8]],
                   evidence=['I'],
                   evidence_card=[2])
'''
# Associating the CPDs with the network
model.add_cpds([cpd_emotion_node]+cpds_list)

infer = VariableElimination(model)
print(infer.query(['G']) ['G'])

# check_model checks for the network structure and CPDs and verifies that the CPDs are correctly 
# defined and sum to 1.
model.check_model()
print model.get_cpds('G')
