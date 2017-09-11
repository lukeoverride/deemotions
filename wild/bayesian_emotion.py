from bayesian_network import BayesianNetwork
import csv
import numpy as np

def main():
    my_bayes = BayesianNetwork()
    is_correct = my_bayes.initModel("./wild_GAF_labels_histogram_train_global.csv")
    print("Model correct: " + str(is_correct))

    predicted_positive = [0,0,0]
    with open("./wild_GAF_labels_val_positive.csv", 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            labels_list = my_bayes.returnLabelsListFromNewImage(row)
            if (len(labels_list) > 0):
                posterior = my_bayes.inference(labels_list)
                predicted_positive[np.argmax(posterior ['emotion_node'].values)] += 1
            print predicted_positive
            
    print "************** FINISHED POSITIVE ****************"
    print ""
    print ""
            
    predicted_negative = [0,0,0]
    with open("./wild_GAF_labels_val_negative.csv", 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            labels_list = my_bayes.returnLabelsListFromNewImage(row)
            if (len(labels_list) > 0):
                posterior = my_bayes.inference(labels_list)
                predicted_negative[np.argmax(posterior ['emotion_node'].values)] += 1
            print predicted_negative
            
    print "************** FINISHED NEGATIVE ****************"
    print ""
    print ""            
            
    predicted_neutral = [0,0,0]
    with open("./wild_GAF_labels_val_neutral.csv", 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            labels_list = my_bayes.returnLabelsListFromNewImage(row)
            if (len(labels_list) > 0):
                posterior = my_bayes.inference(labels_list)
                predicted_neutral[np.argmax(posterior ['emotion_node'].values)] += 1
            print predicted_neutral
            
    print "************** FINISHED NEUTRAL ****************"
    print ""
    print ""            


if __name__ == "__main__":
    main()
    
