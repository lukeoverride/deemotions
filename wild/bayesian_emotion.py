from bayesian_network import BayesianNetwork
import csv

def main():
    my_bayes = BayesianNetwork()
    is_correct = my_bayes.initModel("./wild_GAF_labels_histogram_train_global.csv")
    print("Model correct: " + str(is_correct))
    
    with open("./wild_GAF_labels_val_positive.csv", 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            labels_list = my_bayes.returnLabelsListFromNewImage(row)
            print labels_list
            my_bayes.inference(labels_list)


if __name__ == "__main__":
    main()
    
