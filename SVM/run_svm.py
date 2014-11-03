from sklearn.svm import LinearSVC
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def get_confusion_matrix(Y1, Y2):    
    """
    Generate the confusion matrix between
    the expected outputs and the classifier outputs.    
    Y1: 1D array of expected outputs.
    Y2: 1D array of classifier outputs.
    """    
    # Compute confusion matrix
    matrix = confusion_matrix(Y1, Y2)
    
    print(matrix)
    
    # Show confusion matrix in a separate window.
    plt.matshow(matrix)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def cross_validate(samples, labels, nb_folds = 2):
    """
    Runs cross validation on the SVM.
    """   
    if nb_folds < 2:  # k must be at least 2
        nb_folds = 2
    
    output_log = open("svm_output_log.txt", "w")   
    
    # We need to choose these params.    
    c = np.array([0.8,0.9,1,1.1,0.5,2])
    losses = np.array(['l1','l2'])    
    penalties = np.array(['l1','l2'])
    duals = np.array([True,False])
    valid_accuracies = np.zeros(nb_folds)
    train_accuracies = np.zeros(nb_folds)
    
    #Grid search over the parameters.
    #This will take a while!
    for i in range(len(c)):
        for j in range(len(losses)):
            for k in range(len(penalties)):
                for l in range(len(duals)):
                                         
                    if((losses[j] == 'l1' and penalties[k] == 'l1') \
                    or (penalties[k] == 'l1' and duals[l] == True) \
                    or (losses[j]) == 'l1' and penalties[k] == 'l2' and duals[l] == False \
                    or (losses[j] == 'l1' and duals[l] == True)): #These combinations do not make sense.
                        continue

                    info = " //C penalty: " + str(c[i]) + \
                               " //Loss: " + str(losses[j]) + \
                               " //Penalty: " + str(penalties[k]) + \
                               " //Dual: " + str(duals[l]) +  "\n"
                    print info
                    folds = StratifiedKFold(labels, shuffle=False, n_folds=nb_folds) # Generate the fold indexes.

                    a = 0;
                    #Builds the sets and cross-validate.
                    for train_index, test_index in folds:
                        
                        #Build the sets:
                        train_set, valid_set = samples[train_index], samples[test_index]
                        train_labels, valid_labels = labels[train_index], labels[test_index]
                        
                        #Run and train the SVM.
                        svm = LinearSVC(penalty = penalties[k], loss = losses[j], dual = duals[l], C = c[i])
                        #Training SVM...
                        print("Training...")
                        svm.fit(train_set, train_labels)    
                        #Testing SVM...
                        print("Testing...")
                        valid_accuracies[a] = svm.score(valid_set, valid_labels)    
                        train_accuracies[a] = svm.score(train_set, train_labels)                             
                        a += 1
                                                 
                    valid_accuracy = np.sum(valid_accuracies) / nb_folds  #Average cross-valid accuracy. 
                    valid_stdev = np.std(valid_accuracies) 
                    train_accuracy = np.sum(train_accuracies) / nb_folds  #Average cross-valid accuracy. 
                    train_stdev = np.std(train_accuracies) 
                    line = "Avg valid Accuracy: " + str(valid_accuracy) + \
                               " Stdev: " + str(valid_stdev) + \
                               " //Avg train Accuracy: " + str(train_accuracy) + \
                               " Stdev: " + str(train_stdev) + \
                               " //C penalty: " + str(c[i]) + \
                               " //Loss: " + str(losses[j]) + \
                               " //Penalty: " + str(penalties[k]) + \
                               " //Dual: " + str(duals[l]) +  "\n"
                    output_log.write(line)                             
                    print(line) #Print to keep track of what's happening!    
    output_log.close()
    
###################
# Run cross-validation
###################

print "Loading datasets..."
samples = np.load("train_inputs.npy", 'r')
labels = np.load("train_outputs.npy", 'r')

#pca = PCA(n_components='mle')
#new_samples = pca.fit_transform(samples)

cross_validate(samples, labels, 4) #4-fold cross-valid.
#cross_validate(new_samples, labels, 4) #4-fold cross-valid.