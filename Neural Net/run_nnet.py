import numpy as np
import copy
from nnet import NeuralNetwork
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
    Runs cross validation on the neural net.
    """   
    if nb_folds < 2:  # k must be at least 2
        nb_folds = 2
    
    output_log = open("nn_output_log.txt", "w")   
    
    # We need to choose these params.    
    learning_rates = np.array([0.001,0.0001])
    decrease_constants = np.array([0,0.00000000001])
    sizes = np.array([[25,25],[35,35],[35,25,25]])
    l2s = np.array([0,0.00001])
    l1s = np.array([0,0.00001])
    functions = np.array([True,False])
    valid_accuracies = np.zeros(nb_folds)
    train_accuracies = np.zeros(nb_folds)
    
    #Grid search over the parameters.
    #This will take a while!
    for i in range(len(learning_rates)):
        for j in range(len(decrease_constants)):
            for k in range(len(sizes)):
                for l in range(len(l2s)):
                    for m in range(len(l1s)):  
                        for n in range(len(functions)):
                            
                            folds = StratifiedKFold(labels, shuffle=False, n_folds=nb_folds) # Generate the fold indexes.
                            
                            a = 0;
                            #Builds the sets and cross-validate.
                            for train_index, test_index in folds:
                                
                                #Build the sets:
                                train_set, valid_set = samples[train_index], samples[test_index]
                                train_labels, valid_labels = labels[train_index], labels[test_index]
                                
                                #Run and train the neural net.
                                valid_accuracies[a], train_accuracies[a] = validate_net(train_set, train_labels, valid_set, valid_labels, \
                                        learning_rates[i], decrease_constants[j], sizes[k], l2s[l], l1s[m], functions[n])                              
                                a += 1
                                                         
                            valid_accuracy = np.sum(valid_accuracies) / nb_folds  #Average cross-valid accuracy. 
                            valid_stdev = np.std(valid_accuracies)
                            train_accuracy = np.sum(train_accuracies) / nb_folds  #Average cross-valid accuracy. 
                            train_stdev = np.std(train_accuracies) 
                            line = "Avg valid accuracy: " + str(valid_accuracy) + \
                                       " Stdev: " + str(valid_stdev) + \
                                       " Avg train accuracy: " + str(train_accuracy) + \
                                       " Stdev: " + str(train_stdev) + \
                                       " //Learning rate: " + str(learning_rates[i]) + \
                                       " //Decrease constant: " + str(decrease_constants[j]) + \
                                       " //Hidden layers: " + str(sizes[k]) + \
                                       " //L2: " + str(l2s[l]) + \
                                       " //L1: " + str(l1s[m]) + \
                                       " //Tanh(T) or Sigm(F): " + str(functions[n]) + "\n"
                            output_log.write(line)                             
                            print(line) #Print to keep track of what's happening!    
    output_log.close()
    
def validate_net(train_set, train_labels, valid_set, valid_labels, learning_rate, \
            decrease_constant, size, l2, l1, function) :
    """
    Train and validate the neural net with 
    a given set of parameters.
    Return the best accuracy.
    """    
    neuralNet = NeuralNetwork(lr=learning_rate, dc=decrease_constant, sizes=size, L2=l2, L1=l1,
                     seed=5678, tanh=function, n_epochs=10)
    
    n_classes = 10
    
    print "Training..."
    # Early stopping code @Hugo Larochelle (partially)
    best_val_error = np.inf # Begin with infinite error
    best_it = 0 # Iteration of the best neural net so far wrt valid error
    look_ahead = 5
    n_incr_error = 0
    for current_stage in range(1,500+1,1):
        
        #Stop training when NN has not improved for 5 turns.
        if not n_incr_error < look_ahead:
            break
        neuralNet.n_epochs = current_stage
        neuralNet.train(train_set, train_labels, n_classes)
        n_incr_error += 1
        
        outputs, errors, train_accuracy = neuralNet.test(train_set, train_labels)
        print 'Epoch',current_stage,'|',
        print 'Training accuracy: ' + '%.3f'%train_accuracy+',', ' |',
        outputs, errors, valid_accuracy = neuralNet.test(valid_set, valid_labels)
        print 'Validation accuracy: ' + '%.3f'%valid_accuracy
        
        # Check if this model is better than the previous:
        error = 1.0 - valid_accuracy
        if error < best_val_error:
            best_val_error = error
            best_train_accuracy = train_accuracy            
            n_incr_error = 0
    
    return 1 - best_val_error, best_train_accuracy

def test_net(train_set, train_labels, valid_set, valid_labels, test_set, learning_rate, \
            decrease_constant, size, l2, l1, function) :
    """
    Train and validate the neural net with 
    a given set of parameters.
    Returns the final test output.
    """    
    neuralNet = NeuralNetwork(lr=learning_rate, dc=decrease_constant, sizes=size, L2=l2, L1=l1,
                     seed=5678, tanh=function, n_epochs=10)
    
    n_classes = 10
    
    print "Training..."
    # Early stopping code
    best_val_error = np.inf # Begin with infinite error
    best_it = 0 # Iteration of the best neural net so far wrt valid error
    look_ahead = 5
    n_incr_error = 0
    for current_stage in range(1,500+1,1):
        
        #Stop training when NN has not improved for 5 turns.
        if not n_incr_error < look_ahead:
            break
        neuralNet.n_epochs = current_stage
        neuralNet.train(train_set, train_labels, n_classes)
        n_incr_error += 1
        
        outputs, errors, accuracy = neuralNet.test(train_set, train_labels)
        print 'Epoch',current_stage,'|',
        print 'Training accuracy: ' + '%.3f'%accuracy+',', ' |',
        outputs, errors, accuracy = neuralNet.test(valid_set, valid_labels)
        print 'Validation accuracy: ' + '%.3f'%accuracy
        
        # Check if this model is better than the previous:
        error = 1.0 - accuracy
        if error < best_val_error:
            best_val_error = error
            best_it = current_stage
            n_incr_error = 0
            best_model = copy.deepcopy(neuralNet) # Save the model.
    
    #TODO Clear train and valid set to free memory.
    #Load test set
    outputs = best_model.predict(test_set)
    #TODO save output to CSV.

###################
# Run cross-valid
###################

print "Loading datasets..."
samples = np.load("train_inputs.npy", 'r')
labels = np.load("train_outputs.npy", 'r')

#pca = PCA(n_components='mle')
#new_samples = pca.fit_transform(samples)

cross_validate(samples, labels, 4) #4-fold cross-valid.
#cross_validate(new_samples, labels, 4) #4-fold cross-valid.
