import numpy as np
from cmath import log

class NeuralNetwork():
    """ 
    "lr": learning rate
    "dc": decrease constant
    "sizes": list of hidden layer sizes
    "L2": l2 regularization weight
    "L2": L1 regularization weight
    "seed": random seed
    "tanh": True to use tanh activation, False to use sigmoid
    "n_epoch": Number of training epochs.
    """    
    def __init__(self, lr=0.001, dc=1e-10, sizes=[200,100,50], L2=0.001, L1=0,
                 seed=1234, tanh=False, n_epochs=10):
        self.lr=lr
        self.dc=dc
        self.sizes=sizes
        self.L2=L2
        self.L1=L1
        self.seed=seed
        self.tanh=tanh
        self.n_epochs=n_epochs
        self.updates = 0  # counter for the number of updates

        # internal variable keeping track of the number of training iterations since initialization
        self.epoch = 0 
 
    def initialize(self,input_size,n_classes):
        """
        Allocate memory for the required matrices and initializes
        parameters.
        """
        # Start @Hugo Larochelle 
        self.n_classes = n_classes
        self.input_size = input_size
        n_hidden_layers = len(self.sizes)
        
        #############################################################################
        # Allocate space for the hidden and output layers, as well as the gradients #
        #############################################################################
        
        self.hs = []
        self.grad_hs = []
        for h in range(n_hidden_layers):         
            self.hs += [np.zeros((self.sizes[h],))]       # hidden layer
            self.grad_hs += [np.zeros((self.sizes[h],))]  # ... and gradient
        self.hs += [np.zeros((self.n_classes,))]       # output layer
        self.grad_hs += [np.zeros((self.n_classes,))]  # ... and gradient
        
        ##################################################################
        # Allocate space for the neural network parameters and gradients #
        ##################################################################
        
        self.weights = [np.zeros((self.input_size,self.sizes[0]))]       # input to 1st hidden layer weights
        self.grad_weights = [np.zeros((self.input_size,self.sizes[0]))]  # ... and gradient

        self.biases = [np.zeros((self.sizes[0]))]                        # 1st hidden layer biases
        self.grad_biases = [np.zeros((self.sizes[0]))]                   # ... and gradient

        for h in range(1,n_hidden_layers):
            self.weights += [np.zeros((self.sizes[h-1],self.sizes[h]))]        # h-1 to h hidden layer weights
            self.grad_weights += [np.zeros((self.sizes[h-1],self.sizes[h]))]   # ... and gradient

            self.biases += [np.zeros((self.sizes[h]))]                   # hth hidden layer biases
            self.grad_biases += [np.zeros((self.sizes[h]))]              # ... and gradient

        self.weights += [np.zeros((self.sizes[-1],self.n_classes))]      # last hidden to output layer weights
        self.grad_weights += [np.zeros((self.sizes[-1],self.n_classes))] # ... and gradient

        self.biases += [np.zeros((self.n_classes))]                   # output layer biases
        self.grad_biases += [np.zeros((self.n_classes))]              # ... and gradient
        # End @Hugo Larochelle   
         
        #########################
        # Initialize parameters #
        #########################

        self.rng = np.random.mtrand.RandomState(self.seed)   # create random number generator
        
        self.sample_ranges = np.ndarray(len(self.weights), float)    # allocate space for the ranges of weights
            
        self.sample_ranges[0] = self.compute_range(self.sizes[0], self.input_size)      # compute range for input layer        
        for i in range(1, len(self.sample_ranges)-1):                                   # compute each range for the hidden layers 
            self.sample_ranges[i] = self.compute_range(self.sizes[i], self.sizes[i-1])            
        self.sample_ranges[-1] = self.compute_range(self.n_classes, self.sizes[-1])     # compute range for output layer       

        for i in range(len(self.weights)):   # populate the weights with random values within a specified range             
            for x in np.nditer(self.weights[i], op_flags=['readwrite']): 
                x[...] = self.rng.uniform(-self.sample_ranges[i], self.sample_ranges[i]) 
            
        # Note: The biases are already initialized to 0. 
        
        self.n_updates = 0 # To keep track of the number of updates, to decrease the learning rate
        
    def compute_range(self, current_layer_size, previous_layer_size):
        """
        Compute the range of an uniform distribution for a given hidden layer.
        """
        return (6. ** 0.5) / ((current_layer_size + previous_layer_size) ** 0.5)

    def forget(self):
        """
        Resets the neural network to its original state.
        """
        self.initialize(self.input_size,self.targets)
        self.epoch = 0
        
    def train(self,trainset,labels,n_classes):
        """
        Trains the neural network until it reaches a total number of
        training epochs of "self.n_epochs".
        If "self.epoch == 0", first initialize the model.
        """

        if self.epoch == 0:
            input_size = len(trainset[0])
            self.initialize(input_size,n_classes)
            
        for i in range(self.epoch,self.n_epochs):
            for j in range (len(trainset)):
                self.fprop(trainset[j],labels[j])
                self.bprop(trainset[j],labels[j])
                self.update()
        self.epoch = self.n_epochs
        
    def fprop(self,input,target):
        """
        Forward propagation: 
        - fills the hidden layers and output layer in self.hs
        - returns the training loss, i.e. the 
          regularized negative log-likelihood for this (``input``,``target``) pair
        Argument ``input`` is a Numpy 1D array and ``target`` is an
        integer between 0 and nb. of classes - 1.
        """       
        #########################  
        # Compute h1:    
        #########################    
        transposed_weights = np.transpose(self.weights[0])
        # Preactivation:
        preactivation = np.dot(transposed_weights, input[:, np.newaxis]) + self.biases[0][:, np.newaxis]
        # Activation:
        self.hs[0] = self.compute_activation(preactivation[:,0]).real  
        
        ######################### 
        # Compute the other hs:
        ######################### 
        for i in range(1, len(self.weights) - 1):               # loop over remaining hidden layers
            transposed_weights = np.transpose(self.weights[i])  
            # Preactivation:
            preactivation = np.dot(transposed_weights, self.hs[i-1][:, np.newaxis]) + self.biases[i][:, np.newaxis]
            # Activation:
            self.hs[i] = self.compute_activation(preactivation[:,0]).real             

        ######################### 
        # Compute the output:   
        ######################### 
        transposed_weights = np.transpose(self.weights[-1]) 
        # Preactivation:
        preactivation = np.dot(transposed_weights, self.hs[-2][:, np.newaxis]) + self.biases[-1][:, np.newaxis]  
        # Activation/output:  
        self.hs[-1] = self.softmax(preactivation[:,0])
        
        return self.training_loss(self.hs[-1], target)
        
    def compute_activation(self, preactivation):
        """
        Compute the activation based on tanh or sigmoid.
        Takes the preactivation value.
        """        
        output = np.zeros(len(preactivation))
        if self.tanh == True:
            return np.tanh(preactivation)
        else:  
            output = self.sigmoid(preactivation)            
            return output
     
    def softmax(self, array):
        """
        Compute the softmax of an array:
        exp(input)/sum(exp(input)) 
        """
        array_exp = np.exp(array)
        return array_exp / (1.0 * np.sum(array_exp))
            
    def sigmoid(self, array):
        """
        Compute the sigmoid of an array:
        sigm(input) = 1/(1+exp(-input)) 
        """
        return 1.0 / (1.0 + np.exp(-1.0 * array))
        
    def training_loss(self,output,target):
        """
        Computation of the loss:
        - returns the regularized negative log-likelihood loss associated with the
          given output vector (probabilities of each class) and target (class ID)
        """
        omega = 0.0
        regularization1 = 0.0        
        regularization2 = 0.0
                
        if self.L2 != 0.0:
            for i in range(len(self.weights)):                      # loop over each layer
                omega += np.linalg.norm(self.weights[i]) ** 2       # sums the square of the frobenius norm
            regularization2 = self.L2 * omega
        if self.L1 != 0.0:              
            for i in range(len(self.weights)):    
                omega += np.sum(np.abs(self.weights[i]))                 # sums the absolute value of all weights
            regularization1 = self.L1 * omega
             
        return (-log(output[target])).real + regularization1 + regularization2         # compute loss

    def bprop(self,input,target):
        """
        Backpropagation:
        - fills in the hidden layers and output layer gradients in self.grad_hs
        - fills in the neural network gradients of weights and biases in self.grad_weights and self.grad_biases
        - returns nothing
        Argument ``input`` is a Numpy 1D array and ``target`` is an
        integer between 0 and nb. of classe - 1.
        """
        # Compute the output pre-activation gradient
        e = np.zeros((self.n_classes))
        e[target] = 1
        current_preactivation_gradient = -(e - self.hs[-1])
                       
        for k in range (len(self.weights)-1, 0, -1): # Loop over weights L to 2
            #Compute the gradients of the hidden layer parameters:
            self.grad_weights[k] = np.transpose(np.outer(current_preactivation_gradient, self.hs[k-1])) 
            self.grad_weights[k] += (self.L2 * 2 * self.weights[k]) + (self.L1 * np.sign(self.weights[k])) #regularization gradient
            self.grad_biases[k] = current_preactivation_gradient
        
            #Compute the activation gradient of layer k
            self.grad_hs[k-1] = np.dot(self.weights[k], current_preactivation_gradient)
            
            #Compute the preactivation gradient at the hidden layer below                        
            activation_derivatives = np.ones(len(self.hs[k-1]))
            if self.tanh == True:
                activation_derivatives -= self.hs[k-1] ** 2               
            else:  
                activation_derivatives *= self.hs[k-1]                
                activation_derivatives -= self.hs[k-1] ** 2             
            current_preactivation_gradient = self.grad_hs[k-1] * activation_derivatives  
            
        #compute gradients of the input layer parameters         
        self.grad_weights[0] = np.transpose(np.outer(current_preactivation_gradient, input))
        self.grad_weights[0] += (self.L2 * 2 * self.weights[0]) + (self.L1 * np.sign(self.weights[0])) #regularization gradient
        self.grad_biases[0] = current_preactivation_gradient       
 
    def update(self):
        """
        Stochastic gradient update:
        - performs a gradient step update of the neural network parameters self.weights and self.biases,
          using the gradients in self.grad_weights and self.grad_biases
        """              
        # Update the weight matrix:        
        for k in range(len(self.weights)): 
            self.weights[k] -= self.lr * self.grad_weights[k]  
          
        # Update the bias matrix:
        self.biases -= self.lr * np.array(self.grad_biases)        
                 
        # Update the learning rate.  
        self.updates += 1  
        self.lr /= 1 + self.dc * self.updates  
           
    def predict(self,dataset):
        """
        Computes and returns the outputs of the Learner for
        ``dataset``:
        - the outputs are a Numpy 2D array of size
          len(dataset).
        - the ith row of the array contains the predicted class for the ith example.
        """
        outputs = np.zeros(len(dataset))       
        
        for i in range (len(dataset)): # loop over each example in the dataset
            self.fprop(dataset[i], 0)  # classify the example
            # store the prediction in the output array:
            outputs[i] = np.argmax(self.hs[-1]) # return the index of the maximum element (predicted class) 
                
        return outputs
        
    def test(self,dataset,target_labels):
        """
        Computes and returns the outputs of the Learner, the errors of 
        those outputs for ``dataset`` as well as the accuracy:
        - the errors should be a Numpy 2D array of size
          len(dataset). The ith row of the array contains the 
          0/1 errors for the ith example
        """
        outputs = self.predict(dataset)
        errors = np.zeros(len(dataset))  
        error_sum = 0     
        
        for i in range (len(dataset)): # loop over each example in the dataset           
            if(outputs[i] == target_labels[i]): # if the prediction matches the target, no error
                errors[i] = 0
            else:                
                errors[i] = 1
                error_sum += 1
                
        accuracy = 1.0 - (1.0 * error_sum / len(dataset))  
                 
        return outputs, errors, accuracy
   

