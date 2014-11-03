import cPickle as pickle
import numpy as np
from sklearn.cross_validation import StratifiedKFold

print "Loading datasets..."
train_samples = np.load("train_inputs.npy", 'r')
labels = np.load("train_outputs.npy", 'r')
test_set = np.load("test_inputs.npy", 'r')
fake_test_labels = np.zeros(len(test_set))

folds = StratifiedKFold(labels, shuffle=False, n_folds=8) # Generate the fold indexes.

#Build the valid and test sets.
for train_index, test_index in folds:

	#Build the sets:
	train_set, valid_set = train_samples[train_index], train_samples[test_index]
	train_labels, valid_labels = labels[train_index], labels[test_index]
                 
	break

dataset = []

print "Building tuples..."
#Create dataset tuples
train_tuple = (train_set, train_labels)
valid_tuple = (valid_set, valid_labels)
test_tuple = (test_set, fake_test_labels)

#Save tuples to list
dataset.append(train_tuple)
dataset.append(valid_tuple)
dataset.append(test_tuple)

print "Saving to Pickle"
#Save list to pickle file
pickle.dump( dataset, open( "dataset.pkl", "wb" ) )
