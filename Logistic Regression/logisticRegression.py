# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

# import some data 
samples = np.load("train_inputs.npy", 'r')
labels = np.load("train_outputs.npy", 'r')
test = np.load("test_inputs.npy", 'r')

logreg = linear_model.LogisticRegression()
logreg = logreg.fit(samples,labels)
Z=logreg.predict(test)
print Z
logreg.score(samples,labels)
# we create an instance of Neighbours Classifier and fit the data.
#result=logreg.predict(test)
