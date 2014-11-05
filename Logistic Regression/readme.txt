%% Logistic Regression for Difficult Digits classification user guide
1. Running methods
Just run main.m. This function will call other functions to train a multi-class logistic regression classifier and predict the test dataset.

2. Function description
(1) main.m: main function that trains multi-class logistic regression classifier and predict test dataset
(2) oneVsAll.m: trains multiple logistic regression classifiers and returns all
the classifiers in a matrix
(3) lrGradientDescent.m: Compute cost and gradient for logistic regression
(4) confusion_matrix.m: draw confusion matrix to show the classification results
(5) predictOneVsAll.m: Predict the label with a trained one-vs-all classifier
(6) create_nth_kfold_crossvalidation.m: seperate dataset into training set and cross-validation set
(7) sigmoid.m: Compute sigmoid function
