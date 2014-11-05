%% Initialization
clear ; close all; clc

%% Setup parameters
num_labels = 10;          % 10 labels, from 0 to 9   

fprintf('\nTraining One-vs-All Logistic Regression for difficult digits classification...\n')

%% ------Part 1: loading data ----------
%note that we have read data from the original csv file and write them 
%into the following mat file
%for details, see csv2mat.m and readpca.py in readData folder
%load PCA dataset
%load './trainXmle.mat' %trainXmle
%load './testXmle.mat' %testXmle

%load raw dataset
load '../trainXdata.mat' %trainData
load '../testXdata.mat' %testXData
load '../trainYdata.mat' %y

%% ------Part 2: cross validation ----------
% 4-fold crossvalidation
num_iters = 300;  %iteration times for gradient descent
foldNums = 4;
lambda = 0;  %not use regularization term for Logistic Regression
alpha_pool = [0.005 0.01 0.05 0.1 0.5 1];
alphaNums = length(alpha_pool);

for kk = 1:foldNums
    %saperate the dataset into training set and cross-valid set
    [Xtrain, Ytrain, Xval, Yval] = create_nth_kfold_crossvalidation(trainXmle, y, foldNums, kk);
    for i = 1:alphaNums
  	    %pick up one alpha value
          alpha = alpha_pool(i);  
	    %gradient descent for one-vs-all logistic regression        
          [all_theta] = oneVsAll(Xtrain, Ytrain, num_labels, lambda, alpha, num_iters);
	    %predict validation dataset
          pred = predictOneVsAll(all_theta, Xval);
	    %predict training dataset
          pred2 = predictOneVsAll(all_theta, Xtrain);
          fprintf('\nfold %d with alpha %f lambda %f train: %f, test:%f\n', kk, alpha, lambda, mean(double(pred2 == Ytrain)) * 100, mean(double(pred == Yval)) * 100);
   end
end

%% ------Part 3: test ----------
%optimal alpha for PCA data
%alpha = 0.5;
%optimal alpha for raw data
alpha = 0.1;

num_iters = 300;
%train all the training data with optimal alpha
%mledata
%[all_theta] = oneVsAll(trainXmle, y, num_labels, lambda, alpha, num_iters);
%raw_data
[all_theta] = oneVsAll(trainData, y, num_labels, lambda, alpha, num_iters);
%training data accuracy
%for PCA data
%pred = predictOneVsAll(all_theta, trainXmle);
%for raw data
pred = predictOneVsAll(all_theta, trainData);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

% draw the confusion matrix
confusion_matrix(pred ,y);

%predict the test data and write results
%PCA data
%pred1 = predictOneVsAll(all_theta, testXmle);
%raw data
pred1 = predictOneVsAll(all_theta, testXData);
fid = fopen('test_outputs.csv','wt');
len = length(pred1);
fprintf(fid,'Id,Prediction\n');
for i = 1:len
    fprintf(fid, '%d,%d\n',i,pred1(i));
end

fclose(fid);

