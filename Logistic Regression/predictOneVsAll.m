function p = predictOneVsAll(all_theta, X)

% Predict the label with a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = predictOneVsAll(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class.

m = size(X, 1);
num_labels = size(all_theta, 1);

p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

tempval = sigmoid(X * all_theta');

%pick the label that has maximum value
[Y,I] = max(tempval,[],2);
p = I - 1;

end
