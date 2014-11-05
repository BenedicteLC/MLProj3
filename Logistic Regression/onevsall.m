function [all_theta] = oneVsAll(X, y, num_labels, lambda, alpha, num_iters)
%one-vs-all trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = oneVsAll(X, y, num_labels, lambda, alpha, num_iters) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% number of samples
m = size(X, 1);
% number of features
n = size(X, 2);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Set Initial theta
initial_theta = zeros(n + 1, 1);

% call lrGradientDescent to do gradient descent
theta_tmp = [];

for c = 0:num_labels-1
    [theta] = lrGradientDescent(initial_theta, X, (y == c), lambda, alpha, num_iters);
    theta_tmp = [theta_tmp; theta'];  
end
all_theta = theta_tmp;


end
