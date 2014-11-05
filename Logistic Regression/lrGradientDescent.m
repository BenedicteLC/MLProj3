function [theta] = lrGradientDescent(theta, X, y, lambda, alpha, num_iters)

%lrGradientDescent Compute cost and gradient for logistic regression without regularization (lambda = 0)
%   J = lrGradientDescent(theta, X, y, lambda, alpha, num_iters) computes the cost of using
%   theta as the parameter for logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

%initialization
J = 0;
grad = zeros(size(theta));

%interates for gradient descent
for iter = 1:num_iters
    h_theta = 1./(1+exp(-X*theta));
    theta1 = theta;
    theta1(1) = 0;
    %gradient
    grad = ((h_theta - y)'*X + lambda*theta1')' / m;
    theta = theta - alpha * grad;
end

end
