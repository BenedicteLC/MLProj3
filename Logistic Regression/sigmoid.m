function g = sigmoid(z)
%Compute sigmoid function of z

g = 1.0 ./ (1.0 + exp(-z));
end
