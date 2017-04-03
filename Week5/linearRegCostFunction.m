function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


theta_new=theta;
theta_new(1) = 0;
sum_theta_new_2 = sum(theta_new.^2);
reg_term = lambda/(2*m)* sum_theta_new_2;


J = (1/(2*m) * (X * theta - y)' * (X * theta - y))+reg_term;


%[J, grad] = costFunction(theta, X, y);

% this effectively ignores "theta zero" in the following calculations
%theta_zeroed_first = [0; theta(2:length(theta));];

%J = J + lambda / (2 * m) * sum( theta_zeroed_first .^ 2 );
%J = J + reg_term;
%grad = grad + (lambda / m) * theta_zeroed_first;
grad = (1.0/m) .* X' * (X*theta - y) + (lambda/m) * theta_new;










% =========================================================================

grad = grad(:);

end
