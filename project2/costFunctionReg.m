function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
sum1=1;
sum1=1;
sum2=ones(length(theta),1);
grad = zeros(size(theta));
z=X*theta;
g=sigmoid(z);
J=(-1/m)*((log(g))'*y + (log(1-g))'*(1-y));
theta(1)=0;
sum=((theta'*theta)*(lambda/(2*m)));
J=J + sum;
sum1=sum2'*theta;
grad=(1/m)*X'*(g-y);
grad=grad + (sum1*(lambda/m));












% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
