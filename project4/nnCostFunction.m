function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
z1=zeros(m,hidden_layer_size);
a1=zeros(m,(hidden_layer_size));
z2=zeros(m,num_labels);
a2=zeros(m,(num_labels));
X=[ones(m,1) X];
z1=X*(Theta1');
a1=sigmoid(z1);
m1=length(a1);
a1=[ones(m1,1) a1];
z2=a1*Theta2';
a2=sigmoid(z2);
y=eye(num_labels)(y,:);
J1=sum(sum(Theta1.*Theta1)) + sum(sum(Theta2.*Theta2));
J1=(lambda/(2*m))*J1;
J=(-1/m)*sum(sum( y.*log(a2) + (1-y).*(log(1-a2)) )) + J1;
d3=zeros(size(y,1),size(y,2));
d3=a2-y;
d2=zeros(m,hidden_layer_size);
d2=(d3*Theta2(:,2:end)).*(sigmoidGradient(z1));
Theta1_grad =d2'*X;
Theta2_grad =d3'*a1;
Theta1_grad =(1/m)*Theta1_grad;
Theta2_grad =(1/m)*Theta2_grad;
Theta1(:,1)=0;
Theta2(:,1)=0;
Theta1=(lambda/m)*Theta1;
Theta2=(lambda/m)*Theta2;
Theta1_grad =Theta1_grad +Theta1;
Theta2_grad =Theta2_grad +Theta2;
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
