function p = predict(theta1, theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(theta2, 1);
p = zeros(size(X, 1), 1);
y1=zeros(m,size(theta1,1));
X=[ones(m,1) X];
y1=X*theta1';
fprintf('\n %d %d\n',size(y1,1),size(y1,2));
y1=sigmoid(y1);
y1=[ones(m,1) y1];
y2=zeros(m,size(theta2,1));
fprintf('\n %d %d\n',size(theta2,1),size(theta2,2));
y2=y1*theta2';
fprintf('\n %d %d\n',size(y1,1),size(y1,2));
y2=sigmoid(y2);
y2=y2';
dummy=zeros(size(X, 1), 1);
[dummy,p]=max(y2);
p=p';
fprintf('\n %d %d\n',size(p,1),size(p,2));


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%









% =========================================================================


end
