function J = computeCostMulti(X, y, theta)
m=length(y);
pred=X*theta;
sqr=sum((pred-y).^2);
J=1/(2*m) * sqr;


