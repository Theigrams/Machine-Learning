clear ; 
close all;
clc;
x=1;
y=0;
w=2;
b=2;
eta=0.15;
for i=1:300
    z=w.*x+b;
    a=sigmoid(z);
    E(i)=-log(1-a);
    error=a;
    dw=error*x;
    db=error;
    w=w-eta*dw;
    b=b-eta*db;
    if(E(i)<0.00001)
        break;
    end
end
plot(E)
E(i)
w
b


function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.
    g = 1.0 ./ (1.0 + exp(-z));
end

function g = d_sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.
    g = sigmoid(z).*(1-sigmoid(z));
end