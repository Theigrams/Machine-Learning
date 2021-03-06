%% 解决编码问题的三层神经网络V1.0
%代价函数为交叉熵函数
%增添动量项
%% 初始化
clear ; 
close all;
clc;

%% 设置参数
N=8;
input_layer_size  = N;  
hidden_layer_size = floor(log2(N));  
output_layer_size = N;          
sizes=[input_layer_size,hidden_layer_size,output_layer_size];
maxcount=1000;%设置最大的计数
eta=0.7;%设置学习率
alpha=0.8;%动量学习率
J=zeros(1,maxcount);
%% =========== Part 1: Loading and Visualizing Data =============

% Load Training Data 加载训练数据
%输入集是P，输出为T
load('data3.mat');
m = size(T, 1);  %m为样本个数

%% ================ Part 2: Seting Parameters ================
% set neural network parameters.
w2=rand(sizes(2),sizes(1));
w3=rand(sizes(3),sizes(2));
b2=rand(sizes(2),1);
b3=rand(sizes(3),1);
v_w3=0;v_w2=0;
v_b3=0;v_b2=0;
%% ================ Part 3: Training ANN================
for i=1:maxcount
% Part 1: CostFunction 成本函数
% -------------------------------------------------------------

% w2大小 2 x 2
% w3大小 1 x 2
a1 = T'; 					% 输入层 a1大小 2 x 4
z2 = w2*a1+b2; 			% 第二层输入 z2大小 2 x 4
a2 = sigmoid(z2); 			% 第二层输出
z3 = w3*a2+b3;		% 第三层输入 z3大小 1 x 4
a3 = sigmoid(z3);			% 输出层 得到 1 x 4 的矩阵

cost = -P'.*log(a3)-(1-P').*log(1-a3);
J(i) = sum(sum(cost, 2)) / m; 	% 求和得成本函数

% Part 2: Backpropagation  反向传播
% -------------------------------------------------------------
Error3 =(a3-P')/m; % 第三层的误差
Error2 = (w3)'*Error3 .* d_sigmoid(z2);	% 第二层的误差

d_w3= Error3*a2'; % w2的梯度
d_b3= sum(Error3,2); % b3的梯度

d_w2= Error2*a1'; % w2的梯度
d_b2=sum(Error2,2); % b2的梯度

% Part 3: Gradient descent 梯度下降（momentum 带动量项的）
% -------------------------------------------------------------
v_w3=alpha*v_w3-eta*d_w3;
w3=w3+v_w3;
v_w2=alpha*v_w2-eta*d_w2;
w2=w2+v_w2;
v_b3=alpha*v_b3-eta*d_b3;
b3=b3+v_b3;
v_b2=alpha*v_b2-eta*d_b2;
b2=b2+v_b2;
end
w2
w3
b2
b3


a3
J(i)
plot(J)
% x=1:i;
% plot(x,J)


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



