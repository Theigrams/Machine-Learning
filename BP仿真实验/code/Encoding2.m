%% N层神经网络V2.0
%Encode2
%代价函数为交叉熵函数
%增添动量项
%% 初始化
clear ; 
close all;
clc;

%% ================ Part 1: Loading  Data ===================
% Load Training Data 加载训练数据
%输入集是P，输出为T
load('data4.mat');
m = size(T, 1);  %m为样本个数

%% ================ Part 2: Seting Parameters ================
% set neural network parameters.

sizes=[2,1,4,4];   %设置神经网络层数和每层的神经元数量
num_layers=length(sizes);
for i=2:num_layers
    L(i).b=rand(sizes(i),1);        %第i层神经元的偏置biases
    L(i).db=zeros(size(L(i).b));    %第i层神经元的偏置的梯度
    L(i).vb=0;          %动量项
    L(i).w=rand(sizes(i),sizes(i-1));  %第 i, i-1 层神经元之间的的权值矩阵weights
    L(i).dw=zeros(size(L(i).w));    %权值矩阵的梯度
    L(i).vw=0;
end
%也可以用cell来储存不同维度的矩阵

maxcount=5000;%设置最大的计数
eta=0.1;%设置学习率
alpha=0.8;%动量率
eps=10e-3;
J=zeros(1,maxcount);

%% ================ Part 3: Training ANN===================

for k=1:maxcount
    % Part 1: CostFunction 成本函数
    % -------------------------------------------------------------
    L(1).a=T';
    for i=2:num_layers
            L(i).z=L(i).w*L(i-1).a+L(i).b;
            L(i).a=sigmoid(L(i).z);
    end
    cost = -P'.*log(L(i).a)-(1-P').*log(1-L(i).a);
    J(k) = sum(sum(cost, 2)) / m; 	% 求和得成本函数

    % Part 2: Backpropagation  反向传播
    % -------------------------------------------------------------
    for i=num_layers:-1:2
        if(i==num_layers)
            L(i).e=(L(i).a-P')/m;   %底层的误差
        else
            L(i).e=L(i+1).w'*L(i+1).e.*d_sigmoid(L(i).z);  %第i层的误差
        end
        L(i).dw=L(i).e*L(i-1).a'; % w[i]的梯度
        L(i).db= sum(L(i).e,2); % b[i]的梯度
    end

    % Part 3: Gradient descent 梯度下降（momentum 带动量项的）
    % -------------------------------------------------------------
    for i=num_layers:-1:2
        L(i).vw=alpha*L(i).vw-eta*L(i).dw;
        L(i).w=L(i).w+L(i).vw;
        L(i).vb=alpha*L(i).vb-eta*L(i).db;
        L(i).b=L(i).b+L(i).vb;
    end
    if(J(k)<eps)
        break;
    end
end
L(num_layers).a
J=J(1:k);
J(k)
plot(J)


function g = sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z));
end

function g = d_sigmoid(z)
    g = sigmoid(z).*(1-sigmoid(z));
end