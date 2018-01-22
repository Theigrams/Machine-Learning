%% ���������������������V1.0
%���ۺ���Ϊ���κ���
%% ��ʼ��
clear ; 
close all;
clc;

%% ���ò���
input_layer_size  = 2;  
hidden_layer_size = 2;  
output_layer_size = 1;          
sizes=[input_layer_size,hidden_layer_size,output_layer_size];
maxcount=50000;%�������ļ���
eta=0.03;%����ѧϰ��
alpha=0.9;%����ѧϰ��
J=zeros(1,maxcount);
%% ================ Part 1: Loading  Data =============

% Load Training Data ����ѵ������
%���뼯��P�����ΪT
load('data1.mat');
m = size(T, 1);  %mΪ��������

%% ================ Part 2: Seting Parameters ================
% set neural network parameters.

w2=rand(sizes(2),sizes(1));
w3=rand(sizes(3),sizes(2));
b2=rand(sizes(2),1);
b3=rand(sizes(3),1);

%% ================ Part 3: Training ANN================

for i=1:maxcount
% Part 1: CostFunction �ɱ�����
% -------------------------------------------------------------

% w2��С 2 x 2
% w3��С 1 x 2
a1 = T'; 					% ����� a1��С 2 x 4
z2 = w2*a1+b2; 			% �ڶ������� z2��С 2 x 4
a2 = sigmoid(z2); 			% �ڶ������
z3 = w3*a2+b3;		% ���������� z3��С 1 x 4
a3 = sigmoid(z3);			% ����� �õ� 1 x 4 �ľ���

cost = (a3-P').^2;
J(i) = sum(sum(cost, 2)) / m; 	% ��͵óɱ�����

% Part 2: Backpropagation  ���򴫲�
% -------------------------------------------------------------
Error3 =2/m*(a3-P').*d_sigmoid(z3); % ����������
Error2 = (w3)'*Error3 .* d_sigmoid(z2);	% �ڶ�������

d_w3= Error3*a2'; % w2���ݶ�
d_b3= sum(Error3,2); % b3���ݶ�

d_w2= Error2*a1'; % w2���ݶ�
d_b2=sum(Error2,2); % b2���ݶ�

% Part 3: Gradient descent �ݶ��½�
% -------------------------------------------------------------
w3=w3-eta*d_w3;
w2=w2-eta*d_w2;
b3=b3-eta*d_b3;
b2=b2-eta*d_b2;

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



