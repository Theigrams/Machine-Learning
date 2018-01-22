%% �Գ����ж�
%���ۺ���Ϊ�����غ���
%��������
%����minibatch��
%% ��ʼ��
clear ; 
close all;
clc;

%% ================ Part 1: Loading  Data ===================
% Load Training Data ����ѵ������
%���뼯��P�����ΪT
m=64;   %mΪminibatch��size
load('data5.mat');
M = size(T, 1);  %MΪ��������
%% ================ Part 2: Seting Parameters ================
% set neural network parameters.

sizes=[6,2,1];   %���������������ÿ�����Ԫ����
num_layers=length(sizes);
for i=2:num_layers
    L(i).b=rand(sizes(i),1);        %��i����Ԫ��ƫ��biases
    L(i).db=zeros(size(L(i).b));    %��i����Ԫ��ƫ�õ��ݶ�
    L(i).vb=0;          %������
    L(i).w=rand(sizes(i),sizes(i-1));  %�� i, i-1 ����Ԫ֮��ĵ�Ȩֵ����weights
    L(i).dw=zeros(size(L(i).w));    %Ȩֵ������ݶ�
    L(i).vw=0;
end
%Ҳ������cell�����治ͬά�ȵľ���

maxcount=100000;%�������ļ���
eta=1;%����ѧϰ��
alpha=0.8;%������
eps=10e-4;
J=zeros(1,maxcount);

%% ================ Part 3: Training ANN===================

for k=1:maxcount
    sel = randperm(M);
    sel = sel(1:m);
    X=T(sel,:);
    Y=P(sel,:);
    % Part 1: Feedforward ǰ������
    % -------------------------------------------------------------
    L(1).a=X';
    for i=2:num_layers
            L(i).z=L(i).w*L(i-1).a+L(i).b;
            L(i).a=sigmoid(L(i).z);
    end
    cost = -Y'.*log(L(i).a)-(1-Y').*log(1-L(i).a);
    J(k) = sum(sum(cost, 2)) / m; 	% ��͵óɱ�����

    % Part 2: Backpropagation  ���򴫲�
    % -------------------------------------------------------------
    for i=num_layers:-1:2
        if(i==num_layers)
            L(i).e=(L(i).a-Y')/m;   %�ײ�����
        else
            L(i).e=L(i+1).w'*L(i+1).e.*d_sigmoid(L(i).z);  %��i������
        end
        L(i).dw=L(i).e*L(i-1).a'; % w[i]���ݶ�
        L(i).db= sum(L(i).e,2); % b[i]���ݶ�
    end

    % Part 3: Gradient descent �ݶ��½���momentum ��������ģ�
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

Cost=Feedforward(L,T,num_layers)-P'
sum(Cost.^2)
J=J(1:k);
plot(J)


function g = sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z));
end

function g = d_sigmoid(z)
    g = sigmoid(z).*(1-sigmoid(z));
end

function a=Feedforward(L,X,num_layers)
    a=X';
    for i=2:num_layers
            z=L(i).w*a+L(i).b;
            a=sigmoid(z);
    end
end
    