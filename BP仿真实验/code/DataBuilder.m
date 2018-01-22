clear ; 
close all;
clc;

N=4;
T = abs(dec2bin(0:(2^N-1), N))-48;
P=mod(sum(T, 2),2);
save data2.mat T P