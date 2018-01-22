clear ; 
close all;
clc;

N=6;
T = abs(dec2bin(0:(2^N-1), N))-48;
P=(T(:,1)==T(:,6) &T(:,2)==T(:,5)&T(:,3)==T(:,4));
save data5.mat T P