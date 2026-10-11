% EXAMPLE:
clc;close all;clear;
fs = 10000; % sampling frequency
N = 30000; % data amount
t = (1:N)/fs; % time vector
x1 = (2+cos(2*pi*0.5*t)).*cos(2*pi*5*t+15*t.^2);
x2 = cos(2*pi*2*t);
x = x1+x2;
options.display = 1;
options.max_iter = 30;
options.max_pfs = 10;
[pf3, ams3, fms3, ort3] = lmd_public(x,options);
figure;
subplot(2,1,1),plot(t,x1);
subplot(2,1,2),plot(t,x2);