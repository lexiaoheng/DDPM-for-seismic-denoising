%% This code is associated with paper <Seismic Strong Noise Attenuation Based on Diffusion Model and Principal Component Analysis>
% Author: Junheng Peng <junhengpeng@ieee.org>
% You need to cite this paper if you use this code in your work. <10.1109/TGRS.2024.3355460>
% The test environment includes Matlab R2019a and Python version 3.7.
%% If you need to use this code in your work, you should modify 'main.m, ./func/diffusion_model.m' or directly use python files.
%% set parameters
clc;clear all;
addpath('./data');
addpath('./matlab_functions');

% noise parameters
beta=0.00115:0.00015:0.031;
t_add=100;

%% load clean data 
% So far, this code only support data matirx with 128 * 128, and you can
% use your patch segmentation method in this code;

load demo.mat; %data
data_origin=data;
clear data;
[h,w]=size(data_origin);

%% add noise 
noisy_data=data_origin;
for i=1:t_add
    noise = randn(h,w);
    noisy_data = ((1-beta(i))^0.5) * noisy_data + beta(i)^0.5 * noise;
end

%% run diffusion model with anaconda environment and cmd
% Without any prior information, just only noisy data. 
% You should replace the anaconda environment name in ./func/diffusion_model.m 
out=diffusion_model(noisy_data);

%% demo
figure;
subplot(1,4,1);
imagesc(data_origin);
title('ground truth');

subplot(1,4,2);
imagesc(noisy_data);
title('input noisy data');

subplot(1,4,3);
imagesc(out);
title('output');

subplot(1,4,4);
imagesc(data_origin-out);
title('residuals');

colormap('gray');
caxis([min(min(data_origin)) max(max(data_origin))]);













