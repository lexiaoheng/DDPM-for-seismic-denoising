%% 该函数估计图像(信号)的噪声水平
% 这里假设噪声为高斯白噪声
% 由于各种问题，函数只估计1、2、3维数据的噪声，一般而言，这是足够的.....
% An Efficient Statistical Method for Image Noise Level Estimation
% Guangyong Chen1, Fengyuan Zhu1, and Pheng Ann Heng1,2
function [delta,D]=Noisele(M,d)
M=squeeze(M);
SizeM=size(M);
%% 一维情形
if SizeM(2)==1
if nargin<2
d=65;
end
X=zeros(SizeM(1)-d,d);
for ii=1:SizeM(1)-d
F=M(ii:SizeM(1)-d+ii-1);
X(:,ii)=F(:);
end
else
if nargin<2
d=9;
end
%% 二维情形
if length(SizeM)==2
%% 将每一小斑块列化
X=zeros((SizeM(1)-d)*(SizeM(2)-d),d^2);
for ii=1:d
for jj=1:d
F=M(ii:SizeM(1)-d+ii-1,jj:SizeM(2)-d+jj-1);
X(:,jj+(ii-1)*d)=F(:);
end
end
else
%% 三维情形
X=zeros((SizeM(1)-d)*(SizeM(2)-d)*(SizeM(3)-d),d^3);
index=1;
for ii=1:d
for jj=1:d
for kk=1:d
F=M(ii:SizeM(1)-d+ii-1,jj:SizeM(2)-d+jj-1,kk:SizeM(3)-d+kk-1);
X(:,index)=F(:);
index=index+1;
end
end
end
end
end
[delta,D]=NoiseLevel(X);
end