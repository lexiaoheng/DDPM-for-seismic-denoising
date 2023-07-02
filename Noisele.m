%% �ú�������ͼ��(�ź�)������ˮƽ
% �����������Ϊ��˹������
% ���ڸ������⣬����ֻ����1��2��3ά���ݵ�������һ����ԣ������㹻��.....
% An Efficient Statistical Method for Image Noise Level Estimation
% Guangyong Chen1, Fengyuan Zhu1, and Pheng Ann Heng1,2
function [delta,D]=Noisele(M,d)
M=squeeze(M);
SizeM=size(M);
%% һά����
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
%% ��ά����
if length(SizeM)==2
%% ��ÿһС�߿��л�
X=zeros((SizeM(1)-d)*(SizeM(2)-d),d^2);
for ii=1:d
for jj=1:d
F=M(ii:SizeM(1)-d+ii-1,jj:SizeM(2)-d+jj-1);
X(:,jj+(ii-1)*d)=F(:);
end
end
else
%% ��ά����
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