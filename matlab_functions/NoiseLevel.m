function delta=NoiseLevel(X)
[~,mm]=size(X);
%% 求协方差矩阵
F=cov(X);
%% 求协方差矩阵的特征值并降序排列
[~,D]=eig(F);
D=real(diag(D));
D=sort(D,'descend');
%% 估计噪声大小
for ii=1:mm
t=sum(D(ii:mm))/(mm+1-ii);
F=floor((mm+ii)/2);
F1=F-1;
F2=min(F+1,mm);
if (t<=D(F1))&&(t>=D(F2))
delta=sqrt(t);
break;
end
end
end