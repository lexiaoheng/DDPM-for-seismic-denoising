function delta=NoiseLevel(X)
[~,mm]=size(X);
%% ��Э�������
F=cov(X);
%% ��Э������������ֵ����������
[~,D]=eig(F);
D=real(diag(D));
D=sort(D,'descend');
%% ����������С
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