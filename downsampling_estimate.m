function estimation=downsampling_estimate(data)
a=size(data); %default 128*128
h=a(1);
l=a(2);
estimation=0;

while(h>10 && l>10)
    est=Noisele(data);
    if est>estimation
        estimation = est;
    end
    h=ceil(h/2);
    l=ceil(l/2);
    data=imresize(data,[h,l]);
end
    