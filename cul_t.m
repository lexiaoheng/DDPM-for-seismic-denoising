function out=cul_t(data)
min=1;
out=0;
norm_factor=downsampling_estimate(data);
for t=1:1:200
    if abs((0.04170+0.01009*t-3.066*10^(-5)*t^2+1.7917*10^(-8)*t^3)-norm_factor)<min
       out=t;
       min=abs((0.0417+0.01009*t-3.066*10^(-5)*t^2+1.797*10^(-8)*t^3)-norm_factor);
    end
end
end