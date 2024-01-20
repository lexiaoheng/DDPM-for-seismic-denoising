function out=diffusion_model(data)
    [h,w]=size(data);
    factor=sum(sum(abs(data)))/(h*w);
    data=data*0.7917/factor;
    
    %% if you need to segment your data, please add segmentation process in there and save patches in 
    % './diffusion_model/dataset/' and name them as '1.mat, 2.mat, 3.mat......'. 
    
    save('./diffusion_model/dataset/1.mat','data');
    t_seq=cul_t(data);
    save('./diffusion_model/t_seq.mat','t_seq');
    fprintf('Running diffusion model. \n')
    %% run python programs with cmd.
    % You should replace the anaconda environment name.  
    [~,~]=system('activate NNPython & cd ./diffusion_model & python main.py & exit ');
    
    %%
    fprintf('Diffusion done. \n');

    load('./diffusion_model/out/1.mat','data');

    out=reshape(data,h,w)*factor/0.7917;


