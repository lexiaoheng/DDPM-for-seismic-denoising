# Publication
## （please cite it if you used this code in your paper）
seismic strong noise attenuation based on diffusion model and principal component analysis（10.1109/TGRS.2024.3355460）. Technically, you should to cite the article if you use this code in your work. 
# Introduction
This code is used for noise level assessment and noise attenuation. Since the use of diffusion models for noise processing requires a more accurate grasp of the noise level (to determine the number of diffusion reduction steps), we propose to use principal component analysis for assessment.

# Quick validation（Matlab codes）
run main.m function for validation. You should modify the 'main.m' and './matlab_functions/diffusion_model.m'.

# Diffusion model
support data formation: xxx.mat with variation name "data" in it (you can use matlab to save it) , and you should name them with numbers, for example, "1.mat, 2.mat……".

validation notes: use matlab generate a list to state t of validation data，which should be named as “t_seq.mat”.

use main.py to training and validation your model, please read notes in main.py

if you need pre-trained model, please contact me (junhengpeng@ieee.com)
