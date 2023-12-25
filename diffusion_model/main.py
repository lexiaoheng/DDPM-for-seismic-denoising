import os
from torch.optim import Adam
from utils.networkHelper import *

from noisePredictModels.Unet.UNet import Unet
from utils.trainNetworkHelper import SimpleDiffusionTrainer
from diffusionModels.simpleDiffusion.simpleDiffusion import DiffusionModel
from utils import dataread


# 数据集加载
data_root_path = "./dataset/"
if not os.path.exists(data_root_path):
    os.makedirs(data_root_path)


image_size = 128
channels = 1
batch_size = 1
data_num = 100


imagenet_data = dataread.Dataset(data_root_path, data_num, image_size, augment_horizontal_flip = False, convert_image_to = None)
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=0)
# 以防cuda出现gpu编号错误情况
'''''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
'''''

device = "cuda" if torch.cuda.is_available() else "cpu"

dim_mults = (1, 2, 4,)
denoise_model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=dim_mults
)

timesteps = 200
schedule_name = "linear_beta_schedule"
DDPM = DiffusionModel(schedule_name=schedule_name,
                      timesteps=timesteps,
                      beta_start=0.00115,
                      beta_end=0.031,
                      denoise_model=denoise_model).to(device)

optimizer = Adam(DDPM.parameters(), lr=1e-3)
epoches = 100

# 训练参数设置
root_path = "./saved_train_models"
setting = "imageSize{}_channels{}_dimMults{}_timeSteps{}_scheduleName{}".format(image_size, channels, dim_mults, timesteps, schedule_name)

saved_path = os.path.join(root_path, setting)
if not os.path.exists(saved_path):
    os.makedirs(saved_path)

# 训练请使用以下的代码
Trainer = SimpleDiffusionTrainer(epoches=epoches,
                                 mode='train',
                                 train_loader=data_loader,
                                 optimizer=optimizer,
                                 device=device,
                                 timesteps=timesteps)
DDPM = Trainer(DDPM, model_save_path=saved_path)

# 验证和处理数据请使用以下的代码并注释上面的代买，请确保验证的数据大小与模型参数一致，有些参数是多余的.
# 需要验证的待处理数据存放在./validation_dataset目录下，采用mat格式，待处理数据变量名为data。每一份数据按照正整数编号。
# 请确保根目录下存在t_seq.mat文件，是一个一维数组，长度与验证数据的数目一致，每个元素与验证数据需要处理的步数t一一对应。（t请使用matlab函数进行估计）
'''''
Trainer = SimpleDiffusionTrainer(epoches=epoches,
                                 mode='validation',
                                 train_loader=data_loader,
                                 optimizer=optimizer,
                                 device=device,
                                 timesteps=timesteps)
DDPM = Trainer(DDPM, model_save_path=saved_path)
'''''


