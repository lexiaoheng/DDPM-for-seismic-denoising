import scipy.io as scio
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

def exists(x):
    return x is not None

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        data_num,
        image_size,
        exts = 'mat',
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        # self.paths = [p for p in Path(f'{folder}').glob(f'**/*.{exts}')]
        self.paths = [('./dataset/'+str(p+1)+'.mat') for p in range(data_num)]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = scio.loadmat(path)['data']
        img = torch.tensor(img)
        img = self.transform(img)
        img = img.unsqueeze(0)
        return img.float()