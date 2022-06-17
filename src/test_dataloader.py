# Import 
from operator import index
import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm.notebook import tqdm
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import pdb
import re
import pandas as pd
from torch.utils.data import random_split

import cv2 as cv
import numpy as np
import sys


# Set seed
seed = 1234
torch.manual_seed(seed)


# ISIC dataloader
class ISIC(torch.utils.data.Dataset):
    def __init__(self, transform, data_path):
        'Initialization'
        self.transform = transform
        self.image_paths = sorted(glob.glob(data_path + '/Images/*.jpg'))
        self.seg_paths = sorted(glob.glob(data_path + '/Segmentations/*.png'))
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        seg_path = self.seg_paths[idx]
        
        image = Image.open(image_path)
        seg = Image.open(seg_path)
        Y = self.transform(seg)
        X = self.transform(image)
        return X, Y



# Load data
dataset = torch.load('datasets/train_allstyles.pt')
train_length = int(0.70*dataset.__len__())
train_dataset, val_dataset = random_split(dataset, [train_length,dataset.__len__()-train_length], generator=torch.Generator().manual_seed(seed))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=0)

for minibatch_no, (image, segmentation) in tqdm(enumerate(train_loader), total=len(train_loader)):
    
    pdb.set_trace()
    print(image)