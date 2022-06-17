import os
import numpy as np
import glob
import PIL.Image as Image
import pdb
import torchvision.transforms as transforms

# pip install torchsummary
import torch


torch.manual_seed(1234)

# Set path of data you want to load
name = 'test_style0'

data_path = './data/' + name



# Transform images
size = 256
transform = transforms.Compose([transforms.Resize((size, size)), 
                                      transforms.ToTensor()])


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
dataset = ISIC(transform=transform, data_path=data_path)
torch.save(dataset,'./datasets/'+name+'.pt')
print('Saved train dataset')

