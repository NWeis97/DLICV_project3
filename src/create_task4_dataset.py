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
name = 'task4_dataset'
data_path = './data/'



# Transform images
size = 256
transform = transforms.Compose([transforms.Resize((size, size)), 
                                      transforms.ToTensor()])


# ISIC dataloader
class ISIC_task4(torch.utils.data.Dataset):
    def __init__(self, transform, data_path):
        'Initialization'
        self.transform = transform
        self.image_paths = []
        self.seg_paths_coarse = []
        self.seg_paths_granular = []

        image_paths_style0 = sorted(glob.glob(data_path + 'train_style0/Images/*.jpg'))
        image_paths_style2 = sorted(glob.glob(data_path + 'train_style2/Images/*.jpg'))
        seg_paths_style0 = sorted(glob.glob(data_path + 'train_style0/Segmentations/*.png'))
        seg_paths_style2 = sorted(glob.glob(data_path + 'train_style2/Segmentations/*.png'))

        image_paths_style2_names = []
        for path in image_paths_style2:
            image_paths_style2_names.append(path.split("/")[-1])

        image_paths = [path for path in image_paths_style0 if path.split("/")[-1] in image_paths_style2_names]

        for image_path in image_paths:
            image_name = image_path.split("/")[-1][:-4]
            all_seg_per_image_style2 = [seg for seg in seg_paths_style2 if seg.split("/")[-1][:len(image_name)]==image_name]
            all_seg_per_image_style0 = [seg for seg in seg_paths_style0 if seg.split("/")[-1][:len(image_name)]==image_name]
            
            # Always pick first
            self.image_paths.append(image_path)
            self.seg_paths_coarse.append(all_seg_per_image_style2[0])
            self.seg_paths_granular.append(all_seg_per_image_style0[0])

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        seg_path_coarse = self.seg_paths_coarse[idx]
        seg_path_granular = self.seg_paths_granular[idx]
        
        # Open images
        image = Image.open(image_path)
        seg_coarse = Image.open(seg_path_coarse)
        seg_granular = Image.open(seg_path_granular)

        # Transform images
        image = transforms.functional.crop(image,36,114,image.size[1]-36-37,image.size[0]-114-102)
        X = self.transform_resize(image)
        target = self.transform_resize(seg_granular)
        seg_init = self.transform_resize(seg_coarse)

        return X, seg_init, target

# Load data
dataset = ISIC_task4(transform=transform, data_path=data_path)
torch.save(dataset,'./datasets/'+name+'.pt')
print('Saved train dataset')

