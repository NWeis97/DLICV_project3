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
name = 'classifier_dataset'

data_path = './data/'



# Transform images
size = 256
transform = transforms.Compose([transforms.Resize((size, size)), 
                                      transforms.ToTensor()])


# ISIC dataloader
class ISIC_classifier(torch.utils.data.Dataset):
    def __init__(self, transform, data_path):
        'Initialization'
        self.transform = transform
        image_paths_background = sorted(glob.glob(data_path + 'background/*.jpg'))
        
        image_paths_style0 = sorted(glob.glob(data_path + 'train_style0/Images/*.jpg'))
        image_paths_style1 = sorted(glob.glob(data_path + 'train_style1/Images/*.jpg'))
        image_paths_style2 = sorted(glob.glob(data_path + 'train_style2/Images/*.jpg'))

        image_paths_style2_names = []
        for path in image_paths_style2:
            image_paths_style2_names.append(path.split("/")[-1])

        # Used for task 4 weak annotations
        image_paths_20 = [path for path in image_paths_style0 if path.split("/")[-1] in image_paths_style2_names]

        image_paths_20_names = []
        for path in image_paths_20:
            image_paths_20_names.append(path.split("/")[-1])

        image_paths_0n20 = [path for path in image_paths_style0 if path.split("/")[-1] not in image_paths_20_names]
        image_paths_1n20 = [path for path in image_paths_style1 if path.split("/")[-1] not in image_paths_20_names]
        image_paths_2n20 = [path for path in image_paths_style2 if path.split("/")[-1] not in image_paths_20_names]
        

        image_paths = []
        image_paths.extend(image_paths_2n20)
        image_paths.extend(image_paths_1n20)
        image_paths.extend(image_paths_0n20)

        self.labels = [0]*len(image_paths_background)
        self.labels.extend([1]*len(image_paths))

        # Make final list of image paths (should match with labels)
        self.image_paths = image_paths_background
        self.image_paths.extend(image_paths)
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path)
        image = transforms.functional.crop(image,36,114,image.size[1]-36-37,image.size[0]-114-102)
        X = self.transform(image)
        return X, label

# Load data
dataset = ISIC_classifier(transform=transform, data_path=data_path)
torch.save(dataset,'./datasets/'+name+'.pt')
print('Saved train dataset')

