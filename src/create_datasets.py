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
name = 'train_style0'
data_type = 'train'

data_path = './data/' + name



# Transform images
size = 256 
transform_resize = transforms.Compose([transforms.Resize((size, size)), 
                                transforms.ToTensor()])

transform = transforms.Compose([transforms.ToTensor()])


# ISIC dataloader
class ISIC(torch.utils.data.Dataset):
    def __init__(self, data_type, transform_resize, transform, data_path, seed=1234):
        'Initialization'
        self.transform = transform
        self.transform_resize = transform_resize
        self.data_type = data_type
        self.seed = seed
        self.image_paths = []
        self.seg_paths = []
        image_paths = sorted(glob.glob(data_path + '/Images/*.jpg'))
        seg_paths = sorted(glob.glob(data_path + '/Segmentations/*.png'))

        # Random permutation
        num_images = len(image_paths)
        np.random.seed(self.seed)
        rand_perm = np.random.permutation(np.arange(0,num_images,1))
        rand_perm = rand_perm[:int(0.7*num_images)]

        if self.data_type == 'train':
            image_paths = np.array(image_paths)[rand_perm].tolist()
        elif self.data_type == 'val':
            image_paths = np.array(image_paths)[[x for x in np.arange(0,num_images,1) if x not in rand_perm]].tolist()
        else:
            self.image_paths = image_paths
            self.seg_paths = seg_paths
        
        
        if (self.data_type == 'train') | (self.data_type == 'val'):
            for image_path in image_paths:
                image_name = image_path.split("/")[-1][:-4]
                all_seg_per_image = [seg for seg in seg_paths if seg.split("/")[-1][:len(image_name)]==image_name]
                num_seg_per_image = len(all_seg_per_image)

                self.image_paths.extend([image_path]*num_seg_per_image)
                self.seg_paths.extend(all_seg_per_image)

        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        seg_path = self.seg_paths[idx]
        
        image = Image.open(image_path)
        seg = Image.open(seg_path)
        if self.data_type == 'test':
            seg = transforms.functional.crop(seg,36,114,seg.size[1]-36-37,seg.size[0]-114-102)
            Y = self.transform_resize(seg)
        else:
            Y = self.transform(seg)
        X = transforms.functional.crop(image,36,114,seg.size[1]-36-37,seg.size[0]-114-102)
        X = self.transform_resize(image)
        return X, Y

# Load data
dataset = ISIC( transform_resize=transform_resize, transform=transform, data_type=data_type, data_path=data_path)
dataset.__getitem__(0)
torch.save(dataset,'./datasets/'+name+'-datatype_'+data_type+'.pt')
print('Saved train dataset')

