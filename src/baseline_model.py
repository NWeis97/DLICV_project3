# IMPORTS
import os
import numpy as np
import glob
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torchviz import make_dot
import pdb
from torch.utils.data import random_split


# Set seed
seed = 1234
torch.manual_seed(seed)

# Define figure size
plt.rcParams['figure.figsize'] = [22, 7]

# Select cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# ISIC dataloader
class ISIC(torch.utils.data.Dataset):
    def __init__(self, data_type, transform_resize, transform, data_path, crop_x, seed=1234):
        'Initialization'
        self.transform = transform
        self.crop_x = crop_x
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
            seg = Image.open(seg_path).convert('L')
            seg = transforms.functional.crop(seg,36,114,seg.size[1]-36-37,seg.size[0]-114-102)
            Y = self.transform_resize(seg)
        else:
            Y = self.transform(seg)
        
        if self.crop_x is True:
            image = transforms.functional.crop(image,36,114,image.size[1]-36-37,image.size[0]-114-102)
            X = self.transform_resize(image)
        else:
            X = self.transform(image)

        return X, Y

# Define loss
def bce_loss(y_real, y_pred):
    y_pred = torch.clamp(y_pred,min=-1e3,max=1e3)
    return torch.mean(y_pred - y_real*y_pred +
                      torch.where(y_pred>37, torch.exp(-y_pred), torch.zeros_like(y_pred)) +
                      torch.where((y_pred<=37) & (y_pred >-18), torch.log1p(torch.exp(-y_pred)), torch.zeros_like(y_pred)) +
                      torch.where((y_pred>-33.3) & (y_pred <=-18),torch.exp(y_pred) - y_pred,torch.zeros_like(y_pred)) + 
                      torch.where(y_pred<=-33.3, -y_pred, torch.zeros_like(y_pred))
                      )


# Train procedure as function
def train(model, opt, loss_fn, epochs, train_loader, test_loader):
    X_test, Y_test = next(iter(test_loader))

    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = model(X_batch)
            loss = loss_fn(Y_batch, Y_pred)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate metrics to show the user
            avg_loss += loss / len(train_loader)
        toc = time()
        print(' - loss: %f' % avg_loss)

        # show intermediate results
        model.eval()  # testing mode
        Y_hat = torch.sigmoid(model(X_test.to(device))).detach().cpu()
        #print('Y hat shape:',Y_hat.shape)
        clear_output(wait=True)
        for k in range(6):
            plt.subplot(2, 6, k+1)
            plt.imshow(np.rollaxis(X_test[k].numpy(), 0, 3), cmap='gray')
            plt.title('Real')
            plt.axis('off')

            plt.subplot(2, 6, k+7)
            plt.imshow(Y_hat[k, 0], cmap='gray')
            plt.title('Output')
            plt.axis('off')
        plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
        plt.show()
        plt.savefig('results/epoch'+str(epoch)+'.png')

# Predict model
def predict(model, data):
    model.eval()  # testing mode
    Y_pred = [F.sigmoid(model(X_batch.to(device))) for X_batch, _ in data]
    return np.array(Y_pred)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(16)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(32)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(64*2, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(64)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(64*2, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(128)  # 32 -> 64
        self.dec_conv3 = nn.Conv2d(64*2, 64, 3, padding=1)
        self.upsample4 = nn.Upsample(256)  # 64 -> 128
        self.dec_conv4 = nn.Conv2d(64*2, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x)))       
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        #pdb.set_trace()
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        d0 = torch.cat([d0,e3],dim=1)
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        d1 = torch.cat([d1,e2],dim=1)
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        d2 = torch.cat([d2,e1],dim=1)
        d3 = F.relu(self.dec_conv3(self.upsample3(d2)))
        d3 = torch.cat([d3,e0],dim=1)
        d4 = self.dec_conv4(self.upsample4(d3))  # no activation
        return d4

# load model and visualize
model = UNet().to(device)
make_dot(model(torch.randn(20, 3, 256, 256).cuda()), params=dict(model.named_parameters()))

# Load training data
train_dataset = torch.load('datasets/train_allstyles-datatype_train.pt')
val_dataset = torch.load('datasets/train_allstyles-datatype_val.pt')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=0)

# Load test data
test_dataset = torch.load('datasets/test_style0-datatype_test.pt')
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

## Train 
torch.cuda.empty_cache()
train(model, optim.Adam(model.parameters()), bce_loss, 20, train_loader, test_loader)
