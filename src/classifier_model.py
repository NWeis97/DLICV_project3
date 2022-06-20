# Import 
import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.models as models

import pdb

# settings
torch.manual_seed(1234)
with_augs = False
lr = 0.0001
with_norm = False
num_epochs = 100


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


# Get CUDA
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


out_dict = {'train_acc': [],
              'val_acc': [],
              'train_loss': [],
              'val_loss': []}

# Load training data
dataset = torch.load('datasets/classifier_dataset.pt')
train_len = int(dataset.__len__()*0.7)
train_dataset, val_dataset = torch.utils.data.random_split(dataset,[train_len,dataset.__len__()-train_len], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0)


# Get model
model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.classifier[6].in_features

# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.classifier[6] = nn.Linear(num_ftrs, 2)
# Set optimizer and model
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

def loss_fun(output, target):
    crit = nn.CrossEntropyLoss()
    return crit(output, target)

out_dict = {'train_acc': [],
              'test_acc': [],
              'train_loss': [],
              'test_loss': []}

for epoch in tqdm(range(num_epochs), unit='epoch'):
    #For each epoch
    train_correct = 0
    train_loss = []
    model.train()
    for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        #Zero the gradients computed for each weight
        optimizer.zero_grad()
        #Forward pass your image through the network
        output = model(data)
        #output = torch.sigmoid(output[:,0])
        #Compute the loss
        loss = loss_fun(output, target)
    
        #Backward pass through the network
        loss.backward()
        #Update the weights
        optimizer.step()
        
        train_loss.append(loss.cpu().item())
        #Compute how many were correctly classified
        predicted = output.argmax(dim=1)

        train_correct += (target==predicted).sum().cpu().item()
    #Comput the test accuracy
    model.eval()
    test_loss = []
    test_correct = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        #output = torch.sigmoid(output[:,0])
        test_loss.append(loss_fun(output, target).cpu().item())
        predicted = output.argmax(dim=1)
        test_correct += (target==predicted).sum().cpu().item()
    train_acc = train_correct/len(train_dataset)
    test_acc = test_correct/len(val_dataset)

    out_dict['train_acc'].append(train_correct/len(train_dataset))
    out_dict['test_acc'].append(test_correct/len(val_dataset))
    out_dict['train_loss'].append(np.mean(train_loss))
    out_dict['test_loss'].append(np.mean(test_loss))
    print("Accuracy train: {train:.1f}%\t test: {test:.1f}%".format(test=100*test_acc, train=100*train_acc))

# Save model
name = "ClassifierModel"
torch.save(model.state_dict(), os.getcwd()+"/models/"+name+".pt")

# Make plot of training curves
fig, ax = plt.subplots(1,2,figsize=(15,7))
ax[0].plot(out_dict['train_acc'])
ax[0].plot(out_dict['test_acc'])
ax[0].legend(('Train accuracy','Test accuracy'))
ax[0].set_xlabel('Epoch number')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Training and test accuracy')
ax[1].plot(out_dict['train_loss'])
ax[1].plot(out_dict['test_loss'])
ax[1].legend(('Train loss','Test loss'))
ax[1].set_xlabel('Epoch number')
ax[1].set_ylabel('Loss')
ax[1].set_title('Training and test loss')
fig.savefig(os.getcwd()+'/figures/training_curve_'+name+'.png')