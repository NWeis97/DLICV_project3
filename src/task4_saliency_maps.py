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
from torchviz import make_dot
import torch.optim as optim

import pdb


# ISIC dataloader
class ISIC_task4(torch.utils.data.Dataset):
    def __init__(self, transform, data_path):
        'Initialization'
        self.transform_resize = transform
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


# Get CUDA
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define loss
def bce_loss(y_real, y_pred):
    y_pred = torch.clamp(y_pred,min=-80,max=80)
    return torch.mean(y_pred - y_real*y_pred +
                      torch.where(y_pred>37, torch.exp(-y_pred), torch.zeros_like(y_pred)) +
                      torch.where((y_pred<=37) & (y_pred >-18), torch.log1p(torch.exp(-y_pred)), torch.zeros_like(y_pred)) +
                      torch.where((y_pred>-33.3) & (y_pred <=-18),torch.exp(y_pred) - y_pred,torch.zeros_like(y_pred)) + 
                      torch.where(y_pred<=-33.3, -y_pred, torch.zeros_like(y_pred))
                      )


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

def IoU(a,b):
    a_tot = torch.sum(a)
    b_tot = torch.sum(b)
    I_tot = torch.sum((a==b) & (a==1))
    U_tot = a_tot+b_tot-I_tot

    return I_tot/U_tot


# load model and visualize
model = UNet().to(device)
opt = optim.Adam(model.parameters(), lr=0.00002)
make_dot(model(torch.randn(20, 3, 256, 256).cuda()), params=dict(model.named_parameters()))


# Load training data
dataset = torch.load('datasets/task4_dataset.pt')
train_len = int(dataset.__len__()*0.7)
batch_size = 8
train_dataset, val_dataset = torch.utils.data.random_split(dataset,[train_len,dataset.__len__()-train_len], generator=torch.Generator().manual_seed(42))
train_dataset = torch.utils.data.ConcatDataset([train_dataset,val_dataset],)

# Clean data
remove_indx = [x for x in np.arange(0,39,1) if x not in [1,5,10,13,20,27,34,37]]
train_dataset = torch.utils.data.Subset(train_dataset, remove_indx)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Load saliency maps
train_saliency_init = np.load("models/train_saliency_maps.npy")
train_saliency_init = torch.Tensor(1.0*(train_saliency_init>0.5))
val_saliency_init = np.load("models/val_saliency_maps.npy")
val_saliency_init = torch.Tensor(1.0*(val_saliency_init>0.5))

train_saliency_init = torch.cat((train_saliency_init,val_saliency_init),dim=0)

#Clean data 2.0
train_saliency_init = train_saliency_init[remove_indx,:,:]

train_target_enum = train_saliency_init.clone()
#val_target_enum = val_saliency_init.clone()




transform = transforms.ToPILImage()
epochs_per_round = 5
rounds = 100

# settings
torch.manual_seed(1234)
import random
random.seed(1234)
np.random.seed(1234)

moved_on_from_init = torch.zeros(train_dataset.__len__(),)
losses = []

for f in os.listdir("images/progress_task4/train0/"):
    os.remove(os.path.join("images/progress_task4/train0/", f))
for f in os.listdir("images/progress_task4/train1/"):
    os.remove(os.path.join("images/progress_task4/train1/", f))
for f in os.listdir("images/progress_task4/train2/"):
    os.remove(os.path.join("images/progress_task4/train2/", f))
for f in os.listdir("images/progress_task4/train3/"):
    os.remove(os.path.join("images/progress_task4/train3/", f))
for f in os.listdir("images/progress_task4/train4/"):
    os.remove(os.path.join("images/progress_task4/train4/", f))

for round in range(rounds):
    if round != 0:
        print("Stuff")

    print('* Epoch %d/%d' % (round+1, rounds))

    model.train()  # train mode
    for epoch in range(epochs_per_round):
        print('* Epoch %d/%d' % (epoch+1, epochs_per_round))

        avg_loss = 0
        for minibatch_no, (data, seg_init, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, seg_init, target = data.to(device), seg_init.to(device), target.to(device)
            train_target = train_target_enum[minibatch_no*batch_size:(minibatch_no+1)*batch_size,:,:].to(device)

            
            img_pil = transform(data[1,:,:,:].cpu())
            img_tar = transform(train_target[1,:,:].cpu())
            img_pil.save('image.png')
            img_tar.save('image_tar.png')

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = model(data)
            loss = bce_loss(train_target, Y_pred)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate metrics to show the user
            avg_loss += loss / len(train_loader)
        
        print(' - loss: %f' % avg_loss)
        losses.append(avg_loss.cpu().item())


    model.eval()  # train mode
    for minibatch_no, (data, seg_init, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data = data.to(device)
        train_target = train_target_enum[minibatch_no*batch_size:(minibatch_no+1)*batch_size,:,:]
        
        # forward
        with torch.no_grad():
            Y_pred = model(data)    

        # Go through step 1 and 2 of box recursive model
        for i in range(Y_pred.shape[0]):
            # Get target
            Y_tar = train_target[i,:,:]
            # Get new target
            Y_hat = torch.sigmoid(Y_pred[i,0,:,:]).cpu()
            Y_hat = torch.round(Y_hat)

            # Remove all pres outside "box" (step 1)
            Y_hat[Y_tar == 0] = 0            

            # Check IoU of "box" and new target (step 2)
            IoU_val_init = IoU(Y_hat,train_saliency_init[minibatch_no*batch_size + i,:,:])
            IoU_val_tar = IoU(Y_hat,Y_tar)

            # Progress good
            # 0:0
            # 1:2
            # 2:3
            # 3:4
            # 4:6
            if minibatch_no*batch_size+i < 5:
                if round == 0:
                    img_pil_init = transform(Y_tar)
                    img_pil_init.save(f"images/progress_task4/train{minibatch_no*batch_size+i}/init_target.png")
                img_pil_update = transform(Y_hat)
                if (IoU_val_init < 0.25) | ((IoU_val_tar < 0.75) & (moved_on_from_init[minibatch_no*batch_size + i] == 1)):
                    img_pil_update.save(f"images/progress_task4/train{minibatch_no*batch_size+i}/round_{round}-accept_False.png")
                else:
                    img_pil_update.save(f"images/progress_task4/train{minibatch_no*batch_size+i}/round_{round}-accept_True.png")

            # Reset target if IoU is below 0.4 else update target
            reset_target = 0
            if (IoU_val_init < 0.25):
                print('reseting to init')
                moved_on_from_init[minibatch_no*batch_size + i] = 0
                train_target_enum[minibatch_no*batch_size + i,:,:] = train_saliency_init[minibatch_no*batch_size + i,:,:]
                reset_target = 1
            elif (IoU_val_tar < 0.75) & (moved_on_from_init[minibatch_no*batch_size + i] == 1):
                print('reseting to prev')
                train_target_enum[minibatch_no*batch_size + i,:,:] = Y_tar
            else:
                print('keeping')
                moved_on_from_init[minibatch_no*batch_size + i] = 1
                train_target_enum[minibatch_no*batch_size + i,:,:] = Y_hat

            if reset_target == 0:
                new_target = train_target_enum[minibatch_no*batch_size + i,:,:]
                assert torch.sum(Y_tar[new_target==1]==0) == 0
            
# Save model
name = "SaliencyModel_UNET"
torch.save(model.state_dict(), os.getcwd()+"/models/"+name+".pt")

# Make plot of training curves
fig, ax = plt.subplots(1,1,figsize=(7,7))
ax.plot(np.arange(1,len(losses)+1,1),losses)
ax.legend(('Train loss'))
ax.set_xlabel('Epoch number')
ax.set_ylabel('Loss')
ax.set_title('Training loss')
ax2 = ax.twiny()
new_tick_locations = np.array(np.arange(1,len(losses)+1,20))

def tick_function(X):
    V = np.floor(X/20)*4
    return [int(z) for z in V]

ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(tick_function(new_tick_locations))
ax2.set_xlabel(r"Round number")

fig.savefig(os.getcwd()+'/figures/training_curve_'+name+'.png')



def specificity_sensiticity(pred,true):
    sensi = torch.sum(pred[true==1]==1)/(torch.sum(pred[true==1]==1)+torch.sum(pred[true==1]==0))
    speci = torch.sum(pred[true==0]==0)/(torch.sum(pred[true==0]==0)+torch.sum(pred[true==0]==1))
    return sensi, speci

sensitivity = []
specificity = []
IoU_vals = []
model.eval()  # train mode
for minibatch_no, (data, seg_init, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
    data = data.to(device)
    train_target = train_target_enum[minibatch_no*batch_size:(minibatch_no+1)*batch_size,:,:]
    
    # forward
    with torch.no_grad():
        Y_pred = model(data)    

    # Go through step 1 and 2 of box recursive model
    for i in range(Y_pred.shape[0]):
        # Get target
        Y_tar = train_target[i,:,:]
        # Get new target
        Y_hat = torch.sigmoid(Y_pred[i,0,:,:]).cpu()
        Y_hat = torch.round(Y_hat)       

        # Check IoU of "box" and new target (step 2)
        Y_hat = train_target_enum[minibatch_no*batch_size + i,:,:]
        IoU_val_true = IoU(Y_hat,target[i,0,:,:])
        sensi, speci = specificity_sensiticity(Y_hat,target[i,0,:,:])
        sensitivity.append(sensi)
        specificity.append(speci)
        IoU_vals.append(IoU_val_true)


sensitivity = np.array(sensitivity)
specificity = np.array(specificity)
IoU_vals = np.array(IoU_vals)
print(f'Sensitivity average: {np.mean(sensitivity):.3f}')
print(f'Specificity average: {np.mean(specificity):.3f}')
print(f'IoU_vals average: {np.mean(IoU_vals):.3f}')

#Sensitivity average: 0.674
#Specificity average: 0.931
#IoU_vals average: 0.500"