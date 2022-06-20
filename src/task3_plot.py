from cmath import nan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb


# Load the results
style0_pred = pd.read_csv('results/style0_pred.csv',header=None)
style0_true = pd.read_csv('results/style0_true.csv',header=None)
style1_pred = pd.read_csv('results/style1_pred.csv',header=None)
style1_true = pd.read_csv('results/style1_true.csv',header=None)
style2_pred = pd.read_csv('results/style2_pred.csv',header=None)
style2_true = pd.read_csv('results/style2_true.csv',header=None)
iou0 = pd.read_csv('results/style0_iou.csv',header=None)
iou1 = pd.read_csv('results/style1_iou.csv',header=None)
iou2 = pd.read_csv('results/style2_iou.csv',header=None)

diff0 = []
diff1 = []
diff2 = []
c0 = 0
c1 = 0
c2 = 0
for i in range(style0_pred.shape[0]):
    for j in range(style0_pred.shape[1]):
        if style0_pred.iloc[i][j] > style0_true.iloc[i][j]:
            c0 += 1
        if style1_pred.iloc[i][j] > style1_true.iloc[i][j]:
            c1 += 1
        if style2_pred.iloc[i][j] > style2_true.iloc[i][j]:
            c2 += 1
        diff0.append((style0_pred.iloc[i][j] - style0_true.iloc[i][j])/style0_true.iloc[i][j])
        diff1.append((style1_pred.iloc[i][j] - style1_true.iloc[i][j])/style1_true.iloc[i][j])
        diff2.append((style2_pred.iloc[i][j] - style2_true.iloc[i][j])/style2_true.iloc[i][j])

diff = [diff0,diff1,diff2]
ious = pd.concat([iou0,iou1,iou2],axis=1).to_numpy()
diffNew = [x for x in diff if x != nan]

#pdb.set_trace()
fig = plt.subplots(1,figsize=(16,9))
plt.hist(diffNew,alpha=0.5,range=[-20,100])
plt.title('Prediction vs true size',fontsize=30)
plt.xlabel('Difference in size (%) between pred and true',fontsize=24)
plt.ylabel('Occurences',fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(['Style0', 'Style1', 'Style 2'],fontsize=22)

plt.savefig('results/size_difference_val.png')

fig = plt.subplots(1,figsize=(16,9))
plt.hist(diffNew,alpha=0.5,range=[-2,2])
plt.title('Prediction vs true size (Zoom-in)',fontsize=30)
plt.xlabel('Difference in size (%) between pred and true',fontsize=24)
plt.ylabel('Occurences',fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(['Style0', 'Style1', 'Style 2'],fontsize=22)

plt.savefig('results/size_difference_zoom.png')

fig = plt.subplots(1,figsize=(16,9))
plt.hist(ious,alpha=0.5)
plt.title('IoU scores for different styles of training annotations',fontsize=30)
plt.xlabel('IoU scores',fontsize=24)
plt.ylabel('Occurences',fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(['Style0', 'Style1', 'Style 2'],fontsize=22)

plt.savefig('results/iou_plot.png')

print('How many times did prediction style 0 overestimate the annotation? -',c0,' out of ',(style0_pred.shape[0])*style0_pred.shape[1])
print('How many times did prediction style 1 overestimate the annotation? -',c1,' out of ',(style1_pred.shape[0])*style1_pred.shape[1])
print('How many times did prediction style 2 overestimate the annotation? -',c2,' out of ',(style2_pred.shape[0])*style2_pred.shape[1])


