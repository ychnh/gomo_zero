from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
import numpy as np

#D = torchvision.datasets.CIFAR10('cifar')
#D = torchvision.datasets.FashionMNIST('data')

class torchdataset(Dataset):

    def __init__(self, tv_dataset):
        self.dataset = tv_dataset
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        x,y = self.dataset[i]
        x = np.array(x)

        if len(x.shape)==3:
            x = torch.FloatTensor(x).permute(2,0,1)/255
        else:
            x = torch.FloatTensor(x)/255
            #x = torch.stack([x,x,x])

        y = torch.LongTensor([y])
        return {'x':x, 'y':y}

