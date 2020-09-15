from importlib import reload
import resnet; reload(resnet)
from resnet import BasicBlock
from torch import nn


class ftemp(nn.Module):

    def __init__(self, K):
        super(ftemp, self).__init__()

        self.b1 = BasicBlock(3,64, ident=False)
        self.b2 = BasicBlock(64,64)
        self.b3 = BasicBlock(64,64)
        self.b4 = BasicBlock(64,64)

        #phead
        self.ph1 = nn.Conv2d(64,2,kernel_size=1, stride=1)
        self.ph2 = nn.BatchNorm2d(2)
        self.ph3 = nn.ReLU6()
        #reshape
        self.ph4 = nn.Linear(2*K*K, K*K)
        self.ph5 = nn.Softmax(1)
        
        #vhead
        self.vh1 = nn.Conv2d(64,1,kernel_size=1, stride=1)
        self.vh2 = nn.BatchNorm2d(1)
        self.vh3 = nn.ReLU6()
        #reshape
        self.vh4 = nn.Linear(K*K, 64)
        self.vh5 = nn.ReLU6()
        self.vh6 = nn.Linear(64, 1)
        self.vh7 = nn.Tanh()
        
    def forward(self, x):
        B = x.shape[0]
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)

        p = self.ph1(x)
        p = self.ph2(p)
        p = self.ph3(p)
        p = p.view(B,-1)
        p = self.ph4(p)
        p = self.ph5(p)

        v = self.vh1(x)
        v = self.vh2(v)
        v = self.vh3(v)
        v = v.view(B,-1)
        v = self.vh4(v)
        v = self.vh5(v)
        v = self.vh6(v)
        v = self.vh7(v) #-1,1
        v = (v+1)/2 #0,1
        return p,v
