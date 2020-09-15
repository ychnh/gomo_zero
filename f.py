from eff_net import MBblock
import torch
import torch.nn as nn
import torch.nn.functional as F


class f_gm(nn.Module):

    def __init__(self, K):
        super(f_gm, self).__init__()

        self.K = K
        self.backbone = backbone()
        #19xblock tower


        #phead
        self.ph1 = nn.Conv2d(256,2,kernel_size=1, stride=1)
        self.ph2 = nn.BatchNorm2d(2)
        self.ph3 = nn.ReLU6()
        #reshape
        self.ph4 = nn.Linear(2*K*K, K*K)
        self.ph5 = nn.Softmax(1)

        #vhead
        self.vh1 = nn.Conv2d(256,1,kernel_size=1, stride=1)
        self.vh2 = nn.BatchNorm2d(1)
        self.vh3 = nn.ReLU6()
        #reshape
        self.vh4 = nn.Linear(K*K, 256)
        self.vh5 = nn.ReLU6()
        self.vh6 = nn.Linear(256, 1)
        self.vh7 = nn.Tanh()

    def forward(self, x):
        B = x.shape[0]
        x = self.backbone(x)

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
        v = self.vh7(v)

        return p,v

#Input x,y,p(0,1)
class backbone(nn.Module):
    def __init__(self):
        super(backbone, self).__init__()
        sqzex = False
        self.b1 = MBblock(3,256,sqzex=sqzex, residual=False)
        self.B = nn.ModuleList()
        for i in range(3):
            self.B.append( MBblock(256,256,sqzex=sqzex) )

    def forward(self,x):
        x = self.b1(x)
        for b in self.B:
            x = b(x)

        return x


