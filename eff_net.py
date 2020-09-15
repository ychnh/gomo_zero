import torch
import torch.nn as nn
import torch.nn.functional as F


#MobileNetV2: Inverted Residuals and Linear Bottlenecks
#https://arxiv.org/pdf/1801.04381.pdf
class MBblock(nn.Module):

    def __init__(self, in_ch, out_ch, strd=1, t=1, sqzex=False, residual=True):
        super(MBblock, self).__init__()
        self.strd = strd
        self.residual = residual

        self.widen = nn.Conv2d(in_ch, t*in_ch, kernel_size=1, stride=1)
        self.BN1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU6()
        self.dconv1 = nn.Conv2d(t*in_ch, t*in_ch, kernel_size=(1,3), stride=strd, padding=(0,1))
        self.dconv2 = nn.Conv2d(t*in_ch, t*in_ch, kernel_size=(3,1), stride=1, padding=(1,0))

        self.BN2 = nn.BatchNorm2d(in_ch)
        self.relu2 = nn.ReLU6()
        self.compress = nn.Conv2d(t*in_ch, out_ch, kernel_size=1)

        if sqzex: self.sqzex = SQZEX(out_ch)
        else: self.sqzex = None

    def forward(self, x):

        add = x

        x = self.widen(x)
        x = self.BN1(x)
        x = self.relu1(x)

        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.BN2(x)
        x = self.relu2(x)

        x = self.compress(x)
        #if isinstance(self.sqzex, type(None)): x = self.sqzex(x)
        if self.sqzex!=None: x = self.sqzex(x)

        if self.strd == 1 and self.residual: x = x + add

        return x

#Squeeze-and-Excitation Networks
#https://arxiv.org/pdf/1709.01507.pdf
class SQZEX(nn.Module):

    def __init__(self, in_ch, r=1):
        super(SQZEX, self).__init__()
        self.FC1 = nn.Linear(in_ch, in_ch//r)
        self.relu1 = nn.ReLU6()
        self.FC2 = nn.Linear(in_ch//r, in_ch)

    def forward(self, x):
        w = torch.mean(x, dim=[2,3])
        w = self.FC1(w)
        w = self.relu1(w)
        w = self.FC2(w)
        w = torch.sigmoid(w)
        w = w.unsqueeze(-1).unsqueeze(-1)
        return w*x
