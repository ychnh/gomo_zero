from eff_net import MBblock

class cifar_effnet(nn.Module):
    def __init__(self):
        super(cifar_effnet, self).__init__()
        sqzex = True
        self.b1 = MBblock(3,256,sqzex=sqzex, residual=False)
        B = nn.ModuleList()
        for i in range(3):
            B.append( MBblock(256,256, sqzex=sqzex))
            B.append( MBblock(256,256, strd=2,sqzex=sqzex,residual=False))
        self.B = B
        
    def forward(self,x):
        x = self.b1(x)
        for b in self.B:
            x = b(x)
            print(x.shape)
        return x

x = D[0]['x'].unsqueeze(0)
model = cifar_effnet()
model.eval()
model(x).shape
    
