from collections import defaultdict
from random import randint, randrange,shuffle
import numpy as np


#F = lambda s:[1/9]*9

model = None
S = None

def F(S): 
    global model
    model.eval()
    with torch.no_grad():
        x = S.get_x().float().unsqueeze(0).cuda()
        P,v = model(x)
    P = P.squeeze(0)


    for a,p in enumerate(P): 
        if not S.A[a]: P[a] = 0
    return P.squeeze(0).cpu().detach()
    #return S.unif_P()

    
    #return S.unif_P()

#S = None
#i = 0
#N,a = None, None
#R = defaultdict(lambda:True)

import torch
from util import Xidx_rc, Xrc_idx, any3, any5, any4

class State:

    def __init__(self, K):
        KK = K**2
        self.K, self.KK = K, KK
        self.unif_p2i= [1/(KK-i) for i in range(KK)]

        self._reset()
        self.node = None
        self.root = Node(self)
        self.node = self.root

    def undo(self, a):
        K = self.K
        self.M[Xrc_idx(a,K)] = 0
        self.A[a] = True
        self.i -= 1
        self.node = self.node.up

    def do(self,a):
        assert self.A[a] == True
        K = self.K
        p = self.i%2
        self.M[Xrc_idx(a,K)] = p+1
        self.A[a] = False
        self.lm = a
        self.i += 1
        self.node = self.node[a]

    def unif_P(self):
        KK,K,A = self.KK, self.K, self.A
        if self.i < len(self.unif_p2i): 
            p = self.unif_p2i[self.i]
            x = [p if A[a] else 0 for a in range(KK)]
            return x
        else: 
            return [0]*(KK)

    def _reset(self):
        K = self.K
        self.M = torch.ByteTensor(K*[K*[0]])
        self.A = defaultdict(lambda:True)
        self.lm = None
        self.i = 0

    def reset(self):
        self._reset()
        self.node = self.root

    def isterminal(self):
        M,lm,K = self.M, self.lm, self.K

        p = (self.i-1)%2

        isTerminal, z = False, None
        if True in any3(M, Xrc_idx(lm,K)): isTerminal,z = True,p
        elif self.i >= len(self): isTerminal,z = True,-1

        if isTerminal: self.node.setterminal()
        return isTerminal, z 

    def get_xPI(self):
        PI = torch.FloatTensor( self.node.PI() )
        x = self.get_x()
        return x,PI

    def get_x(self):
        M = self.M
        p1,p2 = (M==1), (M==2)

        plyr = self.node.p if self.node!=None else 0
        p = torch.full(p1.shape, plyr).bool()
        x = torch.stack([p1,p2,p])
        return x



    def __len__(self):
        return self.KK

S = None

#TODO: How to cache DENOM/More efficient ways of computing PI
#TODO: play(det=): PI incompatible because it is dict
# move away from dicts? pros vs cons?
#TODO: Check if forward/back works from arbi node
class Node:

    def __init__(self,state=None):
        global F
        if state==None: global S
        else: S=state

        self.P = F(S)
        self.W = defaultdict(int)
        self.Q = defaultdict(int)
        self.N = defaultdict(int)

        self.down = defaultdict(Node)
        self.up = S.node
        self.up_a = S.lm
        self.p = S.i%2

    def __getitem__(self, k):
        return self.down[k]

    def PI(self, t=1):
        N, P = self.N, self.P
        Nexpt = [N[i]**t for i in range(len(P))]
        DENOM = sum( Nexpt )
        PI = [ Nexpt[i]/DENOM for i in range(len(P)) ]
        return PI

    def branch(self):
        N,Q,P = self.N , self.Q, self.P

        NT = sum( [N[i] for i in self.N] )
        if NT==0: NT=1 #Initial start? TODO: validate and remove
        bp = lambda i: ( Q[i] + P[i] * NT/(1+N[i]) )

        max_i = randint(0,len(P)-1)
        max_bp = 0


        IDX = list(range(len(P))) 
        shuffle(IDX)
        for i in IDX:
            cur_bp = bp(i)
            if cur_bp > max_bp: max_bp=cur_bp; max_i = i
        return max_i

    def back(self, w, a):
        self.N[a] += 1
        self.W[a] += w
        self.Q[a] = self.W[a]/self.N[a]
        return self.up

    def play(self, det=False):
        PI = self.PI()
        if det:
            return np.argmax(PI)
        else:
            return np.random.choice(len(PI), p=PI)

    def setterminal(self):
        self.W = None
        self.Q = None
        self.N = None
        self.P = None
        self.down = None

    def isterminal(self): return self.down == None









# Structure to instantiate and build 
# (s,pi,_z)
#s = [B1,B2,P]
#pi = 
#_z = pov(z)

#i = 0,1 #i%2


# keep only 1 instance of state s
# sample prior to back to add to database
# during back undo a(i) from s(i) to get s(i-1)
    # more efficient way for W/N?
    # common var to keep track of sigma(N(b))
