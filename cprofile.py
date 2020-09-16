
from importlib import reload
import mcst; reload(mcst)
import util; reload(util)
from mcst import Node, State,S
import random

from collections import defaultdict
import torch


def backprop(z, T= 0, save_idx=[]):
    G = []
    
    while(True):
        a = S.node.up_a
        S.undo(a)
        
        won = (z==S.node.p and z!=-1)
        S.node.back(won, a)
        
        if S.i in save_idx: G.append([*S.get_xPI(),z])
            
        if(S.i == T): break
            
    return G

def forward():
    while(True):
        a = S.node.branch()
        S.do(a)
        
        over,z = S.isterminal()
        if(over): break
            
    return z

import numpy as np

def ply(nsave=0):
    T = S.i
    z = forward()

    B = S.i
    save_idx = random.sample(range(T,B),nsave)
    G = backprop(z, T, save_idx)
    return G


import mcst
#from mcst import S, model, State
from prgs import prgs
import ftemp; reload(ftemp)
from ftemp import ftemp

with torch.no_grad():
    mcst.model = ftemp(K=3).cuda()
    S = State(K=3)
    G = []
    for i in prgs( range(50000) ): G+=ply(0)
    #for i in range(20000): G+=ply(0)
    #trn_dataloader = DataLoader(G, batch_size=64, shuffle=True)
    #b = iter(trn_dataloader).next()
    #mcst.model()

# mcst search ply() need 2 sep trees

