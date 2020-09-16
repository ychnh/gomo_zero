import re 
import numpy as np

i = lambda b:b
r1 = lambda b: b.rot90(1,(2,3))
r2 = lambda b: b.rot90(2,(2,3))
r3 = lambda b: b.rot90(3,(2,3))
s = lambda b: b.flip(3)
sr1 = lambda b: s(r1(b))
sr2 = lambda b: s(r2(b))
sr3 = lambda b: s(r3(b))
# Inverses for convient naming
r1s = lambda b: r1(s(b))
r2s = lambda b: r2(s(b))
r3s = lambda b: r3(s(b))
transforms =     [i,r1,r2,r3,s,sr1,sr2,sr3]
inv_transforms = [i,r3,r2,r1,s,r3s,r2s,r1s]




alphabet = 'abcdefghijklmno'
DCTnum_alph = dict(zip(alphabet, np.arange(15)))
DCTnum_plyr = {'B':1,'W':2}

#G = np.load('records.npy', allow_pickle=True)
def process(M):
    M = M.decode("utf-8")
    M = [ m[-5:-4]+m[-3:-1] for m in M.split(';')]
    M = [ (DCTnum_plyr[p], DCTnum_alph[a], DCTnum_alph[b]) for p,a,b in M]
    return M

def Xrc_idx(idx, K=3):
    r,c = idx//K, idx%K
    return r,c
def Xidx_rc(r,c, K=3):
    return K*r+c

#G = [process(g) for g in G]

class bclr:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def linear_nhd(M,r,c,R):
    K = R+1
    LOBJ = []
    p = M[r,c]
    R,C = M.shape
    l0 = lambda r,c,i: (r,c+i)
    l1 = lambda r,c,i: (r+i,c+i)
    l2 = lambda r,c,i: (r+i,c)
    l3 = lambda r,c,i: (r-i,c+i)
    L = [l0,l1,l2,l3]
    for l in L:
        onboard = False
        lobj = []
        for i in range( -(K-1),(K) ):
            x,y = l(r,c,i)
            wthin = within(R,C,x,y)

            if not wthin:
                if onboard: break
                elif not onboard: pass

            elif wthin:
                if not onboard:
                    onboard = True
                lobj.append(M[x,y])

        # Add line
        if K<=len(lobj): LOBJ.append(lobj)
        #LOBJ.append(lobj)

    return LOBJ

def within(R,C,r,c,): return 0<=r<R and 0<=c<C
def cont_count(V,TV, count):
    if V == TV:
        return count + 1
    else:
        return 0

'''
def find_win(M,lst_m, L=3):
    R = L-1
    r,c = lst_m
    p = M[r,c].item()
    #rule_3 = re.compile(p+p+p)
    lines = linear_nhd(M,r,c,R=R)
    for li in lines:
        for j in range(len(li)-L+1):
            if li[j]==p and li[j+1]==p and li[j+2]==p: return True
    return False
    #lines = [ [str(x.item()) for x in li] for li in lines]
    #lines = [ ''.join(li) for li in lines]
    #return [ rule_3.search(li)!=None for li in lines]
'''


RULE_3 = {1:re.compile('111'), 2:re.compile('222')}
def find_win(M,lst_m, L=None):
    r,c = lst_m
    #p = str(M[r,c].item())
    #rule_3 = re.compile(p+p+p)
    p = M[r,c]
    rule_3 = RULE_3[p]
    lines = linear_nhd(M,r,c,R=2)
    lines = [ ''.join([str(x) for x in li]) for li in lines]
    #lines = [ ''.join(li) for li in lines]
    #return [ rule_3.search(li)!=None for li in lines]
     #for li in lines]
    for li in lines: 
        if rule_3.search(li)!=None: return True
    return  False


def any4(M,lst_m):
    r,c = lst_m
    p = str(M[r,c])
    rule_4 = re.compile(p+p+p+p)
    lines = linear_nhd(M,r,c,R=3)
    lines = [ [str(x) for x in li] for li in lines]
    lines = [ ''.join(li) for li in lines]
    return [ rule_4.search(li)!=None for li in lines]

def any5(M,lst_m):
    r,c = lst_m
    p = str(M[r,c].item())
    rule_5 = re.compile(p+p+p+p+p)
    lines = linear_nhd(M,r,c,R=4)
    lines = [ [str(x.item()) for x in li] for li in lines]
    lines = [ ''.join(li) for li in lines]
    return [ rule_5.search(li)!=None for li in lines]



import string
def getM_lastm(g,idx):
    M = np.zeros((15,15)).astype(np.uint8)
    for k, (p,i,j) in enumerate(g):
        M[i,j] = p
        if idx == k:
            break
    return M, (i,j)

def draw(M, HEAD=False, EM=[]):
    key = {0:'.', 1:bclr.OKBLUE+'O'+bclr.ENDC, 2:bclr.FAIL+'X'+bclr.ENDC}
    R,C = M.shape
    out = []
    H = (string.digits + string.ascii_uppercase)[:R]
    if HEAD: out.append( ' '.join('+'+H) )
    for r in range(R):
        outstr = ''
        if HEAD: outstr+= H[r]
        for c in range(C):
            x = M[r,c]
            if (r,c) in EM: outstr += ' '+bclr.BOLD+bclr.UNDERLINE + key[x] + bclr.ENDC
            else: outstr += ' '+key[x]
        out.append(outstr)
    return '\n'.join(out)


# DISPLAY UTIL
import torch
import numpy as np


def inflate_size(size):
    if isinstance(size, int): return (size,size)
    else: return size
    
def imshow(im, size=10):
    size= inflate_size(size)
    from matplotlib import pyplot as plt
    plt.figure(figsize=size)
    plt.imshow(im)

def visualize(g):
    x,PI,z = g
    
    M = x[0]+x[1]*2
    P = PI.reshape(M.shape)
    p = x[2][0,0]
    M,P,p = M.numpy(),(P.numpy()*100).round(1),p.item()
    print(M)
    print('player ',p+1)
    print(P)
# DATA
