from collections import defaultdict
from random import randint

F = lambda s:s
s = 1

#TODO: How to cache DENOM
#TODO: Need to add parent
class Node:
    ''' Requires mcst.F and mcst.s to be declared if not inputed
    '''
    def __init__(self,P=None):
        if P==None: global s, F; self.P = F(s)
        else: self.P = P

        self.W = defaultdict(int)
        self.Q = defaultdict(int)
        self.N = defaultdict(int)
        self.CHILDREN  = defaultdict(Node)
        self.PARENT = 

    def __getitem__(self, k):
        return self.CHILDREN[k]

    def PI(self, t=1):
        N, PI = self.N, defaultdict(int)
        DENOM = sum( [N[i]**t for i in self.N] )

        for i in self.N: PI[i] = N[i]**t/DENOM

        return PI

    def branch(self):
        N,Q,P = self.N , self.Q, self.P

        NT = sum( [N[i] for i in self.N] )
        bp = lambda i: ( Q[i] + P[i] * NT/(1+N[i]) )

        max_i = randint(0,len(P)-1)
        max_bp = 0
        for i,_ in enumerate(P):
            cur_bp = bp(i)
            if cur_bp > max_bp: max_bp=cur_bp; max_i = i
        return max_i

    def back(self):
        #N +=1
        #W+=1
        #Q = w/N
        pass

    def play(self):
        #sample from PI
        pass






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
