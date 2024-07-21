#
# This file is part of ptens, a C++/CUDA library for permutation 
# equivariant message passing. 
#  
# Copyright (c) 2023, Imre Risi Kondor
#
# This source code file is subject to the terms of the noncommercial 
# license distributed with cnine in the file LICENSE.TXT. Commercial 
# use is prohibited. All redistributed versions of this file (in 
# original or modified form) must retain this copyright notice and 
# must be accompanied by a verbatim copy of the license. 
#
#

import torch

import ptens as p
import ptens_base as pb 


class subgraphlayer2(p.subgraphlayer,p.ptensorlayer2):

    def __new__(cls,G,S,atoms,M):
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(G,p.ggraph)
        assert isinstance(S,p.subgraph)
        assert M.size(0)==atoms.nrows2()
        R=super().__new__(subgraphlayer2,M)
        R.atoms=atoms
        R.G=G
        R.S=S
        return R

    @classmethod
    def zeros(self,G,S,nc,device='cpu'):
        atoms=G.subgraphs(S)
        M=torch.zeros([atoms.nrows2(),nc],device=device)
        return subgraphlayer2(G,S,atoms,M)

    @classmethod
    def randn(self,G,S,nc,device='cpu'):
        atoms=G.subgraphs(S)
        M=torch.randn([atoms.nrows2(),nc],device=device)
        return subgraphlayer2(G,S,atoms,M)

    @classmethod
    def from_matrix(self,G,S,M):
        atoms=G.subgraphs(S)
        return subgraphlayer2(G,S,atoms,M)

    @classmethod
    def from_matrixA(self,G,S,atoms,M):
        return subgraphlayer2(G,S,atoms,M)

    
    # ---- Linmaps ------------------------------------------------------------------------------------------


    @classmethod
    def linmaps(self,x):
        return subgraphlayer2(x.G,x.S,x.atoms,super().linmaps(x))


    # ---- Gather ------------------------------------------------------------------------------------------


    @classmethod
    def gather(self,S,x):
        atoms=x.G.subgraphs(S)
        return subgraphlayer2(x.G,x.S,atoms,super().gather(atoms,x))


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "subgraphlayer2(len="+str(self.size(0))+",nc="+str(self.get_nc())+")"

    def __str__(self,indent=""):
        r=indent+"subgraphlayer2:\n"
        for i in range(len(self)):
            r=r+self[i].to_string(indent+"  ")+""
        return r













    # ---- Operations ----------------------------------------------------------------------------------------


#     def __add__(self,y):
#         assert self.size==y.size
#         assert self.atoms==y.atoms
#         r=self.clone()
#         r+=y
#         return r


