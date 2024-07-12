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

import ptens.ptensorlayer1 as ptensorlayer1


class subgraphlayer1(p.subgraphlayer,ptensorlayer1):


    def __new__(cls,G,S,atoms,M):
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(G,p.ggraph)
        assert isinstance(S,p.subgraph)
        R=super().__new__(subgraphlayer1,M)
        R.atoms=atoms
        R.G=G
        R.S=S
        return R

    @classmethod
    def zeros(self,G,S,nc,device='cpu'):
        atoms=G.subgraphs(S)
        M=torch.zeros([atoms.nrows1(),nc],device=device)
        return subgraphlayer1(G,S,atoms,M)

    @classmethod
    def randn(self,G,S,nc,device='cpu'):
        atoms=G.subgraphs(S)
        M=torch.randn([atoms.nrows1(),nc],device=device)
        return subgraphlayer1(G,S,atoms,M)

    @classmethod
    def from_matrix(self,G,S,M):
        atoms=G.subgraphs(S)
        return subgraphlayer1(G,S,atoms,M)

    
    # ---- Linmaps ------------------------------------------------------------------------------------------


    @classmethod
    def linmaps(self,x):
        return subgraphlayer1(x.G,x.S,x.atoms,super().linmaps(x))


    # ---- Gather ------------------------------------------------------------------------------------------


    @classmethod
    def gather(self,S,x):
        atoms=x.G.subgraphs(S)
        return subgraphlayer1(x.G,x.S,atoms,super().gather(atoms,x))


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "subgraphlayer1(len="+str(self.size(0))+",nc="+str(self.get_nc())+")"

    def __str__(self):
        r="subgraphlayer1:\n"
        for i in range(len(self)):
            r=r+self[i].to_string("  ")+""
        return r





