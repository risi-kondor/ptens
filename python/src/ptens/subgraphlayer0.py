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


#import ptens.ptensorsc_base as ptensorsc_base
#import ptens.ptensor0c as ptensor0c
import ptens.ptensorlayer0 as ptensorlayer0


class subgraphlayer0(p.subgraphlayer,ptensorlayer0):


    def __new__(cls,G,S,atoms,M):
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(G,p.ggraph)
        assert isinstance(S,p.subgraph)
        assert M.size(0)==atoms.nrows0()
        R=super().__new__(subgraphlayer0,M)
        R.atoms=atoms
        R.G=G
        R.S=S
        return R

    @classmethod
    def zeros(self,G,S,nc,device='cpu'):
        atoms=G.subgraphs(S)
        M=torch.zeros([atoms.nrows0(),nc],device=device)
        return subgraphlayer0(G,S,atoms,M)

    @classmethod
    def randn(self,G,S,nc,device='cpu'):
        atoms=G.subgraphs(S)
        M=torch.randn([atoms.nrows0(),nc],device=device)
        return subgraphlayer0(G,S,atoms,M)

    @classmethod
    def from_matrix(self,G,S,M):
        atoms=G.subgraphs(S)
        return subgraphlayer0(G,S,atoms,M)

    @classmethod
    def from_matrixA(self,G,S,atoms,M):
        return subgraphlayer0(G,S,atoms,M)

    
    # ---- Linmaps ------------------------------------------------------------------------------------------


    @classmethod
    def linmaps(self,x):
        return subgraphlayer0(x.G,x.S,x.atoms,super().linmaps(x))


    # ---- Gather ------------------------------------------------------------------------------------------


    @classmethod
    def gather(self,S,x):
        atoms=x.G.subgraphs(S)
        return subgraphlayer0(x.G,x.S,atoms,super().gather(atoms,x))


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "subgraphlayer0(len="+str(self.size(0))+",nc="+str(self.size(1))+")"

    def __str__(self,indent=""):
        r=indent+"subgraphlayer0:\n"
        for i in range(len(self)):
            r=r+self[i].to_string(indent+"  ")+""
        return r




