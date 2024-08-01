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


class batched_subgraphlayer0(p.subgraphlayer,p.batched_ptensorlayer0):


    def __new__(cls,G,S,atoms,M):
        assert isinstance(atoms,pb.batched_atomspack)
        assert isinstance(G,p.batched_ggraph)
        assert isinstance(S,p.subgraph)
        R=super().__new__(batched_subgraphlayer0,M)
        R.atoms=atoms
        R.G=G
        R.S=S
        return R

    @classmethod
    def zeros(self,G,S,nc,device='cpu'):
        atoms=G.subgraphs(S)
        M=torch.zeros([atoms.nrows0(),nc],device=device)
        return batched_subgraphlayer0(G,S,atoms,M)

    @classmethod
    def randn(self,G,S,nc,device='cpu'):
        atoms=G.subgraphs(S)
        M=torch.randn([atoms.nrows0(),nc],device=device)
        return batched_subgraphlayer0(G,S,atoms,M)

    @classmethod
    def from_ptensorlayers(self,list):
        for a in list:
            assert isinstance(a,p.ptensorlayer0)
        atoms=pb.batched_atomspack([a.atoms for a in list])
        M=torch.cat(list,0)
        return batched_subgraphlayer0(G,S,atoms,M)

    @classmethod
    def from_matrix(self,G,S,M):
        assert isinstance(G,p.batched_ggraph)
        assert isinstance(S,p.subgraph)
        assert isinstance(M,torch.Tensor)
        atoms=G.subgraphs(S)
        assert M.dim()==2
        assert M.size(0)==atoms.nrows0()
        return batched_subgraphlayer0(G,S,atoms,M)

    
    # ----- Access -------------------------------------------------------------------------------------------


    def __getitem__(self,i):
        assert i<len(self)
        offs=self.atoms.offset0(i)
        n=self.atoms.nrows0(i)
        M=torch.Tensor(self)[offs:offs+n,:]
        return p.subgraphlayer0.from_matrixA(self.G[i],self.S,self.atoms[i],M)


    # ---- Linmaps ------------------------------------------------------------------------------------------


    @classmethod
    def linmaps(self,x):
        return batched_subgraphlayer0(x.G,x.S,x.atoms,super().linmaps(x))


    # ---- Gather ------------------------------------------------------------------------------------------


    @classmethod
    def gather(self,S,x):
        atoms=x.G.subgraphs(S)
        return batched_subgraphlayer0(x.G,x.S,atoms,super().gather(atoms,x))


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "batched_subgraphlayer0(size="+str(len(self))+",nc="+str(self.size(1))+")"

    def __str__(self,indent=""):
        r=indent+self.__repr__()+":\n"
        for i in range(len(self)):
            r=r+self[i].__str__("  ")+"\n"
        return r




