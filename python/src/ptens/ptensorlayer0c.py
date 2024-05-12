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

import ptens_base as pb 
import ptens.ptensorsc as ptensorsc
import ptens.ptensor0c as ptensor0c


class ptensorlayer0c(ptensorlayerc):

    @classmethod
    def zeros(self,atoms,nc,device='cpu'):
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(nc,int)
        R=ptensorlayer0c(torch.zeros([len(atoms),nc],device=device))
        R.atoms=atoms
        return R

    @classmethod
    def randn(self,atoms,nc,device='cpu'):
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(nc,int)
        R=ptensorlayer0c(torch.randn([len(atoms),nc],device=device))
        R.atoms=atoms
        return R

    @classmethod
    def from_matrix(self,atoms,M):
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(M,torch.Tensor)
        assert M.dim()==2
        assert M.size(0)==atoms.tsize0()
        R=ptensorlayer0c(M)
        R.atoms=atoms
        return R

    def clone(self):
        r=ptensorlayer0c(super().clone())
        r.atoms=self.atoms
        return r


    # ----- Access -------------------------------------------------------------------------------------------


    def getk(self):
        return 0
    
    def __len__(self):
        return len(self.atoms)
    
    def get_nc(self):
        return self.size(1)
    
    def __getitem__(self,i):
        assert i<len(self)
        return ptensor0c.from_matrix(self.atoms[i],torch.Tensor(self)[i])


    # ---- Message passing -----------------------------------------------------------------------------------
    

    @classmethod
    def linmaps(self,x):
        if isinstance(x,ptensorlayer0c):
           return x

    @classmethod
    def gather(self,x,S):
        return Ptensorsb_Gather0Fn.apply(x,S)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "ptensorlayer0c(len="+str(self.size(0))+",nc="+str(self.size(1))+")"

    def __str__(self):
        r=""
        for i in range(len(self)):
            r=r+str(self[i])+"\n\n"
        return r













    # ---- Operations ----------------------------------------------------------------------------------------


#     def __add__(self,y):
#         assert self.size==y.size
#         assert self.atoms==y.atoms
#         r=self.clone()
#         r+=y
#         return r


