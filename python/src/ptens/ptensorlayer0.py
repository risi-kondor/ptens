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
import ptens as p 
import ptens.ptensorlayer as ptensorlayer
import ptens.ptensor0 as ptensor0


class ptensorlayer0(ptensorlayer):

    @classmethod
    def make(self,atoms,M):
        R=ptensorlayer0(M)
        R.atoms=atoms
        return R

    @classmethod
    def zeros(self,atoms,nc,device='cpu'):
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.zeros([len(atoms),nc],device=device))

    @classmethod
    def randn(self,atoms,nc,device='cpu'):
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.randn([len(atoms),nc],device=device))

    @classmethod
    def from_matrix(self,atoms,M):
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(M,torch.Tensor)
        assert M.dim()==2
        assert M.size(0)==atoms.nrows0()
        return self.make(atoms,M)

    def as_ptensors0(self):
        return _ptensors0.view(self.atoms,self)


    # ----- Access -------------------------------------------------------------------------------------------


    def getk(self):
        return 0
    
    def __len__(self):
        return len(self.atoms)
    
    def get_nc(self):
        return self.size(1)
    
    def __getitem__(self,i):
        assert i<len(self)
        return ptensor0.from_matrix(self.atoms[i],torch.Tensor(self)[i])


    # ---- Linmaps -------------------------------------------------------------------------------------------
    

    @classmethod
    def linmaps(self,x):
        if isinstance(x,ptensorlayer0):
            return x
        if isinstance(x,p.ptensorlayer1):
            return self.make(x.atoms,x.reduce0())
        if isinstance(x,p.ptensorlayer1):
            return self.make(x.atoms,x.reduce0())


    # ---- Message passing -----------------------------------------------------------------------------------


    @classmethod
    def gather(self,atoms,x,mmap):
        assert isinstance(x,p.ptensorlayer)

    @classmethod
    def gather_from_overlapping(self,atoms,x):
        assert isinstance(x,p.ptensorlayer)

    @classmethod
    def gather_from_neighbors(self,atoms,x):
        assert isinstance(x,p.ptensorlayer)

        

    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "ptensorlayer0(len="+str(self.size(0))+",nc="+str(self.size(1))+")"

    def __str__(self):
        r=""
        for i in range(len(self)):
            r=r+str(self[i])+"\n"
        return r













    # ---- Operations ----------------------------------------------------------------------------------------


#     def __add__(self,y):
#         assert self.size==y.size
#         assert self.atoms==y.atoms
#         r=self.clone()
#         r+=y
#         return r


#     def __init__(self,atoms,M):
#         assert isinstance(atoms,pb.atomspack)
#         assert isinstance(M,torch.Tensor)
#         assert M.dim()==2
#         assert M.size(0)==atoms.tsize0()
#         super(ptensorlayerc,self).__init__(x)
#         atoms=atoms

    #def clone(self):
    #    r=ptensorlayer0(super().clone())
    #    r.atoms=self.atoms
    #    return r

