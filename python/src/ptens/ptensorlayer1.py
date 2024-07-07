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
from ptens_base import ptensors0 as _ptensors0
from ptens_base import ptensors1 as _ptensors1
from ptens_base import ptensors2 as _ptensors2

import ptens as p
import ptens.ptensorlayer as ptensorlayer
import ptens.ptensor1 as ptensor1


class ptensorlayer1(ptensorlayer):

    @classmethod
    def make(self,atoms,M):
        R=ptensorlayer1(M)
        R.atoms=atoms
        return R

    @classmethod
    def zeros(self,atoms,nc,device='cpu'):
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.zeros([atoms.nrows1(),nc],device=device))

    @classmethod
    def randn(self,atoms,nc,device='cpu'):
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.randn([atoms.nrows1(),nc],device=device))

    @classmethod
    def from_matrix(self,atoms,M):
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(M,torch.Tensor)
        assert M.dim()==2
        assert M.size(0)==atoms.nrows1()
        return self.make(atoms,M)

    def as_ptensors1(self):
        return _ptensors1.view(self.atoms,self)


    # ----- Access -------------------------------------------------------------------------------------------


    def getk(self):
        return 1
    
    def __len__(self):
        return len(self.atoms)
    
    def get_nc(self):
        return self.size(1)
    
    def __getitem__(self,i):
        assert i<len(self)
        offs=self.atoms.row_offset1(i)
        n=self.atoms.nrows1(i)
        return ptensor1.from_matrix(self.atoms[i],torch.Tensor(self)[offs:offs+n])


    # ---- Linmaps -------------------------------------------------------------------------------------------
    

    @classmethod
    def linmaps(self,x):
        nc=x.get_nc()
        if isinstance(x,p.ptensorlayer0):
            r=p.ptensorlayer1.zeros(x.atoms,nc)
            r[:,:]=r.broadcast0(x)
            return r
        if isinstance(x,p.ptensorlayer1):
            r=p.ptensorlayer1.zeros(x.atoms,2*nc)
            r[:,0:nc]=r.broadcast0(x.reduce0())
            r[:,nc:2*nc]=x
            return r


    # ---- Message passing -----------------------------------------------------------------------------------


    @classmethod
    def gather(self,atoms,x,map):
        return Ptensorsb_Gather0Fn.apply(x,S)


    # ---- Reductions -----------------------------------------------------------------------------------------


    def reduce0(self):
        if self.atoms.is_constk():
            k=self.atoms.constk()
            return self.reshape(size(0)/k,k,size(1)).sum(dim=1)
        else:
            r=torch.zeros([self.atoms.nrows0(),self.get_nc()])
            self.as_ptensors1().add_reduce0_to(r)
            return r


    # ---- Broadcasting ---------------------------------------------------------------------------------------


    def broadcast0(self,x):
        if self.atoms.is_constk():
            return x.unsqueeze(1).expand(x.size(0),self.atoms.constk(),x.size(1))
        else:
            r=torch.zeros([self.atoms.nrows1(),x.dim(1)])
            self.as_ptensors1().broadcast0(x)
            return r


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "ptensorlayer1(len="+str(len(self.atoms))+",nc="+str(self.size(1))+")"

    def __str__(self):
        r=""
        for i in range(len(self)):
            r=r+str(self[i])+"\n"
        return r





#     def __init__(self,atoms,M):
#         assert isinstance(atoms,pb.atomspack)
#         assert isinstance(M,torch.Tensor)
#         assert M.dim()==2
#         assert M.size(0)==atoms.nrows1()
#         R=ptensorlayer1(M)
#         R.atoms=atoms
#         return R

