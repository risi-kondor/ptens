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

    def backend(self):
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
        return ptensor1.from_tensor(self.atoms[i],torch.Tensor(self)[offs:offs+n,:])


    # ---- Linmaps -------------------------------------------------------------------------------------------
    

    @classmethod
    def linmaps(self,x):
        nc=x.get_nc()
        if isinstance(x,p.ptensorlayer0):
            return self.broadcast0(x)
        if isinstance(x,p.ptensorlayer1):
            r=ptensorlayer1.zeros(x.atoms,2*nc)
            r[:,0:nc]=self.broadcast0(x.reduce0())
            r[:,nc:2*nc]=x
            return r
        if isinstance(x,p.ptensorlayer2):
            r=ptensorlayer1.zeros(x.atoms,5*nc)
            r[:,0:2*nc]=self.broadcast0(x.reduce0())
            r[:,2*nc:5*nc]=x.reduce1()
            return r



    # ---- Message passing -----------------------------------------------------------------------------------


    @classmethod
    def gather(self,atoms,x,map):
        return Ptensorsb_Gather0Fn.apply(x,S)


    # ---- Reductions -----------------------------------------------------------------------------------------


    def tensorize(self):
        assert self.atoms.is_constk()
        k=self.atoms.constk()
        return self.unsqueeze(0).reshape(int(self.size(0)/k),k,self.size(1))


    def reduce0(self):
        if self.atoms.is_constk():
            k=self.atoms.constk()
            return p.ptensorlayer0.make(self.atoms,self.tensorize().sum(dim=1))
        else:
            return ptensorlayer1_reduce0Fn.apply(self)

    def reduce1(self):
        return self


    # ---- Broadcasting ---------------------------------------------------------------------------------------


    @classmethod
    def broadcast0(self,x):
        if x.atoms.is_constk():
            return self.make(x.atoms,x.unsqueeze(1).expand(x.size(0),x.atoms.constk(),x.size(1)).reshape(x.size(0)*x.atoms.constk(),x.size(1)))
        else:
            return ptensorlayer1_broadcast0Fn.apply(x)

    @classmethod
    def broadcast1(self,x):
        return x


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "ptensorlayer1(len="+str(len(self.atoms))+",nc="+str(self.size(1))+")"

    def __str__(self):
        r=""
        for i in range(len(self)):
            r=r+str(self[i])+"\n"
        return r



# ---- Autograd functions --------------------------------------------------------------------------------------------


class ptensorlayer1_reduce0Fn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        r=p.ptensorlayer0.zeros(x.atoms,x.get_nc())
        x.backend().add_reduce0_to(r)
        return r

    @staticmethod
    def backward(ctx,g):
        r=p.ptensorlayer1.zeros(ctx.x.atoms,ctx.x.get_nc())
        r.backend().broadcast0(g)
        return r


class ptensorlayer1_broadcast0Fn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x):
        r=p.ptensorlayer1.zeros(x.atoms,x.get_nc())
        r.backend().broadcast0(x)
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx,g):
        r=p.ptensorlayer0.zeros(ctx.r.atoms,ctx.r.get_nc())
        g.backend().add_reduce0_to(r)
        return r





#     def __init__(self,atoms,M):
#         assert isinstance(atoms,pb.atomspack)
#         assert isinstance(M,torch.Tensor)
#         assert M.dim()==2
#         assert M.size(0)==atoms.nrows1()
#         R=ptensorlayer1(M)
#         R.atoms=atoms
#         return R

