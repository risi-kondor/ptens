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
        if isinstance(atoms,list):
           atoms=pb.atomspack(atoms)
        assert isinstance(atoms,pb.atomspack)
        R=ptensorlayer1(M)
        R.atoms=atoms
        return R

    @classmethod
    def zeros(self,atoms,nc,device='cpu'):
        if isinstance(atoms,list):
           atoms=pb.atomspack(atoms)
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.zeros([atoms.nrows1(),nc],device=device))

    @classmethod
    def randn(self,atoms,nc,device='cpu'):
        if isinstance(atoms,list):
           atoms=pb.atomspack(atoms)
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.randn([atoms.nrows1(),nc],device=device))

    @classmethod
    def sequential(self,atoms,nc,device='cpu'):
        if isinstance(atoms,list):
           atoms=pb.atomspack(atoms)
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.tensor([i for i in range (0,atoms.nrows1()*nc)],
                                            dtype=torch.float,device=device).reshape(atoms.nrows1(),nc))

    @classmethod
    def from_matrix(self,atoms,M):
        if isinstance(atoms,list):
           atoms=pb.atomspack(atoms)
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(M,torch.Tensor)
        assert M.dim()==2
        assert M.size(0)==atoms.nrows1()
        return self.make(atoms,M)

    def zeros_like(self):
        return ptensorlayer1.zeros(self.atoms,self.get_nc(),device=self.device)
    
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
        return ptensorlayer1_linmapsFn.apply(x)


    # ---- Message passing -----------------------------------------------------------------------------------


    @classmethod
    def gather(self,atoms,x,*args):
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(x,p.ptensorlayer)
        if len(args)==0:
            map=pb.layer_map.overlaps_map(atoms,x.atoms)
        else:
            map=args[0]
        return ptensorlayer1_gatherFn.apply(atoms,x,map)


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
        if not hasattr(self,'atoms'): 
            return super().__repr__()
        return "ptensorlayer1(len="+str(len(self.atoms))+",nc="+str(self.size(1))+")"

    def __str__(self,indent=""):
        if not hasattr(self,'atoms'): 
            return super().__str__()
        r=indent+"Ptensorlayer1:\n"
        r=r+self.backend().__str__(indent+"  ")
        return r


# ---- Autograd functions --------------------------------------------------------------------------------------------


class ptensorlayer1_linmapsFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        r=ptensorlayer1.zeros(x.atoms,x.get_nc()*([1,2,5][x.getk()]),device=x.device)
        r.backend().add_linmaps(x.backend())
        ctx.x=x
        return r

    @staticmethod
    def backward(ctx,g):
        r=ctx.x.zeros_like()
        r.backend().add_linmaps_back(g.backend())
        return r


class ptensorlayer1_reduce0Fn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        r=p.ptensorlayer0.zeros(x.atoms,x.get_nc(),device=x.device)
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
        r=p.ptensorlayer1.zeros(x.atoms,x.get_nc(),device=x.device)
        r.backend().broadcast0(x)
        ctx.r=r
        return r

    @staticmethod
    def backward(ctx,g):
        r=p.ptensorlayer0.zeros(ctx.r.atoms,ctx.r.get_nc())
        g.backend().add_reduce0_to(r)
        return r


class ptensorlayer1_gatherFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,atoms,x,tmap):
        r=ptensorlayer1.zeros(atoms,x.get_nc()*([1,2,5][x.getk()]),device=x.device)
        r.backend().add_gather(x.backend(),tmap)
        ctx.x=x
        ctx.tmap=tmap
        return r

    @staticmethod
    def backward(ctx,g):
        r=ctx.x.zeros_like()
        r.backend().add_gather_back(g.backend(),ctx.tmap)
        return r


