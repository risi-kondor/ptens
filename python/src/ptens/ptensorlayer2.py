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
import ptens.ptensor2 as ptensor2


class ptensorlayer2(ptensorlayer):

    @classmethod
    def make(self,atoms,M):
        R=ptensorlayer2(M)
        R.atoms=atoms
        return R

    @classmethod
    def zeros(self,atoms,nc,device='cpu'):
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.zeros([atoms.nrows2(),nc],device=device))

    @classmethod
    def randn(self,atoms,nc,device='cpu'):
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.randn([atoms.nrows2(),nc],device=device))

    @classmethod
    def from_matrix(self,atoms,M):
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(M,torch.Tensor)
        assert M.dim()==2
        assert M.size(0)==atoms.nrows2()
        return self.make(atoms,M)

    def backend(self):
        return _ptensors2.view(self.atoms,self)


    # ----- Access -------------------------------------------------------------------------------------------


    def getk(self):
        return 2
    
    def __len__(self):
        return len(self.atoms)
    
    def get_nc(self):
        return self.size(1)
    
    def __getitem__(self,i):
        assert i<len(self)
        offs=self.atoms.row_offset2(i)
        n=self.atoms.nrows2(i)
        k=self.atoms.nrows1(i)
        nc=self.size(1)
        return ptensor2.from_tensor(self.atoms[i],torch.Tensor(self)[offs:offs+n,:].reshape(k,k,nc))


    # ---- Linmaps -------------------------------------------------------------------------------------------
    

    @classmethod
    def linmaps(self,x):
        nc=x.get_nc()
        if isinstance(x,p.ptensorlayer0):
            return self.broadcast0(x)
        if isinstance(x,p.ptensorlayer1):
            r=ptensorlayer2.zeros(x.atoms,5*nc)
            r[:,0:2*nc]=self.broadcast0(x.reduce0())
            r[:,2*nc:5*nc]=self.broadcast1(x)
            return r
        if isinstance(x,p.ptensorlayer2):
            r=ptensorlayer2.zeros(x.atoms,15*nc)
            r[:,0:4*nc]=self.broadcast0(x.reduce0())
            r[:,4*nc:13*nc]=self.broadcast1(x.reduce1())
            r[:,13*nc:15*nc]=self.broadcast2(x)
            return r


    # ---- Message passing -----------------------------------------------------------------------------------


    @classmethod
    def gather(self,atoms,x,map):
        return Ptensorsb_Gather0Fn.apply(x,S)


    # ---- Reductions -----------------------------------------------------------------------------------------


    def tensorize(self):
        assert self.atoms.is_constk()
        k=self.atoms.constk()
        return self.unsqueeze(0).reshape(int(self.size(0)/k/k),k,k,self.size(1))


    def reduce0(self):
        if self.atoms.is_constk():
            T=self.tensorize()
            N=T.size(0)
            k=T.size(1)
            nc=T.size(3)
            R=torch.zeros([N,2*nc])
            R[:,0:nc]=T.reshape(N,k*k,nc).sum(1)
            R[:,nc:2*nc]=torch.einsum("ijjc->ic",T)
            return p.ptensorlayer0.make(self.atoms,R)
        else:
            return ptensorlayer2_reduce0Fn.apply(self)

    def reduce1(self):
        if self.atoms.is_constk():
            T=self.tensorize()
            N=T.size(0)
            k=T.size(1)
            nc=T.size(3)
            R=torch.zeros([N,k,3*nc])
            R[:,:,0:nc]=T.sum(1)
            R[:,:,nc:2*nc]=T.sum(2)
            R[:,:,2*nc:3*nc]+=torch.einsum("biic->bic",T)
            return p.ptensorlayer1.make(self.atoms,R.reshape(N*k,3*nc))
        else:
            return ptensorlayer2_reduce1Fn.apply(self)

    def reduce2(self): 
        return self


    # ---- Broadcasting ---------------------------------------------------------------------------------------


    @classmethod
    def broadcast0(self,x):
        if x.atoms.is_constk():
            assert x.dim()==2
            N=x.size(0)
            k=x.atoms.constk()
            nc=x.size(1)
            R=torch.zeros([N,k,k,2*nc])
            S=x.unsqueeze(1).expand(N,k,nc)
            R[:,:,:,0:nc]=S.unsqueeze(1).expand(N,k,k,nc)
            U=R[:,:,:,nc:2*nc].diagonal(0,1,2).permute(0,2,1)
            U+=S
            return self.make(x.atoms,R.reshape(N*k*k,2*nc))
        else:
            return ptensorlayer2_broadcast0Fn.apply(x)

    @classmethod
    def broadcast1(self,x):
        if x.atoms.is_constk():
            T=x.tensorize()
            N=T.size(0)
            k=T.size(1)
            nc=T.size(2)
            R=torch.zeros([N,k,k,3*nc])
            R[:,:,:,0:nc]=T.unsqueeze(1).expand(N,k,k,nc)
            R[:,:,:,nc:2*nc]=T.unsqueeze(2).expand(N,k,k,nc)
            U=R[:,:,:,2*nc:3*nc].diagonal(0,1,2).permute(0,2,1)
            U+=T
            return self.make(x.atoms,R.reshape(N*k*k,3*nc))
        else:
            return ptensorlayer2_broadcast1Fn.apply(x)

    @classmethod
    def broadcast2(self,x):
        if x.atoms.is_constk():
            T=x.tensorize()
            N=T.size(0)
            k=T.size(1)
            nc=T.size(3)
            R=torch.zeros([N,k,k,2*nc])
            R[:,:,:,0:nc]=T
            R[:,:,:,nc:2*nc]=torch.einsum("bijc->bjic",T)
            return self.make(x.atoms,R.reshape(N*k*k,2*nc))
        else:
            return ptensorlayer2_broadcast2Fn.apply(x)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "ptensorlayer1(len="+str(len(self.atoms))+",nc="+str(self.size(1))+")"

    def __str__(self):
        r=""
        for i in range(len(self)):
            r=r+str(self[i])+"\n"
        return r


# ---- Autograd functions --------------------------------------------------------------------------------------------


class ptensorlayer2_reduce0Fn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        r=p.ptensorlayer0.zeros(x.atoms,x.get_nc()*2)
        x.backend().add_reduce0_to(r)
        return r

    @staticmethod
    def backward(ctx,g):
        r=p.ptensorlayer2.zeros(ctx.x.atoms,ctx.x.get_nc())
        r.backend().broadcast0_shrink(g)
        return r


class ptensorlayer2_reduce1Fn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        r=p.ptensorlayer1.zeros(x.atoms,x.get_nc()*3)
        x.backend().add_reduce1_to(r)
        return r

    @staticmethod
    def backward(ctx,g):
        r=p.ptensorlayer2.zeros(ctx.x.atoms,ctx.x.get_nc())
        r.backend().broadcast1_shrink(g)
        return r


class ptensorlayer2_broadcast0Fn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        r=p.ptensorlayer2.zeros(x.atoms,x.get_nc()*2)
        r.backend().broadcast0(x)
        return r

    @staticmethod
    def backward(ctx,g):
        r=p.ptensorlayer0.zeros(ctx.x.atoms,ctx.x.get_nc())
        g.backend().add_reduce0_shrink_to(r)
        return r


class ptensorlayer2_broadcast1Fn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        r=p.ptensorlayer2.zeros(x.atoms,x.get_nc()*3)
        r.backend().broadcast1(x)
        return r

    @staticmethod
    def backward(ctx,g):
        r=p.ptensorlayer1.zeros(ctx.x.atoms,ctx.x.get_nc())
        g.backend().add_reduce1_shrink_to(r)
        return r


class ptensorlayer2_broadcast2Fn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        r=p.ptensorlayer2.zeros(x.atoms,x.get_nc()*2)
        r.backend().broadcast2(x)
        return r

    @staticmethod
    def backward(ctx,g):
        r=p.ptensorlayer2.zeros(ctx.x.atoms,ctx.x.get_nc())
        g.backend().add_reduce2_shrink_to(r)
        return r



#     def __init__(self,atoms,M):
#         assert isinstance(atoms,pb.atomspack)
#         assert isinstance(M,torch.Tensor)
#         assert M.dim()==2
#         assert M.size(0)==atoms.nrows1()
#         R=ptensorlayer1(M)
#         R.atoms=atoms
#         return R

    #def clone(self):
    #    r=ptensorlayer2(super().clone())
    #    r.atoms=self.atoms
    #    return r

