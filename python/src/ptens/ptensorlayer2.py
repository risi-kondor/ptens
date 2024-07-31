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
from ptens_base import ptensors0 as _ptensors0
from ptens_base import ptensors1 as _ptensors1
from ptens_base import ptensors2 as _ptensors2



class ptensorlayer2(p.ptensorlayer):

    @classmethod
    def make(self,atoms,M):
        if isinstance(atoms,list):
           atoms=pb.atomspack(atoms)
        assert isinstance(atoms,pb.atomspack)
        R=ptensorlayer2(M)
        R.atoms=atoms
        return R

    @classmethod
    def zeros(self,atoms,nc,device='cpu'):
        if isinstance(atoms,list):
           atoms=pb.atomspack(atoms)
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.zeros([atoms.nrows2(),nc],device=device))

    @classmethod
    def randn(self,atoms,nc,device='cpu'):
        if isinstance(atoms,list):
           atoms=pb.atomspack(atoms)
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.randn([atoms.nrows2(),nc],device=device))

    @classmethod
    def sequential(self,atoms,nc,device='cpu'):
        if isinstance(atoms,list):
           atoms=pb.atomspack(atoms)
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.tensor([i for i in range (0,atoms.nrows2()*nc)],
                                            dtype=torch.float,device=device).reshape(atoms.nrows2(),nc))
    @classmethod
    def from_matrix(self,atoms,M):
        if isinstance(atoms,list):
           atoms=pb.atomspack(atoms)
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(M,torch.Tensor)
        assert M.dim()==2
        assert M.size(0)==atoms.nrows2()
        return self.make(atoms,M)

    def backend(self):
        return _ptensors2.view(self.atoms,self)

    def zeros_like(self):
        return ptensorlayer0.zeros(self.atoms,self.get_nc(),device=self.device)
    

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
        return p.ptensor2.from_tensor(self.atoms[i],torch.Tensor(self)[offs:offs+n,:].reshape(k,k,nc))


    # ---- Linmaps -------------------------------------------------------------------------------------------
    

    @classmethod
    def linmaps(self,x):
        return ptensorlayer2_linmapsFn.apply(x)


    # ---- Message passing -----------------------------------------------------------------------------------


    @classmethod
    def gather(self,atoms,x,*args):
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(x,p.ptensorlayer)
        if len(args)==0:
            map=pb.layer_map.overlaps_map(atoms,x.atoms)
        else:
            map=args[0]
        return ptensorlayer2_gatherFn.apply(atoms,x,map)


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
            R=torch.zeros([N,2*nc],device=x.device)
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
            R=torch.zeros([N,k,3*nc],device=x.device)
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
            R=torch.zeros([N,k,k,2*nc],device=x.device)
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
            R=torch.zeros([N,k,k,3*nc],device=x.device)
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
            R=torch.zeros([N,k,k,2*nc],device=x.device)
            R[:,:,:,0:nc]=T
            R[:,:,:,nc:2*nc]=torch.einsum("bijc->bjic",T)
            return self.make(x.atoms,R.reshape(N*k*k,2*nc))
        else:
            return ptensorlayer2_broadcast2Fn.apply(x)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "ptensorlayer2(len="+str(len(self.atoms))+",nc="+str(self.size(1))+")"

    def __str__(self,indent=""):
        r=indent+"Ptensorlayer2:\n"
        r=r+self.backend().__str__(indent+"  ")
        return r



# ---- Autograd functions --------------------------------------------------------------------------------------------


class ptensorlayer2_linmapsFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        r=ptensorlayer2.zeros(x.atoms,x.get_nc()*([2,5,15][x.getk()]),device=x.device)
        r.backend().add_linmaps(x.backend())
        ctx.x=x
        return r

    @staticmethod
    def backward(ctx,g):
        r=ctx.x.zeros_like()
        r.backend().add_linmaps_back(g.backend())
        return r


class ptensorlayer2_gatherFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,atoms,x,map):
        r=ptensorlayer2.zeros(atoms,x.get_nc()*([2,5,15][x.getk()]),device=x.device)
        r.backend().add_gather(x.backend(),map)
        ctx.x=x
        ctx.tmap=map
        return r

    @staticmethod
    def backward(ctx,g):
        r=ctx.x.zeros_like()
        r.backend().add_gather_back(g.backend(),ctx.map)
        return r


# ------------------------------------------------------------------------------------------------------------


class ptensorlayer2_reduce0Fn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x):
        ctx.x=x
        r=p.ptensorlayer0.zeros(x.atoms,x.get_nc()*2,device=x.device)
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
        r=p.ptensorlayer1.zeros(x.atoms,x.get_nc()*3,device=x.device)
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
        r=p.ptensorlayer2.zeros(x.atoms,x.get_nc()*2,device=x.device)
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
        r=p.ptensorlayer2.zeros(x.atoms,x.get_nc()*3,device=x.device)
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
        r=p.ptensorlayer2.zeros(x.atoms,x.get_nc()*2,device=x.device)
        r.backend().broadcast2(x)
        return r

    @staticmethod
    def backward(ctx,g):
        r=p.ptensorlayer2.zeros(ctx.x.atoms,ctx.x.get_nc())
        g.backend().add_reduce2_shrink_to(r)
        return r

