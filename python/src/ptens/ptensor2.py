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
from ptens.ptensor import ptensor


class ptensor2(ptensor):

    @classmethod
    def zeros(cls,atoms,_nc,device='cpu'):
        return cls.make(atoms,torch.zeros([len(atoms),len(atoms),_nc],device=device))

    @classmethod
    def randn(cls,atoms,_nc,device='cpu'):
        return cls.make(atoms,torch.randn([len(atoms),len(atoms),_nc],device=device))

    @classmethod
    def sequential(cls,atoms,nc,device='cpu'):
        assert isinstance(nc,int)
        return cls.make(atoms,torch.tensor([i for i in range (0,len(atoms)*len(atoms)*nc)],
                                            dtype=torch.float,device=device).reshape(len(atoms),len(atoms),nc))

    @classmethod
    def from_tensor(cls, atoms, M):
        return cls.make(atoms,M)

    def backend(self):
        return pb.ptensor2.view(self.atoms,self)


    # ----- Access -------------------------------------------------------------------------------------------


    def getd(self):
        return self.size(0)

    def get_nc(self):
        return self.size(2)


    # ---- Linmaps -------------------------------------------------------------------------------------------


    @classmethod
    def linmaps(self,x):
        nc=x.get_nc()
        if isinstance(x,p.ptensor0):
            return self.broadcast0(x)
        if isinstance(x,p.ptensor1):
            r=self.zeros(x.atoms,5*nc)
            r[:,:,0:2*nc]=self.broadcast0(x.reduce0())
            r[:,:,2*nc:5*nc]=self.broadcast1(x)
            return r
        if isinstance(x,p.ptensor2):
            r=self.zeros(x.atoms,15*nc)
            r[:,:,0:4*nc]=self.broadcast0(x.reduce0())
            r[:,:,4*nc:13*nc]=self.broadcast1(x.reduce1())
            r[:,:,13*nc:15*nc]=self.broadcast2(x)
            return r


    # ---- Reductions ---------------------------------------------------------------------------------------


    def reduce0(self):
        d=self.size(0)
        nc=self.size(2)
        R=torch.zeros([2*nc])
        R[0:nc]+=self.sum(0).sum(0)
        R[nc:2*nc]+=torch.einsum("iic->c",self)
        return p.ptensor0.make(self.atoms,R)

    def reduce1(self):
        d=self.size(0)
        nc=self.size(2)
        R=torch.zeros([d,3*nc])
        R[:,0:nc]+=self.sum(0)
        R[:,nc:2*nc]+=self.sum(1)
        R[:,2*nc:3*nc]+=torch.einsum("iic->ic",self)
        return p.ptensor1.make(self.atoms,R)


    # ---- Broadcasting ---------------------------------------------------------------------------------------


    @classmethod
    def broadcast0(self,x):
        assert x.dim()==1
        k=len(x.atoms)
        nc=x.size(0)
        R=torch.zeros([k,k,2*nc])
        S=x.unsqueeze(0).expand(k,nc)
        R[:,:,0:nc]=S.unsqueeze(0).expand(k,k,nc)
        U=torch.t(R[:,:,nc:2*nc].diagonal(0,0,1))
        U+=S
        return self.make(x.atoms,R)

    @classmethod
    def broadcast1(self,x):
        assert x.dim()==2
        k=x.size(0)
        nc=x.size(1)
        R=torch.zeros([k,k,3*nc])
        R[:,:,0:nc]=x.unsqueeze(0).expand(k,k,nc)
        R[:,:,nc:2*nc]=x.unsqueeze(1).expand(k,k,nc)
        U=torch.t(R[:,:,2*nc:3*nc].diagonal())
        U+=x
        return self.make(x.atoms,R)

    @classmethod
    def broadcast2(self,x):
        assert x.dim()==3
        k=x.size(0)
        nc=x.size(2)
        R=torch.zeros([k,k,2*nc])
        R[:,:,0:nc]=x
        R[:,:,nc:2*nc]=torch.einsum("ijc->jic",x)
        return self.make(x.atoms,R)



    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "<ptensor1(atoms="+str(self.atoms)+",nc="+str(self.size(1))+")>"

    def __str__(self):
        return self.backend().str()
