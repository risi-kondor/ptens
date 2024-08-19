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
import ptens.cptensorlayer as cptensorlayer


class cptensorlayer2(cptensorlayer):

    @classmethod
    def make(self,atoms,M):
        assert isinstance(atoms,pb.catomspack)
        R=cptensorlayer2(M)
        R.atoms=atoms
        return R

    @classmethod
    def zeros(self,atoms,nc,device='cpu'):
        assert isinstance(atoms,pb.catomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.zeros([len(atoms),atoms.nvecs(),nc],device=device))

    @classmethod
    def randn(self,atoms,nc,device='cpu'):
        assert isinstance(atoms,pb.catomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.randn([len(atoms),atoms.nvecs(),nc],device=device))

    @classmethod
    def sequential(self,atoms,nc,device='cpu'):
        assert isinstance(atoms,pb.catomspack)
        assert isinstance(nc,int)
        return self.make(atoms,torch.tensor([i for i in range (0,len(atoms)*atoms.nvecs()*nc)],
                                            dtype=torch.float,device=device).reshape(len(atoms),atoms.nvecs(),nc))

    @classmethod
    def from_tensor(self,atoms,M):
        assert isinstance(atoms,pb.catomspack)
        assert isinstance(M,torch.Tensor)
        assert M.dim()==3
        assert M.size(0)==len(atoms)
        assert M.size(1)==atoms.nvecs()
        return self.make(atoms,M)

    def zeros_like(self):
        return cptensorlayer2.zeros(self.atoms,self.get_nc(),device=self.device)
    
    def backend(self):
        return pb.cptensors1.view(self.atoms,self)


    # ----- Access -------------------------------------------------------------------------------------------


    def getk(self):
        return 2
    
    def __len__(self):
        return len(self.atoms)
    
    def get_nvecs(self):
        return self.size(1)
    
    def get_nc(self):
        return self.size(2)
    

    # ---- Linmaps -------------------------------------------------------------------------------------------
    

    @classmethod
    def linmaps(self,x):
        return cptensorlayer2_linmapsFn.apply(x)


    # ---- Message passing -----------------------------------------------------------------------------------


    @classmethod
    def gather(self,atoms,x,*args):
        assert isinstance(atoms,pb.catomspack)
        assert isinstance(x,p.cptensorlayer)
        if len(args)==0:
            map=pb.layer_map.overlaps_map(atoms,x.atoms)
        else:
            map=args[0]
        return cptensorlayer2_gatherFn.apply(atoms,x,map)


    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        if not hasattr(self,'atoms'): 
            return super().__repr__()
        return "cptensorlayer2(len="+str(len(self.atoms))+",nvecs="+str(self.size(1))+",nc="+str(self.size(2))+")"

    def __str__(self,indent=""):
        if not hasattr(self,'atoms'): 
            return super().__str__()
        r=indent+"Cptensorlayer2:\n"
        r=r+self.backend().__str__(indent+"  ")
        return r


# ---- Autograd functions --------------------------------------------------------------------------------------------


class cptensorlayer2_linmapsFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x):
        r=cptensorlayer2.zeros(x.atoms,x.get_nc()*([2,5,15][x.getk()]),device=x.device)
        r.backend().add_linmaps(x.backend())
        ctx.x=x
        return r

    @staticmethod
    def backward(ctx,g):
        r=ctx.x.zeros_like()
        r.backend().add_linmaps_back(g.backend())
        return r


class cptensorlayer2_gatherFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,atoms,x,tmap):
        r=cptensorlayer2.zeros(atoms,x.get_nc()*([2,5,15][x.getk()]),device=x.device)
        r.backend().add_gather(x.backend(),tmap)
        ctx.x=x
        ctx.tmap=tmap
        return r

    @staticmethod
    def backward(ctx,g):
        r=ctx.x.zeros_like()
        r.backend().add_gather_back(g.backend(),ctx.tmap)
        return r


