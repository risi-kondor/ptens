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

    def zeros_like(self):
        return ptensorlayer0.zeros(self.atoms,self.get_nc(),device=self.device)
    
    def backend(self):
        return pb.ptensors0.view(self.atoms,self)


    # ----- Access -------------------------------------------------------------------------------------------


    def getk(self):
        return 0
    
    def __len__(self):
        return len(self.atoms)
    
    def get_nc(self):
        return self.size(1)
    
    def __getitem__(self,i):
        assert i<len(self)
        return ptensor0.from_tensor(self.atoms[i],torch.Tensor(self)[i])


    # ---- Linmaps -------------------------------------------------------------------------------------------
    

    @classmethod
    def linmaps(self,x):
        if isinstance(x,ptensorlayer0):
            return x
        if isinstance(x,p.ptensorlayer1):
            return x.reduce0()
        if isinstance(x,p.ptensorlayer2):
            return x.reduce0()


    # ---- Message passing -----------------------------------------------------------------------------------


    @classmethod
    def gather(self,atoms,x,*args):
        assert isinstance(atoms,pb.atomspack)
        assert isinstance(x,p.ptensorlayer)
        if len(args)==0:
            return ptensorlayer0.gather(atoms,x,pb.tensor_map.overlaps_map(atoms,x.atoms)) 
        else:
            assert isinstance(args[0],pb.tensor_map)
            return ptensorlayer0_gatherFn.apply(atoms,x,args[0])
        

    # ---- I/O ----------------------------------------------------------------------------------------------


    def __repr__(self):
        return "ptensorlayer0(len="+str(self.size(0))+",nc="+str(self.size(1))+")"

    def __str__(self):
        r=""
        for i in range(len(self)):
            r=r+str(self[i])+"\n"
        return r



# ---- Autograd functions --------------------------------------------------------------------------------------------



class ptensorlayer0_gatherFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx,atoms,x,tmap):
        r=ptensorlayer0.zeros(atoms,x.get_nc()*([1,1,2][x.getk()]))
        r.backend().add_gather(x.backend(),tmap)
        ctx.x=x
        ctx.tmap=tmap
        return r

    @staticmethod
    def backward(ctx,g):
        r=ctx.x.zeros_like()
        r.backend().add_gather_back(g.backend(),ctx.tmap)
        return r



